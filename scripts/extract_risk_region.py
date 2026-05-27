import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from glioma_sparse.models.factory import build_model
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret import compute_risk_map, extract_topk_patches
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail


# ============================================================
# SETTINGS
# ============================================================

CLASS_ORDER = ["control", "low_grade", "high_grade"]

THUMBNAIL_EXTS = (".jpg", ".jpeg", ".png")
WSI_EXTS = (".ndpi", ".svs", ".tiff", ".tif", ".mrxs")


# ============================================================
# UTIL
# ============================================================

def load_model(checkpoint_path, model_name, device):
    model = build_model(model_name, num_classes=len(CLASS_ORDER))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def is_wsi(path):
    return path.suffix.lower() in WSI_EXTS


def is_thumbnail(path):
    return path.suffix.lower() in THUMBNAIL_EXTS


def generate_thumbnail_if_needed(input_path, tmp_dir):
    if is_wsi(input_path):
        thumb_path = tmp_dir / f"{input_path.stem}_thumb.jpg"
        if not thumb_path.exists():
            print("  generating thumbnail from WSI...")
            create_wsi_thumbnail(input_path, thumb_path)
        return thumb_path
    return input_path


def save_heatmap(risk_map, save_path):
    plt.imshow(risk_map, cmap="hot")
    plt.colorbar()
    plt.title("Risk Map")
    plt.savefig(save_path)
    plt.close()


def save_overlay(image, risk_map, save_path, label_x=1.5, predicted_class_name=None):

    img_rgb = np.array(image.resize((2048, 2048)))

    heatmap = cv2.resize(
        risk_map,
        (2048, 2048),
        interpolation=cv2.INTER_CUBIC
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img_rgb)

    hm = ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=float(risk_map.min()),
        vmax=float(risk_map.max()),
    )

    ax.axis("off")

    # --------------------------------------------------------
    # COLOURBAR
    # --------------------------------------------------------
    cbar = fig.colorbar(hm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([])

    cbar.ax.text(
        0.5, 1.02, "Strong",
        ha="center",
        va="bottom",
        transform=cbar.ax.transAxes,
    )

    cbar.ax.text(
        0.5, -0.02, "Weak",
        ha="center",
        va="top",
        transform=cbar.ax.transAxes,
    )

    label = (
        f"Evidence for '{predicted_class_name}'"
        if predicted_class_name is not None
        else "Class-associated signal"
    )

    cbar.ax.text(
        1.5, 0.5, label,
        rotation=90,
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_grid_risk_overlay(
    image,
    risk_map,
    save_path,
    alpha=1.0,
    predicted_class_name=None,
):

    img = np.array(image)

    h, w = img.shape[:2]
    grid_h, grid_w = risk_map.shape

    cell_h = h / grid_h
    cell_w = w / grid_w

    # normalise to [0, 1]
    risk_norm = (
        risk_map - risk_map.min()
    ) / (
        risk_map.max() - risk_map.min() + 1e-8
    )

    cmap = plt.get_cmap("jet")

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img)
    ax.axis("off")

    # --------------------------------------------------------
    # DRAW LOW RISK FIRST, HIGH RISK LAST
    # --------------------------------------------------------
    cells = sorted(
        [
            (risk_norm[i, j], i, j)
            for i in range(grid_h)
            for j in range(grid_w)
        ],
        key=lambda x: x[0]
    )

    for r, i, j in cells:

        rect = plt.Rectangle(
            (
                int(j * cell_w),
                int(i * cell_h),
            ),
            int(cell_w),
            int(cell_h),
            linewidth=1 + 3 * r,
            edgecolor=cmap(r),
            facecolor="none",
            alpha=alpha,
        )

        ax.add_patch(rect)

    # --------------------------------------------------------
    # COLOURBAR
    # --------------------------------------------------------
    mappable = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.04,
        pad=0.02,
    )

    cbar.set_ticks([])

    cbar.ax.text(
        0.5, 1.02, "Strong",
        ha="center",
        va="bottom",
        transform=cbar.ax.transAxes,
    )

    cbar.ax.text(
        0.5, -0.02, "Weak",
        ha="center",
        va="top",
        transform=cbar.ax.transAxes,
    )

    label = (
        "Evidence for '{}'".format(predicted_class_name)
        if predicted_class_name is not None
        else "Class-associated signal"
    )

    cbar.ax.text(
        1.5,
        0.5,
        label,
        rotation=90,
        ha="center",
        va="center",
        transform=cbar.ax.transAxes,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


# ============================================================
# PER-SLIDE WORKER (shared by single + batch)
# ============================================================

def _process_one_slide(
    thumbnail_path,
    model,
    transform,
    control_img,
    device,
    k,
    output_size,
    patches_dir,
    label_override=None,
    overlay_path=None,
    grid_overlay_path=None,
    extract_patches=True,
):
    """
    Predict, compute risk map, optionally save overlays, extract patches.
    Returns a summary dict for one slide.
    """
    target_img = Image.open(thumbnail_path).convert("RGB")

    # PREDICT
    x = transform(target_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_ORDER[pred_idx]

    label_for_patches = (
        label_override if label_override is not None else pred_label
    )

    # RISK MAP
    risk_map, _target_class = compute_risk_map(
        target_img=target_img,
        control_img=control_img,
        model=model,
        transform=transform,
        target_class=pred_idx,
        device=device,
    )

    # OVERLAYS (optional)
    if grid_overlay_path is not None:
        grid_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        save_grid_risk_overlay(
            target_img,
            risk_map,
            grid_overlay_path,
            predicted_class_name=pred_label,
        )

    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        save_overlay(
            target_img,
            risk_map,
            overlay_path,
            predicted_class_name=pred_label,
        )

    # PATCHES
    n_patches = 0
    if extract_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)
        result = extract_topk_patches(
            thumbnail_path=thumbnail_path,
            risk_map=risk_map,
            k=k,
            label=label_for_patches,
            output_dir=patches_dir,
            output_size=output_size,
        )
        try:
            n_patches = len(result) if result is not None else 0
        except TypeError:
            n_patches = 0

    return {
        "thumbnail_path": str(thumbnail_path),
        "predicted_class": pred_label,
        "label_used": label_for_patches,
        "prob_control": float(probs[0]),
        "prob_low_grade": float(probs[1]),
        "prob_high_grade": float(probs[2]),
        "n_patches_extracted": n_patches,
    }


# ============================================================
# SINGLE-SLIDE ENTRY
# ============================================================

def run_single_inference(
    input_path,
    checkpoint_path,
    control_image_path,
    model_name="resnet34",
    k=5,
    extract_patches=True,
    output_root=None,
    output_size=2048,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_path = Path(input_path)

    run_name = f"risk_output_{input_path.stem}"

    if output_root is None:
        output_dir = input_path.parent / run_name
    else:
        output_dir = Path(output_root) / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Processing: {input_path.name}")

    # --------------------------------------------------------
    # MODEL + TRANSFORM + CONTROL
    # --------------------------------------------------------
    model = load_model(checkpoint_path, model_name, device)
    transform = build_eval_transform()
    control_img = Image.open(control_image_path).convert("RGB")

    # --------------------------------------------------------
    # PREP INPUT
    # --------------------------------------------------------
    thumbnail_path = generate_thumbnail_if_needed(input_path, tmp_dir)

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------
    summary = _process_one_slide(
        thumbnail_path=thumbnail_path,
        model=model,
        transform=transform,
        control_img=control_img,
        device=device,
        k=k,
        output_size=output_size,
        patches_dir=output_dir / "patches",
        overlay_path=output_dir / "overlay.jpg",
        grid_overlay_path=output_dir / "grid_risk_map.jpg",
        extract_patches=extract_patches,
    )

    print(f"\nPrediction: {summary['predicted_class']}")
    print(
        f"Probabilities: control={summary['prob_control']:.3f}, "
        f"low_grade={summary['prob_low_grade']:.3f}, "
        f"high_grade={summary['prob_high_grade']:.3f}"
    )

    print(f"Saved grid risk map: {output_dir / 'grid_risk_map.jpg'}")
    print(f"Saved overlay:       {output_dir / 'overlay.jpg'}")

    if extract_patches:
        print(
            f"Saved {summary['n_patches_extracted']} patches to: "
            f"{output_dir / 'patches'}"
        )

    print("\nDone.")

    return summary


# ============================================================
# BATCH ENTRY
# ============================================================

def run_batch_inference(
    input_dir,
    checkpoint_path,
    control_image_path,
    model_name="resnet34",
    k=5,
    output_size=2048,
    label_override=None,
    save_overlays=False,
    output_root=None,
):
    """
    Walks input_dir for thumbnails (.jpg/.jpeg/.png) and WSIs and runs
    the same per-slide pipeline as run_single_inference on each.

    All extracted patches go to a single flat directory (output_root/patches)
    so the output is directly usable as a Stage B training/inference cohort.
    Overlays, if enabled, go to output_root/overlays with slide-prefixed
    filenames.

    The batch_summary.csv and batch_failures.csv files are append-aware:
    running this function several times with the same output_root (e.g.
    once per molecular class) accumulates a single summary instead of
    overwriting it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = Path(input_dir)

    if output_root is None:
        output_dir = input_dir.parent / f"risk_output_batch_{input_dir.name}"
    else:
        output_dir = Path(output_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = output_dir / "patches"
    overlay_dir = output_dir / "overlays" if save_overlays else None
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nDevice:            {device}")
    print(f"Input directory:   {input_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Label override:    {label_override}")
    print(f"Save overlays:     {save_overlays}\n")

    # --------------------------------------------------------
    # DISCOVER SLIDES
    # --------------------------------------------------------
    files = sorted(p for p in input_dir.rglob("*") if p.is_file())
    slides = [p for p in files if is_thumbnail(p) or is_wsi(p)]

    if not slides:
        print("No slides found in input_dir.")
        return pd.DataFrame()

    print(f"Found {len(slides)} slides\n")

    # --------------------------------------------------------
    # MODEL + TRANSFORM + CONTROL (loaded once)
    # --------------------------------------------------------
    model = load_model(checkpoint_path, model_name, device)
    transform = build_eval_transform()
    control_img = Image.open(control_image_path).convert("RGB")

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------
    summaries = []
    failures = []

    for i, slide in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}] {slide.name}")

        try:
            thumbnail_path = generate_thumbnail_if_needed(slide, tmp_dir)

            stem = slide.stem
            overlay_path = (
                overlay_dir / f"{stem}_overlay.jpg" if save_overlays else None
            )
            grid_overlay_path = (
                overlay_dir / f"{stem}_grid_risk_map.jpg" if save_overlays else None
            )

            summary = _process_one_slide(
                thumbnail_path=thumbnail_path,
                model=model,
                transform=transform,
                control_img=control_img,
                device=device,
                k=k,
                output_size=output_size,
                patches_dir=patches_dir,
                label_override=label_override,
                overlay_path=overlay_path,
                grid_overlay_path=grid_overlay_path,
                extract_patches=True,
            )
            summary["slide_path"] = str(slide)
            summaries.append(summary)

            print(
                f"  predicted: {summary['predicted_class']}, "
                f"label: {summary['label_used']}, "
                f"patches: {summary['n_patches_extracted']}"
            )

        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append({"slide_path": str(slide), "error": str(e)})

    # --------------------------------------------------------
    # SAVE SUMMARY (append-aware)
    # --------------------------------------------------------
    def _append_csv(df, path):
        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)

    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    if not summary_df.empty:
        _append_csv(summary_df, output_dir / "batch_summary.csv")

    if failures:
        _append_csv(pd.DataFrame(failures), output_dir / "batch_failures.csv")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n=== BATCH COMPLETE ===")
    print(f"Processed: {len(summaries)}")
    print(f"Failed:    {len(failures)}")
    print(f"Output:    {output_dir}")

    return summary_df


# ============================================================
# PYCHARM ENTRY
# ============================================================

if __name__ == "__main__":

    # "single"            single slide or single WSI -> overlays + patches
    # "batch"             one folder, one label, accumulate to output_root
    # "batch_all_classes" iterate over molecular-class mapping, all patches
    #                     accumulate to a single output_root for Stage B
    MODE = "single"

    CHECKPOINT = r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\training_output\20260526_0947_resnet34_cw\best_acc.pth"
    CONTROL = r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\ebrains_thumbnails\control\included\86242943-7775-11eb-827d-001a7dda7111.jpg"
    MODEL_NAME = "resnet34"

    # ----------------------------------------------------
    # SINGLE
    # ----------------------------------------------------
    if MODE == "single":
        run_single_inference(
            input_path=r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\testdata\svs\TCGA-06-0206-01Z-00-DX1.5ede33b2-4778-4e33-bb83-17f81bb35aca.svs",
            checkpoint_path=CHECKPOINT,
            control_image_path=CONTROL,
            model_name=MODEL_NAME,
            k=5,
            extract_patches=True,
            output_root=None,
        )

    # ----------------------------------------------------
    # BATCH (one folder, one label)
    # ----------------------------------------------------
    elif MODE == "batch":
        run_batch_inference(
            input_dir=r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\ebrains_thumbnails\oligo_IDHmt_1p19qdel_G2\included",
            checkpoint_path=CHECKPOINT,
            control_image_path=CONTROL,
            model_name=MODEL_NAME,
            k=5,
            label_override="IDH_mut_1p19qCD",
            save_overlays=False,
            output_root=r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\stage_b_training_data",
        )

    # ----------------------------------------------------
    # BATCH OVER ALL MOLECULAR CLASSES (Stage B dataset build)
    # ----------------------------------------------------
    elif MODE == "batch_all_classes":

        BASE_INPUT = Path(r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\ebrains_thumbnails")
        OUTPUT_ROOT = Path(r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\stage_b_training_data")

        # (subfolder, molecular label for Stage B)
        CLASS_MAPPING = [
            ("oligo_IDHmt_1p19qdel_G2/included", "IDH_mut_1p19qCD"),
            ("oligo_IDHmt_1p19qdel_G3/included", "IDH_mut_1p19qCD"),
            ("astro_IDHmt_G2/included",          "IDH_mut"),
            ("astro_IDHmt_G3/included",          "IDH_mut"),
            ("astro_IDHmt_G4/included",          "IDH_mut"),
            ("GBM_IDHwt/included",               "IDH_wt"),
        ]

        for subdir, label in CLASS_MAPPING:
            input_dir = BASE_INPUT / subdir
            if not input_dir.exists():
                print(f"SKIP (not found): {input_dir}")
                continue

            print(f"\n{'=' * 60}")
            print(f"{subdir}  -->  {label}")
            print(f"{'=' * 60}")

            run_batch_inference(
                input_dir=input_dir,
                checkpoint_path=CHECKPOINT,
                control_image_path=CONTROL,
                model_name=MODEL_NAME,
                k=5,
                label_override=label,
                save_overlays=False,
                output_root=OUTPUT_ROOT,
            )

        print("\n=== ALL CLASSES PROCESSED ===\n")

    else:
        raise ValueError(f"Unknown MODE: {MODE}")
