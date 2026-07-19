"""
Build the Stage B region cohort by p95 patch-injection selection.

Design (five-fold, leakage-safe, reusable for end-to-end inference):
  For each of the five folds, that fold's Stage A model extracts ALL of that
  fold's slides (train, val, test). Each region inherits the slide's role in
  that fold. So Stage B is trained per fold on that fold's train+val regions
  and tested on that fold's test regions, and no slide's TEST regions are ever
  produced by a model that trained on it.
  Every extracted region is written to a combined cohort manifest
  (`stageB_manifest.csv`) as well as a fold-specific manifest
  (`manifests/stageB_fold<k>.csv`). The combined manifest is used for
  end-to-end inference, while the per-fold manifests allow Stage B training
  to filter directly by fold. Patches carry the slide id, fold, split and
  region rank in the filename.

Per slide, per fold:
  1. predict grade (default argmax; tuning stays a separate analysis),
  2. compute the injection risk map R for the predicted class, seeded control,
  3. select p95 tiles (>= 95th percentile of R, capped at 4) after tissue filter,
  4. map each tile back to WSI level-0 via the thumbnail JSON sidecar and
     re-extract at high resolution,
  5. write a manifest row.

Run (from repo root):
    python scripts/build_stageB_cohort.py \
        --splits-dir splits \
        --checkpoints-glob "training_output/*_split_0*/best_auc.pth" \
        --control-image data/included/control/<id>.jpg \
        --control-image data/included/control/<id>.jpg \
        --sidecar-root data/ebrains_thumbnails \
        --wsi-root path/to/WHO2021_data \
        --out-dir data/stage_b_cohort_RN50cl

Sidecars and their thumbnails live in <sidecar-root>/<subtype>/included/ and are
resolved by slide id, since the split CSVs point at grade-organised thumbnails
that do not carry the JSON. Molecular subtype is derived from the WSI class
folder (astro_IDHmt_* -> IDH_mt, oligo_IDHmt_1p19qdel_* -> IDH_mt_1p19q,
GBM_IDHwt -> IDH_wt). Control slides are skipped. Patches are written to
patches/fold_<k>/<subtype>/ and the train/val/test role is kept in the manifest.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from glioma_sparse.models.factory import build_model
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import (
    load_mapping, thumbnail_bbox_to_wsi, grid_cell_bbox,
)

import openslide


CLASS_ORDER = ["control", "low_grade", "high_grade"]
GRID = (8, 8)
CONTROL_SHUFFLE_SEED = 42


# ============================================================
# TISSUE FILTER + P95 SELECTION
# ============================================================

def tile_tissue_fraction(thumbnail, row, col, grid):
    rows, cols = grid
    w, h = thumbnail.size
    tw, th = w // cols, h // rows
    tile = thumbnail.crop((col * tw, row * th, col * tw + tw, row * th + th))
    arr = np.array(tile).astype(np.float32) / 255.0
    maxc, minc = arr.max(axis=2), arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-8)
    mask = (maxc < 0.9) & (sat > 0.05)
    return float(mask.sum()) / float(mask.size)


def select_p95_tiles(risk_map, thumbnail, grid, percentile=95, cap=4,
                     min_tissue=0.5):
    """p95 on the full map -> keep >= thresh -> tissue filter -> sort -> cap."""
    rows, cols = grid
    thresh = float(np.percentile(risk_map.flatten(), percentile))
    above = [(r, c) for r in range(rows) for c in range(cols)
             if risk_map[r, c] >= thresh]
    kept = []
    for r, c in above:
        tf = tile_tissue_fraction(thumbnail, r, c, grid)
        if tf >= min_tissue:
            kept.append({"row": r, "col": c, "risk": float(risk_map[r, c]),
                         "tissue_fraction": tf})
    kept.sort(key=lambda d: -d["risk"])
    return kept[:cap], thresh, len(above)


# ============================================================
# MODEL + IO HELPERS
# ============================================================

def load_model(checkpoint_path, model_name, device):
    model = build_model(model_name, num_classes=len(CLASS_ORDER))
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_class(model, transform, thumbnail, device):
    x = transform(thumbnail).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


# cache for recursive filename search under --wsi-root
_WSI_INDEX = {}

# cache for sidecar thumbnails (stem -> thumbnail path with a .json beside it)
_SIDECAR_INDEX = {}


def get_sidecar_thumbnail(stem, sidecar_root):
    """Find the thumbnail whose JSON sidecar we need, by slide id (stem).

    The ebrains thumbnail tree holds both the .jpg and its .json, keyed by the
    slide stem, under <sidecar_root>/<subtype>/included/. Indexed once.
    """
    if sidecar_root not in _SIDECAR_INDEX:
        index = {}
        for jpg in Path(sidecar_root).rglob("*.jpg"):
            if jpg.with_suffix(".json").exists():
                index.setdefault(jpg.stem, jpg)
        _SIDECAR_INDEX[sidecar_root] = index
    return _SIDECAR_INDEX[sidecar_root].get(stem)


def _norm(p):
    """Normalize a stored path so Windows/Unix separators both work."""
    return str(p).replace("\\", "/")


def _build_wsi_index(root):
    """Map filename -> full path for every WSI under root (once)."""
    exts = {".ndpi", ".svs", ".tif", ".tiff", ".mrxs"}
    index = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            index.setdefault(p.name, str(p))
    return index


def resolve_wsi_path(stored_path, args):
    """Turn the sidecar's stored WSI path into a path that exists here.

    Order: (1) stored path as-is, (2) prefix swap, (3) recursive search by
    filename under --wsi-root, (4) flat lookup under --wsi-root. Raises a clear
    error listing what was tried if nothing resolves.
    """
    tried = []

    # 1. stored path unchanged
    if Path(stored_path).exists():
        return stored_path
    tried.append(stored_path)

    # 2. prefix swap (separator-insensitive)
    if args.wsi_old_prefix and args.wsi_new_prefix:
        swapped = _norm(stored_path).replace(
            _norm(args.wsi_old_prefix), _norm(args.wsi_new_prefix), 1)
        swapped = swapped.replace("/", os.sep)
        if Path(swapped).exists():
            return swapped
        tried.append(swapped)

    # 3 + 4. search under --wsi-root
    if args.wsi_root:
        fname = Path(_norm(stored_path)).name

        # recursive search (cached), preserves whatever nesting exists
        if args.wsi_root not in _WSI_INDEX:
            _WSI_INDEX[args.wsi_root] = _build_wsi_index(args.wsi_root)
        hit = _WSI_INDEX[args.wsi_root].get(fname)
        if hit and Path(hit).exists():
            return hit

        # flat lookup directly under the root
        flat = str(Path(args.wsi_root) / fname)
        if Path(flat).exists():
            return flat
        tried.append(f"{args.wsi_root}\\**\\{fname}")

    raise FileNotFoundError(
        "Could not locate WSI for sidecar path.\n  tried:\n    "
        + "\n    ".join(tried)
        + "\n  Pass --wsi-root <folder> to search by filename, or fix "
          "--wsi-old-prefix/--wsi-new-prefix."
    )


def derive_true_labels(folder_name):
    """True labels from the ebrains subtype folder, e.g. 'astro_IDHmt_G2'.

    Returns entity, true_mutation (IDH_mt / IDH_mt_1p19q / IDH_wt),
    true_grade (2/3/4), and true_grade_class (low = grade 2, high = grades 3/4).
    """
    n = folder_name.lower()

    if "gbm" in n or "idhwt" in n:
        entity, mutation = "glioblastoma", "IDH_wt"
    elif "oligo" in n and "1p19q" in n:
        entity, mutation = "oligodendroglioma", "IDH_mt_1p19q"
    elif "astro" in n and "idhmt" in n:
        entity, mutation = "astrocytoma", "IDH_mt"
    else:
        entity, mutation = "unknown", "unknown"

    m = re.search(r"_g(\d)", n)
    if m:
        grade = int(m.group(1))
    elif entity == "glioblastoma":
        grade = 4
    else:
        grade = None

    grade_class = "unknown" if grade is None else ("low" if grade == 2 else "high")

    return {"entity": entity, "true_mutation": mutation,
            "true_grade": grade, "true_grade_class": grade_class,
            "true_subtype_folder": folder_name}


def pred_grade_class(pred_name):
    """Map the Stage A predicted class to a low/high grade class."""
    if pred_name == "low_grade":
        return "low"
    if pred_name == "high_grade":
        return "high"
    return "control"


def load_split_rows(split_csv):
    """Return list of (thumbnail_path, role) for ALL slides in the fold."""
    out = []
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            out.append((Path(row["path"]), row["split"]))
    return out


def resolve_checkpoints(split_csvs, checkpoints_glob):
    """Map split file stem (split_0X) to a checkpoint by the token in its path."""
    import glob as _glob
    ckpts = _glob.glob(checkpoints_glob)
    out = {}
    for sc in split_csvs:
        match = [c for c in ckpts if sc.stem in str(c)]
        if match:
            out[sc.stem] = str(sorted(match)[0])
    return out


def extract_region_highres(slide, mapping, row, col, grid, output_size):
    bbox_thumb = grid_cell_bbox(row, col, grid[0], grid[1], mapping.thumbnail_size)
    wsi_bbox = thumbnail_bbox_to_wsi(mapping, bbox_thumb)
    if wsi_bbox is None:
        return None, None
    wx, wy, ww, wh = wsi_bbox
    patch = slide.read_region((wx, wy), 0, (ww, wh)).convert("RGB")
    if patch.size != (output_size, output_size):
        patch = patch.resize((output_size, output_size), Image.BICUBIC)
    return patch, wsi_bbox


# ============================================================
# PER-SLIDE PROCESS
# ============================================================

MANIFEST_FIELDS = [
    # identity + split
    "slide_id", "fold", "split", "split_csv",
    # true labels (from the ebrains subtype folder)
    "entity", "true_subtype_folder", "true_grade", "true_grade_class",
    "true_mutation",
    # Stage A prediction
    "grade_pred", "pred_grade_class", "grade_class_correct",
    "prob_control", "prob_low_grade", "prob_high_grade",
    # region selection
    "rank", "row", "col", "risk", "p95_threshold", "n_above_p95",
    "tissue_fraction",
    # provenance
    "base_mpp", "target_mpp", "wsi_bbox",
    "patch_path", "thumbnail_path", "wsi_path",
]


def process_slide(thumb_path, role, model, transform, control_img, device,
                  fold_idx, split_csv, args, patches_dir):
    # Stage B only uses tumour slides. Skip control by the class folder the
    # split thumbnail sits in, so a misclassified control stays out.
    if thumb_path.parent.name == "control":
        return []

    # The JSON sidecar and its thumbnail live in the ebrains tree, resolved by
    # slide id (stem). The subtype/grade folder there gives the true labels.
    stem = thumb_path.stem
    sc_thumb = get_sidecar_thumbnail(stem, args.sidecar_root)
    if sc_thumb is None:
        raise RuntimeError(
            f"no sidecar thumbnail for {stem} under {args.sidecar_root}")

    subtype_folder = sc_thumb.parent.parent.name  # e.g. astro_IDHmt_G2
    labels = derive_true_labels(subtype_folder)

    thumbnail = Image.open(sc_thumb).convert("RGB")
    pred_idx, probs = predict_class(model, transform, thumbnail, device)
    pred_name = CLASS_ORDER[pred_idx]
    pgc = pred_grade_class(pred_name)

    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img, model=model,
        transform=transform, grid=GRID, target_class=pred_idx, device=device,
        control_shuffle_seed=CONTROL_SHUFFLE_SEED,
    )

    selected, thresh, n_above = select_p95_tiles(
        risk_map, thumbnail, GRID, percentile=args.percentile,
        cap=args.cap, min_tissue=args.min_tissue)
    if not selected:
        return []

    mapping = load_mapping(sc_thumb)
    if not mapping.has_wsi_mapping():
        raise RuntimeError("no WSI mapping in sidecar (base_mpp was None)")

    wsi_path = resolve_wsi_path(mapping.wsi_path, args)
    slide = openslide.OpenSlide(wsi_path)

    # patches/fold_<k>/<mutation>/  (train/val/test role stays in the manifest)
    dest_dir = patches_dir / f"fold_{fold_idx}" / labels["true_mutation"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_mpp = getattr(mapping, "base_mpp", None)
    target_mpp = getattr(mapping, "target_mpp", None)

    rows = []
    try:
        for rank, sel in enumerate(selected, 1):
            patch, wsi_bbox = extract_region_highres(
                slide, mapping, sel["row"], sel["col"], GRID, args.output_size)
            if patch is None:
                continue
            fname = (f"{stem}_fold{fold_idx}_{role}_rank{rank:02d}"
                     f"_r{sel['row']}c{sel['col']}.jpg")
            patch_path = dest_dir / fname
            patch.save(patch_path, quality=90)

            rows.append({
                "slide_id": stem, "fold": fold_idx, "split": role,
                "split_csv": split_csv.name,
                "entity": labels["entity"],
                "true_subtype_folder": labels["true_subtype_folder"],
                "true_grade": labels["true_grade"],
                "true_grade_class": labels["true_grade_class"],
                "true_mutation": labels["true_mutation"],
                "grade_pred": pred_name, "pred_grade_class": pgc,
                "grade_class_correct": int(pgc == labels["true_grade_class"]),
                "prob_control": round(float(probs[0]), 4),
                "prob_low_grade": round(float(probs[1]), 4),
                "prob_high_grade": round(float(probs[2]), 4),
                "rank": rank, "row": sel["row"], "col": sel["col"],
                "risk": round(sel["risk"], 6),
                "p95_threshold": round(thresh, 6), "n_above_p95": n_above,
                "tissue_fraction": round(sel["tissue_fraction"], 4),
                "base_mpp": base_mpp, "target_mpp": target_mpp,
                "wsi_bbox": list(wsi_bbox), "patch_path": str(patch_path),
                "thumbnail_path": str(sc_thumb), "wsi_path": str(wsi_path),
            })
    finally:
        slide.close()
    return rows


# ============================================================
# MAIN
# ============================================================

def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    control_img = Image.open(args.control_image).convert("RGB")

    root_dir = Path(args.out_dir)

    patches_dir = root_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = root_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = sorted(Path(args.splits_dir).glob(args.split_glob))
    if not split_csvs:
        raise SystemExit(
            f"No split CSVs matching {args.split_glob} in {args.splits_dir}"
        )

    ckpt_map = resolve_checkpoints(split_csvs, args.checkpoints_glob)

    rows_written = slides_done = slides_failed = 0

    # Combined manifest for the entire cohort
    manifest_path = root_dir / "stageB_manifest.csv"

    with open(manifest_path, "w", newline="") as mf:
        combined_writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        combined_writer.writeheader()

        for split_csv in split_csvs:
            m = re.search(r"split_(\d+)", split_csv.stem)
            if m is None:
                raise ValueError(
                    f"Could not determine fold from {split_csv.name}"
                )

            fold_idx = int(m.group(1))

            fold_manifest = manifests_dir / f"stageB_fold{fold_idx}.csv"

            ckpt = ckpt_map.get(split_csv.stem)

            if ckpt is None:
                print(f"[fold {fold_idx}] no checkpoint for {split_csv.stem}, skip")
                continue

            model = load_model(ckpt, args.model, device)
            slides = load_split_rows(split_csv)

            print(
                f"\n[fold {fold_idx}] {split_csv.name}: {len(slides)} slides "
                f"| ckpt {Path(ckpt).name}"
            )

            with open(fold_manifest, "w", newline="") as ff:
                fold_writer = csv.DictWriter(ff, fieldnames=MANIFEST_FIELDS)
                fold_writer.writeheader()

                for thumb_path, role in slides:
                    try:
                        rows = process_slide(
                            thumb_path,
                            role,
                            model,
                            transform,
                            control_img,
                            device,
                            fold_idx,
                            split_csv,
                            args,
                            patches_dir,
                        )

                        for row in rows:
                            combined_writer.writerow(row)
                            fold_writer.writerow(row)

                        rows_written += len(rows)
                        slides_done += 1

                    except Exception as e:
                        slides_failed += 1
                        print(f"    FAILED {thumb_path.name}: {e}")

    print("\n=== DONE ===")
    print(f"Slide-fold passes: {slides_done} | failed: {slides_failed}")
    print(f"Regions written:   {rows_written}")
    print(f"Combined manifest: {manifest_path}")
    print(f"Per-fold manifests: {manifests_dir}")
    print(f"Patches:           {patches_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=str, default="splits")
    p.add_argument("--split-glob", type=str, default="split_0*.csv")
    p.add_argument("--checkpoints-glob", type=str,
                   default="training_output/*_split_0*/best_auc.pth")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--control-image", type=str, required=True)
    p.add_argument("--sidecar-root", type=str,
                   default="data/ebrains_thumbnails",
                   help="Tree holding the thumbnails and JSON sidecars, keyed "
                        "by slide id under <root>/<subtype>/included/.")
    p.add_argument("--wsi-root", type=str, default=None,
                   help="Flat fallback: read WSIs from this dir by filename, "
                        "ignoring the sidecar's subfolder nesting.")
    p.add_argument("--wsi-old-prefix", type=str, default=None,
                   help="Old path prefix stored in the sidecar to replace "
                        "(preserves subfolder nesting). Use with --wsi-new-prefix.")
    p.add_argument("--wsi-new-prefix", type=str, default=None,
                   help="New path prefix to substitute for --wsi-old-prefix.")
    p.add_argument("--out-dir", type=str, default="data/stage_b_cohort_RN50")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--min-tissue", type=float, default=0.5)
    p.add_argument("--output-size", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
    # python - m
    # scripts.stage_b.build_stageB_cohort - -splits - dir
    # splits - -checkpoints - glob
    # "training_output/*_split_0*/best_auc.pth" - -model
    # resnet18 - -control - image
    # "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\control\86242943-7775-11eb-827d-001a7dda7111.jpg" - -sidecar - root
    # "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\ebrains_thumbnails" - -wsi - root
    # "C:\Users\TAbde\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data" - -out - dir
    # stage_b_cohort