import torch
import numpy as np
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
    return path.suffix.lower() in [".ndpi", ".svs", ".tiff", ".tif"]


def generate_thumbnail_if_needed(input_path, tmp_dir):
    if is_wsi(input_path):
        print("Generating thumbnail from WSI...")
        thumb_path = tmp_dir / f"{input_path.stem}_thumb.jpg"

        create_wsi_thumbnail(
            input_path,
            thumb_path,
        )
        return thumb_path
    else:
        return input_path


def save_heatmap(risk_map, save_path):
    plt.imshow(risk_map, cmap="hot")
    plt.colorbar()
    plt.title("Risk Map")
    plt.savefig(save_path)
    plt.close()


def save_overlay(image, risk_map, save_path):
    img = np.array(image.resize((2048, 2048)))

    heatmap = cv2.resize(risk_map, (2048, 2048))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(str(save_path), overlay)


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_single_inference(
    input_path,
    checkpoint_path,
    control_image_path,
    model_name="resnet34",
    k=5,
    extract_patches=True,
    output_dir=None,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_path = Path(input_path)
    checkpoint_path = Path(checkpoint_path)
    control_image_path = Path(control_image_path)

    output_dir = Path(output_dir) if output_dir else input_path.parent / "risk_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Processing: {input_path.name}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    model = load_model(checkpoint_path, model_name, device)
    transform = build_eval_transform()

    # --------------------------------------------------------
    # LOAD CONTROL
    # --------------------------------------------------------
    control_img = Image.open(control_image_path).convert("RGB")

    # --------------------------------------------------------
    # PREP INPUT
    # --------------------------------------------------------
    thumbnail_path = generate_thumbnail_if_needed(input_path, tmp_dir)
    target_img = Image.open(thumbnail_path).convert("RGB")

    # --------------------------------------------------------
    # PREDICT CLASS (IMPORTANT)
    # --------------------------------------------------------
    x = transform(target_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    pred_label = CLASS_ORDER[pred_idx]

    print(f"Prediction: {pred_label}")
    print(f"Probabilities: {probs}")

    # --------------------------------------------------------
    # RISK MAP
    # --------------------------------------------------------
    risk_map, uncertainty_map = compute_risk_map(
        target_img=target_img,
        control_img=control_img,
        model=model,
        transform=transform,
        target_class=pred_idx,
        device=device,
    )

    # --------------------------------------------------------
    # SAVE HEATMAP
    # --------------------------------------------------------
    heatmap_path = output_dir / "risk_map.png"
    save_heatmap(risk_map, heatmap_path)

    # --------------------------------------------------------
    # SAVE OVERLAY
    # --------------------------------------------------------
    overlay_path = output_dir / "overlay.png"
    save_overlay(target_img, risk_map, overlay_path)

    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved overlay: {overlay_path}")

    # --------------------------------------------------------
    # EXTRACT PATCHES
    # --------------------------------------------------------
    if extract_patches:

        patch_dir = output_dir / "patches"
        patch_dir.mkdir(exist_ok=True)

        print(f"\nExtracting top-{k} patches...")

        extract_topk_patches(
            thumbnail_path=thumbnail_path,
            risk_map=risk_map,
            k=k,
            label=pred_label,
            output_dir=patch_dir,
            output_size=2048,
        )

        print(f"Saved patches to: {patch_dir}")

    print("\nDone.")


# ============================================================
# PYCHARM ENTRY
# ============================================================

if __name__ == "__main__":

    run_single_inference(
        input_path=r"D:\Thinkpad_Backup\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data\GBM_IDHwt\GBM_IDHwt\a194f783-357f-11eb-9ee2-001a7dda7111.ndpi",
        checkpoint_path=r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\training_output\20260526_0947_resnet34_cw\best_acc.pth",
        control_image_path=r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\ebrains_thumbnails\control\included\86242943-7775-11eb-827d-001a7dda7111.jpg",
        model_name="resnet34",
        k=5,
        extract_patches=True,
    )