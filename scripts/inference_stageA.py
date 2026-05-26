import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from PIL import Image

from glioma_sparse.models.factory import build_model
from glioma_sparse.data_utils.transforms import build_eval_transform


# ============================================================
# OUTPUT HANDLING
# ============================================================

def create_output_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    experiment_dir = checkpoint_path.parent

    inference_root = experiment_dir / "inference"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = inference_root / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    return Image.open(path).convert("RGB")


def get_image_paths(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    input_path,
    checkpoint_path,
    model_name="resnet18",
    device=None,
    output_dir=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {model_name}")

    # --------------------------------------------------
    # OUTPUT DIR
    # --------------------------------------------------
    if output_dir is None:
        run_dir = create_output_dir(checkpoint_path)
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = build_model(model_name, num_classes=3)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # TRANSFORM (CRITICAL FIX)
    # --------------------------------------------------
    transform = build_eval_transform()

    class_names = ["control", "low_grade", "high_grade"]

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------
    image_paths = get_image_paths(input_path)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in {input_path}")

    results = []

    # --------------------------------------------------
    # LOOP
    # --------------------------------------------------
    with torch.no_grad():
        for img_path in image_paths:
            img = load_image(img_path)
            x = transform(img).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = probs.argmax()
            pred_class = class_names[pred_idx]

            results.append({
                "path": str(img_path),
                "prediction": pred_class,
                "prob_control": float(probs[0]),
                "prob_low_grade": float(probs[1]),
                "prob_high_grade": float(probs[2]),
            })

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    df = pd.DataFrame(results)

    csv_path = run_dir / "stageA_results.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "input_path": str(input_path),
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_samples": len(df),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n=== INFERENCE COMPLETE ===")
    print(f"Samples: {len(df)}")
    print(f"Saved to: {run_dir}")

    return df