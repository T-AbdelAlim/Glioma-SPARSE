import torch
import numpy as np
from pathlib import Path
import csv

from torch.utils.data import DataLoader

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.evaluation.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve
)

# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = "20260525_0136_resnet18_os"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "included"

OUT_DIR = REPO_ROOT / "training_output" / EXPERIMENT_NAME
SPLIT_CSV = OUT_DIR / "data_split.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(csv_path):
    split_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[Path(row["path"])] = row["split"]
    return split_map


# ============================================================
# LOAD HISTORY FROM CSV
# ============================================================

def load_history(csv_path):
    history = {
        "train_loss": [],
        "val_loss": [],
        "acc": [],
        "f1": [],
        "auc": []
    }

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["train_loss"].append(float(row["train_loss"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["acc"].append(float(row["accuracy"]))
            history["f1"].append(float(row["f1"]))
            history["auc"].append(float(row["auc"]))

    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, loader, device):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds = np.argmax(all_probs, axis=1)

    return all_labels, preds, all_probs


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== POST-HOC EVALUATION ===\n")
    print("Experiment: {}".format(EXPERIMENT_NAME))

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------
    base_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=None,
        patch_transform=None,
        class_names=None
    )

    class_names = base_dataset.classes
    paths = base_dataset.paths
    labels = base_dataset.labels

    split_map = load_split(SPLIT_CSV)

    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for p, l in zip(paths, labels):
        s = split_map[p]
        if s == "val":
            val_paths.append(p)
            val_labels.append(l)
        elif s == "test":
            test_paths.append(p)
            test_labels.append(l)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    val_dataset = SlideDataset(DATA_DIR, build_eval_transform(), None)
    val_dataset.paths = val_paths
    val_dataset.labels = val_labels

    test_dataset = SlideDataset(DATA_DIR, build_eval_transform(), None)
    test_dataset.paths = test_paths
    test_dataset.labels = test_labels

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model("resnet18", num_classes=len(class_names))
    model.load_state_dict(torch.load(OUT_DIR / "best_auc.pth", map_location=DEVICE))
    model.to(DEVICE)

    print("Loaded best_auc model")

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------
    history = load_history(OUT_DIR / "log.csv")

    plot_training_curves(history, OUT_DIR)

    # --------------------------------------------------------
    # VALIDATION SET
    # --------------------------------------------------------
    print("\nEvaluating VAL set...")
    labels, preds, probs = evaluate_model(model, val_loader, DEVICE)

    CLASS_ORDER = ["control", "low_grade", "high_grade"]

    # raw counts
    plot_confusion_matrix(
        labels,
        preds,
        CLASS_ORDER,
        OUT_DIR / "confusion_matrix.png",
        normalize=False
    )

    # normalized
    plot_confusion_matrix(
        labels,
        preds,
        CLASS_ORDER,
        OUT_DIR / "confusion_matrix_normalized.png",
        normalize=True
    )

    plot_roc_curve(
        labels,
        probs,
        class_names,
        OUT_DIR / "roc_curve_val.png"
    )

    # --------------------------------------------------------
    # TEST SET
    # --------------------------------------------------------
    print("\nEvaluating TEST set...")
    labels, preds, probs = evaluate_model(model, test_loader, DEVICE)

    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        OUT_DIR / "confusion_matrix_test.png"
    )

    plot_roc_curve(
        labels,
        probs,
        class_names,
        OUT_DIR / "roc_curve_test.png"
    )

    print("\nAll plots saved to: {}".format(OUT_DIR))


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()