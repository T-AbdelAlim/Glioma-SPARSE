import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_train_transform, build_eval_transform
from glioma_sparse.data_utils.sampling import oversample_paths
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.trainer import Trainer
from glioma_sparse.evaluation.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

EXPERIMENT_NAME = f"{datetime.now():%Y%m%d_%H%M}_resnet18_os"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "included"

BASE_OUT_DIR = REPO_ROOT / "training_output"
OUT_DIR = BASE_OUT_DIR / EXPERIMENT_NAME
SPLIT_CSV = OUT_DIR / "data_split.csv"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = True
USE_EXISTING_SPLIT = False

BATCH_SIZE = 4
NUM_EPOCHS = 150
SEED = 42


# ============================================================
# UTIL FUNCTIONS
# ============================================================

def save_config(output_dir, config_dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.txt", "w") as f:
        for k, v in config_dict.items():
            f.write(f"{k}: {v}\n")


def save_split_csv(paths, splits, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split"])
        for p, s in zip(paths, splits):
            writer.writerow([str(p), s])


def load_split(csv_path):
    split_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[Path(row["path"])] = row["split"]
    return split_map


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

    print("\n=== TRAINING ===\n")
    print("Experiment:", EXPERIMENT_NAME)
    print("Device:", DEVICE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # BASE DATASET
    # --------------------------------------------------------
    base_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=None,
        patch_transform=None,
        class_names=None
    )

    paths = base_dataset.paths
    labels = base_dataset.labels
    class_names = base_dataset.classes

    # --------------------------------------------------------
    # SPLIT (STRATIFIED)
    # --------------------------------------------------------
    if USE_EXISTING_SPLIT and SPLIT_CSV.exists():

        split_map = load_split(SPLIT_CSV)

        train_paths, val_paths, test_paths = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for p, l in zip(paths, labels):
            s = split_map[p]
            if s == "train":
                train_paths.append(p); train_labels.append(l)
            elif s == "val":
                val_paths.append(p); val_labels.append(l)
            else:
                test_paths.append(p); test_labels.append(l)

    else:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=0.2, stratify=labels, random_state=SEED
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
        )

        splits = []
        for p in paths:
            if p in train_paths:
                splits.append("train")
            elif p in val_paths:
                splits.append("val")
            else:
                splits.append("test")

        save_split_csv(paths, splits, SPLIT_CSV)

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # --------------------------------------------------------
    # OVERSAMPLING
    # --------------------------------------------------------
    if USE_OVERSAMPLING:
        train_paths, train_labels = oversample_paths(train_paths, train_labels)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    train_dataset = SlideDataset(DATA_DIR, build_train_transform(), Patch(8))
    train_dataset.paths = train_paths
    train_dataset.labels = train_labels

    val_dataset = SlideDataset(DATA_DIR, build_eval_transform(), None)
    val_dataset.paths = val_paths
    val_dataset.labels = val_labels

    test_dataset = SlideDataset(DATA_DIR, build_eval_transform(), None)
    test_dataset.paths = test_paths
    test_dataset.labels = test_labels

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model("resnet18", num_classes=len(class_names))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --------------------------------------------------------
    # CONFIG LOG
    # --------------------------------------------------------
    save_config(OUT_DIR, {
        "experiment_name": EXPERIMENT_NAME,
        "model": "resnet18",
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "oversampling": USE_OVERSAMPLING,
        "device": DEVICE,
        "dataset_size": len(paths),
    })

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        output_dir=OUT_DIR
    )

    start = time.time()
    history = trainer.train(NUM_EPOCHS)
    total_time = time.time() - start

    print(f"\nTotal training time: {total_time:.2f} sec")

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------
    plot_training_curves(history, OUT_DIR)

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------
    print("\n=== TEST EVALUATION ===")

    labels, preds, probs = evaluate_model(model, test_loader, DEVICE)

    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        OUT_DIR / "confusion_matrix.png"
    )

    plot_roc_curve(
        labels,
        probs,
        class_names,
        OUT_DIR / "roc_curve.png"
    )

    print("Saved evaluation plots.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()