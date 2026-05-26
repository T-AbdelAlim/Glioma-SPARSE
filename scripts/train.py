import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import json
import time
from datetime import datetime
from collections import Counter
import argparse
import os

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
    plot_roc_curve,
)
from glioma_sparse.training.seeding import set_seed, seed_worker


# ============================================================
# ARGPARSE (CLI + SLURM SUPPORT)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (resnet18, resnet34, resnet50)")
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "included"
BASE_OUT_DIR = REPO_ROOT / "training_output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = False
USE_CLASS_WEIGHTED_LOSS = True
USE_EXISTING_SPLIT = False

BEST_CHECKPOINT_METRIC = "auc"

BATCH_SIZE = 8
NUM_EPOCHS = 150
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
SEED = 42

CLASS_ORDER = ["control", "low_grade", "high_grade"]


# ============================================================
# UTILITIES
# ============================================================

def build_experiment_name(model_name):
    parts = [model_name]
    if USE_OVERSAMPLING:
        parts.append("os")
    if USE_CLASS_WEIGHTED_LOSS:
        parts.append("cw")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{'_'.join(parts)}"


def save_config(output_dir, config_dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


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


def compute_class_weights(train_labels, num_classes, device):
    counts = Counter(train_labels)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        if i not in counts:
            raise ValueError(
                f"Class index {i} ({CLASS_ORDER[i]}) has no training samples."
            )
        weights.append(total / (num_classes * counts[i]))

    return torch.tensor(weights, dtype=torch.float32, device=device)


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

    args = parse_args()

    # Priority: CLI > SLURM env > default
    model_name = (
        args.model
        or os.environ.get("MODEL")
        or "resnet18"
    )

    print(f"\nUsing model: {model_name}")

    assert not (USE_OVERSAMPLING and USE_CLASS_WEIGHTED_LOSS), \
        "Use either oversampling or class-weighted loss, not both"

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------
    set_seed(SEED)

    experiment_name = EXPERIMENT_NAME or build_experiment_name(model_name)
    out_dir = BASE_OUT_DIR / experiment_name
    split_csv = out_dir / "data_split.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== TRAINING ===\n")
    print("Experiment:", experiment_name)
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    base_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=None,
        patch_transform=None,
        class_names=CLASS_ORDER,
    )

    paths = base_dataset.paths
    labels = base_dataset.labels
    class_names = base_dataset.classes

    assert len(paths) > 0, f"No data found in {DATA_DIR}"

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------
    if USE_EXISTING_SPLIT and split_csv.exists():

        split_map = load_split(split_csv)

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

        train_set = set(train_paths)
        val_set = set(val_paths)

        splits = []
        for p in paths:
            if p in train_set:
                splits.append("train")
            elif p in val_set:
                splits.append("val")
            else:
                splits.append("test")

        save_split_csv(paths, splits, split_csv)

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # --------------------------------------------------------
    # OVERSAMPLING
    # --------------------------------------------------------
    if USE_OVERSAMPLING:
        train_paths, train_labels = oversample_paths(train_paths, train_labels)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    train_dataset = SlideDataset(
        DATA_DIR,
        transform=build_train_transform(),
        patch_transform=Patch(8),
        class_names=CLASS_ORDER,
        paths=train_paths,
    )

    val_dataset = SlideDataset(
        DATA_DIR,
        transform=build_eval_transform(),
        patch_transform=None,
        class_names=CLASS_ORDER,
        paths=val_paths,
    )

    test_dataset = SlideDataset(
        DATA_DIR,
        transform=build_eval_transform(),
        patch_transform=None,
        class_names=CLASS_ORDER,
        paths=test_paths,
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model(model_name, num_classes=len(class_names))
    model.to(DEVICE)

    if USE_CLASS_WEIGHTED_LOSS:
        class_weights = compute_class_weights(train_labels, len(CLASS_ORDER), DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        class_weights_list = class_weights.cpu().tolist()
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights_list = None

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # CONFIG LOG
    # --------------------------------------------------------
    save_config(out_dir, {
        "experiment_name": experiment_name,
        "model": model_name,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "class_weights": class_weights_list,
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
        output_dir=out_dir,
    )

    start = time.time()
    history, _, _ = trainer.train(NUM_EPOCHS)
    print(f"\nTraining time: {time.time() - start:.2f} sec")

    plot_training_curves(history, out_dir)

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------
    best_model_path = out_dir / f"best_{BEST_CHECKPOINT_METRIC}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)

    # --------------------------------------------------------
    # VALIDATION EVAL
    # --------------------------------------------------------
    print("\n=== VALIDATION ===")
    y, preds, probs = evaluate_model(model, val_loader, DEVICE)

    plot_confusion_matrix(y, preds, CLASS_ORDER, out_dir / "confusion_matrix_val.png")
    plot_roc_curve(y, probs, CLASS_ORDER, out_dir / "roc_curve_val.png")

    # --------------------------------------------------------
    # TEST EVAL
    # --------------------------------------------------------
    print("\n=== TEST ===")
    y, preds, probs = evaluate_model(model, test_loader, DEVICE)

    plot_confusion_matrix(y, preds, CLASS_ORDER, out_dir / "confusion_matrix_test.png")
    plot_roc_curve(y, probs, CLASS_ORDER, out_dir / "roc_curve_test.png")

    print("\nSaved evaluation plots.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()