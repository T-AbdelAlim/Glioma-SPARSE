import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
import csv
import random

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_train_transform, build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.trainer import Trainer


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "scripts" / "output_thumbnails" / "included"
OUT_DIR = REPO_ROOT / "scripts" / "training_output"
SPLIT_CSV = OUT_DIR / "data_split.csv"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_EXISTING_SPLIT = False


# ============================================================
# SPLIT FUNCTIONS
# ============================================================

def create_and_save_split(dataset, output_csv, seed=42):

    paths = dataset.paths

    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)

    n = len(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    split_map = {}

    for i in train_idx:
        split_map[paths[i]] = "train"
    for i in val_idx:
        split_map[paths[i]] = "val"
    for i in test_idx:
        split_map[paths[i]] = "test"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split"])

        for p in paths:
            writer.writerow([str(p), split_map[p]])

    print(f"Saved split to: {output_csv}")

    return split_map


def load_split(csv_path):

    split_map = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            split_map[Path(row["path"])] = row["split"]

    print(f"Loaded split from: {csv_path}")

    return split_map


def apply_split(dataset, split_map):

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for path, label in zip(dataset.paths, dataset.labels):

        split = split_map.get(path)

        if split == "train":
            train_paths.append(path)
            train_labels.append(label)

        elif split == "val":
            val_paths.append(path)
            val_labels.append(label)

        elif split == "test":
            test_paths.append(path)
            test_labels.append(label)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== FULL TRAIN DEMO ===\n")

    print("Data directory:", DATA_DIR)
    print("Output directory:", OUT_DIR)
    print("Using device:", DEVICE, "\n")

    # --------------------------------------------------------
    # CLEAN OUTPUT DIR
    # --------------------------------------------------------
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # --------------------------------------------------------
    # BASE DATASET (no transforms yet)
    # --------------------------------------------------------
    base_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=None,
        patch_transform=None,
        class_names=["control", "low_grade", "high_grade"]
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------
    if USE_EXISTING_SPLIT and SPLIT_CSV.exists():
        split_map = load_split(SPLIT_CSV)
    else:
        split_map = create_and_save_split(base_dataset, SPLIT_CSV)

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = apply_split(
        base_dataset,
        split_map
    )

    print("Train size:", len(train_paths))
    print("Val size:  ", len(val_paths))
    print("Test size: ", len(test_paths), "\n")

    # --------------------------------------------------------
    # BUILD DATASETS (WITH CORRECT TRANSFORMS)
    # --------------------------------------------------------
    train_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=build_train_transform(),
        patch_transform=Patch(grid_size=8),
        class_names=["control", "low_grade", "high_grade"]
    )
    train_dataset.paths = train_paths
    train_dataset.labels = train_labels

    val_dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=build_eval_transform(),
        patch_transform=None,
        class_names=["control", "low_grade", "high_grade"]
    )
    val_dataset.paths = val_paths
    val_dataset.labels = val_labels

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model("resnet18", num_classes=3)
    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --------------------------------------------------------
    # TRAINER
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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    trainer.train(num_epochs=3)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
