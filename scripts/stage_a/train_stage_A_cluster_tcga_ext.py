"""
Stage-A training on the EBRAINS+TCGA combined pool
(splits_tcga_ext/split_0X.csv from extend_splits_with_tcga.py), with
optional stain/colour augmentation. Builds train/val/test paths+labels
directly from the split CSV rows rather than scanning a root_dir.

Usage:
    python -m scripts.stage_a.train_stage_A_cluster_tcga_ext \
        --model resnet50 --split-csv splits_tcga_ext/split_01.csv
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import json
from datetime import datetime
from collections import Counter
import argparse
import os

import numpy as np

from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.data_utils.transforms_ext import build_train_transform_ext
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.trainer import Trainer
from glioma_sparse.evaluation.plots import (
    plot_training_curves, plot_confusion_matrix, plot_roc_curve,
)
from glioma_sparse.training.seeding import set_seed, seed_worker, make_generator

from scripts.lib.eval_metrics import compute_metrics, confusion
from scripts.lib.profiling import get_env_info, count_parameters, measure_flops, EnergyMeter


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--split-csv", type=str, required=True,
                         help="A splits_tcga_ext/split_0X.csv (path,split,source,tcga_class).")
    parser.add_argument("--no-stain-jitter", action="store_true")
    parser.add_argument("--no-color-jitter", action="store_true")
    parser.add_argument("--color-jitter-strength", type=str, default="mild",
                         choices=["mild", "strong"])
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = None
REPO_ROOT = Path(__file__).resolve().parents[2]
# SlideDataset requires root_dir/<class> to exist for each class in
# class_names, as a sanity check -- individual sample paths still come from
# the split CSV and may live anywhere (see load_split_rows/_relkey). This
# must be a folder that actually HAS low_grade/high_grade subfolders on every
# machine this runs on; data/included/ (the original EBRAINS-only tree) isn't
# guaranteed to exist under that name everywhere (e.g. it's been renamed on
# the cluster), but data/included_ebrains-tcga/ -- the merged tree this
# script's own samples all live under -- always will be.
DATA_DIR = REPO_ROOT / "data" / "included_ebrains-tcga"
BASE_OUT_DIR = REPO_ROOT / "training_output_tcga_ext"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = False
USE_CLASS_WEIGHTED_LOSS = True
BEST_CHECKPOINT_METRIC = "auc"

BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
SEED = 42

# Control excluded for this training round (see extend_splits_with_tcga.py --
# TCGA has no control-tissue slides, and this run is deliberately trained as
# low_grade/high_grade only, "ensemble 1"). If you regenerate the splits with
# --include-control, change this back to ["control", "low_grade", "high_grade"]
# and rerun -- the split CSV's class folders and this list must match, or
# SlideDataset's class-folder sanity check / class_to_idx mapping will be wrong.
CLASS_ORDER = ["low_grade", "high_grade"]


# ============================================================
# UTILITIES
# ============================================================

def build_experiment_name(model_name):
    parts = [model_name, "tcgaext"]
    if USE_CLASS_WEIGHTED_LOSS:
        parts.append("cw")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{'_'.join(parts)}"


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _relkey(pth):
    """Repo-relative POSIX key from data/ onward, so a split CSV built on
    Windows still resolves on the Linux cluster."""
    p = Path(str(pth).replace("\\", "/"))
    try:
        p = p.resolve().relative_to(REPO_ROOT)
    except (ValueError, OSError):
        pass
    s = p.as_posix()
    i = s.find("data/")
    return s[i:] if i >= 0 else s


def load_split_rows(split_csv):
    """(path, split, source, class) from the extended split CSV."""
    rows = []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = _relkey(row["path"])
            p = REPO_ROOT / key
            if not p.exists():
                raise FileNotFoundError(
                    f"Resolved path does not exist: {p} (from {row['path']} in {split_csv})\n"
                    f"Make sure data/included_ebrains-tcga/ is present at this repo root "
                    f"(run generate_tcga_thumbnails.py + extend_splits_with_tcga.py locally, "
                    f"then sync data/included_ebrains-tcga/ here at the same relative path).")
            cls = p.parent.name
            if cls not in CLASS_ORDER:
                raise ValueError(f"Unrecognised class folder {cls!r} for {p}")
            rows.append({
                "path": p, "split": row["split"],
                "source": row.get("source", "ebrains"),
                "class": cls, "label": CLASS_ORDER.index(cls),
            })
    return rows


def compute_class_weights(train_labels, num_classes, device):
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        if i not in counts:
            raise ValueError(f"Class index {i} ({CLASS_ORDER[i]}) has no training samples.")
        weights.append(total / (num_classes * counts[i]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_probs.argmax(axis=1), all_probs


def profiled_evaluate(model, loader, device, warmup=True):
    model.eval()
    if warmup:
        with torch.no_grad():
            for imgs, _ in loader:
                model(imgs.to(device))
                break
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    meter = EnergyMeter()
    with meter:
        y, preds, probs = evaluate_model(model, loader, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    perf = dict(meter.result)
    n = int(len(y))
    perf["n_images"] = n
    if perf.get("elapsed_sec", 0) > 0:
        perf["throughput_img_per_sec"] = round(n / perf["elapsed_sec"], 3)
        perf["latency_ms_per_image"] = round(1000 * perf["elapsed_sec"] / n, 4)
    if "energy_joules" in perf and n > 0:
        perf["energy_mj_per_image"] = round(1000 * perf["energy_joules"] / n, 4)
    if torch.cuda.is_available():
        perf["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
    return y, preds, probs, perf


def run_eval_split(model, loader, split_name, out_dir):
    print(f"\n=== {split_name.upper()} ===")
    y, preds, probs, perf = profiled_evaluate(model, loader, DEVICE)
    np.savez(out_dir / f"{split_name}_predictions.npz",
             y_true=y, y_prob=probs, class_order=np.array(CLASS_ORDER))
    metrics = compute_metrics(y, probs, CLASS_ORDER)
    save_json(out_dir / f"metrics_{split_name}.json", metrics)
    plot_confusion_matrix(y, preds, CLASS_ORDER, out_dir / f"confusion_matrix_{split_name}.png")
    plot_roc_curve(y, probs, CLASS_ORDER, out_dir / f"roc_curve_{split_name}.png")
    print(f"  acc={metrics['accuracy']:.3f}  macro_auc={metrics['macro_auc']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}")
    return metrics, perf


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    model_name = args.model or os.environ.get("MODEL") or "resnet18"
    print(f"\nUsing model: {model_name}")

    set_seed(SEED)

    experiment_name = build_experiment_name(model_name)
    experiment_name = f"{experiment_name}_{Path(args.split_csv).stem}"
    out_dir = BASE_OUT_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment:", experiment_name, "| Device:", DEVICE)
    print("Split CSV:", args.split_csv)

    # --------------------------------------------------------
    # DATA (built directly from the extended split CSV)
    # --------------------------------------------------------
    all_rows = load_split_rows(args.split_csv)
    by_split = {"train": [], "val": [], "test": []}
    for r in all_rows:
        by_split[r["split"]].append(r)

    train_paths = [r["path"] for r in by_split["train"]]
    train_labels = [r["label"] for r in by_split["train"]]
    val_paths = [r["path"] for r in by_split["val"]]
    val_labels = [r["label"] for r in by_split["val"]]
    test_paths = [r["path"] for r in by_split["test"]]
    test_labels = [r["label"] for r in by_split["test"]]

    source_counts = {
        split: dict(Counter(r["source"] for r in rows))
        for split, rows in by_split.items()
    }
    class_counts_train = {
        cls: dict(Counter(r["source"] for r in by_split["train"] if r["class"] == cls))
        for cls in CLASS_ORDER
    }
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print("Train source counts:", source_counts["train"])
    print("Train per-class source breakdown:", class_counts_train)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    use_stain_jitter = not args.no_stain_jitter
    use_color_jitter = not args.no_color_jitter
    train_transform = build_train_transform_ext(
        use_stain_jitter=use_stain_jitter, use_color_jitter=use_color_jitter,
        color_jitter_strength=args.color_jitter_strength)
    print(f"Augmentation: stain_jitter={use_stain_jitter} "
          f"color_jitter={use_color_jitter} ({args.color_jitter_strength})")

    train_dataset = SlideDataset(DATA_DIR, transform=train_transform,
                                  patch_transform=Patch(8), class_names=CLASS_ORDER,
                                  paths=train_paths)
    val_dataset = SlideDataset(DATA_DIR, transform=build_eval_transform(),
                                patch_transform=None, class_names=CLASS_ORDER,
                                paths=val_paths)
    test_dataset = SlideDataset(DATA_DIR, transform=build_eval_transform(),
                                 patch_transform=None, class_names=CLASS_ORDER,
                                 paths=test_paths)

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    pin_memory = torch.cuda.is_available()
    loader_generator = make_generator(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                               generator=loader_generator, pin_memory=pin_memory,
                               persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                             pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                              pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model(model_name, num_classes=len(CLASS_ORDER))
    model.to(DEVICE)

    class_weights = compute_class_weights(train_labels, len(CLASS_ORDER), DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    param_info = count_parameters(model)
    sample_img, _ = val_dataset[0]
    flop_info = measure_flops(model, tuple(sample_img.shape), DEVICE)
    env_info = get_env_info()

    save_json(out_dir / "config.json", {
        "experiment_name": experiment_name, "model": model_name,
        "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "seed": SEED,
        "class_weights": class_weights.cpu().tolist(),
        "split_source": str(args.split_csv),
        "split_counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "train_source_counts": source_counts["train"],
        "train_class_source_breakdown": class_counts_train,
        "use_stain_jitter": use_stain_jitter, "use_color_jitter": use_color_jitter,
        "color_jitter_strength": args.color_jitter_strength,
        "environment": env_info, "model_params": param_info, "model_flops": flop_info,
    })

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                       optimizer=optimizer, criterion=criterion, device=DEVICE,
                       output_dir=out_dir)
    train_meter = EnergyMeter()
    with train_meter:
        history, _, _ = trainer.train(NUM_EPOCHS)
    train_perf = dict(train_meter.result)
    epochs_run = len(history.get("auc", []))
    train_perf["epochs_run"] = epochs_run
    train_perf["epochs_max"] = NUM_EPOCHS
    print(f"\nTraining time: {train_perf.get('elapsed_sec'):.2f} sec | {epochs_run}/{NUM_EPOCHS} epochs")

    plot_training_curves(history, out_dir)

    best_model_path = out_dir / f"best_{BEST_CHECKPOINT_METRIC}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)

    val_metrics, val_perf = run_eval_split(model, val_loader, "val", out_dir)
    test_metrics, test_perf = run_eval_split(model, test_loader, "test", out_dir)

    save_json(out_dir / "efficiency.json", {
        "experiment_name": experiment_name, "model": model_name, "device": DEVICE,
        "environment": env_info, "model_params": param_info, "model_flops": flop_info,
        "train": train_perf, "inference_val": val_perf, "inference_test": test_perf,
    })
    print("\nSaved predictions, metrics, plots, and efficiency log.")


if __name__ == "__main__":
    main()
