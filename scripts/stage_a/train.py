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
from glioma_sparse.training.seeding import set_seed, seed_worker, make_generator

# Sibling modules in scripts/ (run as: python scripts/train.py)
from scripts.lib.eval_metrics import compute_metrics, confusion
from scripts.lib.profiling import get_env_info, count_parameters, measure_flops, EnergyMeter


# ============================================================
# ARGPARSE (CLI + SLURM SUPPORT)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (resnet18, resnet34, resnet50)")
    parser.add_argument("--split-csv", type=str, default=None,
                        help="Path to a pre-generated split CSV "
                             "(path,split). Overrides on-the-fly splitting.")
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "included"
BASE_OUT_DIR = REPO_ROOT / "training_output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = False
USE_CLASS_WEIGHTED_LOSS = True
USE_EXISTING_SPLIT = False

BEST_CHECKPOINT_METRIC = "f1"

BATCH_SIZE = 8
NUM_EPOCHS = 80
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


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_config(output_dir, config_dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config_dict)


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


def profiled_evaluate(model, loader, device, warmup=True):
    """Run evaluate_model while measuring time, throughput, energy and memory.

    Returns (y_true, preds, probs, perf_dict).
    """
    model.eval()

    # Warmup pass so first-batch/one-off costs stay out of the timing.
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
        perf["peak_gpu_mem_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 1e6, 2
        )
    return y, preds, probs, perf


def run_eval_split(model, loader, split_name, out_dir):
    """Evaluate, save raw predictions + metrics + plots, return (metrics, perf)."""
    print(f"\n=== {split_name.upper()} ===")
    y, preds, probs, perf = profiled_evaluate(model, loader, DEVICE)

    # 1. raw predictions (this is what aggregate_splits.py reads)
    np.savez(
        out_dir / f"{split_name}_predictions.npz",
        y_true=y, y_prob=probs, class_order=np.array(CLASS_ORDER),
    )

    # 2. metrics
    metrics = compute_metrics(y, probs, CLASS_ORDER)
    save_json(out_dir / f"metrics_{split_name}.json", metrics)

    # 3. plots (unchanged behaviour)
    plot_confusion_matrix(y, preds, CLASS_ORDER,
                          out_dir / f"confusion_matrix_{split_name}.png")
    plot_roc_curve(y, probs, CLASS_ORDER,
                   out_dir / f"roc_curve_{split_name}.png")

    print(f"  acc={metrics['accuracy']:.3f}  "
          f"macro_auc={metrics['macro_auc']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  "
          f"qwk={metrics['quadratic_weighted_kappa']:.3f}")
    if perf.get("latency_ms_per_image") is not None:
        line = (f"  {perf['latency_ms_per_image']:.2f} ms/slide  "
                f"({perf.get('throughput_img_per_sec')} slides/s)")
        if "energy_mj_per_image" in perf:
            line += f"  {perf['energy_mj_per_image']:.2f} mJ/slide"
        print(line)

    return metrics, perf


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
    if args.split_csv is not None:
        experiment_name = f"{experiment_name}_{Path(args.split_csv).stem}"
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
    use_provided_split = args.split_csv is not None

    if use_provided_split or (USE_EXISTING_SPLIT and split_csv.exists()):

        source_csv = Path(args.split_csv) if use_provided_split else split_csv
        assert source_csv.exists(), f"Split CSV not found: {source_csv}"
        print(f"Using split: {source_csv}")

        split_map = load_split(source_csv)

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

        # keep a copy of the split used, inside this run's output folder
        save_split_csv(paths, [split_map[p] for p in paths], split_csv)
        split_source = str(source_csv)

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
        split_source = "on_the_fly_sklearn"

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
    loader_generator = make_generator(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=loader_generator,
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
    # MODEL COMPLEXITY (params + FLOPs per slide)
    # --------------------------------------------------------
    param_info = count_parameters(model)
    sample_img, _ = val_dataset[0]
    input_shape = tuple(sample_img.shape)  # (C, H, W)
    flop_info = measure_flops(model, input_shape, DEVICE)
    env_info = get_env_info()

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
        # provenance: which split produced this run
        "split_source": split_source,
        "split_counts": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        # reproducibility / hardware context
        "environment": env_info,
        "model_params": param_info,
        "model_flops": flop_info,
    })

    # --------------------------------------------------------
    # TRAIN (timed + energy-metered)
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

    train_meter = EnergyMeter()
    with train_meter:
        history, _, _ = trainer.train(NUM_EPOCHS)
    train_perf = dict(train_meter.result)

    # Epochs actually run (one history entry per completed epoch). With early
    # stopping this is below NUM_EPOCHS, so it explains cross-split time/energy.
    epochs_run = len(history.get("f1", []))
    train_perf["epochs_run"] = epochs_run
    train_perf["epochs_max"] = NUM_EPOCHS
    train_perf["early_stopped"] = epochs_run < NUM_EPOCHS
    if epochs_run > 0 and train_perf.get("elapsed_sec"):
        train_perf["sec_per_epoch"] = round(train_perf["elapsed_sec"] / epochs_run, 3)
        if "energy_wh" in train_perf:
            train_perf["wh_per_epoch"] = round(train_perf["energy_wh"] / epochs_run, 6)

    msg = f"\nTraining time: {train_perf.get('elapsed_sec'):.2f} sec"
    msg += f" | {epochs_run}/{NUM_EPOCHS} epochs"
    msg += (f" | {train_perf['energy_wh']:.4f} Wh"
            if "energy_wh" in train_perf else " | energy: n/a")
    print(msg)

    plot_training_curves(history, out_dir)

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------
    best_model_path = out_dir / f"best_{BEST_CHECKPOINT_METRIC}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)

    # --------------------------------------------------------
    # EVALUATION (val + test): predictions, metrics, plots, efficiency
    # --------------------------------------------------------
    val_metrics, val_perf = run_eval_split(model, val_loader, "val", out_dir)
    test_metrics, test_perf = run_eval_split(model, test_loader, "test", out_dir)

    # --------------------------------------------------------
    # EFFICIENCY LOG (single source for the compute-cost story)
    # --------------------------------------------------------
    save_json(out_dir / "efficiency.json", {
        "experiment_name": experiment_name,
        "model": model_name,
        "device": DEVICE,
        "environment": env_info,
        "model_params": param_info,
        "model_flops": flop_info,
        "train": train_perf,
        "inference_val": val_perf,
        "inference_test": test_perf,
    })

    print("\nSaved predictions, metrics, plots, and efficiency log.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()