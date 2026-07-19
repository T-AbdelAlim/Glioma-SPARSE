import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import csv
import json
from datetime import datetime
from collections import Counter, defaultdict
import argparse
import os

import numpy as np
from PIL import Image

from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_train_transform, build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.trainer import Trainer
from glioma_sparse.evaluation.plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
)
from glioma_sparse.training.seeding import set_seed, seed_worker, make_generator

# Sibling library modules (run as: python -m scripts.stage_b.train_stageB)
from scripts.lib.eval_metrics import compute_metrics, confusion
from scripts.lib.profiling import get_env_info, count_parameters, measure_flops, EnergyMeter


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True,
                        help="Which fold to train (1-5). Reads that fold's manifest.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (resnet18, resnet34, resnet50)")
    parser.add_argument("--cohort-dir", type=str, default=None,
                        help="Stage B cohort dir (default data/stage_b_cohort).")
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = None

REPO_ROOT = Path(__file__).resolve().parents[2]
COHORT_DIR = REPO_ROOT / "data" / "stage_b_cohort"
BASE_OUT_DIR = REPO_ROOT / "training_output_stageB"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = False
USE_CLASS_WEIGHTED_LOSS = True
USE_PATCH_SHUFFLE = True          # Patch(8) augmentation on train, as in Stage A

BEST_CHECKPOINT_METRIC = "f1"

BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
SEED = 42

# Molecular subtype classes (folder names in the cohort).
STAGE_B_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
CLASS_TO_IDX = {c: i for i, c in enumerate(STAGE_B_CLASSES)}


# ============================================================
# DATASET
# ============================================================

class ManifestPatchDataset(Dataset):
    """High-resolution patches for one fold+split, read from the cohort manifest.

    Patch paths are rebuilt from the cohort dir so the dataset survives the
    cohort folder being moved (paths in the manifest may be stale).
    Returns (image_tensor, label). slide_ids and labels are exposed in order for
    slide-level soft-voting at evaluation.
    """

    def __init__(self, rows, cohort_dir, transform=None, patch_transform=None):
        self.transform = transform
        self.patch_transform = patch_transform
        self.samples = []      # (path, label_idx)
        self.slide_ids = []
        self.labels = []

        cohort_dir = Path(cohort_dir)
        for r in rows:
            mut = r["true_mutation"]
            if mut not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[mut]
            fname = Path(r["patch_path"]).name
            path = (cohort_dir / "patches" / f"fold_{r['fold']}" / mut / fname)
            self.samples.append((path, label))
            self.slide_ids.append(r["slide_id"])
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.patch_transform is not None:
            img = self.patch_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============================================================
# UTILITIES
# ============================================================

def build_experiment_name(model_name, fold):
    parts = [model_name, "stageB", f"fold{fold}"]
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


def read_fold_manifest(cohort_dir, fold):
    mpath = Path(cohort_dir) / "manifests" / f"stageB_fold{fold}.csv"
    if not mpath.exists():
        raise SystemExit(f"Fold manifest not found: {mpath}")
    rows = []
    with open(mpath) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]


def oversample_rows(rows):
    """Duplicate minority-class rows up to the majority count. Random patch
    permutation each epoch (Patch(8)) makes duplicates differ during training."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["true_mutation"]].append(r)
    if not by_class:
        return rows
    target = max(len(v) for v in by_class.values())
    out = []
    rng = np.random.default_rng(SEED)
    for cls, items in by_class.items():
        out.extend(items)
        if len(items) < target:
            extra = rng.choice(len(items), size=target - len(items), replace=True)
            out.extend(items[i] for i in extra)
    return out


def compute_class_weights(train_labels, num_classes, device):
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        if i not in counts:
            raise ValueError(
                f"Class index {i} ({STAGE_B_CLASSES[i]}) has no training patches.")
        weights.append(total / (num_classes * counts[i]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# EVALUATION (patch inference + slide-level soft vote)
# ============================================================

def evaluate_patches(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def profiled_patch_inference(model, dataset, device, warmup=True):
    """Patch-level inference with time/energy/memory, order preserved."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=NUM_WORKERS > 0)
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
        y_patch, probs_patch = evaluate_patches(model, loader, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    perf = dict(meter.result)
    n = int(len(y_patch))
    perf["n_patches"] = n
    if perf.get("elapsed_sec", 0) > 0:
        perf["throughput_patch_per_sec"] = round(n / perf["elapsed_sec"], 3)
        perf["latency_ms_per_patch"] = round(1000 * perf["elapsed_sec"] / n, 4)
    if "energy_joules" in perf and n > 0:
        perf["energy_mj_per_patch"] = round(1000 * perf["energy_joules"] / n, 4)
    if torch.cuda.is_available():
        perf["peak_gpu_mem_mb"] = round(
            torch.cuda.max_memory_allocated(device) / 1e6, 2)
    return y_patch, probs_patch, perf


def soft_vote(y_patch, probs_patch, slide_ids):
    """Average patch probabilities per slide -> slide-level y_true, y_prob."""
    psum = defaultdict(lambda: np.zeros(len(STAGE_B_CLASSES)))
    pcount = defaultdict(int)
    strue = {}
    for i, sid in enumerate(slide_ids):
        psum[sid] += probs_patch[i]
        pcount[sid] += 1
        strue[sid] = int(y_patch[i])
    slides = sorted(psum.keys())
    y_slide = np.array([strue[s] for s in slides], dtype=int)
    prob_slide = np.array([psum[s] / pcount[s] for s in slides], dtype=float)
    return slides, y_slide, prob_slide


def run_eval_stageB(model, dataset, split_name, out_dir):
    print(f"\n=== {split_name.upper()} ===")
    y_patch, probs_patch, perf = profiled_patch_inference(model, dataset, DEVICE)

    slides, y_slide, prob_slide = soft_vote(y_patch, probs_patch, dataset.slide_ids)

    # raw outputs: slide-level is the primary artifact (aggregation reads this),
    # patch-level saved alongside for completeness.
    np.savez(out_dir / f"{split_name}_predictions.npz",
             y_true=y_slide, y_prob=prob_slide,
             class_order=np.array(STAGE_B_CLASSES), slide_ids=np.array(slides))
    np.savez(out_dir / f"{split_name}_predictions_patch.npz",
             y_true=y_patch, y_prob=probs_patch,
             class_order=np.array(STAGE_B_CLASSES),
             slide_ids=np.array(dataset.slide_ids))

    metrics_slide = compute_metrics(y_slide, prob_slide, STAGE_B_CLASSES)
    metrics_patch = compute_metrics(y_patch, probs_patch, STAGE_B_CLASSES)
    save_json(out_dir / f"metrics_{split_name}.json",
              {"level": "slide", **metrics_slide,
               "patch_level": metrics_patch})

    preds_slide = prob_slide.argmax(axis=1)
    plot_confusion_matrix(y_slide, preds_slide, STAGE_B_CLASSES,
                          out_dir / f"confusion_matrix_{split_name}.png")
    plot_roc_curve(y_slide, prob_slide, STAGE_B_CLASSES,
                   out_dir / f"roc_curve_{split_name}.png")

    # slide-level efficiency derived from patch inference
    n_slides = len(slides)
    perf["n_slides"] = n_slides
    if perf.get("elapsed_sec", 0) > 0 and n_slides > 0:
        perf["latency_ms_per_slide"] = round(1000 * perf["elapsed_sec"] / n_slides, 4)
    if "energy_joules" in perf and n_slides > 0:
        perf["energy_mj_per_slide"] = round(1000 * perf["energy_joules"] / n_slides, 4)

    print(f"  [slide] acc={metrics_slide['accuracy']:.3f}  "
          f"macro_auc={metrics_slide['macro_auc']:.3f}  "
          f"macro_f1={metrics_slide['macro_f1']:.3f}")
    print(f"  [patch] acc={metrics_patch['accuracy']:.3f}  "
          f"macro_auc={metrics_patch['macro_auc']:.3f}  "
          f"({perf['n_patches']} patches, {n_slides} slides)")
    return metrics_slide, perf


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    model_name = args.model or os.environ.get("MODEL") or "resnet18"
    cohort_dir = Path(args.cohort_dir) if args.cohort_dir else COHORT_DIR
    fold = args.fold

    print(f"\nStage B | model {model_name} | fold {fold} | cohort {cohort_dir}")
    assert not (USE_OVERSAMPLING and USE_CLASS_WEIGHTED_LOSS), \
        "Use either oversampling or class-weighted loss, not both"

    set_seed(SEED)

    experiment_name = EXPERIMENT_NAME or build_experiment_name(model_name, fold)
    out_dir = BASE_OUT_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Experiment:", experiment_name, "| Device:", DEVICE)

    # --------------------------------------------------------
    # DATA (from the fold manifest)
    # --------------------------------------------------------
    rows = read_fold_manifest(cohort_dir, fold)
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")
    assert train_rows and val_rows and test_rows, \
        "Fold manifest missing one of train/val/test."

    if USE_OVERSAMPLING:
        train_rows = oversample_rows(train_rows)

    patch_tf = Patch(8) if USE_PATCH_SHUFFLE else None
    train_dataset = ManifestPatchDataset(
        train_rows, cohort_dir, transform=build_train_transform(),
        patch_transform=patch_tf)
    val_dataset = ManifestPatchDataset(
        val_rows, cohort_dir, transform=build_eval_transform(), patch_transform=None)
    test_dataset = ManifestPatchDataset(
        test_rows, cohort_dir, transform=build_eval_transform(), patch_transform=None)

    n_slides = lambda ds: len(set(ds.slide_ids))
    print(f"Train: {len(train_dataset)} patches / {n_slides(train_dataset)} slides | "
          f"Val: {len(val_dataset)} / {n_slides(val_dataset)} | "
          f"Test: {len(test_dataset)} / {n_slides(test_dataset)}")

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    pin_memory = torch.cuda.is_available()
    gen = make_generator(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=gen,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)

    # --------------------------------------------------------
    # MODEL + LOSS
    # --------------------------------------------------------
    model = build_model(model_name, num_classes=len(STAGE_B_CLASSES))
    model.to(DEVICE)

    if USE_CLASS_WEIGHTED_LOSS:
        class_weights = compute_class_weights(train_dataset.labels,
                                              len(STAGE_B_CLASSES), DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        class_weights_list = class_weights.cpu().tolist()
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights_list = None

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # MODEL COMPLEXITY
    # --------------------------------------------------------
    param_info = count_parameters(model)
    sample_img, _ = val_dataset[0]
    input_shape = tuple(sample_img.shape)
    flop_info = measure_flops(model, input_shape, DEVICE)
    env_info = get_env_info()

    # --------------------------------------------------------
    # CONFIG LOG
    # --------------------------------------------------------
    def class_counts(ds):
        return {STAGE_B_CLASSES[i]: int(c)
                for i, c in sorted(Counter(ds.labels).items())}

    save_config(out_dir, {
        "experiment_name": experiment_name,
        "stage": "B",
        "model": model_name,
        "fold": fold,
        "classes": STAGE_B_CLASSES,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "use_patch_shuffle": USE_PATCH_SHUFFLE,
        "use_oversampling": USE_OVERSAMPLING,
        "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "class_weights": class_weights_list,
        "cohort_dir": str(cohort_dir),
        "fold_manifest": str(Path(cohort_dir) / "manifests" / f"stageB_fold{fold}.csv"),
        "split_counts_patches": {
            "train": len(train_dataset), "val": len(val_dataset),
            "test": len(test_dataset)},
        "split_counts_slides": {
            "train": n_slides(train_dataset), "val": n_slides(val_dataset),
            "test": n_slides(test_dataset)},
        "train_class_counts_patches": class_counts(train_dataset),
        "environment": env_info,
        "model_params": param_info,
        "model_flops": flop_info,
    })

    # --------------------------------------------------------
    # TRAIN (timed + energy-metered)
    # --------------------------------------------------------
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=DEVICE, output_dir=out_dir)

    train_meter = EnergyMeter()
    with train_meter:
        history, _, _ = trainer.train(NUM_EPOCHS)
    train_perf = dict(train_meter.result)

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
    # EVALUATION (val + test): slide-level soft vote + patch level
    # --------------------------------------------------------
    val_metrics, val_perf = run_eval_stageB(model, val_dataset, "val", out_dir)
    test_metrics, test_perf = run_eval_stageB(model, test_dataset, "test", out_dir)

    # --------------------------------------------------------
    # EFFICIENCY LOG
    # --------------------------------------------------------
    save_json(out_dir / "efficiency.json", {
        "experiment_name": experiment_name,
        "stage": "B", "model": model_name, "fold": fold, "device": DEVICE,
        "environment": env_info,
        "model_params": param_info,
        "model_flops": flop_info,
        "train": train_perf,
        "inference_val": val_perf,
        "inference_test": test_perf,
    })

    print("\nSaved predictions (slide + patch), metrics, plots, and efficiency log.")


if __name__ == "__main__":
    main()
#python -m scripts.stage_b.train_stageB --fold 1