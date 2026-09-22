"""
Stage-B training on the EBRAINS+TCGA combined pool, with optional
stain/colour augmentation. Reads one manifest per fold from
build_stageB_cohort_tcga_ext.py (--cohort-dir/manifests/stageB_fold<k>.csv,
a `source` column tags each row). Not compatible with the old EBRAINS-only
cohorts (data/stage_b_cohort_RN50/ etc.) -- those used the 3-class Stage-A
model.

Usage:
    python -m scripts.stage_b.train_stage_B_cluster_tcga_ext \
        --model resnet50 --fold 1 \
        --cohort-dir data/stage_b_cohort_tcga_ext
"""

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
import time

import numpy as np
from PIL import Image

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
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cohort-dir", type=str, default="data/stage_b_cohort_tcga_ext",
                         help="Combined EBRAINS+TCGA Stage-B cohort from "
                              "build_stageB_cohort_tcga_ext.py (both sources, "
                              "one manifest per fold). NOT one of the "
                              "pre-existing EBRAINS-only cohorts.")
    parser.add_argument("--no-stain-jitter", action="store_true")
    parser.add_argument("--no-color-jitter", action="store_true")
    parser.add_argument("--color-jitter-strength", type=str, default="mild",
                         choices=["mild", "strong"])
    parser.add_argument("--only-correct-grade", action="store_true",
                         help="Drop train+val patches from slides Stage A "
                              "graded incorrectly (grade_class_correct==0) -- "
                              "their p95 regions were selected relative to the "
                              "wrong target class. Test is never filtered, "
                              "regardless of this flag, so it stays comparable "
                              "to the unfiltered run.")
    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = None
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_OUT_DIR = REPO_ROOT / "training_output_stageB_tcga_ext"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OVERSAMPLING = False
USE_CLASS_WEIGHTED_LOSS = True
USE_PATCH_SHUFFLE = True
BEST_CHECKPOINT_METRIC = "auc"

BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
SEED = 42

STAGE_B_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
CLASS_TO_IDX = {c: i for i, c in enumerate(STAGE_B_CLASSES)}


# ============================================================
# DATASET (resolves patches via each row's own patch_path)
# ============================================================

class ManifestPatchDatasetExt(Dataset):
    """Resolves each patch as <cohort_dir>/patches/fold_<fold>/<mutation>/<filename>,
    rebuilt from --cohort-dir rather than trusting the manifest's stored path."""

    def __init__(self, rows, cohort_dir, transform=None,
                 patch_transform=None, patch_subdir="patches"):
        self.transform = transform
        self.patch_transform = patch_transform
        self.samples = []
        self.slide_ids = []
        self.labels = []
        root = Path(cohort_dir)
        for r in rows:
            mut = r["true_mutation"]
            if mut not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[mut]
            fname = str(r["patch_path"]).replace("\\", "/").rsplit("/", 1)[-1]
            path = root / patch_subdir / f"fold_{r['fold']}" / mut / fname
            self.samples.append((path, label))
            self.slide_ids.append(r["slide_id"])
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = _open_image_retrying(path)
        if self.patch_transform is not None:
            img = self.patch_transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _open_image_retrying(path, attempts=5, base_delay=0.3):
    """Retry on transient OSError -- the cluster mount occasionally throws
    BlockingIOError on a file that's actually fine."""
    last_err = None
    for attempt in range(attempts):
        try:
            return Image.open(path).convert("RGB")
        except OSError as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise OSError(f"Failed to open {path} after {attempts} attempts") from last_err


# ============================================================
# UTILITIES
# ============================================================

def build_experiment_name(model_name, fold, only_correct_grade=False):
    parts = [model_name, "stageB", f"fold{fold}", "tcgaext"]
    if USE_CLASS_WEIGHTED_LOSS:
        parts.append("cw")
    if only_correct_grade:
        parts.append("correctgradeonly")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_{'_'.join(parts)}"


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_manifest(mpath):
    if not mpath.exists():
        return []
    with open(mpath, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            import csv as _csv
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(f, delimiter=delimiter))


def load_fold_rows(cohort_dir, fold):
    manifest_path = Path(cohort_dir) / "manifests" / f"stageB_fold{fold}.csv"
    rows = read_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"Fold manifest missing/empty: {manifest_path}\n"
                          f"Run build_stageB_cohort_tcga_ext.py first (and make "
                          f"sure it actually reached this fold, not just earlier ones).")
    # manifest stores source ("ebrains"/"tcga") in the split_csv column
    for r in rows:
        r["source"] = r["split_csv"]
    n_ebrains = sum(1 for r in rows if r["source"] == "ebrains")
    n_tcga = sum(1 for r in rows if r["source"] == "tcga")
    print(f"Loaded {manifest_path.name}: {len(rows)} rows ({n_ebrains} ebrains, {n_tcga} tcga)")
    return rows


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]


def compute_class_weights(train_labels, num_classes, device):
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        if i not in counts:
            raise ValueError(f"Class index {i} ({STAGE_B_CLASSES[i]}) has no training patches.")
        weights.append(total / (num_classes * counts[i]))
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    if torch.cuda.is_available():
        perf["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
    return y_patch, probs_patch, perf


def soft_vote(y_patch, probs_patch, slide_ids):
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

    np.savez(out_dir / f"{split_name}_predictions.npz",
             y_true=y_slide, y_prob=prob_slide,
             class_order=np.array(STAGE_B_CLASSES), slide_ids=np.array(slides))
    np.savez(out_dir / f"{split_name}_predictions_patch.npz",
             y_true=y_patch, y_prob=probs_patch,
             class_order=np.array(STAGE_B_CLASSES), slide_ids=np.array(dataset.slide_ids))

    metrics_slide = compute_metrics(y_slide, prob_slide, STAGE_B_CLASSES)
    metrics_patch = compute_metrics(y_patch, probs_patch, STAGE_B_CLASSES)
    save_json(out_dir / f"metrics_{split_name}.json",
              {"level": "slide", **metrics_slide, "patch_level": metrics_patch})

    preds_slide = prob_slide.argmax(axis=1)
    plot_confusion_matrix(y_slide, preds_slide, STAGE_B_CLASSES,
                           out_dir / f"confusion_matrix_{split_name}.png")
    plot_roc_curve(y_slide, prob_slide, STAGE_B_CLASSES, out_dir / f"roc_curve_{split_name}.png")

    print(f"  [slide] acc={metrics_slide['accuracy']:.3f}  macro_auc={metrics_slide['macro_auc']:.3f}  "
          f"macro_f1={metrics_slide['macro_f1']:.3f}")
    return metrics_slide, perf


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    model_name = args.model or os.environ.get("MODEL") or "resnet18"
    fold = args.fold

    print(f"\nStage B (EBRAINS+TCGA) | model {model_name} | fold {fold}")
    set_seed(SEED)

    experiment_name = build_experiment_name(model_name, fold, args.only_correct_grade)
    out_dir = BASE_OUT_DIR / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Experiment:", experiment_name, "| Device:", DEVICE)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------
    rows = load_fold_rows(args.cohort_dir, fold)
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")
    assert train_rows and val_rows and test_rows, "Fold manifests missing one of train/val/test."

    if args.only_correct_grade:
        # test stays unfiltered so it's comparable to the unfiltered run
        n_train_before, n_val_before = len(train_rows), len(val_rows)
        train_rows = [r for r in train_rows if r["grade_class_correct"] == "1"]
        val_rows = [r for r in val_rows if r["grade_class_correct"] == "1"]
        print(f"--only-correct-grade: train {n_train_before} -> {len(train_rows)} "
              f"({n_train_before - len(train_rows)} dropped), "
              f"val {n_val_before} -> {len(val_rows)} "
              f"({n_val_before - len(val_rows)} dropped)")
        assert train_rows and val_rows, \
            "--only-correct-grade left train or val empty for this fold."

    source_counts = dict(Counter(r["source"] for r in train_rows))
    class_source_counts = {
        cls: dict(Counter(r["source"] for r in train_rows if r["true_mutation"] == cls))
        for cls in STAGE_B_CLASSES
    }
    print("Train source counts (patches):", source_counts)
    print("Train per-class source breakdown:", class_source_counts)

    use_stain_jitter = not args.no_stain_jitter
    use_color_jitter = not args.no_color_jitter
    train_transform = build_train_transform_ext(
        use_stain_jitter=use_stain_jitter, use_color_jitter=use_color_jitter,
        color_jitter_strength=args.color_jitter_strength)
    print(f"Augmentation: stain_jitter={use_stain_jitter} "
          f"color_jitter={use_color_jitter} ({args.color_jitter_strength})")

    patch_tf = Patch(8) if USE_PATCH_SHUFFLE else None
    ds_kwargs = dict(cohort_dir=args.cohort_dir)
    train_dataset = ManifestPatchDatasetExt(train_rows, transform=train_transform,
                                             patch_transform=patch_tf, **ds_kwargs)
    val_dataset = ManifestPatchDatasetExt(val_rows, transform=build_eval_transform(),
                                           patch_transform=None, **ds_kwargs)
    test_dataset = ManifestPatchDatasetExt(test_rows, transform=build_eval_transform(),
                                            patch_transform=None, **ds_kwargs)

    if len(train_dataset):
        sample_path = train_dataset.samples[0][0]
        if not Path(sample_path).exists():
            raise SystemExit(f"Patch not found: {sample_path}")

    n_slides = lambda ds: len(set(ds.slide_ids))
    print(f"Train: {len(train_dataset)} patches / {n_slides(train_dataset)} slides | "
          f"Val: {len(val_dataset)} / {n_slides(val_dataset)} | "
          f"Test: {len(test_dataset)} / {n_slides(test_dataset)}")

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------
    pin_memory = torch.cuda.is_available()
    gen = make_generator(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                               generator=gen, pin_memory=pin_memory,
                               persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                             pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0)

    # --------------------------------------------------------
    # MODEL + LOSS
    # --------------------------------------------------------
    model = build_model(model_name, num_classes=len(STAGE_B_CLASSES))
    model.to(DEVICE)
    class_weights = compute_class_weights(train_dataset.labels, len(STAGE_B_CLASSES), DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    param_info = count_parameters(model)
    sample_img, _ = val_dataset[0]
    flop_info = measure_flops(model, tuple(sample_img.shape), DEVICE)
    env_info = get_env_info()

    def class_counts(ds):
        return {STAGE_B_CLASSES[i]: int(c) for i, c in sorted(Counter(ds.labels).items())}

    save_json(out_dir / "config.json", {
        "experiment_name": experiment_name, "stage": "B", "model": model_name, "fold": fold,
        "classes": STAGE_B_CLASSES, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "seed": SEED,
        "use_patch_shuffle": USE_PATCH_SHUFFLE,
        "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "class_weights": class_weights.cpu().tolist(),
        "cohort_dir": str(args.cohort_dir),
        "only_correct_grade": args.only_correct_grade,
        "train_source_counts": source_counts, "train_class_source_breakdown": class_source_counts,
        "use_stain_jitter": use_stain_jitter, "use_color_jitter": use_color_jitter,
        "color_jitter_strength": args.color_jitter_strength,
        "split_counts_patches": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        "split_counts_slides": {"train": n_slides(train_dataset), "val": n_slides(val_dataset), "test": n_slides(test_dataset)},
        "train_class_counts_patches": class_counts(train_dataset),
        "environment": env_info, "model_params": param_info, "model_flops": flop_info,
    })

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                       optimizer=optimizer, criterion=criterion, device=DEVICE, output_dir=out_dir)
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

    val_metrics, val_perf = run_eval_stageB(model, val_dataset, "val", out_dir)
    test_metrics, test_perf = run_eval_stageB(model, test_dataset, "test", out_dir)

    save_json(out_dir / "efficiency.json", {
        "experiment_name": experiment_name, "stage": "B", "model": model_name, "fold": fold,
        "device": DEVICE, "environment": env_info, "model_params": param_info, "model_flops": flop_info,
        "train": train_perf, "inference_val": val_perf, "inference_test": test_perf,
    })
    print("\nSaved predictions (slide + patch), metrics, plots, and efficiency log.")


if __name__ == "__main__":
    main()
