"""
Re-run val+test evaluation for an already-trained Stage-A tcga_ext
checkpoint, without retraining.

Usage:
    python -m scripts.stage_a.reeval_stage_A_tcga_ext \
        --split-csv splits_tcga_ext/split_01.csv \
        --checkpoint training_output_tcga_ext/20260803_1920_resnet50_tcgaext_cw_split_01/best_auc.pth
"""

import argparse
from pathlib import Path

import torch

from scripts.stage_a.train_stage_A_cluster_tcga_ext import (
    CLASS_ORDER, DATA_DIR, DEVICE, load_split_rows, run_eval_split,
)
from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.seeding import seed_worker
from torch.utils.data import DataLoader

BATCH_SIZE = 8
NUM_WORKERS = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet50")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.checkpoint).parent

    all_rows = load_split_rows(args.split_csv)
    val_paths = [r["path"] for r in all_rows if r["split"] == "val"]
    test_paths = [r["path"] for r in all_rows if r["split"] == "test"]

    val_dataset = SlideDataset(DATA_DIR, transform=build_eval_transform(),
                                patch_transform=None, class_names=CLASS_ORDER,
                                paths=val_paths)
    test_dataset = SlideDataset(DATA_DIR, transform=build_eval_transform(),
                                 patch_transform=None, class_names=CLASS_ORDER,
                                 paths=test_paths)

    pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                             pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                              pin_memory=pin_memory)

    model = build_model(args.model, num_classes=len(CLASS_ORDER))
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)

    print(f"Re-evaluating {args.checkpoint}")
    print(f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    run_eval_split(model, val_loader, "val", out_dir)
    run_eval_split(model, test_loader, "test", out_dir)

    print("\nDone. metrics_val.json and metrics_test.json (+ plots) written to", out_dir)


if __name__ == "__main__":
    main()
