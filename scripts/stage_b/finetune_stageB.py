"""
Stage B fine-tuning (Option 1): warm start from an existing Stage B checkpoint.

Stage B is already trained on the regions that the same fold's Stage A selects
(build_stageB_cohort.py), so what this script changes is HOW Stage B continues
to learn from them. Two recipes share the same warm start, cohort, epoch budget,
patch-level validation F1 checkpoint rule and slide-level soft voting:

  --recipe continue   matched CONTROL. Original hyper-parameters (Adam, constant
                      lr 1e-4, plain cross-entropy, Patch(8) shuffle only).
  --recipe refined    lr 2e-5 with cosine decay, label smoothing 0.1, weight
                      decay 1e-4 (AdamW), plus flips / 90-degree rotations and
                      mild colour jitter on top of Patch(8).

Every run also writes slide-level predictions with 8-view dihedral test-time
augmentation (TTA) next to the plain ones, so TTA can be evaluated on its own.

The checkpoint is chosen on the validation split only (patch-level macro-F1,
like the original Trainer); epoch 0 (the untouched warm start) is not a
candidate. Test slides are never used for any decision.

Run (from repo root), one fold per call:
    python -m scripts.stage_b.finetune_stageB --model resnet18 --fold 1 \
        --cohort-dir data/stage_b_cohort_RN18_F1 --recipe refined
"""

import argparse
import glob
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.transforms import build_eval_transform, build_train_transform
from glioma_sparse.data_utils.tta import predict_probs
from glioma_sparse.evaluation.metrics import compute_classification_metrics
from glioma_sparse.models.factory import build_model
from glioma_sparse.training.seeding import make_generator, seed_worker, set_seed
from scripts.lib.eval_metrics import compute_metrics
from scripts.lib.profiling import EnergyMeter, get_env_info
from scripts.stage_b.train_stageB import (
    STAGE_B_CLASSES, ManifestPatchDataset, compute_class_weights,
    read_fold_manifest, soft_vote, split_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RECIPES = {
    "continue": dict(lr=1e-4, label_smoothing=0.0, weight_decay=0.0,
                     aug="none", schedule="constant"),
    "refined": dict(lr=2e-5, label_smoothing=0.1, weight_decay=1e-4,
                    aug="geo_color", schedule="cosine"),
}


class DihedralJitter:
    """Random dihedral transform (flip / 90-degree rotation) on a PIL image."""

    def __call__(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        k = random.randint(0, 3)
        if k:
            img = img.transpose({1: Image.ROTATE_90, 2: Image.ROTATE_180,
                                 3: Image.ROTATE_270}[k])
        return img


def train_transform(aug):
    pre = []
    if aug in ("geo", "geo_color"):
        pre.append(DihedralJitter())
    if aug == "geo_color":
        pre.append(transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                          saturation=0.1, hue=0.02))
    return transforms.Compose(pre + [build_train_transform()])


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--cohort-dir", required=True,
                   help="Stage B cohort to train/evaluate on.")
    p.add_argument("--recipe", choices=list(RECIPES), required=True)
    p.add_argument("--init-ckpt", default=None,
                   help="Stage B checkpoint to warm-start from (default: "
                        "<init-root>/*fold<k>*/best_f1.pth).")
    p.add_argument("--init-root", default=None,
                   help="Default ResNet{18,50}_output/training_output_stageB_F1.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Train in fp32. Default: bf16 autocast + channels-last for the "
                        "training forward/backward only (2048x2048 inputs do not fit a "
                        "16 GB GPU in fp32 at batch 8); evaluation is always fp32.")
    p.add_argument("--out-root", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--limit-train", type=int, default=0,
                   help="Smoke test: only N training patches.")
    return p.parse_args()


def patch_probs(model, loader, tta):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, y in loader:
            ps.append(predict_probs(model, imgs.to(DEVICE), tta).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    a = parse_args()
    hp = RECIPES[a.recipe]
    arch_tag = {"resnet18": "ResNet18", "resnet50": "ResNet50"}.get(a.model, a.model)
    cohort = Path(a.cohort_dir)
    init_root = Path(a.init_root or REPO_ROOT / f"{arch_tag}_output" /
                     "training_output_stageB_F1")
    if a.init_ckpt:
        init_ckpt = Path(a.init_ckpt)
    else:
        hits = sorted(glob.glob(str(init_root / f"*fold{a.fold}_*" / "best_f1.pth")))
        assert hits, f"no init checkpoint under {init_root} for fold {a.fold}"
        init_ckpt = Path(hits[0])

    set_seed(a.seed)
    out_root = Path(a.out_root or REPO_ROOT / f"{arch_tag}_output" /
                    f"training_output_stageB_ft_{a.recipe}")
    name = a.tag or f"{time.strftime('%Y%m%d_%H%M')}_{a.model}_stageB_{a.recipe}_fold{a.fold}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage B {a.recipe} | {a.model} | fold {a.fold} | cohort {cohort}\n"
          f"init {init_ckpt}\n-> {out_dir}")

    rows = read_fold_manifest(cohort, a.fold)
    tr, va, te = (split_rows(rows, s) for s in ("train", "val", "test"))
    train_ds = ManifestPatchDataset(tr, cohort, transform=train_transform(hp["aug"]),
                                    patch_transform=Patch(8))
    val_ds = ManifestPatchDataset(va, cohort, transform=build_eval_transform())
    test_ds = ManifestPatchDataset(te, cohort, transform=build_eval_transform())
    if a.limit_train:
        step = max(1, len(train_ds.samples) // a.limit_train)
        sel = list(range(0, len(train_ds.samples), step))[:a.limit_train]
        train_ds.samples = [train_ds.samples[i] for i in sel]
        train_ds.labels = [train_ds.labels[i] for i in sel]
        train_ds.slide_ids = [train_ds.slide_ids[i] for i in sel]
    print(f"patches train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} | "
          f"slides {len(set(train_ds.slide_ids))}/{len(set(val_ds.slide_ids))}/"
          f"{len(set(test_ds.slide_ids))}")

    pin = torch.cuda.is_available()
    mk = lambda ds, shuffle, gen=None: DataLoader(
        ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers,
        worker_init_fn=seed_worker, generator=gen, pin_memory=pin,
        persistent_workers=a.workers > 0)
    train_loader = mk(train_ds, True, make_generator(a.seed))
    val_loader, test_loader = mk(val_ds, False), mk(test_ds, False)

    model = build_model(a.model, num_classes=len(STAGE_B_CLASSES))
    model.load_state_dict(torch.load(init_ckpt, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    use_amp = a.amp and DEVICE == "cuda"
    if use_amp:
        model.to(memory_format=torch.channels_last)

    w = compute_class_weights(train_ds.labels, len(STAGE_B_CLASSES), DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=hp["label_smoothing"])
    opt = (torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
           if hp["weight_decay"] > 0 else
           torch.optim.Adam(model.parameters(), lr=hp["lr"]))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=a.epochs, eta_min=0.1 * hp["lr"])
             if hp["schedule"] == "cosine" else None)

    (out_dir / "config.json").write_text(json.dumps({
        **vars(a), "hyperparameters": hp, "init_ckpt": str(init_ckpt),
        "class_weights": w.cpu().tolist(),
        "train_class_counts_patches": {STAGE_B_CLASSES[k]: v for k, v in
                                       sorted(Counter(train_ds.labels).items())},
        "split_counts_patches": {"train": len(train_ds), "val": len(val_ds),
                                 "test": len(test_ds)},
        "selection_rule": "patch-level validation macro-F1 (epoch 0 excluded)",
        "git_commit": git_hash(), "environment": get_env_info()},
        indent=2, default=str))
    log = out_dir / "log.csv"
    log.write_text("epoch,train_loss,val_loss,val_acc,val_f1,val_auc,epoch_sec\n")

    best, since, epochs_run = -1.0, 0, 0
    meter = EnergyMeter()
    with meter:
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            model.train()
            tl, nb = 0.0, 0
            for imgs, y in train_loader:
                imgs, y = imgs.to(DEVICE), y.to(DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(imgs.contiguous(memory_format=torch.channels_last)
                                if use_amp else imgs)
                loss = crit(out.float(), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item(); nb += 1
            if sched:
                sched.step()
            yv, pv = patch_probs(model, val_loader, tta=False)
            m = compute_classification_metrics(pv, yv)
            vl = float(nn.functional.nll_loss(torch.log(torch.tensor(pv) + 1e-12),
                                              torch.tensor(yv)))
            dt = time.time() - t0
            with open(log, "a") as f:
                f.write(f"{ep},{tl/nb:.5f},{vl:.5f},{m['accuracy']:.4f},"
                        f"{m['f1']:.4f},{m['auc']:.4f},{dt:.1f}\n")
            print(f"ep {ep}/{a.epochs} loss {tl/nb:.3f} | val patch F1 {m['f1']:.3f} "
                  f"AUC {m['auc']:.3f} acc {m['accuracy']:.3f} | {dt:.0f}s")
            epochs_run = ep
            torch.save(model.state_dict(), out_dir / "last.pth")
            if m["f1"] > best:
                best, since = m["f1"], 0
                torch.save(model.state_dict(), out_dir / "best_f1.pth")
                print("  saved best")
            else:
                since += 1
                if since >= a.patience:
                    print("early stop")
                    break
    train_perf = dict(meter.result)
    train_perf.update(epochs_run=epochs_run, epochs_max=a.epochs)

    model.load_state_dict(torch.load(out_dir / "best_f1.pth", map_location=DEVICE,
                                     weights_only=True))
    model.to(DEVICE)
    summary = {}
    for split, loader, ds in (("val", val_loader, val_ds), ("test", test_loader, test_ds)):
        for tta in (False, True):
            y, p = patch_probs(model, loader, tta)
            slides, ys, ps = soft_vote(y, p, ds.slide_ids)
            suffix = "_tta" if tta else ""
            np.savez(out_dir / f"{split}_predictions{suffix}.npz",
                     y_true=ys, y_prob=ps, class_order=np.array(STAGE_B_CLASSES),
                     slide_ids=np.array(slides))
            if not tta:
                np.savez(out_dir / f"{split}_predictions_patch.npz",
                         y_true=y, y_prob=p, class_order=np.array(STAGE_B_CLASSES),
                         slide_ids=np.array(ds.slide_ids))
            sm = compute_metrics(ys, ps, STAGE_B_CLASSES)
            summary[f"{split}{suffix}"] = sm
            print(f"{split.upper()}{' +TTA' if tta else ''}  slide acc "
                  f"{sm['accuracy']:.3f} macro-AUC {sm['macro_auc']:.3f} "
                  f"macro-F1 {sm['macro_f1']:.3f}")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "efficiency.json").write_text(json.dumps(
        {"train": train_perf, "device": DEVICE}, indent=2, default=str))


if __name__ == "__main__":
    main()
