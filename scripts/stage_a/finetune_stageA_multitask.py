"""
Stage A fine-tuning with an auxiliary molecular-subtype head (Option 2).

Warm-starts the grade network of one fold from its existing Stage A checkpoint
and continues training with a second linear head that predicts the molecular
subtype (IDH_mt / IDH_mt_1p19q / IDH_wt) from the same 2048x2048 thumbnail
features. Loss = CE_grade + lambda * CE_subtype. Control slides carry no
subtype and are ignored in the subtype term. The grade head is what drives
patch injection, so the cascade is otherwise unchanged; the auxiliary head only
shapes the shared features (and can optionally be fused with Stage B, see
scripts/analysis/compare_systems.py).

--aux-weight 0 gives the matched CONTROL: identical warm start, optimiser,
schedule, epochs and checkpoint rule, grade loss only. Comparing it with the
multitask run separates the effect of the auxiliary task from the effect of
simply training longer.

Checkpoint selection uses the validation split only:
  lambda > 0 : mean(grade macro-F1, subtype macro-F1 on tumour slides)
  lambda = 0 : grade macro-F1
Epoch 0 (the untouched warm start) is never a candidate.

Run (from repo root), one fold per call:
    python -m scripts.stage_a.finetune_stageA_multitask --model resnet18 \
        --split-csv splits/split_01.csv --aux-weight 0.5
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glioma_sparse.data_utils.patches import Patch
from glioma_sparse.data_utils.slide_dataset import SlideDataset
from glioma_sparse.data_utils.subtype_labels import (
    SUBTYPE_CLASSES, IGNORE_INDEX, build_subtype_index, subtype_label,
)
from glioma_sparse.data_utils.transforms import build_eval_transform, build_train_transform
from glioma_sparse.evaluation.metrics import compute_classification_metrics
from glioma_sparse.models.multitask import GradeAuxNet
from glioma_sparse.training.seeding import make_generator, seed_worker, set_seed
from scripts.lib.eval_metrics import compute_metrics
from scripts.lib.profiling import EnergyMeter, get_env_info
from scripts.stage_a.train import _relkey, compute_class_weights, load_split

CLASS_ORDER = ["control", "low_grade", "high_grade"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultiTaskSlideDataset(SlideDataset):
    """SlideDataset that also returns the subtype label (IGNORE_INDEX for controls)."""

    def __init__(self, *a, subtype_index, **kw):
        super().__init__(*a, **kw)
        self.sub_labels = [subtype_label(Path(p).stem, subtype_index)
                           for p in self.paths]

    def __getitem__(self, idx):
        img, grade = super().__getitem__(idx)
        return img, grade, self.sub_labels[idx]


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--split-csv", required=True)
    p.add_argument("--init-ckpt", default=None,
                   help="Stage A grade checkpoint to warm-start from "
                        "(default: <init-root>/*<split stem>*/best_f1.pth).")
    p.add_argument("--init-root", default=None,
                   help="Default ResNet{18,50}_output/training_output.")
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="lambda for the subtype loss; 0 = grade-only control.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-5, help="backbone + grade head")
    p.add_argument("--head-lr", type=float, default=1e-3, help="new subtype head")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Train in fp32. Default: bf16 autocast + channels-last for the "
                        "training forward/backward only (2048x2048 inputs do not fit a "
                        "16 GB GPU in fp32 at batch 8); evaluation is always fp32.")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "included"))
    p.add_argument("--sidecar-root", default=str(REPO_ROOT / "data" / "ebrains_thumbnails"))
    p.add_argument("--out-root", default=None,
                   help="Default training_output_stageA_mt (or _ctrl when lambda=0), "
                        "inside ResNet{18,50}_output-style folder under the repo root.")
    p.add_argument("--tag", default=None)
    p.add_argument("--limit-train", type=int, default=0,
                   help="Smoke test: use only N training slides.")
    return p.parse_args()


def evaluate(model, loader, use_aux):
    """Grade + subtype probabilities over a loader."""
    model.eval()
    g_prob, g_true, s_prob, s_true = [], [], [], []
    with torch.no_grad():
        for imgs, grade, sub in loader:
            g, s = model(imgs.to(DEVICE))
            g_prob.append(torch.softmax(g, 1).cpu().numpy())
            s_prob.append(torch.softmax(s, 1).cpu().numpy())
            g_true.append(grade.numpy())
            s_true.append(sub.numpy())
    return (np.concatenate(g_true), np.concatenate(g_prob),
            np.concatenate(s_true), np.concatenate(s_prob))


def val_scores(gt, gp, st, sp, use_aux):
    gm = compute_classification_metrics(gp, gt)
    m = st != IGNORE_INDEX
    sm = (compute_classification_metrics(sp[m], st[m]) if m.any()
          else {"f1": float("nan"), "auc": float("nan"), "accuracy": float("nan")})
    select = 0.5 * (gm["f1"] + sm["f1"]) if use_aux else gm["f1"]
    return gm, sm, select


def main():
    a = parse_args()
    use_aux = a.aux_weight > 0
    arm = "mt" if use_aux else "ctrl"
    arch_tag = {"resnet18": "ResNet18", "resnet50": "ResNet50"}.get(a.model, a.model)
    split_csv = Path(a.split_csv)
    split_stem = split_csv.stem

    init_root = Path(a.init_root or REPO_ROOT / f"{arch_tag}_output" / "training_output")
    if a.init_ckpt:
        init_ckpt = Path(a.init_ckpt)
    else:
        hits = sorted(glob.glob(str(init_root / f"*{split_stem}*" / "best_f1.pth")))
        assert hits, f"no init checkpoint under {init_root} for {split_stem}"
        init_ckpt = Path(hits[0])

    set_seed(a.seed)
    out_root = Path(a.out_root or REPO_ROOT / f"{arch_tag}_output" /
                    f"training_output_stageA_{arm}")
    name = a.tag or f"{time.strftime('%Y%m%d_%H%M')}_{a.model}_{arm}_{split_stem}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage A {arm} | {a.model} | {split_stem} | init {init_ckpt}\n-> {out_dir}")

    # ---- data ----------------------------------------------------------
    data_dir = Path(a.data_dir)
    base = SlideDataset(root_dir=data_dir, class_names=CLASS_ORDER)
    split_map = load_split(split_csv)
    parts = {"train": [], "val": [], "test": []}
    for p in base.paths:
        parts[split_map[_relkey(p)]].append(p)
    if a.limit_train:
        step = max(1, len(parts["train"]) // a.limit_train)
        parts["train"] = parts["train"][::step][:a.limit_train]
    sub_index = build_subtype_index(a.sidecar_root)

    def make(split, train):
        return MultiTaskSlideDataset(
            data_dir, transform=build_train_transform() if train else build_eval_transform(),
            patch_transform=Patch(8) if train else None,
            class_names=CLASS_ORDER, paths=parts[split], subtype_index=sub_index)

    ds = {s: make(s, s == "train") for s in parts}
    print({s: len(d) for s, d in ds.items()},
          "| tumour slides without subtype label:",
          sum(1 for p, l in zip(ds["train"].paths, ds["train"].labels)
              if l != 0 and ds["train"].sub_labels[ds["train"].paths.index(p)] == IGNORE_INDEX))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ds["train"], batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        worker_init_fn=seed_worker, generator=make_generator(a.seed),
        pin_memory=pin, persistent_workers=a.workers > 0)
    eval_loaders = {s: DataLoader(ds[s], batch_size=a.batch_size, shuffle=False,
                                  num_workers=a.workers, worker_init_fn=seed_worker,
                                  pin_memory=pin, persistent_workers=a.workers > 0)
                    for s in ("val", "test")}

    # ---- model / loss / optimiser ---------------------------------------
    model = GradeAuxNet(a.model, n_grade=3, n_subtype=len(SUBTYPE_CLASSES))
    model.load_grade(init_ckpt, map_location=DEVICE)
    model.to(DEVICE)
    if a.amp and DEVICE == "cuda":
        model.to(memory_format=torch.channels_last)

    grade_w = compute_class_weights(ds["train"].labels, 3, DEVICE)
    grade_loss_fn = nn.CrossEntropyLoss(weight=grade_w)
    sub_train = [s for s in ds["train"].sub_labels if s != IGNORE_INDEX]
    sub_w = compute_class_weights(sub_train, len(SUBTYPE_CLASSES), DEVICE)
    sub_loss_fn = nn.CrossEntropyLoss(weight=sub_w, ignore_index=IGNORE_INDEX)

    opt = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": a.lr},
        {"params": model.aux.parameters(), "lr": a.head_lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs, eta_min=0.1 * a.lr)

    config = {**vars(a), "arm": arm, "init_ckpt": str(init_ckpt),
              "class_weights_grade": grade_w.cpu().tolist(),
              "class_weights_subtype": sub_w.cpu().tolist(),
              "split_counts": {s: len(d) for s, d in ds.items()},
              "git_commit": git_hash(), "environment": get_env_info(),
              "selection_rule": ("mean(grade macro-F1, subtype macro-F1)" if use_aux
                                 else "grade macro-F1")}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    log_path = out_dir / "log.csv"
    log_path.write_text("epoch,train_loss,train_loss_grade,train_loss_subtype,"
                        "val_grade_f1,val_grade_auc,val_grade_acc,"
                        "val_sub_f1,val_sub_auc,val_sub_acc,select,epoch_sec\n")

    # ---- train -----------------------------------------------------------
    best, since = -1.0, 0
    meter = EnergyMeter()
    epochs_run = 0
    with meter:
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            model.train()
            tl = tg = ts = 0.0
            nb = 0
            for imgs, grade, sub in train_loader:
                imgs, grade, sub = imgs.to(DEVICE), grade.to(DEVICE), sub.to(DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=a.amp and DEVICE == "cuda"):
                    g_logit, s_logit = model(
                        imgs.contiguous(memory_format=torch.channels_last)
                        if a.amp and DEVICE == "cuda" else imgs)
                g_logit, s_logit = g_logit.float(), s_logit.float()
                lg = grade_loss_fn(g_logit, grade)
                if use_aux and (sub != IGNORE_INDEX).any():
                    ls = sub_loss_fn(s_logit, sub)
                else:
                    ls = torch.zeros((), device=DEVICE)
                loss = lg + a.aux_weight * ls
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item(); tg += lg.item(); ts += ls.item(); nb += 1
            sched.step()

            gt, gp, st, sp = evaluate(model, eval_loaders["val"], use_aux)
            gm, sm, sel = val_scores(gt, gp, st, sp, use_aux)
            dt = time.time() - t0
            with open(log_path, "a") as f:
                f.write(f"{ep},{tl/nb:.5f},{tg/nb:.5f},{ts/nb:.5f},"
                        f"{gm['f1']:.4f},{gm['auc']:.4f},{gm['accuracy']:.4f},"
                        f"{sm['f1']:.4f},{sm['auc']:.4f},{sm['accuracy']:.4f},"
                        f"{sel:.4f},{dt:.1f}\n")
            print(f"ep {ep}/{a.epochs} loss {tl/nb:.3f} | val grade F1 {gm['f1']:.3f} "
                  f"AUC {gm['auc']:.3f} | val subtype F1 {sm['f1']:.3f} AUC {sm['auc']:.3f} "
                  f"| select {sel:.3f} | {dt:.0f}s")
            epochs_run = ep
            model.save(out_dir / "last.pth", out_dir / "last_aux.pth")
            if sel > best:
                best, since = sel, 0
                model.save(out_dir / "best_f1.pth", out_dir / "best_aux.pth")
                print("  saved best")
            else:
                since += 1
                if since >= a.patience:
                    print("early stop")
                    break
    train_perf = dict(meter.result)
    train_perf.update(epochs_run=epochs_run, epochs_max=a.epochs)

    # ---- final evaluation with the selected checkpoint --------------------
    model.load_grade(out_dir / "best_f1.pth", map_location=DEVICE)
    model.load_aux(out_dir / "best_aux.pth", map_location=DEVICE)
    model.to(DEVICE)
    results = {}
    for split in ("val", "test"):
        gt, gp, st, sp = evaluate(model, eval_loaders[split], use_aux)
        m = st != IGNORE_INDEX
        paths = [str(p) for p in ds[split].paths]
        np.savez(out_dir / f"{split}_predictions.npz",
                 y_true=gt, y_prob=gp, class_order=np.array(CLASS_ORDER),
                 sub_true=st, sub_prob=sp, sub_class_order=np.array(SUBTYPE_CLASSES),
                 slide_ids=np.array([Path(p).stem for p in paths]))
        gm = compute_metrics(gt, gp, CLASS_ORDER)
        sm = compute_metrics(st[m], sp[m], SUBTYPE_CLASSES)
        results[split] = {"grade": gm, "subtype_head_tumour_only": sm}
        (out_dir / f"metrics_{split}.json").write_text(
            json.dumps({"grade": gm, "subtype_head_tumour_only": sm}, indent=2))
        print(f"{split.upper()}  grade acc {gm['accuracy']:.3f} macro-AUC "
              f"{gm['macro_auc']:.3f} | subtype-head acc {sm['accuracy']:.3f} "
              f"macro-AUC {sm['macro_auc']:.3f}")
    (out_dir / "efficiency.json").write_text(json.dumps(
        {"train": train_perf, "device": DEVICE}, indent=2, default=str))


if __name__ == "__main__":
    main()
