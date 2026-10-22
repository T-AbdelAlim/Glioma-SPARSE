"""
End-to-end held-out evaluation of several Stage B variants on the same regions.

Runs the exact per-slide pipeline of `inference_end_to_end.py --testset`
(Stage A grade -> patch-injection risk map -> p95 tile selection -> high-res
extraction from the WSI -> Stage B soft vote) but scores any number of Stage B
checkpoints per fold on the SAME extracted patches, so systems that share a
Stage A differ only in Stage B and are compared on identical inputs. It reuses
the original helper functions unchanged; only the bookkeeping differs:

  * every system's slide-level Stage B probabilities are written, without and
    with 8-view dihedral test-time augmentation (TTA);
  * if an auxiliary Stage A subtype head is given (--use-aux), its thumbnail-
    level subtype probabilities are written too, so fusion with Stage B can be
    evaluated offline (scripts/analysis/compare_systems.py);
  * validation slides can be processed as well (--roles val,test) so that fusion
    weights are chosen on validation data only.

No integration rule is applied here: per-slide probabilities are saved and the
end-to-end decision (grade must be right AND subtype must match) is applied in
the analysis script, which reproduces the original `end_to_end_correct`.

Example:
    python -m scripts.inference.e2e_multi --run-name e2e_multi_rn18_A0 \
        --stage-a-glob "ResNet18_output/training_output/*_split_0*/best_f1.pth" \
        --system B0=ResNet18_output/training_output_stageB_F1/*fold*/best_f1.pth \
        --roles val,test --tta
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image, PngImagePlugin

# Some source WSIs carry very large embedded metadata (e.g. scanner XML) that PIL copies into
# a cached patch's PNG tEXt chunk on save; Pillow's default 1 MB decompressed-text-chunk guard
# then refuses to read that cache file back on a later run. Raise it generously (100 MB) since
# these are our own locally-written cache files, not untrusted input.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.data_utils.tta import predict_probs
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, load_mapping, thumbnail_bbox_to_wsi
from glioma_sparse.models.factory import build_model
from glioma_sparse.models.multitask import build_aux_head, pooled_features
from scripts.inference.inference_end_to_end import (
    GRADE_CLASSES, GRID, RESULTS_ROOT, SEED, SUBTYPE_CLASSES, _rel, _sidecar_for,
    _true_labels, extract_highres, resolve_wsi, select_tiles,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUB_KEYS = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def load_net(ckpt, name, n_classes):
    m = build_model(name, num_classes=n_classes, pretrained=False)
    m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()


def reroot(path):
    """Split CSVs store absolute paths from the machine that wrote them; keep the
    part from data/ onward and re-anchor it in this repo."""
    s = str(path).replace("\\", "/")
    i = s.rfind("/data/")
    return REPO_ROOT / s[i + 1:] if i >= 0 else Path(path)


def union_cols(rows):
    """Column names over all rows, in first-seen order (rows differ: no Stage B
    columns for slides that stop at Stage A)."""
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def pick(ckpts, token):
    hits = sorted(c for c in ckpts if token in str(c))
    return hits[0] if hits else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--stage-a-glob", required=True,
                   help="Per-fold Stage A checkpoints; the split stem (split_0k) must "
                        "appear in the path.")
    p.add_argument("--use-aux", action="store_true",
                   help="Also load best_aux.pth beside each Stage A checkpoint.")
    p.add_argument("--system", action="append", required=True, metavar="NAME=GLOB",
                   help="Stage B system: per-fold checkpoints; 'fold<k>' must appear "
                        "in the path. Repeatable.")
    p.add_argument("--roles", default="test", help="comma list of split roles to run.")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--sidecar-root", default=r"data\ebrains_thumbnails")
    p.add_argument("--wsi-root", default=r"C:\Users\TAbde\Documents\EMC_postdoc"
                                         r"\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data")
    p.add_argument("--control-image",
                   default="data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--min-tissue", type=float, default=0.25)
    p.add_argument("--output-size", type=int, default=2048)
    p.add_argument("--patch-cache", default="data/e2e_patch_cache",
                   help="Directory of lossless PNG patches keyed by (slide, row, col); "
                        "'' disables the cache.")
    p.add_argument("--external-labels", default=None,
                   help="CSV of an external cohort (columns slide_id, path, true_grade_class, true_mutation). "
                        "Every slide is scored by every fold model (role 'external'); thumbnails are created "
                        "on demand in --thumb-dir.")
    p.add_argument("--thumb-dir", default=None, help="thumbnail cache (external mode)")
    p.add_argument("--ext-stain-profile", default="none",
                   help="Stage A/B stain profile for external slides ('none' = no correction).")
    p.add_argument("--risk-batch", type=int, default=8,
                   help="batch size of the patch-injection forward passes (lower = less GPU memory; results unchanged)")
    p.add_argument("--slide-ids", default=None, help="Comma list of slide ids: only these slides are scored.")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: slides per fold.")
    p.add_argument("--resume", action="store_true",
                   help="Skip (fold, role) parts that already have a CSV.")
    return p.parse_args()


def main():
    a = parse_args()
    a.stain_profile = None if a.ext_stain_profile in (None, '', 'none') else a.ext_stain_profile
    systems = {}
    for spec in a.system:
        name, pattern = spec.split("=", 1)
        systems[name] = sorted(glob.glob(str(_rel(pattern))))
        assert systems[name], f"system {name}: no checkpoints match {pattern}"
    a_ckpts = sorted(glob.glob(str(_rel(a.stage_a_glob))))
    assert a_ckpts, "no Stage A checkpoints matched"
    roles = ["external"] if a.external_labels else [r.strip() for r in a.roles.split(",")]
    folds = [int(f) for f in a.folds.split(",")]

    if a.patch_cache:
        a.patch_cache = str(_rel(a.patch_cache))
        Path(a.patch_cache).mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_ROOT / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = build_eval_transform()
    control_img = Image.open(_rel(a.control_image)).convert("RGB")
    split_csvs = {int(re.search(r"split_(\d+)", p.stem).group(1)): p
                  for p in sorted(_rel(a.splits_dir).glob("split_0*.csv"))}

    used = {"git_commit": git_hash(), "args": vars(a), "checkpoints": {}}
    for fold in folds:
        sc = split_csvs[fold]
        a_ck = pick(a_ckpts, sc.stem)
        b_ck = {n: pick(c, f"fold{fold}") for n, c in systems.items()}
        assert a_ck and all(b_ck.values()), f"fold {fold}: missing checkpoint"
        used["checkpoints"][fold] = {"A": str(a_ck), **{n: str(c) for n, c in b_ck.items()}}
        model_a = load_net(a_ck, a.model, len(GRADE_CLASSES))
        aux = None
        if a.use_aux:
            aux = build_aux_head(a.model, len(SUB_KEYS)).to(DEVICE).eval()
            aux.load_state_dict(torch.load(Path(a_ck).with_name("best_aux.pth"),
                                           map_location=DEVICE, weights_only=True))
        models_b = {n: load_net(c, a.model, len(SUBTYPE_CLASSES)) for n, c in b_ck.items()}

        for role in roles:
            part = out_dir / f"per_slide_fold{fold}_{role}.csv"
            if a.resume and part.exists():
                print(f"[fold {fold} {role}] exists, skip")
                continue
            items = external_items(a) if a.external_labels else internal_items(a, sc, role)
            if a.slide_ids:
                keep = set(a.slide_ids.split(","))
                items = [it for it in items if it["stem"] in keep]
            if a.limit:
                items = items[:a.limit]
            print(f"[fold {fold} {role}] {len(items)} slides", flush=True)
            rows = []
            t_fold = time.time()
            for si, it in enumerate(items, 1):
                try:
                    row = process_slide(it["src"], it["stem"], it["true"], it["is_control"], fold, role,
                                        model_a, aux, models_b, transform, control_img, a,
                                        wsi_path=it.get("wsi_path"))
                    if it.get("extra"):
                        row.update(it["extra"])
                    rows.append(row)
                except Exception as e:
                    print(f"   FAILED {it['stem']}: {e}")
                if si % 10 == 0:
                    print(f"   {si}/{len(items)}  ({time.time()-t_fold:.0f}s)", flush=True)
            if rows:
                cols = union_cols(rows)
                with open(part, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
            print(f"[fold {fold} {role}] done in {time.time()-t_fold:.0f}s", flush=True)

    # merge parts
    all_rows = []
    for part in sorted(out_dir.glob("per_slide_fold*_*.csv")):
        all_rows.extend(csv.DictReader(open(part)))
    if all_rows:
        cols = union_cols(all_rows)
        with open(out_dir / "per_slide.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
    (out_dir / "run_config.json").write_text(json.dumps(used, indent=2, default=str))
    print(f"\nWrote {out_dir / 'per_slide.csv'} ({len(all_rows)} rows)")


def internal_items(a, sc, role):
    """Test/val slides of one split CSV (EBRAINS) as work items."""
    out = []
    for r in csv.DictReader(open(sc)):
        if r["split"] != role:
            continue
        split_path = reroot(r["path"])
        stem = split_path.stem
        if split_path.parent.name == "control":
            out.append({"src": split_path, "stem": stem, "is_control": True,
                        "true": {"grade_class": "control", "mutation": "NA"}})
            continue
        sc_thumb = _sidecar_for(stem, _rel(a.sidecar_root))
        if sc_thumb is None:
            print(f"   no sidecar for {stem}")
            continue
        out.append({"src": sc_thumb, "stem": stem, "is_control": False, "true": _true_labels(sc_thumb)})
    return out


def external_items(a):
    """External cohort slides as work items; thumbnails (2048 px, 4 um/px) + sidecars are created once."""
    import pandas as pd
    from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto
    lab = pd.read_csv(_rel(a.external_labels))
    tdir = Path(_rel(a.thumb_dir))
    tdir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in lab.itertuples():
        thumb = tdir / f"{r.slide_id}.jpg"
        if not (thumb.exists() and thumb.with_suffix(".json").exists()):
            create_wsi_thumbnail_auto(r.path, output_path=str(thumb), target_mpp=4.0, output_size=2048,
                                      stain_profile=a.stain_profile, verbose=False)
        out.append({"src": thumb, "stem": r.slide_id, "is_control": False, "wsi_path": r.path,
                    "true": {"grade_class": r.true_grade_class, "mutation": r.true_mutation},
                    "extra": {"cohort_type": r.type, "true6": r.true6, "source": getattr(r, "source", "")}})
    return out


def process_slide(src, stem, true, is_control, fold, role, model_a, aux, models_b,
                  transform, control_img, a, wsi_path=None):
    thumbnail = Image.open(src).convert("RGB")
    x = transform(thumbnail).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs_a = torch.softmax(model_a(x), dim=1).cpu().numpy()[0]
        sub_a = (torch.softmax(aux(pooled_features(model_a, x)), dim=1).cpu().numpy()[0]
                 if aux is not None else None)
    pred_idx = int(probs_a.argmax())
    grade_name = GRADE_CLASSES[pred_idx]
    grade_class = {"low_grade": "low", "high_grade": "high"}.get(grade_name, "control")

    row = {"slide_id": stem, "fold": fold, "role": role,
           "true_grade_class": true["grade_class"], "true_mutation": true["mutation"],
           "grade_pred": grade_name, "grade_class": grade_class,
           "prob_control": float(probs_a[0]), "prob_low_grade": float(probs_a[1]),
           "prob_high_grade": float(probs_a[2])}
    if sub_a is not None:
        for k, v in zip(SUB_KEYS, sub_a):
            row[f"stageA_sub_p_{k}"] = float(v)

    patches = []
    if not (grade_name == "control" or is_control):
        risk_map, _ = compute_risk_map(
            target_img=thumbnail, control_img=control_img, model=model_a,
            transform=transform, grid=GRID, target_class=pred_idx, device=DEVICE,
            control_shuffle_seed=SEED, score="logit", batch_size=a.risk_batch)
        selected, _ = select_tiles(risk_map, thumbnail, GRID, a.percentile,
                                   a.cap, a.min_tissue)
        row['risk_map'] = json.dumps([[round(float(v), 4) for v in rr] for rr in risk_map])
        if selected:
            mapping = load_mapping(src)
            if not mapping.has_wsi_mapping():
                raise RuntimeError("no WSI mapping")
            wsi_file = resolve_wsi(mapping.wsi_path, a, wsi_path)
            boxes = []
            for sel_ in selected:
                b_ = thumbnail_bbox_to_wsi(mapping, grid_cell_bbox(sel_['row'], sel_['col'], GRID[0], GRID[1],
                                                                  mapping.thumbnail_size))
                if b_ is not None:
                    boxes.append([int(v) for v in b_])
            row['bboxes'] = json.dumps(boxes)
            row['selected_rc'] = json.dumps([[sel_['row'], sel_['col']] for sel_ in selected])

            def read(sel):
                # A region is identified by (slide, row, col). Extracted patches are
                # kept losslessly (PNG), so systems that select the same region reuse
                # identical pixels instead of re-reading level 0 of the WSI (~7-30 s).
                cached = (Path(a.patch_cache) / f"{stem}_r{sel['row']}c{sel['col']}.png"
                          if a.patch_cache else None)
                if cached is not None and cached.exists():
                    return Image.open(cached).convert("RGB")
                slide = openslide.OpenSlide(wsi_file)
                try:
                    patch = extract_highres(slide, mapping, sel["row"], sel["col"],
                                            GRID, a.output_size)[0]
                finally:
                    slide.close()
                if patch is not None and cached is not None:
                    tmp = cached.with_suffix(f".{os.getpid()}.tmp.png")
                    patch.info = {}  # drop any inherited metadata before it gets written as a PNG text chunk
                    patch.save(tmp, compress_level=1)
                    os.replace(tmp, cached)
                return patch

            with ThreadPoolExecutor(max_workers=len(selected)) as ex:
                patches = [p for p in ex.map(read, selected) if p is not None]
    row["n_regions"] = len(patches)

    if patches:
        xb = torch.stack([transform(p) for p in patches]).to(DEVICE)
        for name, mb in models_b.items():
            with torch.no_grad():
                p = predict_probs(mb, xb, tta=False).mean(0).cpu().numpy()
                for k, v in zip(SUB_KEYS, p):
                    row[f"{name}__p_{k}"] = float(v)
                if a.tta:
                    pt = predict_probs(mb, xb, tta=True).mean(0).cpu().numpy()
                    for k, v in zip(SUB_KEYS, pt):
                        row[f"{name}__tta_p_{k}"] = float(v)
    return row


if __name__ == "__main__":
    main()
