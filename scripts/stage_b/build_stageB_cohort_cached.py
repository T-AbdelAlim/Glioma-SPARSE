"""
Stage B cohort builder with a patch cache (same output as build_stageB_cohort.py).

The selection logic is the original one (that fold's Stage A model predicts the
grade, computes the injection risk map for the predicted class, keeps the p95
tiles after the tissue filter, capped) and the manifest schema is unchanged, so
the result is a drop-in cohort for train_stageB.py / finetune_stageB.py. Only how
the high-resolution patch is obtained differs:

  * A region is fully identified by (slide, grid row, grid col). If a patch for
    the same region already exists (from an earlier cohort, or from another fold
    of this build) the JPEG is reused instead of re-reading level 0 of the WSI
    (~7 s per patch from the external drive).
  * Patches of one slide that are not cached are read in parallel threads.
  * --resume skips folds whose per-fold manifest is already complete.

The cache is only trusted after --verify-cache confirms that a freshly extracted
patch is pixel-identical to its cached JPEG (default: 3 random cached regions).

Example:
    python -m scripts.stage_b.build_stageB_cohort_cached \
        --checkpoints-glob "ResNet18_output/training_output_stageA_mt/*_split_0*/best_f1.pth" \
        --cache-dirs data/stage_b_cohort_RN18_F1 --model resnet18 \
        --control-image data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg \
        --wsi-root "<WHO2021_data>" --out-dir data/stage_b_cohort_RN18_A_mt
"""

import argparse
import csv
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openslide
import torch
from PIL import Image

from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import load_mapping
from scripts.stage_b.build_stageB_cohort import (
    CONTROL_SHUFFLE_SEED, GRID, MANIFEST_FIELDS, derive_true_labels,
    extract_region_highres, get_sidecar_thumbnail, load_model, load_split_rows,
    pred_grade_class, predict_class, resolve_checkpoints, resolve_wsi_path,
    select_p95_tiles,
)
from scripts.stage_b.build_stageB_cohort import CLASS_ORDER

NAME_RE = re.compile(r"^(?P<stem>.+)_fold\d+_(?:train|val|test)_rank\d+_r(?P<r>\d+)c(?P<c>\d+)\.jpg$")


def index_cache(dirs):
    """(slide stem, row, col) -> path of an existing high-res patch JPEG."""
    idx = {}
    for d in dirs:
        for p in Path(d).glob("patches/fold_*/*/*.jpg"):
            m = NAME_RE.match(p.name)
            if m:
                idx.setdefault((m["stem"], int(m["r"]), int(m["c"])), p)
    return idx


def verify_cache(idx, args, n=3, seed=0):
    """Fresh extraction of a few cached regions must equal their cached JPEG."""
    rng = random.Random(seed)
    keys = rng.sample(sorted(idx), min(n, len(idx)))
    for stem, r, c in keys:
        sc_thumb = get_sidecar_thumbnail(stem, args.sidecar_root)
        mapping = load_mapping(sc_thumb)
        slide = openslide.OpenSlide(resolve_wsi_path(mapping.wsi_path, args))
        try:
            fresh, _ = extract_region_highres(slide, mapping, r, c, GRID, args.output_size)
        finally:
            slide.close()
        tmp = Path(args.out_dir) / f"_verify_{os.getpid()}.jpg"
        fresh.save(tmp, quality=90)
        a = Image.open(tmp).convert("RGB")
        b = Image.open(idx[(stem, r, c)]).convert("RGB")
        diff = sum(1 for x, y in zip(a.tobytes(), b.tobytes()) if x != y)
        tmp.unlink()
        print(f"  cache check {stem[:8]} r{r}c{c}: differing bytes = {diff}")
        if diff:
            raise SystemExit("cache verification failed: extraction is not reproducible; "
                             "run without --cache-dirs")


def get_patch(sel, stem, dest, wsi_path, mapping, idx, args):
    """Return (wsi_bbox, cache_hit). Writes the JPEG to dest."""
    key = (stem, sel["row"], sel["col"])
    hit = idx.get(key)
    from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, thumbnail_bbox_to_wsi
    bbox = thumbnail_bbox_to_wsi(
        mapping, grid_cell_bbox(sel["row"], sel["col"], GRID[0], GRID[1],
                                mapping.thumbnail_size))
    if bbox is None:
        return None, False
    if hit is not None and hit.exists():
        shutil.copyfile(hit, dest)
        return bbox, True
    # shared lossless cache (also filled by scripts/inference/e2e_multi.py); a PNG
    # re-encoded at q90 equals a fresh extraction saved at q90
    png = (Path(args.png_cache) / f"{stem}_r{sel['row']}c{sel['col']}.png"
           if args.png_cache else None)
    if png is not None and png.exists():
        Image.open(png).convert("RGB").save(dest, quality=90)
        idx[key] = dest
        return bbox, True
    slide = openslide.OpenSlide(wsi_path)
    try:
        patch, bbox = extract_region_highres(slide, mapping, sel["row"], sel["col"],
                                             GRID, args.output_size)
    finally:
        slide.close()
    if patch is None:
        return None, False
    if png is not None:
        tmp = png.with_suffix(f".{os.getpid()}.tmp.png")
        patch.save(tmp, compress_level=1)
        os.replace(tmp, png)
    patch.save(dest, quality=90)
    idx[key] = dest
    return bbox, False


def process_slide(thumb_path, role, model, transform, control_img, device, fold_idx,
                  split_csv, args, patches_dir, idx, stats):
    if thumb_path.parent.name == "control":
        return []
    stem = thumb_path.stem
    sc_thumb = get_sidecar_thumbnail(stem, args.sidecar_root)
    if sc_thumb is None:
        raise RuntimeError(f"no sidecar thumbnail for {stem}")
    labels = derive_true_labels(sc_thumb.parent.parent.name)

    thumbnail = Image.open(sc_thumb).convert("RGB")
    pred_idx, probs = predict_class(model, transform, thumbnail, device)
    pred_name = CLASS_ORDER[pred_idx]
    pgc = pred_grade_class(pred_name)
    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img, model=model,
        transform=transform, grid=GRID, target_class=pred_idx, device=device,
        control_shuffle_seed=CONTROL_SHUFFLE_SEED)
    selected, thresh, n_above = select_p95_tiles(
        risk_map, thumbnail, GRID, percentile=args.percentile, cap=args.cap,
        min_tissue=args.min_tissue)
    if not selected:
        return []

    mapping = load_mapping(sc_thumb)
    if not mapping.has_wsi_mapping():
        raise RuntimeError("no WSI mapping in sidecar")
    wsi_path = resolve_wsi_path(mapping.wsi_path, args)
    dest_dir = patches_dir / f"fold_{fold_idx}" / labels["true_mutation"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rank, sel in enumerate(selected, 1):
        fname = (f"{stem}_fold{fold_idx}_{role}_rank{rank:02d}"
                 f"_r{sel['row']}c{sel['col']}.jpg")
        jobs.append((rank, sel, dest_dir / fname))
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(lambda j: get_patch(j[1], stem, j[2], wsi_path, mapping,
                                              idx, args), jobs))

    rows = []
    for (rank, sel, patch_path), (bbox, hit) in zip(jobs, res):
        if bbox is None:
            continue
        stats["hit" if hit else "miss"] += 1
        rows.append({
            "slide_id": stem, "fold": fold_idx, "split": role,
            "split_csv": split_csv.name, "entity": labels["entity"],
            "true_subtype_folder": labels["true_subtype_folder"],
            "true_grade": labels["true_grade"],
            "true_grade_class": labels["true_grade_class"],
            "true_mutation": labels["true_mutation"],
            "grade_pred": pred_name, "pred_grade_class": pgc,
            "grade_class_correct": int(pgc == labels["true_grade_class"]),
            "prob_control": round(float(probs[0]), 4),
            "prob_low_grade": round(float(probs[1]), 4),
            "prob_high_grade": round(float(probs[2]), 4),
            "rank": rank, "row": sel["row"], "col": sel["col"],
            "risk": round(sel["risk"], 6), "p95_threshold": round(thresh, 6),
            "n_above_p95": n_above, "tissue_fraction": round(sel["tissue_fraction"], 4),
            "base_mpp": getattr(mapping, "base_mpp", None),
            "target_mpp": getattr(mapping, "target_mpp", None),
            "wsi_bbox": list(bbox), "patch_path": str(patch_path),
            "thumbnail_path": str(sc_thumb), "wsi_path": str(wsi_path)})
    return rows


def reroot(path, repo_root):
    s = str(path).replace("\\", "/")
    i = s.rfind("/data/")
    return repo_root / s[i + 1:] if i >= 0 else Path(path)


def build(args):
    repo = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    control_img = Image.open(args.control_image).convert("RGB")
    root = Path(args.out_dir)
    patches_dir = root / "patches"
    manifests = root / "manifests"
    patches_dir.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    if args.png_cache:
        Path(args.png_cache).mkdir(parents=True, exist_ok=True)
    idx = index_cache(args.cache_dirs)
    print(f"cache: {len(idx)} regions from {len(args.cache_dirs)} cohort dir(s)")
    if idx and args.verify_cache:
        verify_cache(idx, args, n=args.verify_cache)

    split_csvs = sorted(Path(args.splits_dir).glob(args.split_glob))
    ckpt_map = resolve_checkpoints(split_csvs, args.checkpoints_glob)
    only = {int(f) for f in args.folds.split(",")}
    stats = {"hit": 0, "miss": 0}
    for split_csv in split_csvs:
        fold = int(re.search(r"split_(\d+)", split_csv.stem).group(1))
        if fold not in only:
            continue
        fold_manifest = manifests / f"stageB_fold{fold}.csv"
        done_flag = manifests / f"stageB_fold{fold}.done"
        if args.resume and done_flag.exists():
            print(f"[fold {fold}] done, skip")
            continue
        ckpt = ckpt_map.get(split_csv.stem)
        if ckpt is None:
            print(f"[fold {fold}] no checkpoint, skip")
            continue
        model = load_model(ckpt, args.model, device)
        slides = [(reroot(p, repo), role) for p, role in load_split_rows(split_csv)]
        if args.limit_slides:
            step = max(1, len(slides) // args.limit_slides)
            slides = slides[::step][:args.limit_slides]
        print(f"\n[fold {fold}] {len(slides)} slides | ckpt {ckpt}", flush=True)
        t0 = time.time()
        failed = 0
        with open(fold_manifest, "w", newline="") as ff:
            w = csv.DictWriter(ff, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for i, (thumb_path, role) in enumerate(slides, 1):
                try:
                    for row in process_slide(thumb_path, role, model, transform,
                                             control_img, device, fold, split_csv,
                                             args, patches_dir, idx, stats):
                        w.writerow(row)
                except Exception as e:
                    failed += 1
                    print(f"    FAILED {thumb_path.name}: {e}")
                if i % 25 == 0:
                    print(f"   {i}/{len(slides)} ({time.time()-t0:.0f}s) "
                          f"cache hit/miss {stats['hit']}/{stats['miss']}", flush=True)
        done_flag.write_text(f"failed={failed}\n")
        print(f"[fold {fold}] done in {time.time()-t0:.0f}s, failed={failed}")

    # combined manifest from the per-fold ones
    with open(root / "stageB_manifest.csv", "w", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for m in sorted(manifests.glob("stageB_fold*.csv")):
            w.writerows(csv.DictReader(open(m)))
    print(f"\nDONE. cache hits {stats['hit']}, extracted {stats['miss']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--split-glob", default="split_0*.csv")
    p.add_argument("--checkpoints-glob", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--control-image", required=True)
    p.add_argument("--sidecar-root", default="data/ebrains_thumbnails")
    p.add_argument("--wsi-root", default=None)
    p.add_argument("--wsi-old-prefix", default=None)
    p.add_argument("--wsi-new-prefix", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dirs", nargs="*", default=[])
    p.add_argument("--verify-cache", type=int, default=3)
    p.add_argument("--png-cache", default="data/e2e_patch_cache",
                   help="Shared lossless patch cache (same as e2e_multi); '' disables.")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--min-tissue", type=float, default=0.5)
    p.add_argument("--output-size", type=int, default=2048)
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit-slides", type=int, default=0,
                   help="Smoke test / selection check: every k-th slide, N per fold.")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
