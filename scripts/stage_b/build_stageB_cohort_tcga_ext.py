"""
Stage-B patch extraction for the combined EBRAINS+TCGA cohort, per fold,
using that fold's 2-class Stage-A checkpoint. Reads splits_tcga_ext/split_0X.csv
directly (source=ebrains/tcga rows) so region selection matches what Stage-A
trained on.

No control class this round, so --control-image just needs a stable
injection canvas, e.g. a clear low_grade slide.

Usage (from repo root, after Stage-A combined training + checkpoint pull):
    python -m scripts.stage_b.build_stageB_cohort_tcga_ext ^
        --splits-dir splits_tcga_ext ^
        --checkpoints-glob "training_output_tcga_ext/*_split_0*/best_f1.pth" ^
        --model resnet50 ^
        --control-image data/included_ebrains-tcga/low_grade/<a_clear_low_grade_id>.jpg ^
        --sidecar-root data/ebrains_thumbnails ^
        --tcga-thumb-root data/tcga_thumbnails ^
        --wsi-root path/to/WHO2021_data ^
        --out-dir data/stage_b_cohort_tcga_ext
"""

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image

from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.wsi_mapping import load_mapping
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.models.factory import build_model

# Reuse, don't reimplement: everything below is imported unmodified from the
# existing EBRAINS Stage-B cohort builder. CLASS_ORDER and load_model are
# deliberately NOT imported -- see the CONTROL-FREE STAGE-A note above.
from scripts.stage_b.build_stageB_cohort import (
    GRID, CONTROL_SHUFFLE_SEED, MANIFEST_FIELDS,
    predict_class, select_p95_tiles, extract_region_highres,
    derive_true_labels, pred_grade_class, resolve_checkpoints, resolve_wsi_path,
    get_sidecar_thumbnail,
)

import openslide

# This round's Stage-A output classes (see train_stage_A_cluster_tcga_ext.py).
CLASS_ORDER = ["low_grade", "high_grade"]


def load_model_n(checkpoint_path, model_name, device, num_classes):
    """Like build_stageB_cohort.load_model, but with an explicit num_classes."""
    model = build_model(model_name, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_combined_fold_rows(split_csv):
    """(slide_id, source, tcga_class, split) for every slide in this fold."""
    out = []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["path"]).stem
            out.append((stem, row["source"], row["tcga_class"], row["split"]))
    return out


def resolve_thumbnail_and_labels(stem, source, tcga_class, args):
    """Returns (sidecar_carrying_jpg_path, subtype_folder, labels) or raises."""
    if source == "tcga":
        jpg_path = Path(args.tcga_thumb_root) / tcga_class / "included" / f"{stem}.jpg"
        if not jpg_path.exists():
            raise FileNotFoundError(f"TCGA sidecar thumbnail not found: {jpg_path}")
        subtype_folder = tcga_class
    else:  # ebrains
        sc_thumb = get_sidecar_thumbnail(stem, args.sidecar_root)
        if sc_thumb is None:
            raise RuntimeError(f"no EBRAINS sidecar thumbnail for {stem} under {args.sidecar_root}")
        jpg_path = sc_thumb
        subtype_folder = sc_thumb.parent.parent.name  # e.g. astro_IDHmt_G2

    labels = derive_true_labels(subtype_folder)
    return jpg_path, subtype_folder, labels


def process_slide_ext(stem, source, subtype_folder, role, jpg_path, labels,
                       model, transform, control_img, device, fold_idx, args,
                       patches_dir):
    thumbnail = Image.open(jpg_path).convert("RGB")
    pred_idx, probs = predict_class(model, transform, thumbnail, device)
    pred_name = CLASS_ORDER[pred_idx]
    pgc = pred_grade_class(pred_name)

    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img, model=model,
        transform=transform, grid=GRID, target_class=pred_idx, device=device,
        control_shuffle_seed=CONTROL_SHUFFLE_SEED,
    )
    selected, thresh, n_above = select_p95_tiles(
        risk_map, thumbnail, GRID, percentile=args.percentile,
        cap=args.cap, min_tissue=args.min_tissue)
    if not selected:
        return []

    mapping = load_mapping(jpg_path)
    if not mapping.has_wsi_mapping():
        raise RuntimeError(f"no WSI mapping in sidecar for {stem}")

    wsi_path = resolve_wsi_path(mapping.wsi_path, args)
    slide = openslide.OpenSlide(wsi_path)

    dest_dir = patches_dir / f"fold_{fold_idx}" / labels["true_mutation"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    slide_id = f"tcga_{stem}" if source == "tcga" else stem

    rows = []
    try:
        for rank, sel in enumerate(selected, 1):
            patch, wsi_bbox = extract_region_highres(
                slide, mapping, sel["row"], sel["col"], GRID, args.output_size)
            if patch is None:
                continue

            if args.stain_profile and source == "tcga":
                import numpy as np
                from glioma_sparse.preprocessing.stain_normalization import normalize_with_profile
                patch = Image.fromarray(
                    normalize_with_profile(np.array(patch), args.stain_profile, stage="B"))

            fname = (f"{slide_id}_fold{fold_idx}_{role}_rank{rank:02d}"
                     f"_r{sel['row']}c{sel['col']}.jpg")
            patch_path = dest_dir / fname
            patch.save(patch_path, quality=90)

            rows.append({
                "slide_id": slide_id, "fold": fold_idx, "split": role,
                "split_csv": source,
                "entity": labels["entity"], "true_subtype_folder": subtype_folder,
                "true_grade": labels["true_grade"],
                "true_grade_class": labels["true_grade_class"],
                "true_mutation": labels["true_mutation"],
                "grade_pred": pred_name, "pred_grade_class": pgc,
                "grade_class_correct": int(pgc == labels["true_grade_class"]),
                **{f"prob_{c}": round(float(p), 4)
                   for c, p in zip(CLASS_ORDER, probs)},
                "prob_control": 0.0,  # no control class this round, kept for MANIFEST_FIELDS
                "rank": rank, "row": sel["row"], "col": sel["col"],
                "risk": round(sel["risk"], 6),
                "p95_threshold": round(thresh, 6), "n_above_p95": n_above,
                "tissue_fraction": round(sel["tissue_fraction"], 4),
                "base_mpp": getattr(mapping, "base_mpp", None),
                "target_mpp": getattr(mapping, "target_mpp", None),
                "wsi_bbox": list(wsi_bbox), "patch_path": str(patch_path),
                "thumbnail_path": str(jpg_path), "wsi_path": str(wsi_path),
            })
    finally:
        slide.close()
    return rows


def load_done_slide_ids(fold_manifest):
    """slide_ids already in the fold manifest, for resume."""
    if not fold_manifest.exists():
        return set()
    done = set()
    try:
        with open(fold_manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("slide_id"):
                    done.add(row["slide_id"])
    except Exception:
        pass
    return done


def rebuild_combined_manifest(root_dir, manifests_dir):
    """Rewrite stageB_manifest.csv from the current stageB_fold*.csv files."""
    manifest_path = root_dir / "stageB_manifest.csv"
    fold_manifests = sorted(manifests_dir.glob("stageB_fold*.csv"))
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for fm in fold_manifests:
            with open(fm, newline="") as f:
                for row in csv.DictReader(f):
                    writer.writerow(row)
    return manifest_path


def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    control_img = Image.open(args.control_image).convert("RGB")

    root_dir = Path(args.out_dir)
    patches_dir = root_dir / "patches"
    manifests_dir = root_dir / "manifests"
    patches_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = sorted(Path(args.splits_dir).glob("split_0*.csv"))
    if not split_csvs:
        raise SystemExit(f"No split_0X.csv found in {args.splits_dir} "
                          f"-- run extend_splits_with_tcga.py first.")

    ckpt_map = resolve_checkpoints(split_csvs, args.checkpoints_glob)
    rows_written = slides_done = slides_failed = slides_skipped = 0

    for split_csv in split_csvs:
        m = re.search(r"split_(\d+)", split_csv.stem)
        fold_idx = int(m.group(1))
        fold_key = f"split_{fold_idx:02d}"

        ckpt = ckpt_map.get(fold_key)
        if ckpt is None:
            print(f"[fold {fold_idx}] no checkpoint matching {fold_key}, skip")
            continue

        fold_rows = load_combined_fold_rows(split_csv)
        n_ebrains = sum(1 for r in fold_rows if r[1] == "ebrains")
        n_tcga = sum(1 for r in fold_rows if r[1] == "tcga")

        fold_manifest = manifests_dir / f"stageB_fold{fold_idx}.csv"
        already_done = load_done_slide_ids(fold_manifest)
        def slide_id_of(stem, source):
            return f"tcga_{stem}" if source == "tcga" else stem
        remaining = [r for r in fold_rows if slide_id_of(r[0], r[1]) not in already_done]

        print(f"\n[fold {fold_idx}] ckpt {Path(ckpt).name} | "
              f"{len(fold_rows)} slides ({n_ebrains} ebrains, {n_tcga} tcga)")
        if already_done:
            print(f"    resuming: {len(already_done)} slides already in "
                  f"{fold_manifest.name}, {len(remaining)} remaining")

        if not remaining:
            print(f"    fold {fold_idx} already complete, skipping")
            slides_skipped += len(already_done)
            continue

        model = load_model_n(ckpt, args.model, device, len(CLASS_ORDER))

        write_header = not fold_manifest.exists()
        failed_log = manifests_dir / f"stageB_fold{fold_idx}_failed.csv"
        failed_write_header = not failed_log.exists()
        with open(fold_manifest, "a", newline="") as ff, \
             open(failed_log, "a", newline="") as flf:
            fold_writer = csv.DictWriter(ff, fieldnames=MANIFEST_FIELDS)
            if write_header:
                fold_writer.writeheader()
            failed_writer = csv.DictWriter(
                flf, fieldnames=["fold", "source", "slide_id", "error_type", "error"])
            if failed_write_header:
                failed_writer.writeheader()

            for i, (stem, source, tcga_class, role) in enumerate(remaining, 1):
                try:
                    jpg_path, subtype_folder, labels = resolve_thumbnail_and_labels(
                        stem, source, tcga_class, args)
                    rows = process_slide_ext(
                        stem, source, subtype_folder, role, jpg_path, labels,
                        model, transform, control_img, device, fold_idx, args,
                        patches_dir)
                    for row in rows:
                        fold_writer.writerow(row)
                    ff.flush()  # crash-safety: each completed slide is durable
                    # on disk immediately, so an interruption here loses at
                    # most the slide in progress, never anything already done.
                    rows_written += len(rows)
                    slides_done += 1
                except Exception as e:
                    slides_failed += 1
                    print(f"    FAILED {source}:{stem}: {e}")
                    failed_writer.writerow({
                        "fold": fold_idx, "source": source, "slide_id": stem,
                        "error_type": type(e).__name__, "error": str(e),
                    })
                    flf.flush()

                if i % 25 == 0 or i == len(remaining):
                    print(f"    [fold {fold_idx}] {i}/{len(remaining)} processed")

        # Rebuild the combined manifest after every fold, not just at the
        # end -- if you stop here, stageB_manifest.csv already reflects
        # whatever folds are actually done, no separate step needed.
        rebuild_combined_manifest(root_dir, manifests_dir)

    manifest_path = rebuild_combined_manifest(root_dir, manifests_dir)
    print("\n=== DONE ===")
    print(f"Slide-fold passes: {slides_done} | failed: {slides_failed} | "
          f"skipped (already done): {slides_skipped}")
    print(f"Regions written:   {rows_written}")
    print(f"Combined manifest: {manifest_path}")
    print(f"Per-fold manifests: {manifests_dir}")
    print(f"Patches:           {patches_dir}")
    print("\ndata/stage_b_cohort_RN50/ (and other existing cohorts) were not modified.")
    print("\nIf interrupted, just rerun the exact same command -- already-")
    print("completed slides (per fold manifest) are skipped automatically.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=str, default="splits_tcga_ext",
                    help="Dir with the combined split_0X.csv from extend_splits_with_tcga.py.")
    p.add_argument("--checkpoints-glob", type=str, required=True,
                    help="Stage-A checkpoints from the EBRAINS+TCGA combined "
                         "training run, e.g. 'training_output_tcga_ext/*_split_0*/best_f1.pth'.")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--control-image", type=str, required=True,
                    help="Fixed injection-canvas thumbnail. This round has no "
                         "control class, so pass a representative, clear "
                         "low_grade slide instead of literal control tissue "
                         "-- the mechanism just needs a stable canvas.")
    p.add_argument("--sidecar-root", type=str, default="data/ebrains_thumbnails",
                    help="Tree holding EBRAINS thumbnails + JSON sidecars, "
                         "keyed by slide id under <root>/<subtype>/included/.")
    p.add_argument("--tcga-thumb-root", type=str, default="data/tcga_thumbnails",
                    help="Tree holding TCGA thumbnails + JSON sidecars, from "
                         "generate_tcga_thumbnails.py.")
    p.add_argument("--wsi-root", type=str, default=None,
                    help="Fallback search root if a sidecar's stored wsi_path "
                         "no longer resolves (e.g. different machine).")
    p.add_argument("--wsi-old-prefix", type=str, default=None)
    p.add_argument("--wsi-new-prefix", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="data/stage_b_cohort_tcga_ext")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--min-tissue", type=float, default=0.5)
    p.add_argument("--output-size", type=int, default=2048)
    p.add_argument("--stain-profile", type=str, default=None,
                    help="Optional fitted profile (e.g. TCGA-Glioma) applied "
                         "to each extracted TCGA high-res patch before saving "
                         "(EBRAINS patches are never normalized -- they ARE "
                         "the reference distribution).")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
