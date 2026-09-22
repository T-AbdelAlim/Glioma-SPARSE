r"""
End-to-end inference for the EBRAINS+TCGA combined pipeline. Sibling of
inference_end_to_end.py (which stays EBRAINS-only, 3-class): here Stage A is
2-class (low_grade/high_grade) and test slides come from
splits_tcga_ext/split_0X.csv (mixed ebrains/tcga sources).

Usage (from repo root, checkpoints already trained + pulled locally):
    python -m scripts.inference.inference_end_to_end_tcga_ext ^
        --splits-dir splits_tcga_ext ^
        --stage-a-glob "training_output_tcga_ext/*_split_0*/best_f1.pth" ^
        --stage-b-glob "training_output_stageB_tcga_ext/*_fold*_tcgaext_cw/best_auc.pth" ^
        --sidecar-root data/ebrains_thumbnails ^
        --tcga-thumb-root data/tcga_thumbnails ^
        --wsi-root path/to/WHO2021_data ^
        --wsi-old-prefix "D:\Thinkpad_Backup" --wsi-new-prefix "E:\Thinkpad_Backup" ^
        --control-image data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg ^
        --no-stage-b-riskmap ^
        --run-name e2e_testset_tcga_ext
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import openslide

from scripts.stage_b.build_stageB_cohort import derive_true_labels, get_sidecar_thumbnail
from scripts.inference.inference_end_to_end import (
    REPO_ROOT, RESULTS_ROOT, GRID, SEED, SUBTYPE_CLASSES, INTEGRATED, WSI_EXTS,
    load_model, forward_probs, confidence, mutant_status, abbrev, _rel,
    select_tiles, extract_highres, extract_subtile,
    occlusion_map, top_occlusion_cells, save_stage_a_riskmap, save_stage_b_occlusion,
    resolve_wsi, write_xlsx_report, score_row, _write_testset_outputs,
)

# This round's Stage-A output classes -- no "control" (see
# train_stage_A_cluster_tcga_ext.py / build_stageB_cohort_tcga_ext.py).
GRADE_CLASSES = ["low_grade", "high_grade"]


def grade_to_class(grade_name):
    return "low" if grade_name == "low_grade" else "high"


# ============================================================
# PER-SLIDE PIPELINE (like inference_end_to_end.run_slide, no control branch)
# ============================================================

def run_slide_tcga_ext(thumb_source, model_a, model_b, transform, control_img, args,
                       slide_dir, wsi_path=None, generate_thumb=False, thumb_out=None):
    """thumb_out lets callers share a thumbnail cache across fold passes."""
    slide_id = Path(thumb_source).stem if thumb_source else Path(wsi_path).stem
    riskmap_dir = slide_dir / "riskmap"
    patches_dir = slide_dir / "patches"
    riskmap_dir.mkdir(parents=True, exist_ok=True)

    thumb_path = Path(thumb_source) if thumb_source else None
    if generate_thumb:
        from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto as create_wsi_thumbnail
        thumb_path = Path(thumb_out) if thumb_out else (slide_dir / f"{slide_id}.jpg")
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        create_wsi_thumbnail(wsi_path, output_path=str(thumb_path),
                             target_mpp=args.target_mpp, output_size=args.thumb_size,
                             stain_profile=args.stain_profile)
    thumbnail = Image.open(thumb_path).convert("RGB")

    # Stage A grade + risk map
    probsA, _ = forward_probs(model_a, transform, thumbnail, args.device)
    cA = confidence(probsA)
    grade_name = GRADE_CLASSES[cA["pred_idx"]]
    grade_class = grade_to_class(grade_name)

    risk_map, _ = compute_risk_map_(
        thumbnail, control_img, model_a, transform, cA["pred_idx"], args.device)
    selected, thresh = select_tiles(risk_map, thumbnail, GRID,
                                    args.percentile, args.cap, args.min_tissue)
    save_stage_a_riskmap(thumbnail, risk_map, selected,
                         riskmap_dir / "stageA_riskmap.jpg",
                         f"{slide_id}  Stage A: {grade_name} ({cA['confidence']:.2f})")

    result = {
        "slide_id": slide_id, "grade_pred": grade_name, "grade_class": grade_class,
        "prob_low_grade": round(float(probsA[0]), 4),
        "prob_high_grade": round(float(probsA[1]), 4),
        "stageA_confidence": cA["confidence"], "stageA_margin": cA["margin"],
        "p_threshold": round(thresh, 6), "n_regions": len(selected),
    }

    if not selected:
        result.update({"subtype_pred": "NA", "integrated_diagnosis": "No region extracted",
                       "stageB_confidence": "", "integrated_confidence": ""})
        return result

    mapping = load_mapping_(thumb_path)
    if not mapping.has_wsi_mapping():
        raise RuntimeError(f"no WSI mapping for {slide_id}")
    wsi_real = resolve_wsi(mapping.wsi_path, args, explicit=wsi_path)
    slide = openslide.OpenSlide(wsi_real)
    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_probs = []
    try:
        for rank, sel in enumerate(selected, 1):
            patch, patch_bbox = extract_highres(slide, mapping, sel["row"], sel["col"],
                                                GRID, args.output_size,
                                                stain_profile=args.stain_profile)
            if patch is None:
                continue
            tag = f"{slide_id}_rank{rank:02d}_r{sel['row']}c{sel['col']}"
            patch.save(patches_dir / f"{tag}.jpg", quality=90)
            pB, _ = forward_probs(model_b, transform, patch, args.device)
            patch_probs.append(pB)

            if args.stage_b_riskmap:
                imp = occlusion_map(model_b, transform, patch, int(np.argmax(pB)), args.device)
                tops = top_occlusion_cells(imp, args.occlusion_topk)
                save_stage_b_occlusion(
                    patch, imp, tops, riskmap_dir / f"stageB_{tag}_occlusion.jpg",
                    f"{tag}  Stage B occlusion ({SUBTYPE_CLASSES[int(np.argmax(pB))]})")
                for ti, (sr, sc, val) in enumerate(tops, 1):
                    zoom = extract_subtile(slide, patch_bbox, sr, sc, args.output_size)
                    if zoom is not None:
                        zoom.save(riskmap_dir / f"stageB_{tag}_top{ti}_r{sr}c{sc}_zoom.jpg",
                                 quality=90)
    finally:
        slide.close()

    if not patch_probs:
        result.update({"subtype_pred": "NA", "integrated_diagnosis": "No region extracted",
                       "stageB_confidence": "", "integrated_confidence": ""})
        return result

    mean_prob = np.mean(patch_probs, axis=0)
    cB = confidence(mean_prob)
    subtype = SUBTYPE_CLASSES[cB["pred_idx"]]
    integrated = INTEGRATED.get((grade_class, subtype), "Unresolved")
    result.update({
        "subtype_pred": subtype,
        "prob_IDH_mt": round(float(mean_prob[0]), 4),
        "prob_IDH_mt_1p19q": round(float(mean_prob[1]), 4),
        "prob_IDH_wt": round(float(mean_prob[2]), 4),
        "stageB_confidence": cB["confidence"], "stageB_margin": cB["margin"],
        "integrated_diagnosis": integrated,
        "integrated_confidence": round(cA["confidence"] * cB["confidence"], 4),
        "n_patches_used": len(patch_probs),
    })
    return result


def compute_risk_map_(thumbnail, control_img, model_a, transform, pred_idx, device):
    from glioma_sparse.interpret.patch_injection import compute_risk_map
    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img, model=model_a,
        transform=transform, grid=GRID, target_class=pred_idx,
        device=device, control_shuffle_seed=SEED, score="logit")
    return risk_map, pred_idx


def load_mapping_(thumb_path):
    from glioma_sparse.interpret.wsi_mapping import load_mapping
    return load_mapping(thumb_path)


# ============================================================
# COMBINED-SOURCE TEST SLIDE RESOLUTION (ebrains + tcga)
# ============================================================

def resolve_test_slide(stem, source, tcga_class, args):
    """(thumb_path, true_labels) for one test-split row, or raises."""
    if source == "tcga":
        thumb_path = Path(args.tcga_thumb_root) / tcga_class / "included" / f"{stem}.jpg"
        if not thumb_path.exists():
            raise FileNotFoundError(f"TCGA sidecar thumbnail not found: {thumb_path}")
        subtype_folder = tcga_class
    else:
        sc_thumb = get_sidecar_thumbnail(stem, args.sidecar_root)
        if sc_thumb is None:
            raise RuntimeError(f"no EBRAINS sidecar thumbnail for {stem} under {args.sidecar_root}")
        thumb_path = sc_thumb
        subtype_folder = sc_thumb.parent.parent.name

    labels = derive_true_labels(subtype_folder)
    true = {"grade_class": labels["true_grade_class"], "mutation": labels["true_mutation"]}
    return thumb_path, true


def load_combined_test_rows(split_csv):
    """(stem, source, tcga_class) for this fold's test-split slides."""
    out = []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            stem = Path(row["path"]).stem
            out.append((stem, row["source"], row["tcga_class"]))
    return out


# ============================================================
# EXTERNAL COHORT (e.g. Radboud) LABEL RESOLUTION
# ============================================================

def _labels_from_folder_name_ext(folder):
    """Keyword match on an ancestor folder name, e.g. 'astro_IDHmt_G2'.
    Gunknown folders return None (not "high") -- grade isn't established."""
    f = folder.lower()
    if "gunknown" in f:
        return None
    if "gbm" in f or "idhwt" in f:
        return {"grade_class": "high", "mutation": "IDH_wt"}
    if "oligo" in f and "1p19q" in f:
        return {"grade_class": "low" if "_g2" in f else "high",
                "mutation": "IDH_mt_1p19q"}
    if "astro" in f and "idhmt" in f:
        return {"grade_class": "low" if "_g2" in f else "high",
                "mutation": "IDH_mt"}
    return None


def _true_labels_from_ancestor_ext(wsi_path):
    """Walk up ancestor folders for the first subtype match; "unknown" if none."""
    for ancestor in Path(wsi_path).parents:
        labels = _labels_from_folder_name_ext(ancestor.name)
        if labels is not None:
            return labels
    return {"grade_class": "unknown", "mutation": "unknown"}


# ============================================================
# EXTERNAL DRIVER (e.g. Radboud -- one checkpoint pair, no per-fold holdout)
# ============================================================

def run_external_tcga_ext(args):
    """Score an external cohort (Radboud, ...) with one checkpoint pair."""
    from glioma_sparse.data_utils.transforms import build_eval_transform
    transform = build_eval_transform()
    control_img = Image.open(_rel(args.control_image)).convert("RGB")
    model_a = load_model(_rel(args.stage_a_ckpt), args.model, len(GRADE_CLASSES), args.device)
    model_b = load_model(_rel(args.stage_b_ckpt), args.model, len(SUBTYPE_CLASSES), args.device)
    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    input_root = _rel(args.input)
    slides = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in WSI_EXTS)
    print(f"[external] {len(slides)} slides found under {input_root} "
         f"(checkpoint_label={args.checkpoint_label}, stain_profile={args.stain_profile})")

    all_rows = []
    n_unknown = 0
    for i, wsi in enumerate(slides, 1):
        true = _true_labels_from_ancestor_ext(wsi)
        if true["grade_class"] == "unknown":
            n_unknown += 1
            print(f"  [{i}/{len(slides)}] SKIP (no subtype folder matched, or Gunknown): {wsi}")
            continue
        stem = wsi.stem
        try:
            res = run_slide_tcga_ext(None, model_a, model_b, transform, control_img, args,
                                     run_dir / abbrev(stem), wsi_path=str(wsi), generate_thumb=True)
            res["fold"] = args.checkpoint_label
            res.update(score_row(true, res))
            all_rows.append(res)
            print(f"  [{i}/{len(slides)}] {stem}: true={true['grade_class']}/{true['mutation']} "
                 f"pred={res['grade_class']}/{res.get('subtype_pred','NA')} "
                 f"e2e_correct={res.get('end_to_end_correct')}")
        except Exception as e:
            print(f"  [{i}/{len(slides)}] FAILED {stem}: {e}")

    if n_unknown:
        print(f"\n[external] {n_unknown} slide(s) skipped: no ancestor folder matched a known "
             f"subtype pattern (or it was a Gunknown folder) -- NOT included in the report.")
    _write_testset_outputs(all_rows, run_dir)
    print(f"\nExternal report: {run_dir / 'e2e_testset_report.xlsx'}")


# ============================================================
# EXTERNAL DRIVER, ALL 5 FOLDS
# ============================================================

def run_external_5fold_tcga_ext(args):
    from glioma_sparse.data_utils.transforms import build_eval_transform
    transform = build_eval_transform()
    control_img = Image.open(_rel(args.control_image)).convert("RGB")
    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    thumb_cache_dir = run_dir / "_thumb_cache"
    thumb_cache_dir.mkdir(parents=True, exist_ok=True)

    a_ckpts = sorted(glob.glob(str(_rel(args.stage_a_glob))))
    b_ckpts = sorted(glob.glob(str(_rel(args.stage_b_glob))))
    if not (a_ckpts and b_ckpts):
        raise SystemExit(f"No checkpoints matched.\n  stage A glob: "
                         f"{_rel(args.stage_a_glob)} -> {len(a_ckpts)} hits\n  "
                         f"stage B glob: {_rel(args.stage_b_glob)} -> {len(b_ckpts)} hits")

    def pick(ckpts, token):
        m = [c for c in ckpts if token in str(c)]
        return m[0] if m else None

    # Resolve ground truth and skip unmatched/Gunknown slides ONCE, not per fold.
    input_root = _rel(args.input)
    slides = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in WSI_EXTS)
    slide_labels, n_unknown = [], 0
    for wsi in slides:
        true = _true_labels_from_ancestor_ext(wsi)
        if true["grade_class"] == "unknown":
            n_unknown += 1
            continue
        slide_labels.append((wsi, true))
    print(f"[external 5-fold] {len(slide_labels)} labeled slides under {input_root} "
         f"({n_unknown} skipped: unmatched or Gunknown), stain_profile={args.stain_profile}")

    all_rows = []
    for fold_idx in range(1, 6):
        a = pick(a_ckpts, f"split_{fold_idx:02d}")
        b = pick(b_ckpts, f"fold{fold_idx}_")
        if not (a and b):
            print(f"\n[fold {fold_idx}] missing checkpoint (stage_a={a}, stage_b={b}), skip")
            continue
        print(f"\n[fold {fold_idx}] stage_a={Path(a).parent.name}  stage_b={Path(b).parent.name}")
        model_a = load_model(a, args.model, len(GRADE_CLASSES), args.device)
        model_b = load_model(b, args.model, len(SUBTYPE_CLASSES), args.device)
        fold_dir = run_dir / f"fold_{fold_idx}"

        for i, (wsi, true) in enumerate(slide_labels, 1):
            stem = wsi.stem
            try:
                cached = thumb_cache_dir / f"{stem}.jpg"
                already_cached = cached.exists()
                res = run_slide_tcga_ext(
                    cached if already_cached else None,
                    model_a, model_b, transform, control_img, args,
                    fold_dir / abbrev(stem), wsi_path=str(wsi),
                    generate_thumb=not already_cached, thumb_out=cached)
                res["slide_id"] = stem
                res["fold"] = fold_idx
                res.update(score_row(true, res))
                all_rows.append(res)
            except Exception as e:
                print(f"    [fold {fold_idx}] FAILED {stem}: {e}")
            if i % 25 == 0 or i == len(slide_labels):
                print(f"    [fold {fold_idx}] {i}/{len(slide_labels)} processed")

    _write_testset_outputs(all_rows, run_dir)
    print(f"\nExternal 5-fold report: {run_dir / 'e2e_testset_report.xlsx'}")


# ============================================================
# TESTSET DRIVER
# ============================================================

def run_testset_tcga_ext(args):
    from glioma_sparse.data_utils.transforms import build_eval_transform
    transform = build_eval_transform()
    control_img = Image.open(_rel(args.control_image)).convert("RGB")
    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = sorted(_rel(args.splits_dir).glob("split_0*.csv"))
    a_ckpts = sorted(glob.glob(str(_rel(args.stage_a_glob))))
    b_ckpts = sorted(glob.glob(str(_rel(args.stage_b_glob))))
    if not split_csvs:
        raise SystemExit(f"No split_0*.csv found in {_rel(args.splits_dir)}.")
    if not (a_ckpts and b_ckpts):
        raise SystemExit(f"No checkpoints matched.\n  stage A glob: "
                         f"{_rel(args.stage_a_glob)} -> {len(a_ckpts)} hits\n  "
                         f"stage B glob: {_rel(args.stage_b_glob)} -> {len(b_ckpts)} hits")

    def pick(ckpts, token):
        m = [c for c in ckpts if token in str(c)]
        return m[0] if m else None

    all_rows = []
    for fold_idx, sc in enumerate(split_csvs, 1):
        a = pick(a_ckpts, sc.stem)
        b = pick(b_ckpts, f"fold{fold_idx}_")
        if not (a and b):
            print(f"[fold {fold_idx}] missing checkpoint (stage_a={a}, stage_b={b}), skip")
            continue
        model_a = load_model(a, args.model, len(GRADE_CLASSES), args.device)
        model_b = load_model(b, args.model, len(SUBTYPE_CLASSES), args.device)
        test_rows = load_combined_test_rows(sc)
        n_ebrains = sum(1 for r in test_rows if r[1] == "ebrains")
        n_tcga = sum(1 for r in test_rows if r[1] == "tcga")
        print(f"[fold {fold_idx}] {len(test_rows)} test slides "
              f"({n_ebrains} ebrains, {n_tcga} tcga)")
        fold_dir = run_dir / f"fold_{fold_idx}"

        for stem, source, tcga_class in test_rows:
            try:
                thumb_path, true = resolve_test_slide(stem, source, tcga_class, args)
                slide_id = f"tcga_{stem}" if source == "tcga" else stem
                res = run_slide_tcga_ext(thumb_path, model_a, model_b, transform,
                                         control_img, args, fold_dir / abbrev(slide_id))
                res["slide_id"] = slide_id
                res["fold"] = fold_idx
                res.update(score_row(true, res))
                all_rows.append(res)
            except Exception as e:
                print(f"    FAILED {source}:{stem}: {e}")

    _write_testset_outputs(all_rows, run_dir)
    print(f"\nTestset report: {run_dir / 'e2e_testset_report.xlsx'}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--external", action="store_true", default=False,
                   help="score a labeled external cohort (e.g. Radboud) against ground truth "
                       "inferred from its subtype folder names, using a single given "
                       "--stage-a-ckpt/--stage-b-ckpt (no fold globbing: external data isn't "
                       "held out per fold, run once per checkpoint pair you want to cover and "
                       "combine the reports afterward)")
    p.add_argument("--external-5fold", action="store_true", default=False,
                   help="manuscript-grade external validation: score the external cohort with "
                       "EACH of the 5 internal fold checkpoint pairs (via --stage-a-glob/"
                       "--stage-b-glob, same as testset mode) and combine into one report with "
                       "mean +/- std across folds -- avoids reporting a single, possibly "
                       "cherry-picked fold's number. Thumbnails are generated once and cached "
                       "across folds (thumbnailing doesn't depend on the model); risk-map/patch "
                       "selection reruns per fold since it does.")
    p.add_argument("--checkpoint-label", default="fold1",
                   help='tags every row of an --external run (e.g. "fold1") so multiple runs '
                       "against different checkpoint pairs can be concatenated and aggregated "
                       "the same way internal fold results are")
    p.add_argument("--input", default=None,
                   help="--external mode: root folder of the external cohort's raw WSIs, "
                       "organized by subtype (e.g. astro_IDHmt_G2/, GBM_IDHwt/, ...)")
    p.add_argument("--stage-a-ckpt", default=None, help="--external mode: single Stage-A checkpoint")
    p.add_argument("--stage-b-ckpt", default=None, help="--external mode: single Stage-B checkpoint")
    p.add_argument("--target-mpp", type=float, default=4.0,
                   help="--external mode: thumbnail microns-per-pixel (fresh thumbnail generation)")
    p.add_argument("--thumb-size", type=int, default=2048,
                   help="--external mode: thumbnail output size (fresh thumbnail generation)")
    p.add_argument("--run-name", default="e2e_testset_tcga_ext")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--splits-dir", default="splits_tcga_ext")
    p.add_argument("--stage-a-glob", default="training_output_tcga_ext/*_split_0*/best_f1.pth")
    p.add_argument("--stage-b-glob",
                   default="training_output_stageB_tcga_ext/*_fold*_tcgaext_cw/best_f1.pth")
    p.add_argument("--sidecar-root", default="data/ebrains_thumbnails")
    p.add_argument("--tcga-thumb-root", default="data/tcga_thumbnails")
    p.add_argument("--wsi-root", default=None)
    p.add_argument("--wsi-old-prefix", default=None)
    p.add_argument("--wsi-new-prefix", default=None)
    p.add_argument("--control-image", default=r"data\included\control\86242943-7775-11eb-827d-001a7dda7111.jpg")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--min-tissue", type=float, default=0.25)
    p.add_argument("--output-size", type=int, default=2048)
    p.add_argument("--stain-profile", dest="stain_profile", default=None,
                   help='name of a fitted stain profile to normalize against, e.g. "RD-mrxs" '
                       "for Radboud (see glioma_sparse.preprocessing.stain_normalization); "
                       "omit for no correction")
    p.add_argument("--stage-b-riskmap", action="store_true", default=True)
    p.add_argument("--no-stage-b-riskmap", dest="stage_b_riskmap", action="store_false",
                   help="disable the Stage-B occlusion sweep -- interpretability only, "
                       "not needed for accuracy/AUC, and dominates runtime on large batches.")
    p.add_argument("--occlusion-topk", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.external and args.external_5fold:
        raise SystemExit("--external and --external-5fold are mutually exclusive")
    if args.external and not (args.input and args.stage_a_ckpt and args.stage_b_ckpt):
        raise SystemExit("--external requires --input, --stage-a-ckpt, and --stage-b-ckpt")
    if args.external_5fold and not args.input:
        raise SystemExit("--external-5fold requires --input")
    return args


def main():
    args = parse_args()
    if args.external_5fold:
        run_external_5fold_tcga_ext(args)
    elif args.external:
        run_external_tcga_ext(args)
    else:
        run_testset_tcga_ext(args)


if __name__ == "__main__":
    main()
