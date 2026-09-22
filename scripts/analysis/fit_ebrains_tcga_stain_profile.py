"""
Fit a stain-normalization profile from the EBRAINS+TCGA combined training
data itself (both sources, unnormalized), rather than reusing RD-mrxs /
TCGA-Glioma, which predate TCGA being added to training. Class-stratified
(grade for Stage A, subtype for Stage B) to avoid weighting toward whichever
class has the most images.

Usage (from repo root):
    python -m scripts.analysis.fit_ebrains_tcga_stain_profile
"""

from collections import defaultdict
from pathlib import Path
import csv

from glioma_sparse.preprocessing.stain_normalization import (
    fit_reference_stratified, save_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "EBRAINS-TCGA-train-ref"
SAMPLE_N_PER_CLASS = 100
MIN_N = 10
SEED = 0


def collect_stage_a_paths():
    """{grade_class: [thumbnail paths]} from the merged Stage-A training tree."""
    root = REPO_ROOT / "data" / "included_ebrains-tcga"
    out = {}
    for grade in ("low_grade", "high_grade"):
        paths = sorted((root / grade).glob("*.jpg"))
        out[grade] = paths
        print(f"  stage A / {grade}: {len(paths)} thumbnails")
    return out


def collect_stage_b_paths():
    """{true_mutation: [patch paths]}, TRAIN split only, pooled across all 5 folds."""
    cohort_dir = REPO_ROOT / "data" / "stage_b_cohort_tcga_ext"
    by_class = defaultdict(list)
    for fold in range(1, 6):
        manifest = cohort_dir / "manifests" / f"stageB_fold{fold}.csv"
        if not manifest.exists():
            print(f"  [fold {fold}] manifest not found, skip")
            continue
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != "train":
                    continue
                fname = Path(row["patch_path"]).name
                patch_path = cohort_dir / "patches" / f"fold_{row['fold']}" / row["true_mutation"] / fname
                if patch_path.exists():
                    by_class[row["true_mutation"]].append(patch_path)
    for cls, paths in by_class.items():
        print(f"  stage B / {cls}: {len(paths)} train patches (pooled across folds)")
    return dict(by_class)


def main():
    print("Collecting Stage A thumbnail paths (grade-stratified)...")
    stage_a_paths = collect_stage_a_paths()
    print("\nCollecting Stage B patch paths (mutation-stratified, train split only)...")
    stage_b_paths = collect_stage_b_paths()

    print(f"\nFitting Stage A reference (n_per_class={SAMPLE_N_PER_CLASS})...")
    stageA = fit_reference_stratified(
        stage_a_paths, min_n=MIN_N, sample_n_per_class=SAMPLE_N_PER_CLASS, seed=SEED, verbose=True)
    print(f"  classes included: {stageA['classes_included']}")
    print(f"  classes excluded (too few): {stageA['classes_excluded_too_few']}")
    print(f"  n_used={stageA['n_used']} n_failed={stageA['n_failed']}")

    print(f"\nFitting Stage B reference (n_per_class={SAMPLE_N_PER_CLASS})...")
    stageB = fit_reference_stratified(
        stage_b_paths, min_n=MIN_N, sample_n_per_class=SAMPLE_N_PER_CLASS, seed=SEED, verbose=True)
    print(f"  classes included: {stageB['classes_included']}")
    print(f"  classes excluded (too few): {stageB['classes_excluded_too_few']}")
    print(f"  n_used={stageB['n_used']} n_failed={stageB['n_failed']}")

    # per_class sub-dicts are diagnostic only, not needed in the saved profile
    stageA_to_save = {k: v for k, v in stageA.items() if k != "per_class"}
    stageB_to_save = {k: v for k, v in stageB.items() if k != "per_class"}

    path = save_profile(
        PROFILE_ID,
        label="EBRAINS+TCGA combined training reference",
        description=(
            "Class-stratified reference fitted directly from the EBRAINS+TCGA "
            "combined training data (both sources, unnormalized -- TCGA patches "
            "went into training without correction). Supersedes RD-mrxs / "
            "TCGA-Glioma for validating models trained on the combined pool: "
            "those two profiles were fitted before TCGA was added to training "
            "and reflect an EBRAINS-only reference distribution."),
        stageA=stageA_to_save, stageB=stageB_to_save,
        fitted_from_stageA=(
            f"Stage-A training thumbnails: data/included_ebrains-tcga/"
            f"{{low_grade,high_grade}}/*.jpg (both ebrains + tcga sources, "
            f"grade-stratified, {SAMPLE_N_PER_CLASS}/class)."),
        fitted_from_stageB=(
            f"Stage-B training patches: data/stage_b_cohort_tcga_ext/patches/"
            f"fold_*/<mutation>/*.jpg, TRAIN split only per "
            f"manifests/stageB_fold*.csv, pooled across all 5 folds, "
            f"mutation-stratified, {SAMPLE_N_PER_CLASS}/class."),
    )
    print(f"\nSaved profile: {path}")


if __name__ == "__main__":
    main()
