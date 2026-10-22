#!/usr/bin/env bash
# External validation (TCGA, 180 slides): UNI2-h at Stage B, all five EBRAINS fold heads applied unchanged.
# Requires: run_external_resnet.sh already run for the matching cfg (needs its per_slide.csv for regions),
# and the TRIDENT+UNI2-h features for the 180 TCGA slides already extracted.
# Usage: run_external_resuni.sh rn18_f1 | rn50_f1 | rn50_auc
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
cfg=${1:?config}
case $cfg in
  rn18_f1)  cohort=data/stage_b_cohort_RN18_F1 ;;
  rn50_f1)  cohort=data/stage_b_cohort_RN50_F1 ;;
  rn50_auc) cohort=data/stage_b_cohort_RN50 ;;
  *) echo "unknown config (rn18_f1 | rn50_f1 | rn50_auc)"; exit 1 ;;
esac
$PY -u -m scripts.analysis.resuni_predictions --config $cfg --cohort $cohort \
  --external-regions results/e2e_external_${cfg}_tcga/per_slide.csv \
  --external-features-dir "../mahmoodlab/trident_tcga_val/20x_256px_0px_overlap/features_uni_v2" \
  --external-labels results/external_TCGA_val/labels.csv --external-tag tcga --external-only \
  2>&1 | tee -a logs/external_resuni_${cfg}_tcga.log
