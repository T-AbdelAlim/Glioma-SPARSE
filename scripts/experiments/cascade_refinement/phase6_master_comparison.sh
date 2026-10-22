#!/usr/bin/env bash
# Master comparison of all systems (ResNet cascades, refinement study, ResUNI, CLAM benchmark).
# Needs: results/e2e_regions_{rn18_f1,rn50_f1,rn50_auc}.csv (scripts.analysis.e2e_regions) -- see docs/experiments/MASTER_COMPARISON.md
source "$(dirname "$0")/common.sh"
$PY -m scripts.analysis.compute_costs
for cfg in "rn18_f1 data/stage_b_cohort_RN18_F1" "rn50_f1 data/stage_b_cohort_RN50_F1" "rn50_auc data/stage_b_cohort_RN50"; do
  set -- $cfg
  [ -f results/resuni/$1/predictions.csv ] || $PY -u -m scripts.analysis.resuni_predictions --config $1 --cohort $2 --e2e-regions results/e2e_regions_$1.csv
done
$PY -m scripts.analysis.master_comparison
$PY -m scripts.analysis.build_master_doc
$PY -m scripts.analysis.plot_all_experiments
