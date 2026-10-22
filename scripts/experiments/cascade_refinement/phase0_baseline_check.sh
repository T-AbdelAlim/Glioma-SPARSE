#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
$PY -m scripts.inference.e2e_multi --run-name e2e_multi_rn18_baseline_check \
  --stage-a-glob "$A0_GLOB" --system "B0=$B0_GLOB" --roles test --tta --resume \
  2>&1 | tee -a logs/phase0_baseline_check.log
$PY -m scripts.analysis.compare_systems \
  --verify results/e2e_multi_rn18_baseline_check/per_slide.csv B0 \
  results/e2e_testset_rn18_f1/e2e_testset_report_RN18_f1.xlsx | tee logs/phase0_verify.log
