#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
$PY -m scripts.analysis.compare_systems --config configs/cascade_rn18.json \
  2>&1 | tee logs/phase5_analysis.log
$PY -m scripts.analysis.plot_cascade_refinement --dir results/cascade_refinement_rn18 \
  2>&1 | tee -a logs/phase5_analysis.log
$PY -m pip freeze > results/cascade_refinement_rn18/pip_freeze.txt
