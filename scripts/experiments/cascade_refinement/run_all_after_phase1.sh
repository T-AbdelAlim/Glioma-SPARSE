#!/usr/bin/env bash
# Chains the remaining phases after phase 1 finished, so the GPU never idles.
# Stops at the first failing phase.
cd "$(dirname "$0")/../../.." || exit 1
S=scripts/experiments/cascade_refinement
until [ -f ResNet18_output/training_output_stageB_ft_refined/rn18_A0_stageB_refined_fold5/efficiency.json ]; do sleep 60; done
bash $S/phase4_e2e.sh A0        || exit 1
bash $S/phase2_option2_stageA.sh || exit 1
bash $S/phase3_cohorts_and_stageB.sh || exit 1
bash $S/phase4_e2e.sh Amt       || exit 1
bash $S/phase4_e2e.sh Actrl     || exit 1
bash $S/phase5_analysis.sh
echo ALL_DONE
