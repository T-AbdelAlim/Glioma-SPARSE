#!/usr/bin/env bash
# Resume of the chain from phase 3 (phases 0-2 and phase 4 for A0 are done). Stops at the first failure.
cd "$(dirname "$0")/../../.." || exit 1
S=scripts/experiments/cascade_refinement
bash $S/phase3_cohorts_and_stageB.sh || exit 1
bash $S/phase4_e2e.sh Amt       || exit 1
bash $S/phase4_e2e.sh Actrl     || exit 1
bash $S/phase5_analysis.sh
echo ALL_DONE
