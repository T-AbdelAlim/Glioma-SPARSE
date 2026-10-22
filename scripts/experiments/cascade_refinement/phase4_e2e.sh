#!/usr/bin/env bash
# Held-out end-to-end scoring (test folds). Systems sharing a Stage A are scored on
# identical regions. Usage: phase4_e2e.sh A0 | Actrl | Amt
# Two fold-parallel processes per Stage A (region extraction is CPU/IO bound; two
# processes is what fits in 16 GB of GPU memory); the shared lossless patch cache is
# filled once; a final call merges the per-fold tables.
source "$(dirname "$0")/common.sh"
FT=ResNet18_output/training_output_stageB_ft
run() {  # run <name> <stage-a-glob> <extra args...>
  local name=$1 aglob=$2; shift 2
  for folds in 1,2,3 4,5; do
    $PY -m scripts.inference.e2e_multi --run-name $name --stage-a-glob "$aglob" \
      --roles test --tta --resume --folds $folds "$@" \
      2>&1 | tee -a logs/phase4_${name}_${folds//,/_}.log &
  done
  wait
  $PY -m scripts.inference.e2e_multi --run-name $name --stage-a-glob "$aglob" \
      --roles test --tta --resume "$@" 2>&1 | tee -a logs/phase4_${name}_merge.log
}
case "${1:-}" in
  A0) run e2e_multi_rn18_A0 "$A0_GLOB" \
        --system "B0=$B0_GLOB" \
        --system "Bc=${FT}_continue/rn18_A0_stageB_continue_fold*/best_f1.pth" \
        --system "Br=${FT}_refined/rn18_A0_stageB_refined_fold*/best_f1.pth" ;;
  Actrl) run e2e_multi_rn18_Actrl "ResNet18_output/training_output_stageA_ctrl/*_split_0*/best_f1.pth" \
        --system "B0=$B0_GLOB" ;;
  Amt) run e2e_multi_rn18_Amt "ResNet18_output/training_output_stageA_mt/*_split_0*/best_f1.pth" --use-aux \
        --system "B0=$B0_GLOB" \
        --system "Bc=${FT}_continue/rn18_Amt_stageB_continue_fold*/best_f1.pth" \
        --system "Br=${FT}_refined/rn18_Amt_stageB_refined_fold*/best_f1.pth" ;;
  *) echo "usage: $0 A0|Actrl|Amt"; exit 1 ;;
esac
