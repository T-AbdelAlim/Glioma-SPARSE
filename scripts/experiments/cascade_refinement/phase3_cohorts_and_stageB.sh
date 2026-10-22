#!/usr/bin/env bash
# Rebuild the Stage B regions of the multitask Stage A (patch caches reuse identical
# regions), then fine-tune Stage B on them: "continue" (S2) and "refined" (S12).
# The cohort build runs as 2 parallel processes over folds (region extraction is
# CPU/IO bound); --resume makes every fold restartable.
source "$(dirname "$0")/common.sh"
cohort="data/stage_b_cohort_RN18_A_mt"
build() {
  $PY -m scripts.stage_b.build_stageB_cohort_cached --model $MODEL     --checkpoints-glob "ResNet18_output/training_output_stageA_mt/*_split_0*/best_f1.pth"     --control-image $CONTROL_IMG --wsi-root "$WSI_ROOT" --out-dir $cohort     --cache-dirs $COHORT_A0 --verify-cache 0 --folds "$1" --resume 2>&1 | tee -a logs/phase3_cohort_folds_${1//,/_}.log
}
build 1,2,3 & build 4,5 & wait
# stop if any fold is missing (never train on a partial cohort)
for f in 1 2 3 4 5; do [ -f $cohort/manifests/stageB_fold$f.done ] || { echo "cohort fold $f incomplete"; exit 1; }; done
# combined manifest (each parallel process wrote only its own folds' part)
$PY -m scripts.stage_b.build_stageB_cohort_cached --model $MODEL   --checkpoints-glob "ResNet18_output/training_output_stageA_mt/*_split_0*/best_f1.pth"   --control-image $CONTROL_IMG --wsi-root "$WSI_ROOT" --out-dir $cohort   --cache-dirs $COHORT_A0 --folds 1,2,3,4,5 --resume 2>&1 | tee -a logs/phase3_cohort_merge.log
for recipe in continue refined; do
  for fold in 1 2 3 4 5; do
    out="ResNet18_output/training_output_stageB_ft_${recipe}"
    tag="rn18_Amt_stageB_${recipe}_fold${fold}"
    [ -f "$out/$tag/efficiency.json" ] && { echo "skip $tag"; continue; }
    $PY -m scripts.stage_b.finetune_stageB --model $MODEL --fold $fold       --cohort-dir $cohort --recipe $recipe --tag $tag --workers 10       2>&1 | tee -a logs/phase3_Amt_${recipe}.log
  done
done
