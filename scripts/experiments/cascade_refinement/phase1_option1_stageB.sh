#!/usr/bin/env bash
# Option 1: Stage B fine-tuning on the original Stage A's regions.
source "$(dirname "$0")/common.sh"
for recipe in continue refined; do
  for fold in 1 2 3 4 5; do
    out="ResNet18_output/training_output_stageB_ft_${recipe}"
    tag="rn18_A0_stageB_${recipe}_fold${fold}"
    [ -f "$out/$tag/efficiency.json" ] && { echo "skip $tag"; continue; }
    $PY -m scripts.stage_b.finetune_stageB --model $MODEL --fold $fold \
      --cohort-dir $COHORT_A0 --recipe $recipe --tag $tag --workers 10 \
      2>&1 | tee -a logs/phase1_${recipe}.log
  done
done
