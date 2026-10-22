#!/usr/bin/env bash
# Option 2: Stage A fine-tuning with (mt) and without (ctrl) the subtype head.
source "$(dirname "$0")/common.sh"
for arm in mt ctrl; do
  w=0.5; [ $arm = ctrl ] && w=0
  for fold in 1 2 3 4 5; do
    out="ResNet18_output/training_output_stageA_${arm}"
    tag="rn18_stageA_${arm}_split_0${fold}"
    [ -f "$out/$tag/efficiency.json" ] && { echo "skip $tag"; continue; }
    $PY -m scripts.stage_a.finetune_stageA_multitask --model $MODEL \
      --split-csv splits/split_0${fold}.csv --aux-weight $w --tag $tag \
      2>&1 | tee -a logs/phase2_${arm}.log
  done
done
