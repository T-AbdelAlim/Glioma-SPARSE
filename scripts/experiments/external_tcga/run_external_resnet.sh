#!/usr/bin/env bash
# External validation (TCGA, 180 slides, subtype-balanced): ResNet cascades, all five EBRAINS fold models
# applied unchanged.
# Usage: run_external_resnet.sh rn18_f1 | rn18_auc | rn50_f1 | rn50_auc
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
cfg=${1:?config}
case $cfg in
  rn18_f1)  model=resnet18; A="ResNet18_output/training_output/*_split_0*/best_f1.pth";  B="ResNet18_output/training_output_stageB_F1/*fold*/best_f1.pth" ;;
  rn18_auc) model=resnet18; A="ResNet18_output/training_output/*_split_0*/best_auc.pth"; B="ResNet18_output/training_output_stageB/*fold*/best_auc.pth" ;;
  rn50_f1)  model=resnet50; A="ResNet50_output/training_output/*_split_0*/best_f1.pth";  B="ResNet50_output/training_output_stageB_F1/*fold*/best_f1.pth" ;;
  rn50_auc) model=resnet50; A="ResNet50_output/training_output/*_split_0*/best_auc.pth"; B="ResNet50_output/training_output_stageB/*fold*/best_auc.pth" ;;
  *) echo "unknown config"; exit 1 ;;
esac
$PY -u -m scripts.inference.e2e_multi --run-name e2e_external_${cfg}_tcga --model $model \
  --stage-a-glob "$A" --system "B0=$B" \
  --external-labels results/external_TCGA_val/labels.csv --thumb-dir results/external_TCGA_val/thumbs_none \
  --patch-cache data/e2e_patch_cache_ext_tcga --ext-stain-profile none --resume \
  2>&1 | tee -a logs/external_${cfg}_tcga.log
