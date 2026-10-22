#!/usr/bin/env bash
# Shared settings for the cascade refinement study (RN18).
set -u
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"
MODEL=resnet18
A0_GLOB="ResNet18_output/training_output/*_split_0*/best_f1.pth"          # original Stage A
B0_GLOB="ResNet18_output/training_output_stageB_F1/*fold*/best_f1.pth"    # original Stage B
COHORT_A0="data/stage_b_cohort_RN18_F1"                                   # regions of the original Stage A
CONTROL_IMG="data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg"
WSI_ROOT="C:/Users/TAbde/Documents/EMC_postdoc/Virtual_Biopsy/data/WSI_ebrains/WHO2021_data"
mkdir -p logs
