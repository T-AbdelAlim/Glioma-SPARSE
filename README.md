<p align="right">
  <img src="https://raw.githubusercontent.com/T-AbdelAlim/Glioma-SPARSE/main/docs/logo_small.png" width="220"/>
</p>

# GLIOMA-SPARSE

GLIOMA-SPARSE is a lightweight, interpretable computational pathology framework for glioma classification from routine H&E whole-slide images.

The method follows a coarse-to-fine strategy:
- Stage A: low-resolution analysis to identify informative regions
- Stage B: high-resolution analysis on selected regions

The goal is to minimize compute while preserving diagnostic signal.

---

## INDEX

1. [Installation](#1-installation)
2. [Repository Structure](#2-repository-structure)
3. [Preprocessing](#3-preprocessing)
4. [Stage A Pipeline (Current)](#4-stage-a-pipeline-current)
5. [Cluster Usage (SLURM)](#5-cluster-usage-slurm)
6. [Outputs](#6-outputs)

---

## 1. INSTALLATION

### Recommended setup (fast & stable)

Create environment:

```
conda create -n glioma-sparse python=3.10
conda activate glioma-sparse
```

Install dependencies:

```
pip install -r requirements.txt
```

Install repository:

```
pip install -e .
```

---

### GPU support (required for training speed)

Verify:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### OpenSlide (Windows requirement)

Download binaries:  
https://openslide.org/download/

Add to PATH:

```
C:\path\to\openslide\bin
```

---

## 2. REPOSITORY STRUCTURE

```
Glioma-SPARSE/

configs/        → configuration files (future)
data/           → processed thumbnails per class
docs/           → documentation
notebooks/      → experiments
scripts/        → runnable scripts (entry points)

src/
  glioma_sparse/

    preprocessing/
      create_wsi_thumbnail.py
      process_dataset.py

    data_utils/
      slide_dataset.py
      patches.py
      transforms.py
      sampling.py

    models/
      factory/

    training/
      trainer.py
      seeding.py        ← ensures reproducibility

    evaluation/
      metrics.py
      plots.py

tests/
```

---

## 3. PREPROCESSING

Converts WSIs into thumbnails + tissue metrics.

---

### 3.1 Thumbnail Generation

Function:

```
create_wsi_thumbnail()
```

Returns:

```
img, tissue_fraction, effective_tissue_fraction
```

Steps:
1. Read WSI (OpenSlide)
2. Downsample to target MPP
3. Compute tissue mask
4. Compute tissue_fraction (before padding)
5. Pad to square
6. Resize
7. Compute effective_tissue_fraction (after resizing)

---

### 3.2 Dataset Processing

Function:

```
process_wsi_folder()
```

What it does:
- Recursively finds WSIs (.svs, .ndpi, .mrxs, .tif, .tiff)
- Generates thumbnails
- Computes:
  - tissue_fraction
  - effective_tissue_fraction
- Splits into:
  - included/
  - low_tissue/
- Logs everything to metadata.csv

Run batch processing:

```
python scripts/demo_wsi_to_thumbnail.py
```

---

### 3.3 Key Concepts

#### tissue_fraction
- Computed before padding
- Reflects biological content

#### effective_tissue_fraction
- Computed after padding/resizing
- Reflects what the model actually sees

#### threshold_metric

```
"tissue"    → biological filtering
"effective" → model-input-aware filtering
```

---

## 4. STAGE A PIPELINE (CURRENT)

Run:

```
python scripts/train.py
```

This is a complete, reproducible training pipeline.

---

### 4.1 Dataset

Class:

```
SlideDataset
```

- One thumbnail = one sample
- Explicit class order enforced:
  
```
["control", "low_grade", "high_grade"]
```

- Ensures mapping:

```
0 = control  
1 = low_grade  
2 = high_grade
```

- Supports:
  - transforms
  - patch shuffling
  - predefined splits via `paths`

---

### 4.2 Reproducibility (IMPORTANT)

Seeding is centralized:

```
glioma_sparse.training.seeding
```

Includes:
- torch
- numpy
- random
- CUDA

Ensures:
- deterministic splits
- reproducible training runs

---

### 4.3 Train / Val / Test Split

- Default: 80 / 10 / 10 split
- Stratified on labels
- Saved to:

```
training_output/<experiment_name>/data_split.csv
```

- Can be reused via:

```
USE_EXISTING_SPLIT = True
```

---

### 4.4 Patch Shuffle (Core idea)

Module:

```
Patch(grid_size=8)
```

- Splits image into grid
- Randomly permutes patches
- Applied at every sample access

Ensures stochasticity even with limited data.

---

### 4.5 Class Imbalance Handling

Two strategies:

#### Option A — Oversampling

```
USE_OVERSAMPLING = True
```

- Balances dataset via duplication
- Works well with patch shuffle

#### Option B — Class-weighted loss (recommended)

```
USE_CLASS_WEIGHTED_LOSS = True
```

- Uses inverse frequency weighting
- More stable than oversampling

⚠️ Do NOT combine both unless carefully tuned.

---

### 4.6 Model

```
build_model("resnet18", num_classes=3)
```

- Lightweight baseline
- Suitable for resource-efficient experiments
- Replaceable with:
  - ResNet50
  - EfficientNet
  - custom architectures

---

### 4.7 Training

Handled by:

```
Trainer
```

Includes:

- Forward + backward pass
- Validation loop
- Metrics:
  - Accuracy
  - F1
  - AUC
- Logging:
  - per epoch
  - CSV output
- Checkpointing:
  - best_auc.pth
  - best_f1.pth
  - best_acc.pth
  - last.pth
- Early stopping (optional)

---

### 4.8 Evaluation Outputs

Generated automatically:

- Training curves:
  - loss_curve.png
  - auc_curve.png
  - f1_curve.png
  - accuracy_curve.png

- Confusion matrices:
  - confusion_matrix_val_raw.png
  - confusion_matrix_val_norm.png
  - confusion_matrix_test_raw.png
  - confusion_matrix_test_norm.png

- ROC curves:
  - roc_curve_val.png
  - roc_curve_test.png

All plots use fixed class order.

---

### 4.9 Compute Tracking

Trainer logs:

- Epoch time
- Total training time
- AUC per hour

Supports reporting of:
- efficiency
- scalability

---

## 5. CLUSTER USAGE (SLURM)

Training can be run on GPU clusters (e.g. oaks-lab).

---

### 5.1 SLURM Script

Example:

```
train_glioma_sparse.slurm
```

Uses:
- 1 GPU
- 8 CPUs
- 64 GB RAM
- containerized environment

Key paths:

```
/data/pathology/projects/tareq/glioma-sparse
```

---

### 5.2 Submit Job

From project directory:

```
sbatch train_glioma_sparse.slurm
```

---

### 5.3 Monitor Job

```
squeue -u $USER
```

Logs:

```
/home/<user>/logs/slurm-<jobid>.out
```

---

## 6. OUTPUTS

```
training_output/<experiment_name>/

config.txt
log.csv
data_split.csv

best_auc.pth
best_f1.pth
best_acc.pth
last.pth

loss_curve.png
auc_curve.png
f1_curve.png
accuracy_curve.png

confusion_matrix_val_raw.png
confusion_matrix_val_norm.png
confusion_matrix_test_raw.png
confusion_matrix_test_norm.png

roc_curve_val.png
roc_curve_test.png
```

---

### metadata.csv (preprocessing)

```
slide_path
thumbnail_path
tissue_fraction
effective_tissue_fraction
included
```

---

## NOTES

- Thumbnails saved as JPG (quality=90)
- Explicit class ordering enforced across:
  - dataset
  - training
  - evaluation
- Designed for:
  - local machines
  - GPU clusters
- Fully reproducible experiment structure

---

## NEXT STEPS

- YAML config system
- Mixed precision (AMP)
- Inference timing per slide
- Stage B (patch-level model)
- End-to-end pipeline
- External validation datasets