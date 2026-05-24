# GLIOMA-SPARSE

GLIOMA-SPARSE is a lightweight, interpretable computational pathology framework for glioma classification from routine H&E whole-slide images.

The method follows a coarse-to-fine strategy:
- Stage A: low-resolution analysis to identify informative regions
- Stage B: high-resolution analysis on selected regions

The goal is to minimize compute while preserving diagnostic signal.

---

## INDEX

1. Installation
2. Repository Structure
3. Preprocessing
4. Stage A Pipeline (Current)
5. Outputs

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
scripts/        → runnable scripts

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

This is now a complete training pipeline.

---

### 4.1 Dataset

Class:

```
SlideDataset
```

- One thumbnail = one sample
- Folder structure defines labels
- Stores:
  - paths
  - labels
- Supports:
  - transforms
  - patch shuffling

---

### 4.2 Train / Val / Test Split

- Default: 80 / 10 / 10 split
- Saved to:

```
training_output/<experiment_name>/data_split.csv
```

- Can be reused via:

```
USE_EXISTING_SPLIT = True
```

Note:
- For very small datasets, validation may be empty
- Stratified splitting is recommended for real experiments

---

### 4.3 Patch Shuffle (Core idea)

Module:

```
Patch(grid_size=8)
```

- Splits image into grid
- Randomly permutes patches
- Applied at every sample access

Ensures stochasticity even with limited data.

---

### 4.4 Oversampling (Class Balancing)

Module:

```
data_utils/sampling.py
```

Usage:

```
if USE_OVERSAMPLING:
    train_dataset.paths, train_dataset.labels = oversample_paths(...)
```

- Balances class distribution
- Applied before DataLoader
- Combined with patch shuffle → unique samples

---

### 4.5 Model

```
build_model("resnet18", num_classes=3)
```

- Default: ResNet-18
- Input: 2048 × 2048
- Easily replaceable

---

### 4.6 Training

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
  - best AUC
  - best F1
  - best Accuracy
  - last model
  - optional per-epoch checkpoints
- Early stopping

---

### 4.7 Evaluation Outputs

After training, the pipeline generates:

- Training curves:
  - loss_curve.png
  - auc_curve.png
  - f1_curve.png
  - accuracy_curve.png
- Confusion matrix:
  - confusion_matrix.png
- ROC curve:
  - roc_curve.png

All plotting is handled via:

```
glioma_sparse.evaluation.plots
```

---

### 4.8 Compute Tracking

Trainer logs:

- Epoch time
- Total training time
- AUC per hour

Enables reporting of compute efficiency.

---

## 5. OUTPUTS

```
training_output/<experiment_name>/

log.csv
data_split.csv

best_auc.pth
best_f1.pth
best_acc.pth
last.pth
epoch_*.pth   (optional)

loss_curve.png
auc_curve.png
f1_curve.png
accuracy_curve.png
confusion_matrix.png
roc_curve.png
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
- Designed for:
  - local machines
  - GPU clusters
- Uses:
  - patch shuffling
  - oversampling
  - explicit dataset splits
- Fully reproducible experiment folders

---

## NEXT STEPS

- YAML config system
- Mixed precision (AMP)
- Inference timing per slide
- Stage B (patch-level model)
- End-to-end pipeline
- External validation datasets