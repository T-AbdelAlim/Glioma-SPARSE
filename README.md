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

Create and activate environment:

```bash
conda env create -f environment.yml
conda activate glioma-sparse
```

Install repository in editable mode:

```bash
pip install -e .
```

---

## 2. REPOSITORY STRUCTURE

```
Glioma-SPARSE/

configs/        → configuration files (future)
data/           → raw WSIs (ignored by git)
docs/           → documentation
notebooks/      → experiments
scripts/        → runnable demos

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

tests/          → optional
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
img, tissue_fraction, tissue_pixels
```

Steps:
1. Read WSI (OpenSlide)  
2. Downsample (MPP-based)  
3. Compute tissue mask  
4. Compute tissue_fraction (before padding)  
5. Pad to square  
6. Resize  
7. Return image + metrics  

---

### 3.2 Dataset Processing

Function:

```
process_wsi_folder()
```

What it does:
- Finds WSIs recursively  
- Generates thumbnails  
- Computes:
  - tissue_fraction
  - effective_tissue_fraction  
- Splits into:
  - included/
  - low_tissue/  
- Writes metadata.csv  

Run:

```bash
python scripts/process_dataset_demo.py
```

Config:

```
TISSUE_THRESHOLD = 0.3
THRESHOLD_METRIC = "tissue" or "effective"
```

---

### 3.3 Key Concepts

#### tissue_fraction
- Computed before padding  
- Reflects biological content  

#### effective_tissue_fraction
- Computed after padding/resizing  
- Reflects model input quality  

#### threshold_metric

```
"tissue"    → biological filtering  
"effective" → model-input-aware filtering  
```

---

## 4. STAGE A PIPELINE (CURRENT)

Pipeline implemented and runnable via:

```bash
python scripts/train_demo.py
```

---

### 4.1 Dataset

Class:

```
SlideDataset
```

- One thumbnail = one sample  
- Folder structure defines labels  
- Supports:
  - transforms
  - patch shuffling  

---

### 4.2 Patch Shuffle (Core idea)

Module:

```
Patch(grid_size=8)
```

- Splits image into grid  
- Randomly permutes patches  
- Applied at every sample access  

→ Ensures stochasticity even for duplicated samples  

---

### 4.3 Oversampling (Class Balancing)

Module:

```
data_utils/sampling.py
```

Usage in training script:

```python
if USE_OVERSAMPLING:
    train_dataset.paths, train_dataset.labels = oversample_paths(...)
```

- Balances class distribution  
- Happens BEFORE DataLoader  
- Combined with patch shuffle → unique samples  

---

### 4.4 Model

```
build_model("resnet18", num_classes=3)
```

- Default: ResNet-18  
- Input: 2048 × 2048  
- Easily switchable  

---

### 4.5 Training

Trainer handles:

- Forward + backward pass  
- AMP support (if enabled later)  
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

---

### 4.6 Compute Tracking

Trainer logs:

- Epoch time  
- Total training time  
- AUC per hour  

→ Directly supports compute-efficiency reporting

---

## 5. OUTPUTS

```
training_output/

metrics.csv
best_auc.pth
best_f1.pth
best_acc.pth
last.pth
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
- Patch shuffle + oversampling provide strong stochastic training  

---

## NEXT STEPS

- Proper train/val split  
- YAML config system  
- Stage A scaling experiments  
- Inference timing per slide  
- Stage B (patch-level model)  
- End-to-end pipeline  