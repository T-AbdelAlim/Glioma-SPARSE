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
   3.1 Thumbnail Generation (single WSI)  
   3.2 Dataset Processing (batch)  
   3.3 Key Concepts  
4. Outputs  

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

Use this as a quick mental map:

```
Glioma-SPARSE/

configs/        → configuration files (future)
data/           → small test WSIs (ignored by git)
docs/           → documentation
notebooks/      → experiments / exploration
scripts/        → runnable demo scripts
src/
  glioma_sparse/
    preprocessing/
      create_wsi_thumbnail.py   → single WSI processing
      process_dataset.py        → batch processing
tests/         → unit tests (optional)
```

---

## 3. PREPROCESSING

This stage converts raw WSIs into thumbnails and computes tissue statistics.

---

### 3.1 Thumbnail Generation (single WSI)

Core function:

```
create_wsi_thumbnail()
```

Location:

```
src/glioma_sparse/preprocessing/create_wsi_thumbnail.py
```

What it does:

1. Reads WSI using OpenSlide  
2. Downsamples to target physical resolution (MPP)  
3. Computes tissue mask  
4. Computes:
   - tissue_fraction (before padding)
5. Pads image to square  
6. Resizes to output size  
7. Computes:
   - effective_tissue_fraction (after resizing)
8. Returns:

```
img, tissue_fraction, effective_fraction
```

Important:
- Tissue is computed BEFORE padding  
- Effective fraction is computed AFTER resizing  
- All metric computation happens inside this function  

---

### 3.2 Dataset Processing (batch)

Core function:

```
process_wsi_folder()
```

Location:

```
src/glioma_sparse/preprocessing/process_dataset.py
```

What it does:

- Recursively finds WSIs  
- Generates thumbnails  
- Uses metrics returned from `create_wsi_thumbnail`  
- Splits outputs into:
  - included/
  - low_tissue/
- Logs everything to CSV  

No metric recomputation is performed in this script.

---

### Demo script

Run:

```bash
python scripts/process_dataset_demo.py
```

Config inside script:

```
TISSUE_THRESHOLD = 0.3
THRESHOLD_METRIC = "tissue"   # or "effective"
```

---

### 3.3 Key Concepts

#### tissue_fraction

```
fraction of tissue pixels BEFORE padding
```

- Computed BEFORE padding  
- Reflects biological content  
- Use when you care about actual tissue  

---

#### effective_tissue_fraction

```
fraction of tissue pixels AFTER resizing
```

- Computed AFTER padding and resizing  
- Reflects what the model actually sees  
- Captures:
  - excessive padding
  - thin tissue regions
  - scanner artifacts  

---

#### threshold_metric

Controls filtering:

```
"tissue"    → biological filtering (default)
"effective" → model-input-aware filtering
```

Guideline:

- Use "tissue" for dataset curation  
- Use "effective" for model robustness  

---

## 4. OUTPUTS

After running dataset processing:

```
output_dir/

included/      → thumbnails above threshold
low_tissue/    → thumbnails below threshold
metadata.csv   → full log
```

---

### metadata.csv

Columns:

```
slide_path
thumbnail_path
tissue_fraction
effective_tissue_fraction
included
```

---

## NOTES

- Thumbnails are saved as JPG (quality=90)
- Tissue masks (optional) are saved as PNG
- Designed to work on:
  - local machines (Windows/Linux)
  - remote GPU clusters

---

## NEXT STEPS

- Stage A: low-resolution risk mapping  
- Stage B: patch-based classification  
- End-to-end inference pipeline  