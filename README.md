# GLIOMA-SPARSE

GLIOMA-SPARSE is a lightweight, interpretable computational pathology framework for glioma classification from routine H&E whole-slide images.

The method follows a coarse-to-fine strategy, identifying informative regions at low resolution and selectively refining predictions at higher resolution. This enables accurate classification while minimizing computational cost.

---

# Index

1. Installation  
2. Preprocessing  
   2.1 Thumbnail Generation  
   2.2 Single-slide processing  
   2.3 Batch processing  

---

# 1. Installation

## Create environment

```bash
conda env create -f environment.yml
conda activate Glioma-SPARSE
```

## Install package (important)

```bash
pip install -e .
```

This allows imports like:

```python
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
```

---

# 2. Preprocessing

This stage converts raw WSIs into standardized thumbnails and computes tissue content.

Core idea:
- Always generate thumbnails  
- Always compute tissue fraction  
- Never discard data silently  
- Use threshold only for grouping  

---

## 2.1 Thumbnail Generation

### Core script (single WSI)

Location:
```
src/glioma_sparse/preprocessing/create_wsi_thumbnail.py
```

Main function:
```python
create_wsi_thumbnail(
    slide_path,
    output_path=None,
    target_mpp=4.0,
    output_size=2048,
    tissue_threshold=None,
    save_mask=False
)
```

Returns:
```
(thumbnail, tissue_fraction)
```

Notes:
- No printing  
- No folder logic  
- Used in both batch and inference pipelines  

---

## 2.2 Single-slide processing

Used for:
- debugging  
- development  
- future inference pipeline  

Example:

```python
from pathlib import Path
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail

slide_path = Path("path/to/slide.svs")

thumbnail, tissue_fraction = create_wsi_thumbnail(
    slide_path,
    output_path="output.jpg",
    target_mpp=4.0,
    output_size=2048,
    tissue_threshold=None,
    save_mask=True
)

print("Tissue fraction:", tissue_fraction)
```

---

## 2.3 Batch processing

### Script

Location:
```
src/glioma_sparse/preprocessing/process_dataset.py
```

Main function:
```python
process_wsi_folder(
    input_dir,
    output_dir,
    tissue_threshold=0.3
)
```

### What it does

- Recursively finds WSIs  
- Generates thumbnails for all slides  
- Computes tissue fraction  
- Splits outputs into:
  - kept/ (included)
  - low_tissue/ (below threshold)  
- Writes metadata to CSV  

### Output structure

```
output_dir/
    kept/
    low_tissue/
    metadata.csv
```

### CSV format

```
slide_path, thumbnail_path, tissue_fraction, included
```

---

## Running batch processing

### Demo script

Location:
```
scripts/demo_thumbnail.py
```

Run from project root:

```bash
python scripts/demo_thumbnail.py
```

---

## Configuration (important parameters)

These appear in different places but should be kept consistent:

### Resolution
```
target_mpp = 4.0
```

### Output size
```
output_size = 2048
```

### Tissue filtering (for grouping only)
```
tissue_threshold = 0.3
```

---

## Notes to self (important)

- Do NOT filter slides during generation  
- Threshold is only for grouping, not exclusion  
- Always log tissue_fraction  
- Always keep metadata.csv  
- MRXS files may behave differently so check tissue masks  

---

## Next steps (future work)

- Stage A: low-resolution risk mapping  
- Stage B: high-resolution patch classification  
- End-to-end inference pipeline  
- Visualization (mask overlays)  