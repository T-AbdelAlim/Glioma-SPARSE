
# GLIOMA-SPARSE
<img align="right" src="docs/logo.png" width="220px" />
GLIOMA-SPARSE is a lightweight, interpretable computational pathology framework for glioma classification from routine H&E whole-slide images.

The method follows a coarse-to-fine strategy:
- Stage A: low-resolution analysis to identify informative regions
- Stage B: high-resolution analysis on selected regions

The goal is to minimize compute while preserving diagnostic signal.



## INDEX

1. [Installation](#1-installation)
2. [Repository Structure](#2-repository-structure)
3. [Preprocessing](#3-preprocessing)
4. [Stage A Pipeline (Training)](#4-stage-a-pipeline-training)
5. [Patch Injection and Stage B Dataset Generation](#5-patch-injection-and-stage-b-dataset-generation)
6. [Cluster Usage (SLURM)](#6-cluster-usage-slurm)
7. [Outputs](#7-outputs)

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
scripts/        → runnable entry points
  train.py                   → Stage A training
  process_wsi_folder.py      → batch preprocessing runner
  extract_risk_region.py     → Stage A interpretability + Stage B dataset build
  inference_stageA.py        → Stage A inference on new slides

src/
  glioma_sparse/

    preprocessing/
      create_wsi_thumbnail.py   → single-WSI thumbnail + mapping sidecar
      process_dataset.py        → batch preprocessing + CSV log

    data_utils/
      slide_dataset.py
      patches.py
      transforms.py
      sampling.py

    models/
      factory.py

    training/
      trainer.py

    evaluation/
      metrics.py
      plots.py

    interpret/                  ← Stage A interpretability + Stage B input generation
      wsi_mapping.py            → thumbnail ↔ WSI level-0 coordinate mapping
      patch_injection.py        → risk map by injecting target patches into a control
      highres_extraction.py     → top-k high-resolution patch extraction from the WSI

    utils/
      seeding.py                ← ensures reproducibility

tests/
```

---

## 3. PREPROCESSING

Converts WSIs into thumbnails + tissue metrics + a mapping sidecar.

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
2. Downsample to target MPP (physical resolution in µm per pixel)
3. Compute tissue mask
4. Compute tissue_fraction (before padding)
5. Pad to square
6. Resize to thumbnail size
7. Compute effective_tissue_fraction (after resizing)
8. Write mapping sidecar (`<slide_id>.json`) when an output_path is provided

---

### 3.2 Mapping Sidecar (IMPORTANT for Stage B)

Each thumbnail JPG is accompanied by a `<slide_id>.json` sidecar containing every parameter needed to map a thumbnail-pixel bounding box back to WSI level-0 pixel coordinates without re-opening the slide:

```json
{
  "wsi_path": "/data/.../slide.svs",
  "wsi_level0_dim": [60000, 50000],
  "base_mpp": 0.5001,
  "target_mpp": 4.0,
  "tissue_image_dim": [7500, 6249],
  "canvas_size": 7500,
  "tissue_offset_in_canvas": [0, 626],
  "thumbnail_size": 2048
}
```

These sidecars are required by `interpret/` for Stage B (re-extracting high-resolution patches from selected thumbnail regions). Old thumbnails generated before sidecar support need to be re-preprocessed.

---

### 3.3 Dataset Processing

Function:

```
process_wsi_folder()
```

What it does:
- Recursively finds WSIs (.svs, .ndpi, .mrxs, .tif, .tiff)
- Generates thumbnails (+ sidecars) via a staging directory so JPG and JSON always travel as a pair
- Computes:
  - tissue_fraction
  - effective_tissue_fraction
- Splits into:
  - included/
  - low_tissue/
- Logs everything to metadata.csv (including sidecar paths)

Run batch processing:

```
python scripts/process_wsi_folder.py
```

---

### 3.4 Key Concepts

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

## 4. STAGE A PIPELINE (TRAINING)

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
glioma_sparse.utils.seeding
```

Includes:
- torch
- numpy
- random
- CUDA
- per-worker DataLoader seeds

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

Ensures the model is invariant to tile position, which is a prerequisite for the patch-injection interpretability of Stage A (Section 5).

---

### 4.5 Class Imbalance Handling

Two strategies:

#### Option A. Oversampling

```
USE_OVERSAMPLING = True
```

- Balances dataset via duplication
- Works well with patch shuffle

#### Option B. Class-weighted loss (recommended)

```
USE_CLASS_WEIGHTED_LOSS = True
```

- Uses normalised inverse-frequency weighting:
  `w_i = total / (num_classes * count_i)`
- Weights average to 1.0 in the balanced case, so loss magnitudes are
  comparable to an unweighted run.
- More stable than oversampling

The two strategies are mutually exclusive (an assertion enforces this).

---

### 4.6 Model

```
build_model("resnet18", num_classes=3)
```

- Lightweight baseline
- Suitable for resource-efficient experiments
- Replaceable with:
  - ResNet34
  - ResNet50
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

Final validation and test evaluation are run on the best checkpoint (selected by `BEST_CHECKPOINT_METRIC`, default "auc"), not on the last-epoch state.

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

All plots use the fixed class order.

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

## 5. PATCH INJECTION AND STAGE B DATASET GENERATION

The `interpret/` module turns Stage A predictions into spatially-localised explanations and, in the same step, produces the high-resolution patches that feed Stage B. The script `scripts/extract_risk_region.py` is the user-facing entry point that wraps the module for single-slide inspection, per-class batch processing, and full Stage B dataset construction.

---

### 5.1 Patch-Injection Risk Map

Module:

```
glioma_sparse.interpret.patch_injection
```

Idea. For a target slide that Stage A predicts as class `p`, replace the tile at position (r, c) of a control slide with the target slide's tile at the same position, run the model on the modified image, and record how much the predicted probability of class `p` changes. The control slide is shuffled once with a fixed seed so that its own tile arrangement carries no positional information.

```
risk_map[r, c] = P_with_injected_target_tile(class p)  -  P_control_alone(class p)
```

Positive values mean the injected tile increased the model's confidence in class `p`. Higher values therefore mean "more indicative of class p".

Why this works for our pipeline. Training uses random patch permutation (Section 4.4), which makes the model invariant to tile position, so the risk score reflects only the tile's content rather than its location on the slide.

Direct module usage (rarely needed; prefer the script in 5.4):

```python
from glioma_sparse.interpret import compute_risk_map

risk_map, target_class = compute_risk_map(
    target_img=target_thumbnail,
    control_img=control_thumbnail,
    model=stage_a_model,
    transform=eval_transform,
    grid=(8, 8),
    device="cuda",
)
```

---

### 5.2 Coordinate Mapping (Thumbnail → WSI)

Module:

```
glioma_sparse.interpret.wsi_mapping
```

A naive mapping `scale = wsi_dim / thumbnail_dim` is wrong, because the thumbnail goes through three transformations during preprocessing:

1. WSI level 0 resampled to target MPP, producing a tissue image of size `(Wt, Ht)`.
2. Tissue image padded to a square canvas of side `S = max(Wt, Ht)`.
3. Canvas resized to the final thumbnail size `T` (e.g. 2048).

Each thumbnail's sidecar (Section 3.2) records every parameter of these transformations, so the mapping is recovered exactly without re-opening the WSI:

```python
from glioma_sparse.interpret import load_mapping, thumbnail_bbox_to_wsi

mapping  = load_mapping("path/to/thumbnail.jpg")
wsi_bbox = thumbnail_bbox_to_wsi(mapping, (px_left, py_top, px_right, py_bottom))
# wsi_bbox = (wx, wy, ww, wh) suitable for openslide.read_region((wx, wy), 0, (ww, wh))
# or None if the thumbnail bbox falls entirely in padding
```

The function handles padding correctly: a tile that overlaps the padding band is automatically clipped to the tissue region before being mapped, and a tile that lies entirely in padding returns `None`.

---

### 5.3 High-Resolution Patch Extraction

Module:

```
glioma_sparse.interpret.highres_extraction
```

Given a thumbnail, its Stage-A risk map, and `k`, this function selects the top-`k` highest-risk tiles (after filtering out tiles whose thumbnail content is mostly padding), maps each one back to WSI level-0 coordinates using the sidecar, reads the region with OpenSlide, and writes a `(output_size × output_size)` JPG (default 2048 × 2048) per patch.

Direct module usage:

```python
from glioma_sparse.interpret import extract_topk_patches

results = extract_topk_patches(
    thumbnail_path="data/included/slide.jpg",
    risk_map=risk_map,
    k=5,
    output_size=2048,
    label=class_names[target_class],
)
```

Output files are named `<slide_id>_rankNN_r<r>c<c>_<label>.jpg`.

---

### 5.4 Script: extract_risk_region.py

Script:

```
scripts/extract_risk_region.py
```

The script loads the Stage A model and the canonical control image once, then runs the per-slide pipeline (predict, compute risk map, save overlays, extract top-k patches) in three operational modes selected by the `MODE` variable at the top of the file.

#### Mode: single

Process one thumbnail or one raw WSI. If the input is a WSI (`.ndpi`, `.svs`, `.mrxs`, `.tif`, `.tiff`), a thumbnail and matching sidecar are generated on the fly in a `tmp/` subdirectory.

Outputs (per slide):

```
risk_output_<stem>/
  overlay.jpg              ← smooth heatmap on the thumbnail
  grid_risk_map.jpg        ← per-tile coloured grid + colourbar
  patches/                 ← top-k high-resolution patches
  tmp/                     ← intermediate thumbnail + sidecar
```

Use this mode for figure generation, qualitative inspection, and demos.

#### Mode: batch

Process every thumbnail or WSI inside one folder with one label. Used to build the Stage B input cohort for a single molecular class.

Required setting: `label_override`, a string carried into the filename of every extracted patch (e.g. `"IDH_mut_1p19qCD"`).

Outputs:

```
<output_root>/
  patches/                 ← flat directory, all patches across the batch
  overlays/                ← optional, only if save_overlays=True
  batch_summary.csv        ← per-slide prediction + probabilities + n_patches
  batch_failures.csv       ← only if any slide failed
  tmp/                     ← intermediate thumbnails for WSI inputs
```

`batch_summary.csv` and `batch_failures.csv` are append-aware. Running the script again with the same `output_root` (for example to add a new class) concatenates results rather than overwriting them.

#### Mode: batch_all_classes

Iterate over the molecular-class mapping in one run. This builds the complete Stage B training dataset.

Mapping (WHO 2021):

```
oligo_IDHmt_1p19qdel_G2   →  IDH_mut_1p19qCD
oligo_IDHmt_1p19qdel_G3   →  IDH_mut_1p19qCD
astro_IDHmt_G2            →  IDH_mut
astro_IDHmt_G3            →  IDH_mut
astro_IDHmt_G4            →  IDH_mut
GBM_IDHwt                 →  IDH_wt
```

Control is excluded (no molecular subtype to predict).

Output structure is identical to `batch`, with patches from all classes accumulating in `output_root/patches/`. Filenames encode the molecular class:

```
<output_root>/patches/
  <slide_id>_rank01_r<r>c<c>_IDH_mut_1p19qCD.jpg
  <slide_id>_rank02_r<r>c<c>_IDH_mut_1p19qCD.jpg
  ...
  <slide_id>_rank01_r<r>c<c>_IDH_mut.jpg
  ...
  <slide_id>_rank01_r<r>c<c>_IDH_wt.jpg
  ...
```

This folder is the input directory for Stage B training (`DATA_DIR` in the upcoming `train_stage_b.py`).

#### Design notes

- The model, transform, and control image are loaded once per script invocation, not per slide.
- The risk map is always computed for the **predicted** class from Stage A. The `label_override` only sets the molecular label written into patch filenames.
- Overlays are off by default in batch mode (slow, not needed for training data); turn them on with `save_overlays=True` if you want a visual record.
- The canonical control thumbnail is fixed per project. Change it only if you know why.

---

## 6. CLUSTER USAGE (SLURM)

Training can be run on GPU clusters (e.g. oaks-lab).

---

### 6.1 SLURM Script

Example:

```
train_glioma_sparse.slurm
```

Uses:
- 1 GPU
- 8 CPUs
- 64 GB RAM
- conda environment (no container required)

Key paths:

```
/data/pathology/projects/tareq/Glioma-SPARSE
```

---

### 6.2 Submit Job

From project directory:

```
sbatch train_glioma_sparse.slurm resnet18
```

---

### 6.3 Monitor Job

```
squeue -u $USER
```

Logs:

```
/home/<user>/logs/slurm-<jobid>.out
```

---

## 7. OUTPUTS

### Stage A training

```
training_output/<experiment_name>/

config.json
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

### Preprocessing

```
data/<class>/
  included/
    <slide_id>.jpg
    <slide_id>.json        ← mapping sidecar
  low_tissue/
    <slide_id>.jpg
    <slide_id>.json
  metadata.csv
```

`metadata.csv` columns:

```
slide_path
thumbnail_path
sidecar_path
tissue_fraction
effective_tissue_fraction
included
processing_time_sec
success
error
```

---

### Stage A interpretation, single mode

```
risk_output_<slide_stem>/
  overlay.jpg
  grid_risk_map.jpg
  patches/
    <slide_id>_rank01_r<r>c<c>_<predicted_class>.jpg
    ...
```

---

### Stage B input cohort (batch or batch_all_classes mode)

```
stage_b_training_data/
  patches/
    <slide_id>_rank01_r<r>c<c>_IDH_mut_1p19qCD.jpg
    <slide_id>_rank01_r<r>c<c>_IDH_mut.jpg
    <slide_id>_rank01_r<r>c<c>_IDH_wt.jpg
    ...
  overlays/                  ← only if save_overlays=True
  batch_summary.csv
  batch_failures.csv         ← if any slide failed
```

---

## NOTES

- Thumbnails saved as JPG (quality=90)
- Each thumbnail has a JSON sidecar with the WSI mapping
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

- Stage B training script (`scripts/train_stage_b.py`)
- End-to-end inference pipeline (Stage A → patch extraction → Stage B → soft-vote)
- YAML config system
- Mixed precision (AMP)
- Inference timing per slide
- External validation datasets
