# GLIOMA-SPARSE
<img align="right" src="docs/logo.png" width="220px" />
GLIOMA-SPARSE is a lightweight, interpretable computational pathology framework for glioma classification from routine H&E whole-slide images.

The method follows a coarse-to-fine strategy:
- **Stage A** grades the slide from a low-resolution thumbnail (control / low_grade / high_grade) and, through patch injection, identifies the most informative regions.
- **Stage B** predicts the molecular subtype (IDH_mt / IDH_mt_1p19q / IDH_wt) from high-resolution patches re-extracted at the Stage A regions.

The design keeps compute low while preserving diagnostic signal, so the whole pipeline runs on a single modest GPU.

---

## INDEX

1. [Installation](#1-installation)
2. [Repository Structure](#2-repository-structure)
3. [Preprocessing](#3-preprocessing)
4. [Cross-Validation Splits](#4-cross-validation-splits)
5. [Stage A Training](#5-stage-a-training)
6. [Threshold Tuning (Stage A)](#6-threshold-tuning-stage-a)
7. [Patch Injection and the Stage B Cohort](#7-patch-injection-and-the-stage-b-cohort)
8. [Stage B Training](#8-stage-b-training)
9. [End-to-End Inference](#9-end-to-end-inference)
10. [Aggregating Results Across Folds](#10-aggregating-results-across-folds)
11. [Ablations](#11-ablations)
12. [Outputs](#12-outputs)
13. [Notes and Next Steps](#13-notes-and-next-steps)

---

## 1. INSTALLATION

Create the environment:

```
conda create -n glioma-sparse python=3.10
conda activate glioma-sparse
pip install -r requirements.txt
pip install -e .
```

### GPU support

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect a version tag like `2.x.x+cu121 True`. If it reports `+cpu` or `False`, reinstall PyTorch with a CUDA build matching your driver. The `cu121` and `cu124` wheels ship their own CUDA runtime, so any recent driver works:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The `CUDA Version` shown in `nvidia-smi` is the maximum runtime the driver supports, so a driver reporting a higher version than the wheel is fine.

### Optional dependencies for efficiency logging

```
pip install nvidia-ml-py thop
```

`nvidia-ml-py` (imported as `pynvml`) enables GPU energy measurement, and `thop` enables FLOP counting. Both degrade gracefully: without them, the relevant fields are logged as null and everything still runs.

### OpenSlide (Windows)

Download the binaries from https://openslide.org/download/ and add `C:\path\to\openslide\bin` to PATH.

---

## 2. REPOSITORY STRUCTURE

Entry points are run as modules from the repo root, for example `python -m scripts.stage_a.train`.

```
Glioma-SPARSE/

data/
  included/                     grade-organised thumbnails (control/low_grade/high_grade)
  ebrains_thumbnails/           thumbnails + JSON sidecars, per subtype/included
  stage_b_cohort/               generated Stage B patches + manifests
splits/                         the 5 frozen cross-validation split CSVs
results/                        end-to-end inference outputs, per run and slide

scripts/
  lib/
    eval_metrics.py             shared metric definitions (Stage A + B + aggregation)
    profiling.py                env capture, params/FLOPs, GPU energy meter
  data/
    generate_splits.py          write splits/split_01..05.csv
    aggregate_splits.py         aggregate per-fold results -> summary + figure
    process_wsi_folder.py       batch preprocessing runner
  stage_a/
    train.py                    Stage A training (one fold per run)
    tune_threshold.py           post-hoc operating-point tuning on validation
    inference_stageA.py         Stage A inference on new slides
    injection_site_ablation.py  R stability across injection sites
    fixed_site_injection.py     inject every tile at one fixed site
  stage_b/
    build_stageB_cohort.py      per-fold p95 patch extraction + manifests
    stageB_dataset.py           resolve train/val/test per fold from the manifest
    train_stageB.py             Stage B training (one fold per run)
  inference/
    inference_end_to_end.py     Stage A + Stage B end-to-end (single/folder/testset)
  demo/
    injection_demo.py           explainer figure of the injection mechanism

src/glioma_sparse/
  preprocessing/
    create_wsi_thumbnail.py     single-WSI thumbnail + mapping sidecar
    process_dataset.py          batch preprocessing + CSV log
  data_utils/
    slide_dataset.py, patches.py, transforms.py, sampling.py
  models/factory.py
  training/
    trainer.py                  training loop, checkpoints, early stopping
    seeding.py                  set_seed, seed_worker, make_generator
  evaluation/plots.py
  interpret/
    wsi_mapping.py              thumbnail <-> WSI level-0 coordinate mapping
    patch_injection.py          risk map by injecting target tiles into a control
    highres_extraction.py       p95 / top-k high-resolution patch extraction

tests/
```

The shared helpers in `scripts/lib/` are imported as `from scripts.lib.eval_metrics import ...`, so run entry points as modules (`python -m scripts.stage_a.train`) from the repo root.

---

## 3. PREPROCESSING

Preprocessing converts each WSI into a 2048x2048 thumbnail resampled to a target resolution (4.0 µm/pixel), plus a JSON sidecar that records the exact transformation from thumbnail pixels back to WSI level-0 coordinates.

### 3.1 Thumbnail generation

`create_wsi_thumbnail()` returns `img, tissue_fraction, effective_tissue_fraction` and, when an output path is given, writes the thumbnail JPG and its `<slide_id>.json` sidecar. Steps: read the WSI, pick the coarsest pyramid level at or below the target downsample, resize to the target MPP, measure tissue fraction, pad to a square, resize to the thumbnail size, and record the mapping.

### 3.2 Mapping sidecar (required for Stage B)

Each thumbnail carries a sidecar with everything needed to project a thumbnail bounding box to WSI level-0 without re-opening the slide:

```json
{
  "wsi_path": "/data/.../slide.ndpi",
  "wsi_level0_dim": [60000, 50000],
  "base_mpp": 0.5001,
  "target_mpp": 4.0,
  "tissue_image_dim": [7500, 6249],
  "canvas_size": 7500,
  "tissue_offset_in_canvas": [0, 626],
  "thumbnail_size": 2048
}
```

Stage B depends on these sidecars. A thumbnail generated before sidecar support has to be re-preprocessed.

### 3.3 Batch processing

`process_wsi_folder()` finds WSIs recursively, generates thumbnails and sidecars through a staging directory so the JPG and JSON always travel together, computes the tissue metrics, sorts slides into `included/` and `low_tissue/`, and logs to `metadata.csv`.

```
python -m scripts.data.process_wsi_folder
```

`tissue_fraction` is measured before padding (biological content), `effective_tissue_fraction` after padding (what the model sees), and `threshold_metric` selects which one the include/exclude decision uses.

---

## 4. CROSS-VALIDATION SPLITS

The study uses five stratified 80/10/10 splits, frozen once and reused by Stage A, Stage B, and any benchmark, so every result is comparable across the same partitions.

```
python -m scripts.data.generate_splits
```

This writes `splits/split_01.csv` through `splits/split_05.csv`, each with `path,split` columns and identical per-class proportions. The script prints per-split, per-class counts and confirms the five splits differ. Regenerate the splits whenever `DATA_DIR` changes.

---

## 5. STAGE A TRAINING

Train one fold per run by pointing at that fold's split CSV:

```
python -m scripts.stage_a.train --split-csv splits/split_01.csv
python -m scripts.stage_a.train --split-csv splits/split_02.csv
...
```

The `--split-csv` flag loads the pre-generated split, so the run uses exactly that partition rather than an on-the-fly split. Each run writes to `training_output/<timestamp>_resnet18_cw_split_0X/`.

### 5.1 Class order

Fixed across dataset, training, and evaluation:

```
0 = control, 1 = low_grade, 2 = high_grade
```

### 5.2 Reproducibility

Seeding is centralised in `glioma_sparse.training.seeding`: it seeds torch, numpy, random, and CUDA, sets `PYTHONHASHSEED`, and enables cuDNN determinism. The training DataLoader gets a seeded generator and per-worker seeds. With the seed fixed across all five folds, the only thing varying between folds is the data partition, which is what a split study is meant to isolate.

### 5.3 Patch shuffle

`Patch(8)` splits each thumbnail into an 8x8 grid and permutes the 64 tiles at every training access, applied to the training set only. This makes the model invariant to tile position, which is the property that lets Stage A's patch-injection risk score reflect tile content rather than location (Section 7).

### 5.4 Class imbalance

Two mutually exclusive strategies, enforced by an assertion:
- `USE_CLASS_WEIGHTED_LOSS = True` (default): normalised inverse-frequency weights, `w_i = total / (num_classes * count_i)`.
- `USE_OVERSAMPLING = True`: minority duplication, each duplicate seeing a fresh patch permutation.

### 5.5 Epochs and early stopping

`NUM_EPOCHS = 80` is a ceiling. Early stopping (in `Trainer`, on validation AUC, patience configurable) ends most runs earlier. Keep `NUM_EPOCHS` and patience identical across folds and models so the cross-fold time and energy comparison stays clean. The number of epochs actually run is logged in `efficiency.json`.

### 5.6 What each run saves

- `config.json`: hyperparameters, split provenance, environment, params, FLOPs.
- `log.csv`: per-epoch train/val loss and metrics.
- `data_split.csv`: the split used, for provenance.
- `best_auc.pth`, `best_f1.pth`, `best_acc.pth`, `last.pth`.
- `val_predictions.npz`, `test_predictions.npz`: raw `y_true` and `y_prob` (this is what aggregation and threshold tuning read).
- `metrics_val.json`, `metrics_test.json`: accuracy, balanced accuracy, macro/weighted F1, macro and per-class AUC, per-class recall and precision, quadratic weighted kappa.
- `confusion_matrix_*.png`, `roc_curve_*.png`, training curves.
- `efficiency.json`: params, FLOPs per slide, training time and energy, inference latency, throughput, energy per slide, peak GPU memory, epochs run.

Final val and test evaluation runs on the checkpoint selected by `BEST_CHECKPOINT_METRIC` (default `auc`), not the last epoch.

---

## 6. THRESHOLD TUNING (STAGE A)

Stage A tends to rank low_grade well (high AUC) while the default argmax sends many low_grade slides to an adjacent grade (low recall). `tune_threshold.py` recovers low_grade recall after training, without retraining, by re-weighting the class scores before the argmax. The weights are fit on each fold's validation predictions and applied once to that fold's test predictions, so the test estimate stays honest.

```
python -m scripts.stage_a.tune_threshold --objective macro_f1
python -m scripts.stage_a.tune_threshold --objective macro_f1 --pooled
python -m scripts.stage_a.tune_threshold --pooled --objective low_grade_recall --min-precision 0.6
```

`--pooled` fits one shared operating point on all folds' validation predictions combined, which is steadier than per-fold fitting when validation sets are small, and is the version to report. AUC is unchanged by design (thresholding does not alter the ranking). Outputs land in `training_output/aggregate_tuned/`, with default-versus-tuned summaries and the fitted weights per fold. Report the tuned result as a pre-specified sensitivity analysis alongside the argmax result.

---

## 7. PATCH INJECTION AND THE STAGE B COHORT

### 7.1 Risk map by patch injection

`glioma_sparse.interpret.patch_injection.compute_risk_map` measures, for a target slide predicted as class `p`, how much injecting each target tile into a fixed shuffled control shifts the model's output for `p`:

```
risk_map[r, c] = score_injected(class p) - score_baseline(class p)
```

The score is the **class logit** by default (`score="logit"`). On a confident model the softmax saturates, so a single injected tile barely moves the probability and the map collapses to zero; the logit keeps the contrast. The control is shuffled once with a fixed seed (42), so its own tile arrangement carries no positional information. Because Stage A is trained with patch permutation, the model is position-invariant, so R reflects tile content rather than location.

In the manuscript the injection uses a single **fixed control site** `s`, identical for every injection, so the displaced control tile is held constant and R depends only on the injected tile's content. The value R is stored at the target tile's own coordinate `(r,c)`, keeping the map aligned with the slide. The fixed-site and same-position variants agree closely (Section 11), and the equation is written R = z_p(C[s <- T_{r,c}]) - z_p(C), with z_p the pre-softmax logit for class p.

### 7.2 Coordinate mapping

`glioma_sparse.interpret.wsi_mapping` inverts the three preprocessing transforms (resample to target MPP, pad to square, resize to thumbnail) using the sidecar, so a thumbnail tile maps exactly to a WSI level-0 region. Tiles overlapping padding are clipped; tiles entirely in padding return `None`.

### 7.3 p95 region selection

`glioma_sparse.interpret.highres_extraction` selects the strongest-signal tiles by the **p95 rule**: keep tiles at or above the 95th percentile of the risk map, drop tiles that fail a tissue filter, sort by R, and cap at four per slide. Each selected tile is re-read from the WSI at high resolution (default 2048x2048).

### 7.4 Building the Stage B cohort

`build_stageB_cohort.py` produces the full Stage B dataset, per fold and leakage-safe. For each of the five folds, that fold's Stage A model extracts all of that fold's slides (train, val, test). Each region inherits the slide's split role, so a slide's test regions are never selected by a model that trained on it. Control slides are skipped.

```
python -m scripts.stage_b.build_stageB_cohort ^
    --splits-dir splits ^
    --checkpoints-glob "training_output/*_split_0*/best_auc.pth" ^
    --model resnet18 ^
    --control-image data/included/control/<control_id>.jpg ^
    --sidecar-root data/ebrains_thumbnails ^
    --wsi-root path/to/WHO2021_data ^
    --out-dir data/stage_b_cohort
```

Key behaviours:
- **Sidecar resolution.** The split CSVs point at grade-organised thumbnails that do not carry sidecars, so each slide's sidecar and thumbnail are resolved by slide id under `--sidecar-root` (`<subtype>/included/`), and used for tiling and mapping.
- **WSI resolution.** The stored WSI path is tried as-is, then a prefix swap (`--wsi-old-prefix`/`--wsi-new-prefix`), then a recursive search by filename under `--wsi-root`. This survives the slides being moved.
- **Molecular subtype and grade** are derived from the ebrains subtype folder:

  ```
  astro_IDHmt_G2/G3/G4        -> IDH_mt          (grade 2/3/4)
  oligo_IDHmt_1p19qdel_G2/G3  -> IDH_mt_1p19q    (grade 2/3)
  GBM_IDHwt                   -> IDH_wt          (grade 4)
  ```

  Grade 2 maps to grade class **low**, grades 3 and 4 map to **high**.

### 7.5 Cohort layout and manifests

```
data/stage_b_cohort/
  stageB_manifest.csv          all folds combined
  manifests/
    stageB_fold1.csv           one manifest per fold
    ... stageB_fold5.csv
  patches/
    fold_<k>/IDH_mt/           high-resolution patches
    fold_<k>/IDH_mt_1p19q/
    fold_<k>/IDH_wt/
```

Each manifest row carries the slide id, fold, split role, true entity, true subtype folder, true grade (2/3/4), true grade class (low/high), true mutation, the Stage A predicted grade and grade class with a correctness flag, the three predicted probabilities, the region's grid position, risk value, p95 threshold, tissue fraction, base and target MPP, the WSI bounding box, and the patch, thumbnail, and WSI paths.

Check coverage and split integrity before training Stage B:

```
python -m scripts.stage_b.stageB_dataset data/stage_b_cohort/stageB_manifest.csv
```

This reports per-fold train/val/test counts and confirms every slide sits in exactly one split within a fold.

---

## 8. STAGE B TRAINING

Train one fold per run. Stage B reads that fold's manifest, trains on the high-resolution patches with molecular labels, and soft-votes patch probabilities to a slide-level subtype for evaluation.

```
python -m scripts.stage_b.train_stageB --fold 1
python -m scripts.stage_b.train_stageB --fold 2
...
```

Details:
- **Classes**: `IDH_mt`, `IDH_mt_1p19q`, `IDH_wt`.
- **Splits** come from the manifest's `split` column, so all patches of one slide share a split. Patch paths are rebuilt from `--cohort-dir` (default `data/stage_b_cohort`), so moving the cohort does not break training.
- **Augmentation**: `Patch(8)` on the training set by default (`USE_PATCH_SHUFFLE`), matching Stage A. Validation and test use intact patches.
- **Imbalance**: class-weighted loss by default, oversampling optional.
- **Evaluation**: patch probabilities are averaged per slide (soft-vote) to a slide-level prediction, which is the reported result. Both slide-level and patch-level raw outputs are saved.

Each run writes to `training_output_stageB/<timestamp>_resnet18_stageB_fold<k>_cw/`, with `test_predictions.npz` and `val_predictions.npz` at slide level (plus `_patch.npz`), `metrics_<split>.json`, confusion and ROC plots, `config.json`, checkpoints, and `efficiency.json`.

Quadratic weighted kappa appears in the metrics because the shared metric function always computes it, but the molecular subtypes are not ordinal, so for Stage B rely on accuracy, macro-AUC, macro-F1, and the per-class numbers.

---

## 9. END-TO-END INFERENCE

`scripts/inference/inference_end_to_end.py` runs the full pipeline on new material: thumbnail generation, Stage A grading with a risk map, high-resolution region extraction, Stage B molecular subtyping, and an integrated WHO 2021 diagnosis. It has three modes and can be driven either from the command line or from a `CONFIG` block at the top of the file, so it runs straight from a PyCharm Run button while keeping the flags available for others.

### 9.1 What it does per slide

For each slide the script generates a 2048x2048 thumbnail and its JSON sidecar, predicts the grade with Stage A, builds the logit risk map, and selects the strongest-signal regions at the p95 rule (configurable with `--percentile`, for example 90 or 97). Each region is re-extracted from the WSI at high resolution and passed through Stage B, and the patch probabilities are soft-voted to a slide-level subtype. The grade class and subtype combine into an integrated WHO 2021 diagnosis. Each prediction carries a confidence, defined as one minus the normalised entropy of the probability vector, and the integrated diagnosis carries its own confidence, the product of the Stage A and Stage B confidences.

### 9.2 Modes

Single slide or a folder of WSIs (`.ndpi`, `.svs`, `.mrxs`), using one Stage A and one Stage B checkpoint:

```
python -m scripts.inference.inference_end_to_end \
    --input path/to/slide.ndpi \
    --stage-a-ckpt training_output/<fold>/best_auc.pth \
    --stage-b-ckpt training_output_stageB/<fold>/best_auc.pth \
    --wsi-root path/to/WHO2021_data \
    --control-image data/included/control/<control_id>.jpg \
    --run-name e2e_single [--percentile 95] [--stage-b-riskmap]
```

Testset mode runs each fold's held-out test slides through that fold's own Stage A and Stage B models, which keeps the evaluation leakage-free in the same way the cohort build is. It reads the test slides directly from each `split_0X.csv` and pairs checkpoints to folds by the tokens `split_0X` and `foldX` in their paths:

```
python -m scripts.inference.inference_end_to_end --testset \
    --splits-dir splits \
    --stage-a-glob "training_output/*_split_0*/best_auc.pth" \
    --stage-b-glob "training_output_stageB/*_fold*/best_auc.pth" \
    --sidecar-root data/ebrains_thumbnails \
    --wsi-root path/to/WHO2021_data --run-name e2e_testset
```

Relative paths in `CONFIG` are anchored to the repository root, so the script works whether it is launched as a module or as a plain file with any working directory. Command-line flags override `CONFIG`.

### 9.3 Fair control handling (testset)

Controls are not part of the molecular problem and were never seen by Stage B, so they are scored fairly. A control predicted as control stops after Stage A and counts as correct end-to-end, without entering Stage B. A control predicted as tumour counts as a Stage A error, while its Stage B call is excluded from the subtype metrics. Controls are left out of the subtype accuracy entirely.

### 9.4 Metrics

The testset report gives, per fold and aggregated across folds with mean ± std: grade accuracy over all slides, subtype accuracy over the tumour slides, a wildtype-versus-mutant accuracy that collapses IDH_mt and IDH_mt_1p19q into one mutant group (so an astrocytoma called oligodendroglioma still counts as a correct mutant call), and end-to-end accuracy with the fair control rule. A separate 1p/19q codeletion table reports, for the true IDH-mutant slides, the sensitivity for codeleted (oligodendroglioma) and non-codeleted (astrocytoma) cases, plus the codeletion accuracy within the cases actually called mutant.

### 9.5 Outputs

Everything is written under `results/<run-name>/<slide-abbrev>/`, with `fold_<k>/` between them in testset mode. Each slide folder holds a `riskmap/` and a `patches/` subfolder:

```
results/<run-name>/<slide-abbrev>/
  riskmap/
    stageA_riskmap.jpg                     thumbnail with the 8x8 risk overlay and p95 boxes
    stageB_<region>_occlusion.jpg          Stage B patch with the occlusion importance overlay
    stageB_<region>_top<k>_r<r>c<c>_zoom.jpg  cellular-level zoom of the strongest occlusion tiles
  patches/
    <region>.jpg                           the extracted 2048x2048 Stage B patch
```

The optional Stage B risk map (`--stage-b-riskmap`, or `stage_b_riskmap` in `CONFIG`) uses occlusion rather than injection into a control, as described in the manuscript: each tile of a Stage B patch is occluded and the drop in the predicted-class logit is recorded. The strongest tiles are then re-read from the WSI at 2048x2048, a third zoom on top of the thumbnail and the region, so the IDH-indicative areas can be inspected at cellular level. The number of tiles re-zoomed per patch is set by `occlusion_topk`. All generated figures are 300 dpi JPGs sized around 2048 pixels.

The integrated diagnosis report is an `.xlsx` with per-slide predictions, probabilities, and confidences, coloured by uncertainty (green for confident, amber for moderate, red for low). In testset mode it also carries the per-fold summary and the 1p/19q codeletion table as separate sheets.

---

## 10. AGGREGATING RESULTS ACROSS FOLDS

The same aggregation reads Stage A and Stage B runs, since both save slide-level `test_predictions.npz` and `efficiency.json`.

Stage A:

```
python -m scripts.data.aggregate_splits --base-dir training_output --split test
```

Stage B:

```
python -m scripts.data.aggregate_splits --base-dir training_output_stageB --split test
```

It recomputes metrics per fold with the shared metric function, then writes: `aggregate_metrics_per_split.csv` (one row per fold), `aggregate_summary.csv` (mean, std, and a ready-to-paste "mean ± std" per metric), `aggregate_efficiency.csv` (compute cost, mean ± std), and a four-panel `aggregate_figure.png`/`.pdf` (overall metrics with error bars, per-class recall, mean confusion matrix, and a compute-cost table). It also computes performance-per-compute ratios where efficiency data is present.

Report the headline as the five-fold mean ± std, which captures variation from the data partition on top of within-fold sampling noise.

---

## 11. ABLATIONS

- **Injection-site stability** (`scripts/stage_a/injection_site_ablation.py`): inject one important tile into every one of the 64 control sites and measure the spread of R. A small coefficient of variation is evidence that R reflects tile content, not position.
- **Fixed-site injection** (`scripts/stage_a/fixed_site_injection.py`): inject every target tile into one fixed control slot, so the displaced control tile is constant and R reflects tile content alone. Compare its p95 set to the standard same-position map.
- **Injection demo** (`scripts/demo/injection_demo.py`): explainer figure of the mechanism, with a fixed-site default and a `--same-position` flag.
- **Model size and no-shuffle**: rerun Stage A on the same five splits with `--model resnet50`, or with `USE_PATCH_SHUFFLE = False`, for the compute and interpretability-validity arguments. Anything framed as a performance claim uses the full five folds.

---

## 12. OUTPUTS

### Stage A / Stage B run

```
training_output[_stageB]/<experiment_name>/
  config.json
  log.csv
  data_split.csv                (Stage A)
  best_auc.pth  best_f1.pth  best_acc.pth  last.pth
  val_predictions.npz   test_predictions.npz
  (Stage B also: *_predictions_patch.npz)
  metrics_val.json      metrics_test.json
  confusion_matrix_*.png  roc_curve_*.png  training curves
  efficiency.json
```

### Aggregation

```
<base-dir>/aggregate/
  aggregate_metrics_per_split.csv
  aggregate_summary.csv
  aggregate_efficiency.csv
  aggregate_figure.png / .pdf
```

### End-to-end inference

```
results/<run-name>/
  integrated_diagnosis_report.xlsx        (single/folder mode)
  e2e_testset_report.xlsx                 (testset mode: per-slide + summary + 1p/19q sheets)
  <slide-abbrev>/                          (fold_<k>/<slide-abbrev>/ in testset mode)
    riskmap/  stageA_riskmap.jpg, stageB_*_occlusion.jpg, stageB_*_zoom.jpg
    patches/  <region>.jpg
```

### Preprocessing

```
data/.../included/<slide_id>.jpg + <slide_id>.json
data/.../low_tissue/...
metadata.csv
```

---

## 13. NOTES AND NEXT STEPS

Notes:
- Thumbnails are 2048x2048 JPGs (quality 90) at 4.0 µm/pixel, each with a JSON sidecar.
- The risk score R and the p95 selection are computed on the class logit.
- Class ordering is fixed across dataset, training, and evaluation.
- The pipeline is reproducible end to end: the split CSVs, `config.json`, and `efficiency.json` make every run traceable.
- Report five-fold mean ± std, and treat Stage A threshold tuning as a pre-specified sensitivity analysis.

Next steps:
- Choose a single shipped model per stage for deployment (best fold, ensemble, or refit on all data), for the single-slide and folder inference modes.
- External validation cohort (e.g. TCGA) as a generalisation test, added alongside the internal five-fold result.
- Benchmark rerun on the same five splits for a matched comparison, including the compute measurements.
- YAML config system and mixed precision (AMP).
