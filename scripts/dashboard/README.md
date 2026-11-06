# Glioma-SPARSE dashboard

An interactive interface for the two-stage Glioma-SPARSE pipeline. The browser renders the workflow; a
local FastAPI backend runs the pathology code (OpenSlide, the Stage A ResNet, patch injection, the Stage B
encoder and heads), the same code as the command-line pipeline.

## Run

From the repository root, with the trained models in place (`python -m scripts.check_setup`):

```
python -m scripts.dashboard.run_dashboard
```

This opens the dashboard as a native window. To use a browser instead:
`python -m uvicorn scripts.dashboard.backend:app --host 127.0.0.1 --port 8000` and open http://127.0.0.1:8000.

## Use

1. **Import slide**: paste a whole-slide image path or browse, optionally pick a stain correction, and build
   the 2048 × 2048 thumbnail (4 µm/pixel). Results are saved in a `GS_<slide>` folder next to the slide.
2. **Stage A: grade and patch-regions**: ResNet-50 (default) or ResNet-18 of cross-validation fold 5, with the
   checkpoint of best validation F1, AUC or accuracy, and the signal percentile (default 95, at most four
   patch-regions; lower percentiles select more). The 8 × 8 grid next to the slider shows the cells that leave
   Stage A as patch-regions: while dragging, how many can; after a run, exactly the selected cells on the
   slide's signal map (cells above the percentile with enough tissue).
3. **Stage B: molecular type**: the frozen encoder (DistillPath; UNI2-h in the `full` branch) and the number
   of 20× tiles drawn at random inside the patch-regions: 8, 16, 32, 64 (default), 128, 256, 512, 1024 or
   **all** (every tissue tile in the regions, at most 4,096). Each count uses a head trained with that number
   of tiles; the slider snaps to the nearest count with a trained head. The diagram under the slider shows
   the flow: tiles in the patch-regions, encoder, mean embedding, logistic-regression head. Accuracy
   plateaus from about 32 tiles (manuscript, Supplementary Figure S6).
4. **Display**: signal map, class predictions, raw values (2 decimals) and the **Stage B evidence map** (on
   by default): the sampled tiles coloured by their exact contribution to the molecular prediction (the head
   is linear), with the most influential tile at 20×.
5. **Run**: animates thumbnail → grade → signal map → patch-regions → sampled tiles → encoder → molecular
   type → integrated diagnosis. Changing the percentile afterwards updates the result without a full rerun.
6. **Export & load**: a coloured `.xlsx` report and figures (signal map, regions, Stage B evidence), and
   loading of earlier results. **Open results folder** opens the folder of the saved or loaded result.
7. **Batch processing**: a folder of slides with the same settings; batch runs with DistillPath are tagged
   `DP` in the run folder name.

**Model and integrated diagnosis.** The dashboard uses the models of cross-validation fold 5 (`APP_FOLD` in
`glioma_sparse/models/resuni.py`), which had the highest mean accuracy on its internal test slides and on TCGA. As in
the paper, its grade and molecular-type probabilities give a score for each of the six diagnoses (control;
astrocytoma, IDH-mutant, grade 2 or 3-4; oligodendroglioma, grade 2 or 3; glioblastoma, IDH-wildtype), and the
diagnosis is the class with the highest score (`six_score` and `_integrate` in `backend.py`; with several folds in
`FOLDS`, the scores are averaged, as in the paper's external validation). The grades, and the patch-regions on the TCGA slides
checked, are identical to the paper's fold-5 results; the thumbnail is graded as saved (JPEG), the form in which the models were trained.

Patch injection uses the cropped computation (`method="crop"` in `glioma_sparse.interpret.patch_injection`),
which gives the same signal map as the full computation at about a quarter of the cost.

The header button **Methodology explained** opens a step-by-step explainer (`scripts/dashboard/explainer/`,
served at `/explainer/<file>`). To change it, replace the files there (H.264 MP4 or PNG) and edit the
`STEPS` list in `index.html` (`METHOD` module).

## Stage B encoders

Stage B classifies the molecular type from the mean embedding of the tiles sampled in the patch-regions with the
head of fold 5
of logistic-regression heads of the chosen encoder (`data/resuni_heads/<config>`). Encoder weights are never
bundled.

### DistillPath

DistillPath-KS16-UNI2h (https://huggingface.co/RamonK/DistillPath-KS16-UNI2h, a 22M ViT-S/16 distilled from
UNI2-h) is public under a non-commercial licence. No access request is needed: the app uses a local copy if
there is one, and otherwise downloads it (86 MB) on first use. It checks, in this order:

1. `GLIOMA_SPARSE_DISTILLPATH_CKPT=C:\path\to\model.safetensors`
2. `models\DistillPath-KS16-UNI2h\model.safetensors` next to the packaged exe
3. `<repo>/models/DistillPath-KS16-UNI2h/model.safetensors`
4. the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`)

Heads exist for ResNet-50 and ResNet-18 best-F1 (`rn50_f1_dp_uni2h`, `rn18_f1_dp_uni2h`).

### UNI2-h (`full` branch)

UNI2-h (681M) is gated: request access at https://huggingface.co/MahmoodLab/UNI2-h and download
`pytorch_model.bin`. The app finds it via `GLIOMA_SPARSE_UNI2H_CKPT`, the file located in the app
(**Locate UNI2-h weights…**, remembered in `%APPDATA%\GliomaSPARSE\settings.json`),
`models\UNI2-h\pytorch_model.bin` next to the exe or in the repository, or the Hugging Face cache. Heads exist
for ResNet-50 best-F1 and best-AUC and ResNet-18 best-F1 (`rn50_f1`, `rn50_auc`, `rn18_f1`).

## Apps

```
python scripts/dashboard/build_exe.py
```

builds the app with PyInstaller for the platform it runs on: `dist/GliomaSPARSE-Dashboard-Windows` (Windows,
CPU-only PyTorch), `dist/GliomaSPARSE-Dashboard-Windows-GPU` (Windows, CUDA PyTorch, needs an NVIDIA GPU) or
`dist/GliomaSPARSE-Dashboard-macOS.app` (macOS). The release apps are built by the `apps` workflow
(`.github/workflows/apps.yml`) on the `light-release` branch, from the release's `Glioma-SPARSE-models.zip`. The app carries a model bundle
(staged in `build/exe_bundle`):
the Stage A checkpoints of fold 5, for ResNet-18 and ResNet-50 and each metric with Stage B heads, the control image used for
patch injection, the logo and the Stage B heads of this version's encoders. Started inside a repository with
the trained models, the app uses the repository instead; `GLIOMA_SPARSE_ROOT` overrides both. It opens as a
native window, serves on port 8000 and writes errors to `dashboard.log` next to the app (or in the user's
`GliomaSPARSE` folder). `--check` starts the app without a window, reports the trained models it finds and runs
a small computation on the control image, for testing a build.

## References

The data, models, tools and libraries the dashboard builds on (Digital Brain Tumour Atlas, ResNet, DistillPath,
UNI2-h, OpenSlide, PyTorch, timm, FastAPI, pywebview, PyInstaller and others) are referenced in the
[main README](../../README.md#acknowledgements-and-references).

## Notes

- ResNet-18 versus ResNet-50 as Stage A: grading is equivalent, but the complete pipeline is 1–2 points less
  accurate with ResNet-18 (manuscript, Supplementary Table S3), so ResNet-50 is the default.
- Confidence (Stage A, Stage B and integrated) is one minus the normalised entropy of the predicted
  probabilities; how often predictions at each confidence were correct, internally and on TCGA, is in the
  [main README](../../README.md#dashboard).
- Display-only toggles re-render the last run without recomputing.
- `GET /api/config` reports the encoders, weights and heads found, useful for checking paths.
