# Glioma-SPARSE dashboard

An interactive, animated browser interface for the two-stage Glioma-SPARSE
pipeline. The browser renders the workflow; a local FastAPI backend runs the
real pathology code (OpenSlide, your ResNet checkpoints, patch injection, UNI2-h),
mirroring `inference_report.py` / `inference_end_to_end.py` exactly.

## Layout

Place both files inside your repo so `import glioma_sparse...` resolves:

```
Glioma-SPARSE/
  scripts/
    dashboard/
      backend.py      <- this
      index.html      <- this
```

## Install

```
pip install fastapi uvicorn pillow numpy torch torchvision openslide-python openpyxl matplotlib timm
```

(OpenSlide also needs its native library on Windows: install the OpenSlide
binaries and put the `bin` folder on PATH, or use `openslide-bin`.)

## Run

From the repo root:

```
python -m uvicorn scripts.dashboard.backend:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Use

1. **Import**: paste a full WSI path (e.g. `C:\...\slide.ndpi`) and click Import.
   A `GS_analysis/` folder is created next to the slide, containing the
   thumbnail + `.json` sidecar, and `figures/` and `report/` subfolders.
2. **Models**: Stage A is a ResNet-50 (default) or ResNet-18, best-F1 or best-AUC checkpoint.
   Stage B is always the frozen UNI2-h foundation model with a logistic-regression head
   (the earlier ResNet Stage B options were removed).
3. **Stage B tiles**: number of 20× tiles drawn at random from the selected regions and
   encoded by UNI2-h: 8, 16, 32, 64 (default), 128, 256, 512 or 1024. Each count uses a head
   trained with that number of tiles; counts without a trained head are greyed out
   (ResNet-50 and ResNet-18 best-F1 have all counts, ResNet-50 best-AUC only 64).
   Accuracy plateaus from about 32 tiles (manuscript, Figure 7).
4. **Parameters**: signal percentile (default 95), region cap (default 4 of the 64 grid
   cells), and toggles for the signal map, class predictions, raw values (2 decimals), and
   the **Stage B evidence map**: the tiles that made the prediction, coloured by their exact
   contribution to the molecular prediction (the head is linear), with the most influential
   tile at 20×. No extra encoding is needed, so it is instant.
5. **Run full pipeline**: animates thumbnail → grade → signal map → selected regions →
   64 sampled tiles → UNI2-h → molecular prediction → integrated diagnosis.
6. **Save results**: writes a colored `.xlsx` report to `GS_analysis/report/` and JPGs
   (signal map, regions, Stage B tile evidence and top tile) to `GS_analysis/figures/`.

## Methodology explained

The header button opens a step-by-step explainer built from the animations and figures of
the GL-SPARSE talk (`scripts/dashboard/explainer/`, served at `/explainer/<file>`): slide
size, the standard foundation-model pipeline, the decomposition by complexity, naive
injection, training on shuffled mosaics, patch injection, Stage B with UNI2-h, and Stage B tile
evidence. It presents the method only (results are in the paper). Play all, Prev/Next, the
dots or the arrow keys navigate.
To update it, replace the files in `explainer/` (H.264 MP4 or PNG) and edit the `STEPS` list
in `index.html` (`METHOD` module).

## Stage B: UNI2-h foundation model

Stage B classifies the molecular type from the averaged UNI2-h embeddings of the tiles
sampled inside Stage A's selected regions (5-fold-ensembled logistic head). Same settings in
both the interactive run and batch processing.

It needs a local UNI2-h checkpoint, which is **not bundled** with this repo or the packaged
exe: UNI2-h is a gated model (request access at
https://huggingface.co/MahmoodLab/UNI2-h, then download `pytorch_model.bin`). The app finds
it automatically, checking in this order:

1. `GLIOMA_SPARSE_UNI2H_CKPT=C:\path\to\UNI2-h\pytorch_model.bin`
2. the file located earlier in the app (**Locate UNI2-h weights…** in the Stage B panel, shown
   when no weights are found; the file is checked and remembered in
   `%APPDATA%\GliomaSPARSE\settings.json`)
3. `models\UNI2-h\pytorch_model.bin` next to the packaged exe
4. `<repo>/models/UNI2-h/pytorch_model.bin`, or `models/UNI2-h` in a sibling `mahmoodlab` checkout
5. the Hugging Face cache (`~/.cache/huggingface/hub/models--MahmoodLab--UNI2-h`, or `HF_HUB_CACHE` / `HF_HOME`)

Without it, `GET /api/config` reports `resuni_available: false` and Stage B cannot run (the
rest of the app is unaffected). The trained per-fold heads (`data/resuni_heads/<config>/lr_n<tiles>_fold<k>.npz`)
exist for ResNet18/F1, ResNet50/F1 and ResNet50/AUC; Stage B is unavailable for
ResNet18/AUC. Needs `timm>=1.0` in addition to the packages above.

## Standalone exe

```
.venv\Scripts\python.exe scripts\dashboard\build_exe.py
```

builds `dist\GliomaSPARSE-Dashboard\GliomaSPARSE-Dashboard.exe` (PyInstaller, one folder,
about 3.5 GB with torch/CUDA). The folder can be copied to another Windows machine and run
there: besides `index.html` and the explainer media it carries a model bundle (about
420 MB, staged in `build\exe_bundle`): the Stage A checkpoint of the best fold per metric for
ResNet-18 and ResNet-50 with its validation scores, the control image used for patch
injection, the logo and the Stage B heads. Started inside a repo that has the trained models
(e.g. from `dist\`), the exe uses the repo instead; `GLIOMA_SPARSE_ROOT` overrides both.
The UNI2-h weights are never bundled; the recipient needs their own access and the app finds
or asks for the file as described above. It opens as a native window and serves on port 8000,
so close any dev server on that port first. Errors go to `dashboard.log` next to the exe.

## Notes

- All probabilities and uncertainties are shown to 2 decimals.
- ResNet-18 vs ResNet-50 as Stage A: grading is equivalent, but the complete pipeline is
  1–2 points less accurate with ResNet-18 (manuscript, Supplementary Table S3), so
  ResNet-50 is the default.
- Display-only toggles (signal / predictions / raw values) re-render the last
  run instantly without recomputing.
- Checkpoint paths, control image, and logo path are all set at the top of
  `backend.py`; edit there to point at other folds or models.
- `GET /api/config` reports which checkpoints were found on disk, useful for
  debugging path issues.
