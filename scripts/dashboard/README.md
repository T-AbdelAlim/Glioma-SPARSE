# Glioma-SPARSE dashboard

The dashboard runs the whole pipeline on a slide. The browser shows the workflow and a local FastAPI backend runs the same code as the command-line pipeline.

## Run

From the repository root, with the trained models in place (`python -m scripts.check_setup`):

```
python -m scripts.dashboard.run_dashboard
```

This opens a native window. To use a browser instead, run `python -m uvicorn scripts.dashboard.backend:app --host 127.0.0.1 --port 8000` and open http://127.0.0.1:8000.

## Use

1. **Import slide.** Pick a whole-slide image and build the thumbnail. Results are saved in a `GS_<slide>` folder next to the slide.
2. **Stage A.** Grades the slide and selects the patch-regions. The percentile slider sets how many regions are selected.
3. **Stage B.** Pick the encoder and the number of 20x tiles (64 by default, or all). It predicts the molecular type.
4. **Run.** Shows each step, up to the integrated diagnosis. The evidence map shows how much each tile contributed.
5. **Export.** Saves an `.xlsx` report and the figures. Earlier results can be loaded again.
6. **Batch.** Runs a folder of slides with the same settings.

The **Methodology explained** button opens a step-by-step explainer of the method.

The dashboard uses the models of cross-validation fold 5. Confidence is one minus the normalized entropy of the predicted probabilities. Use it to rank predictions, not as the chance that a prediction is right.

## Encoders

**DistillPath** ([RamonK/DistillPath-KS16-UNI2h](https://huggingface.co/RamonK/DistillPath-KS16-UNI2h)) is public and downloaded on first use. To use a local copy, set `GLIOMA_SPARSE_DISTILLPATH_CKPT` or put it in `models/DistillPath-KS16-UNI2h/`.

**UNI2-h** (`full` branch) is gated. Request access at [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and download `pytorch_model.bin`. Put it in `models/UNI2-h/`, set `GLIOMA_SPARSE_UNI2H_CKPT`, or locate it in the app.

## Apps

```
python scripts/dashboard/build_exe.py
```

This builds the app for the platform it runs on, in `dist/`. The release apps are built by the `apps` workflow on the `light-release` branch. Errors go to `dashboard.log` next to the app. `--check` tests a build without opening a window.
