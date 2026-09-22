# Glioma-SPARSE dashboard

An interactive, animated browser interface for the two-stage Glioma-SPARSE
pipeline. The browser renders the workflow; a local FastAPI backend runs the
real pathology code (OpenSlide, your ResNet checkpoints, patch injection),
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
pip install fastapi uvicorn pillow numpy torch torchvision openslide-python openpyxl matplotlib
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
2. **Models**: switch Stage A and Stage B between RN18 and RN50 independently.
   The fold is locked to `split_02` for this demo (the four checkpoint paths are
   in `backend.py`, `CHECKPOINTS`).
3. **Parameters**: risk percentile (default 95), tile cap (default 4), and
   toggles for the signal map, class predictions, raw values (p and
   uncertainty, 2 decimals), and the Stage B occlusion heatmap (off by default).
4. **Run full pipeline**: animates thumbnail → grade → signal map → selected
   tiles → connector lines to the extracted high-res patches → into Stage B →
   molecular prediction → integrated diagnosis.
5. **Save results**: writes a colored `.xlsx` report to `GS_analysis/report/`
   and 2048×2048 JPGs (risk map, patches, occlusion overlays) to
   `GS_analysis/figures/`.

## Notes

- All probabilities and uncertainties are shown to 2 decimals.
- The occlusion heatmap runs 64 forward passes per patch, so it is off by
  default; turn it on before running if you want Stage B importance maps.
- Display-only toggles (signal / predictions / raw values) re-render the last
  run instantly without recomputing.
- Checkpoint paths, control image, and logo path are all set at the top of
  `backend.py`; edit there to point at other folds or models.
- `GET /api/config` reports which checkpoints were found on disk, useful for
  debugging path issues.
