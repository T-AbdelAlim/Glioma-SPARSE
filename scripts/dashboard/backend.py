r"""
Glioma-SPARSE dashboard backend.

A local FastAPI server that exposes the real two-stage pipeline to the browser
frontend. Every pathology operation mirrors inference_report.py /
inference_end_to_end.py exactly, importing the same glioma_sparse modules, so
the dashboard is guaranteed consistent with the working pipeline.

Run from the repo root (so `import glioma_sparse...` resolves):

    pip install fastapi uvicorn pillow numpy torch torchvision openslide-python
    python -m uvicorn scripts.dashboard.backend:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000  (index.html is served from the same folder).

Pipeline stages, each its own endpoint so the frontend can animate progress:

    POST /api/import      slide_path -> make GS_analysis/, thumbnail + sidecar
    POST /api/stage_a     -> grade softmax + confidence
    POST /api/risk_map    percentile -> risk grid + selected tiles (+ wsi bbox)
    POST /api/extract     tile -> level-0 patch (base64) + coordinate transform
    POST /api/stage_b     -> subtype softmax per patch, soft-voted slide call
    POST /api/stage_b_occlusion  patch -> occlusion importance grid (optional)
    POST /api/integrate   -> WHO 2021 integrated diagnosis
    POST /api/save        -> colored .xlsx report + 2048x2048 jpimages

State is kept per session_id in memory; the frontend passes it back each call.
"""

from pathlib import Path
import base64
import io
import json
import os
import re
import sys
import time
import uuid

import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ---- real pipeline imports (same as inference_report.py) ----
from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto as create_wsi_thumbnail
from glioma_sparse.preprocessing.stain_normalization import normalize_with_profile, list_profiles
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.interpret.patch_injection import (
    compute_risk_map, tile_image, reconstruct_image,
)
from glioma_sparse.interpret.wsi_mapping import (
    load_mapping, thumbnail_bbox_to_wsi, grid_cell_bbox,
)
import openslide


# ============================================================
# CONFIG
# ============================================================

# index.html sits next to this file in dev; a frozen exe (see run_dashboard.py /
# build_exe.py) bundles it next to the executable instead, since __file__
# points into PyInstaller's extracted bundle rather than the real repo tree.
HERE = (Path(sys._MEIPASS) if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
        else Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent)

# repo root: two levels up from this file, or from dist/GliomaSPARSE-Dashboard/
# for the frozen exe. GLIOMA_SPARSE_ROOT overrides both (e.g. exe moved elsewhere).
REPO_ROOT = Path(os.environ.get("GLIOMA_SPARSE_ROOT") or (
    Path(sys.executable).resolve().parents[2] if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[2]))

# checkpoints are resolved dynamically (see resolve_best_checkpoint): for a
# given arch+metric, every fold under the relevant training_output* dir is
# compared on that metric's val score and the winner's best_<metric>.pth is
# loaded. Stage B pools both stageB dirs (best_auc-derived and best_f1-derived
# Stage-A patch cohorts) and picks the single best fold across both.
ARCH_OUTPUT_DIRS = {
    "resnet18": REPO_ROOT / "ResNet18_output",
    "resnet50": REPO_ROOT / "ResNet50_output",
}
STAGE_A_SUBDIR = "training_output"
STAGE_B_SUBDIRS = ["training_output_stageB", "training_output_stageB_F1"]
METRIC_COLUMN = {"f1": "macro_f1", "auc": "macro_auc", "acc": "accuracy"}
DEFAULT_METRIC = "f1"
CONTROL_IMAGE = str(REPO_ROOT / "data" / "included" / "control" / "86242943-7775-11eb-827d-001a7dda7111.jpg")

# preset example slide for the "Methodology explained" walkthrough: a real
# oligodendroglioma, run through the actual pipeline (RN50/RN50, p95, occlusion
# on). Only read when methodology_cache/ is missing; GLIOMA_SPARSE_METHOD_SLIDE
# points it at the slide on another machine.
METHODOLOGY_TARGET = os.environ.get(
    "GLIOMA_SPARSE_METHOD_SLIDE",
    r"C:\Users\TAbde\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data\oligo_IDHmt_1p19qdel_G3\oligo_IDHmt_1p19qdel_G3\a198054a-357f-11eb-bac3-001a7dda7111.ndpi")
LOGO_PATH = str(REPO_ROOT / "docs" / "logo.png")

GRADE_CLASSES = ["control", "low_grade", "high_grade"]
SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
GRID = (8, 8)
SEED = 42
TARGET_MPP = 4.0
THUMB_SIZE = 2048
OUTPUT_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# each slide runs in its own subprocess so a malformed .mrxs can't hang/crash the server
SLIDE_TIMEOUT_SECONDS = 600

DEFAULT_STAIN_PROFILE = None

INTEGRATED = {
    ("low", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 2",
    ("high", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 3-4",
    ("low", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 2",
    ("high", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 3",
    ("low", "IDH_wt"): "Glioblastoma, IDH-wildtype, grade 4",   # fix 1a: wt -> GBM
    ("high", "IDH_wt"): "Glioblastoma, IDH-wildtype, grade 4",
}


# ============================================================
# MODEL CACHE
# ============================================================

_transform = build_eval_transform()
_control_img = None
_models = {}          # (stage, arch, metric) -> model
_best_ckpt_cache = {}  # (stage, arch, metric) -> (ckpt_path, fold_name, val_value)


def control_img():
    global _control_img
    if _control_img is None:
        _control_img = Image.open(CONTROL_IMAGE).convert("RGB")
    return _control_img


def _fold_dirs(base_dir):
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and not p.name.startswith("aggregate"))


def resolve_best_checkpoint(stage, arch, metric):
    """Pick the fold with the highest val score for `metric`, across every
    fold dir under the relevant training_output* tree(s), and return its
    best_<metric>.pth. Stage B pools STAGE_B_SUBDIRS into one comparison."""
    key = (stage, arch, metric)
    if key in _best_ckpt_cache:
        return _best_ckpt_cache[key]
    if metric not in METRIC_COLUMN:
        raise HTTPException(400, f"unknown metric {metric!r}, expected one of {list(METRIC_COLUMN)}")
    if arch not in ARCH_OUTPUT_DIRS:
        raise HTTPException(400, f"unknown arch {arch!r}")

    root = ARCH_OUTPUT_DIRS[arch]
    base_dirs = [root / STAGE_A_SUBDIR] if stage == "stage_a" else [root / d for d in STAGE_B_SUBDIRS]
    col = METRIC_COLUMN[metric]

    best = None
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for fold_dir in _fold_dirs(base_dir):
            metrics_file = fold_dir / "metrics_val.json"
            ckpt = fold_dir / f"best_{metric}.pth"
            if not metrics_file.exists() or not ckpt.exists():
                continue
            value = json.loads(metrics_file.read_text()).get(col)
            if value is None:
                continue
            if best is None or value > best[0]:
                best = (value, fold_dir.name, ckpt)

    if best is None:
        raise HTTPException(400, f"no {metric} checkpoint found for {stage}/{arch} under "
                                 f"{', '.join(str(d) for d in base_dirs)}")
    value, fold_name, ckpt = best
    result = (str(ckpt), fold_name, round(float(value), 4))
    _best_ckpt_cache[key] = result
    return result


def get_model(stage, arch, metric=DEFAULT_METRIC):
    key = (stage, arch, metric)
    if key not in _models:
        ckpt, _, _ = resolve_best_checkpoint(stage, arch, metric)
        n = len(GRADE_CLASSES) if stage == "stage_a" else len(SUBTYPE_CLASSES)
        # pretrained=False: about to load our own checkpoint anyway, so skip the
        # otherwise-wasted ImageNet-weights download (also lets this run fully offline)
        m = build_model(arch, num_classes=n, pretrained=False)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.to(DEVICE).eval()
        _models[key] = m
    return _models[key]


def forward_probs(model, img):
    x = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, logits.cpu().numpy()[0]


def entropy_conf(probs):
    p = np.asarray(probs, float)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    return round(1.0 - ent / np.log(len(p)), 2)


def b64(img, fmt="JPEG", q=88):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def grade_class_short(name):
    return {"low_grade": "low", "high_grade": "high"}.get(name, "control")


def normalize_stage_b_patch(patch, stain_profile):
    """Re-stain a Stage-B high-res patch against a named profile's Stage-B
    reference. stain_profile=None (or falsy) is a no-op passthrough."""
    if not stain_profile:
        return patch
    return Image.fromarray(normalize_with_profile(np.array(patch), stain_profile, stage="B"))


# ============================================================
# OCCLUSION (mirrors inference_end_to_end.occlusion_map, but batched --
# on a CPU-only laptop, 64 individual forward passes is the single slowest
# step in the app; one batched pass is much faster for the same result)
# ============================================================

def occlusion_map(model, patch, pred_idx, grid=GRID, batch_size=16):
    rows, cols = grid
    _, base_logits = forward_probs(model, patch)
    base = float(base_logits[pred_idx])
    tiles = tile_image(patch, grid)
    mean_col = tuple(int(v) for v in np.array(patch).reshape(-1, 3).mean(0))

    occ_imgs = []
    for r in range(rows):
        for c in range(cols):
            occ = list(tiles)
            occ[r * cols + c] = Image.new("RGB", occ[r * cols + c].size, mean_col)
            occ_imgs.append(reconstruct_image(occ, grid, patch.size))

    logits = []
    with torch.no_grad():
        for start in range(0, len(occ_imgs), batch_size):
            batch = occ_imgs[start:start + batch_size]
            x = torch.stack([_transform(im) for im in batch]).to(DEVICE)
            logits.append(model(x).cpu().numpy())
    logits = np.concatenate(logits, axis=0)

    imp = (base - logits[:, pred_idx]).reshape(rows, cols).astype(np.float32)
    return imp


def subtile_wsi_bbox(patch_wsi_bbox, sr, sc, grid=8):
    """WSI level-0 bbox of occlusion sub-tile (sr,sc). Mirrors e2e script."""
    wx, wy, ww, wh = patch_wsi_bbox
    sub_ww, sub_wh = ww // grid, wh // grid
    return (wx + sc * sub_ww, wy + sr * sub_wh, sub_ww, sub_wh)


def extract_subtile(slide, patch_wsi_bbox, sr, sc, out_size, grid=8):
    """Re-read level 0 at the sub-tile's true coordinates: a real high-res zoom,
    not an upscaled crop of the 2048 patch. Mirrors the e2e script."""
    sx, sy, sw, sh = subtile_wsi_bbox(patch_wsi_bbox, sr, sc, grid)
    if sw <= 0 or sh <= 0:
        return None
    tile = slide.read_region((sx, sy), 0, (sw, sh)).convert("RGB")
    return tile.resize((out_size, out_size), Image.BICUBIC)


# ============================================================
# SESSION STATE
# ============================================================

SESSIONS = {}   # session_id -> dict (insertion order = recency, oldest evicted first)
MAX_SESSIONS = 8   # each holds full-res thumbnails/patches in memory; bound it for long sittings


def sess(sid):
    if sid not in SESSIONS:
        raise HTTPException(400, "unknown session; import a slide first")
    return SESSIONS[sid]


def new_session(sid, data):
    SESSIONS[sid] = data
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))


# ============================================================
# API MODELS
# ============================================================

class ImportReq(BaseModel):
    slide_path: str
    stain_profile: str | None = None

class StageAReq(BaseModel):
    session_id: str
    arch: str = "resnet50"
    metric: str = DEFAULT_METRIC

class RiskReq(BaseModel):
    session_id: str
    percentile: float = 95.0
    cap: int = 0            # 0 = use all tiles above the percentile (Option A)
    min_tissue: float = 0.25

class ExtractReq(BaseModel):
    session_id: str
    row: int
    col: int

class StageBReq(BaseModel):
    session_id: str
    arch: str = "resnet50"
    metric: str = DEFAULT_METRIC

class OcclReq(BaseModel):
    session_id: str
    row: int
    col: int
    arch: str = "resnet50"
    metric: str = DEFAULT_METRIC

class SaveReq(BaseModel):
    session_id: str


# ============================================================
# APP
# ============================================================

app = FastAPI(title="Glioma-SPARSE dashboard")


@app.get("/", response_class=HTMLResponse)
def index():
    idx = HERE / "index.html"
    if idx.exists():
        return idx.read_text(encoding="utf-8")
    return "<h1>index.html not found next to backend.py</h1>"


@app.get("/logo.png")
def logo():
    if Path(LOGO_PATH).exists():
        return FileResponse(LOGO_PATH)
    raise HTTPException(404, "logo not found")


def _checkpoint_available(stage, arch, metric):
    try:
        resolve_best_checkpoint(stage, arch, metric)
        return True
    except HTTPException:
        return False


@app.get("/api/config")
def config():
    return {
        "device": DEVICE,
        "grade_classes": GRADE_CLASSES,
        "subtype_classes": SUBTYPE_CLASSES,
        "grid": GRID,
        "archs": ["resnet18", "resnet50"],
        "metrics": list(METRIC_COLUMN),
        "default_metric": DEFAULT_METRIC,
        "checkpoints_present": {
            f"{s}:{a}:{m}": _checkpoint_available(s, a, m)
            for s in ("stage_a", "stage_b") for a in ARCH_OUTPUT_DIRS for m in METRIC_COLUMN
        },
    }


# ---- native "browse for WSI" dialog (runs on the server = this machine) ----
@app.post("/api/browse")
def api_browse():
    """Open a native OS file picker and return the chosen WSI path.

    The browser cannot hand a real filesystem path to the backend, so we open a
    Tk dialog on the machine running uvicorn. Works when the server runs locally
    with a desktop session (the intended setup). If Tk is unavailable, the user
    can still paste a path manually.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a whole-slide image",
            filetypes=[("Whole-slide images",
                        "*.ndpi *.svs *.mrxs *.tif *.tiff"),
                       ("All files", "*.*")])
        root.destroy()
        return {"path": path or ""}
    except Exception as e:
        raise HTTPException(400,
            f"native file dialog unavailable ({e}); paste the path manually")


# ---- 1. IMPORT: build thumbnail + sidecar in a temp dir (no GS folder yet) ----
def _abbrev(stem):
    """Short slide id for the analysis folder name: GS_<abbrev>."""
    head = stem.split("-")[0]
    return head if len(head) >= 6 else stem[:12]


def _safe_slide_folder_name(stem):
    """Short, collision-safe folder name for a slide's results, e.g.
    GS_TCGA-HT-7611_a3f9c1d2. Using the full slide stem (as a first attempt at
    avoiding TCGA collisions) made paths exceed Windows' 260-char MAX_PATH once
    nested under a deep input tree, since TCGA stems already embed a long GUID.
    A short abbreviation plus a hash of the FULL stem keeps the folder name
    short while still being unique per slide (the hash, not the abbreviation,
    guarantees no two different slides collide)."""
    import hashlib
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"GS_{_abbrev(stem)}_{h}"[:64]


def _arch_num(name):
    """'resnet18' -> '18', 'resnet50' -> '50'. Falls back to the raw name,
    uppercased, if it isn't in the resnetNN form (so an unrecognised arch
    doesn't crash folder naming, just produces a less tidy tag)."""
    m = re.match(r"resnet(\d+)$", name.strip().lower())
    return m.group(1) if m else name.strip().upper()


def _arch_tag(archA, archB):
    """e.g. archA='resnet18', archB='resnet50' -> 'RN1850'."""
    return f"RN{_arch_num(archA)}{_arch_num(archB)}"


def _metric_tag(metricA, metricB):
    """e.g. metricA='f1', metricB='auc' -> 'f1auc'."""
    return f"{metricA}{metricB}"


def _tissue_in_thumb(mapping):
    """Rectangle (in thumbnail pixels) where real tissue sits, excluding the
    padding that create_wsi_thumbnail added to square the image. Lets the WSI
    mini-map crop the padding so it aligns with true level-0 coordinates."""
    S = float(mapping.canvas_size)
    T = float(mapping.thumbnail_size)
    Wt, Ht = mapping.tissue_image_dim
    off_x, off_y = mapping.tissue_offset_in_canvas
    scale = T / S   # canvas -> thumbnail
    return {"x": round(off_x * scale, 1), "y": round(off_y * scale, 1),
            "w": round(Wt * scale, 1), "h": round(Ht * scale, 1),
            "thumb": int(T)}


def _wsi_crop_extent(mapping):
    """The level-0 pixel size of the region that was actually read (the crop,
    for .mrxs; the full slide, for anything else where wsi_crop_origin is
    [0,0]). The dashboard's WSI mini-map displays THIS region (stretched to
    fill the mini-map box, via _tissue_in_thumb's clip), not the full slide,
    so marker positions must be normalized against this extent and the crop
    origin, not against the full wsi_level0_dim - otherwise a marker's
    position is computed correctly in absolute WSI coordinates but placed
    wrongly on a mini-map that only shows the cropped sub-region. Returns
    None if there's no WSI mapping (e.g. missing mpp), matching
    has_wsi_mapping()'s convention elsewhere."""
    if not mapping.has_wsi_mapping():
        return None
    r = mapping.target_mpp / mapping.base_mpp
    Wt, Ht = mapping.tissue_image_dim
    return [round(Wt * r), round(Ht * r)]


@app.post("/api/stain_profiles")
def api_stain_profiles():
    """List available stain-normalization profiles for the frontend's picker."""
    return {"profiles": [{"id": None, "label": "None (no correction)",
                          "description": "Use for EBRAINS training/test data, or any "
                                        "dataset with no fitted profile yet."}]
                        + list_profiles()}


@app.post("/api/import")
def api_import(req: ImportReq):
    import tempfile
    slide_path = Path(req.slide_path)
    if not slide_path.exists():
        raise HTTPException(400, f"slide not found: {slide_path}")

    # thumbnail + sidecar go to a temp dir; the GS_<abbrev> folder is only
    # created on save, so importing never writes next to the slide.
    tmp_dir = Path(tempfile.mkdtemp(prefix="gs_dash_"))
    thumb_path = tmp_dir / f"{slide_path.stem}.jpg"
    t0 = time.time()
    thumbnail, tissue_frac, eff_frac = create_wsi_thumbnail(
        slide_path, output_path=str(thumb_path),
        target_mpp=TARGET_MPP, output_size=THUMB_SIZE,
        stain_profile=req.stain_profile)
    mapping = load_mapping(thumb_path)

    sid = uuid.uuid4().hex[:12]

    # planned (not yet created) analysis folder alongside the slide.
    # Naming order matches batch: descriptor first, unique ID last
    # (GS_<slide-abbrev>_<uniqueID>), so repeated saves of the same slide never
    # collide with or overwrite an earlier save.
    gs_dir = slide_path.parent / f"GS_{_abbrev(slide_path.stem)}_{sid[:8]}"

    new_session(sid, {
        "slide_path": str(slide_path),
        "slide_id": slide_path.stem,
        "gs_dir": str(gs_dir),          # created on save, not now
        "tmp_dir": str(tmp_dir),
        "thumb_path": str(thumb_path),
        "thumbnail": thumbnail,
        "mapping": mapping,
        "tissue_fraction": round(float(tissue_frac), 4),
        "patches": {},
        "results": {},
        "stain_profile": req.stain_profile,
    })
    return {
        "session_id": sid,
        "slide_id": slide_path.stem,
        "gs_dir": str(gs_dir),
        "gs_dir_created": False,
        "thumbnail": b64(thumbnail),
        "thumbnail_size": THUMB_SIZE,
        "thumbnail_w": thumbnail.size[0],
        "thumbnail_h": thumbnail.size[1],
        "wsi_level0_dim": mapping.wsi_level0_dim,
        "wsi_crop_origin": mapping.wsi_crop_origin,
        "wsi_crop_extent": _wsi_crop_extent(mapping),
        "tissue_fraction": round(float(tissue_frac), 4),
        "has_wsi_mapping": mapping.has_wsi_mapping(),
        "tissue_in_thumb": _tissue_in_thumb(mapping),
        "elapsed_sec": round(time.time() - t0, 2),
        "stain_profile": req.stain_profile,
    }


# ---- 2. STAGE A: grade ----
@app.post("/api/stage_a")
def api_stage_a(req: StageAReq):
    s = sess(req.session_id)
    model = get_model("stage_a", req.arch, req.metric)
    _, fold_name, val_value = resolve_best_checkpoint("stage_a", req.arch, req.metric)
    probs, logits = forward_probs(model, s["thumbnail"])
    pred_idx = int(np.argmax(probs))
    s["stage_a"] = {"arch": req.arch, "metric": req.metric, "probs": probs, "pred_idx": pred_idx}
    s["results"]["stage_a"] = {
        "arch": req.arch,
        "metric": req.metric,
        "checkpoint_fold": fold_name,
        "grade_pred": GRADE_CLASSES[pred_idx],
        "prob_control": round(float(probs[0]), 2),
        "prob_low_grade": round(float(probs[1]), 2),
        "prob_high_grade": round(float(probs[2]), 2),
        "confidence": entropy_conf(probs),
    }
    return {
        "grade_pred": GRADE_CLASSES[pred_idx],
        "grade_class": grade_class_short(GRADE_CLASSES[pred_idx]),
        "probs": {GRADE_CLASSES[i]: round(float(probs[i]), 2) for i in range(3)},
        "confidence": entropy_conf(probs),
        "arch": req.arch,
        "metric": req.metric,
        "checkpoint_fold": fold_name,
        "val_score": val_value,
    }


# ---- 3. RISK MAP + tile selection ----
def _tissue_fraction(thumbnail, row, col, grid):
    rows, cols = grid
    w, h = thumbnail.size
    tw, th = w // cols, h // rows
    tile = thumbnail.crop((col * tw, row * th, col * tw + tw, row * th + th))
    arr = np.array(tile).astype(np.float32) / 255.0
    maxc, minc = arr.max(2), arr.min(2)
    sat = (maxc - minc) / (maxc + 1e-8)
    mask = (maxc < 0.9) & (sat > 0.05)
    return float(mask.sum()) / float(mask.size)


@app.post("/api/risk_map")
def api_risk_map(req: RiskReq):
    s = sess(req.session_id)
    if "stage_a" not in s:
        raise HTTPException(400, "run stage_a first")
    arch = s["stage_a"]["arch"]
    metric = s["stage_a"]["metric"]
    # reuse the cached risk map if the Stage A model has not changed; only the
    # percentile threshold is re-applied (fast). Recompute only on model change.
    if (s.get("risk_map") is not None and s.get("risk_map_arch") == arch
            and s.get("risk_map_metric") == metric):
        risk_map = s["risk_map"]
    else:
        model = get_model("stage_a", arch, metric)
        risk_map, target_class = compute_risk_map(
            target_img=s["thumbnail"], control_img=control_img(), model=model,
            transform=_transform, grid=GRID, target_class=s["stage_a"]["pred_idx"],
            device=DEVICE, control_shuffle_seed=SEED, score="logit")
        s["risk_map"] = risk_map
        s["risk_map_arch"] = arch
        s["risk_map_metric"] = metric

    # selection: cells >= percentile that pass tissue filter, top-cap by risk
    thresh = float(np.percentile(risk_map.flatten(), req.percentile))
    cand = []
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            if risk_map[r, c] >= thresh:
                tf = _tissue_fraction(s["thumbnail"], r, c, GRID)
                if tf >= req.min_tissue:
                    cand.append({"row": r, "col": c,
                                 "risk": round(float(risk_map[r, c]), 2),
                                 "tissue": round(tf, 2)})
    cand.sort(key=lambda d: -d["risk"])
    selected = cand if req.cap <= 0 else cand[:req.cap]
    s["selected"] = selected
    s["percentile"] = req.percentile

    # normalised grid for heatmap rendering
    rm = risk_map
    norm = (rm - rm.min()) / (np.ptp(rm) + 1e-8)
    return {
        "risk_grid": [[round(float(v), 3) for v in row] for row in norm],
        "raw_grid": [[round(float(v), 2) for v in row] for row in rm],
        "threshold": round(thresh, 2),
        "percentile": req.percentile,
        "selected": selected,
        "n_candidates": len(cand),
        "n_selected": len(selected),
        "cap": req.cap,           # 0 = all above percentile
        "grid": list(GRID),
        "grid_rows": GRID[0],
        "grid_cols": GRID[1],
    }


# ---- 4. EXTRACT high-res patch + coordinate transform ----
@app.post("/api/extract")
def api_extract(req: ExtractReq):
    s = sess(req.session_id)
    mapping = s["mapping"]
    # reuse an already-extracted patch (expensive WSI read) if we have it
    if (req.row, req.col) in s["patches"]:
        patch = s["patches"][(req.row, req.col)]
        wx, wy, ww, wh = s.get("patch_bbox", {}).get((req.row, req.col),
                                                     (0, 0, patch.size[0], patch.size[1]))
        return {
            "patch": b64(patch),
            "bbox_thumb": list(grid_cell_bbox(req.row, req.col, GRID[0], GRID[1],
                                              mapping.thumbnail_size)),
            "wsi_bbox": [wx, wy, ww, wh],
            "wsi_level0_dim": mapping.wsi_level0_dim,
            "wsi_crop_origin": mapping.wsi_crop_origin,
            "wsi_crop_extent": _wsi_crop_extent(mapping),
            "resample_ratio": round(mapping.target_mpp / mapping.base_mpp, 2)
                if mapping.has_wsi_mapping() else None,
            "output_size": OUTPUT_SIZE,
            "cached": True,
        }
    bbox_thumb = grid_cell_bbox(req.row, req.col, GRID[0], GRID[1],
                                mapping.thumbnail_size)
    wsi_bbox = thumbnail_bbox_to_wsi(mapping, bbox_thumb)
    if wsi_bbox is None:
        raise HTTPException(400, "tile maps to padding (no tissue)")
    wx, wy, ww, wh = wsi_bbox
    slide = openslide.OpenSlide(mapping.wsi_path)
    try:
        patch = slide.read_region((wx, wy), 0, (ww, wh)).convert("RGB")
    finally:
        slide.close()
    if patch.size != (OUTPUT_SIZE, OUTPUT_SIZE):
        patch = patch.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
    if Path(mapping.wsi_path).suffix.lower() == ".mrxs":
        from glioma_sparse.preprocessing.mrxs_tissue_crop import mask_black_artifacts
        patch, artifact_frac = mask_black_artifacts(patch)
        if artifact_frac > 0:
            print(f"[api_extract] r{req.row}c{req.col}: cleaned "
                  f"{artifact_frac*100:.1f}% black scan-artifact pixels")
    patch = normalize_stage_b_patch(patch, s.get("stain_profile"))
    s["patches"][(req.row, req.col)] = patch
    s.setdefault("patch_bbox", {})[(req.row, req.col)] = (wx, wy, ww, wh)
    return {
        "patch": b64(patch),
        "bbox_thumb": list(bbox_thumb),
        "wsi_bbox": [wx, wy, ww, wh],
        "wsi_level0_dim": mapping.wsi_level0_dim,
        "wsi_crop_origin": mapping.wsi_crop_origin,
        "wsi_crop_extent": _wsi_crop_extent(mapping),
        "resample_ratio": round(mapping.target_mpp / mapping.base_mpp, 2)
            if mapping.has_wsi_mapping() else None,
        "output_size": OUTPUT_SIZE,
    }


# ---- 5. STAGE B: subtype per patch + soft vote ----
@app.post("/api/stage_b")
def api_stage_b(req: StageBReq):
    s = sess(req.session_id)
    if not s["patches"]:
        raise HTTPException(400, "extract at least one patch first")
    model = get_model("stage_b", req.arch, req.metric)
    _, fold_name, val_value = resolve_best_checkpoint("stage_b", req.arch, req.metric)
    per_patch = []
    probs_stack = []
    for (r, c), patch in s["patches"].items():
        probs, _ = forward_probs(model, patch)
        probs_stack.append(probs)
        per_patch.append({
            "row": r, "col": c,
            "probs": {SUBTYPE_CLASSES[i]: round(float(probs[i]), 2) for i in range(3)},
            "pred": SUBTYPE_CLASSES[int(np.argmax(probs))],
            "confidence": entropy_conf(probs),
        })
    slide_probs = np.mean(np.stack(probs_stack, 0), axis=0)
    pred_idx = int(np.argmax(slide_probs))
    s["stage_b"] = {"arch": req.arch, "metric": req.metric, "slide_probs": slide_probs, "pred_idx": pred_idx}
    s["results"]["stage_b"] = {
        "arch": req.arch,
        "metric": req.metric,
        "checkpoint_fold": fold_name,
        "subtype_pred": SUBTYPE_CLASSES[pred_idx],
        "prob_IDH_mt": round(float(slide_probs[0]), 2),
        "prob_IDH_mt_1p19q": round(float(slide_probs[1]), 2),
        "prob_IDH_wt": round(float(slide_probs[2]), 2),
        "confidence": entropy_conf(slide_probs),
        "n_patches": len(per_patch),
    }
    return {
        "per_patch": per_patch,
        "slide_probs": {SUBTYPE_CLASSES[i]: round(float(slide_probs[i]), 2) for i in range(3)},
        "subtype_pred": SUBTYPE_CLASSES[pred_idx],
        "confidence": entropy_conf(slide_probs),
        "arch": req.arch,
        "metric": req.metric,
        "checkpoint_fold": fold_name,
        "val_score": val_value,
    }


# ---- 5b. STAGE B occlusion map (optional, off by default) ----
@app.post("/api/stage_b_occlusion")
def api_occlusion(req: OcclReq):
    s = sess(req.session_id)
    patch = s["patches"].get((req.row, req.col))
    if patch is None:
        raise HTTPException(400, "extract that patch first")
    model = get_model("stage_b", req.arch, req.metric)
    probs, _ = forward_probs(model, patch)
    pred_idx = int(np.argmax(probs))
    imp = occlusion_map(model, patch, pred_idx)
    norm = (imp - imp.min()) / (np.ptp(imp) + 1e-8)
    s.setdefault("occlusion", {})[(req.row, req.col)] = imp

    # top-1 occlusion cell -> re-read level 0 at its true coordinates (real zoom,
    # mirrors the e2e _zoom.jpg), not an upscaled crop of the 2048 patch.
    flat = [(imp[r, c], r, c) for r in range(GRID[0]) for c in range(GRID[1])]
    flat.sort(key=lambda t: -t[0])
    tv, tr, tc = flat[0]
    patch_bbox = s.get("patch_bbox", {}).get((req.row, req.col))
    zoom = None
    if patch_bbox is not None:
        slide = openslide.OpenSlide(s["mapping"].wsi_path)
        try:
            zoom = extract_subtile(slide, patch_bbox, tr, tc, OUTPUT_SIZE)
        finally:
            slide.close()
    if zoom is None:   # fallback: crop the 2048 patch
        tw, th = patch.size[0] // GRID[1], patch.size[1] // GRID[0]
        zoom = patch.crop((tc*tw, tr*th, tc*tw+tw, tr*th+th)).resize(
            (OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)

    # this zoom is an INDEPENDENT level-0 re-read (extract_subtile above), not
    # derived from s["patches"], so it never went through the artifact
    # cleaning applied at extraction time - same .mrxs gate, same fix.
    if Path(s["mapping"].wsi_path).suffix.lower() == ".mrxs":
        from glioma_sparse.preprocessing.mrxs_tissue_crop import mask_black_artifacts
        zoom, artifact_frac = mask_black_artifacts(zoom)
        if artifact_frac > 0:
            print(f"[api_occlusion] r{req.row}c{req.col} zoom: cleaned "
                  f"{artifact_frac*100:.1f}% black scan-artifact pixels")

    s.setdefault("occl_top", {})[(req.row, req.col)] = {
        "zoom": zoom, "cell": (int(tr), int(tc)), "importance": round(float(tv), 2),
        "parent": (req.row, req.col)}

    return {
        "importance_grid": [[round(float(v), 3) for v in row] for row in norm],
        "raw_grid": [[round(float(v), 2) for v in row] for row in imp],
        "pred": SUBTYPE_CLASSES[pred_idx],
        "top_cell": [int(tr), int(tc)],
        "top_importance": round(float(tv), 2),
        "top_zoom": b64(zoom),
        "grid": GRID,
    }


# ---- 6. INTEGRATE ----
@app.post("/api/integrate")
def api_integrate(req: StageBReq):
    s = sess(req.session_id)
    if "stage_a" not in s or "stage_b" not in s:
        raise HTTPException(400, "run both stages first")
    gc = grade_class_short(GRADE_CLASSES[s["stage_a"]["pred_idx"]])
    st = SUBTYPE_CLASSES[s["stage_b"]["pred_idx"]]
    if gc == "control":
        dx = "Control tissue"
    else:
        dx = INTEGRATED.get((gc, st), "Unresolved")
    conf = round(entropy_conf(s["stage_a"]["probs"]) *
                 entropy_conf(s["stage_b"]["slide_probs"]), 2)
    s["results"]["integrated"] = {"diagnosis": dx, "confidence": conf}
    return {"integrated_diagnosis": dx, "integrated_confidence": conf,
            "grade_class": gc, "subtype": st}


# ---- 7. SAVE: colored xlsx + 2048 jpgs ----
@app.post("/api/save")
def api_save(req: SaveReq):
    s = sess(req.session_id)
    saved = _save_session_folder(s, Path(s["gs_dir"]))
    xlsx = next((p for p in saved if p.endswith("_report.xlsx")), None)
    return {"saved": saved, "report": xlsx}


def _save_session_folder(s, gs, save_patches=True):
    """Write the full analysis folder (figures, report, session.json) for a
    session into gs. Shared by single-slide save and batch processing.
    save_patches=False skips writing the full-resolution 2048x2048 patch JPGs
    (used for large batch runs to save disk and time); everything else
    (risk map figure, occlusion, report, session.json for reload) is
    unaffected."""
    fig_dir = gs / "figures"; rep_dir = gs / "report"
    fig_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    import shutil
    tmp_thumb = Path(s["thumb_path"])
    if tmp_thumb.exists():
        dest_thumb = gs / tmp_thumb.name
        shutil.copy2(tmp_thumb, dest_thumb)
        sidecar = tmp_thumb.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, dest_thumb.with_suffix(".json"))
        saved.append(str(dest_thumb))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_fig(fig, name):
        p = fig_dir / name
        fig.savefig(p, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig); saved.append(str(p))

    if "risk_map" in s:
        rm = s["risk_map"]; thumb = s["thumbnail"]
        fig, ax = plt.subplots(figsize=(2048/300, 2048/300))
        ax.imshow(thumb)
        rows, cols = GRID; w, h = thumb.size; tw, th = w/cols, h/rows
        hn = (rm-rm.min())/(np.ptp(rm)+1e-8)
        ax.imshow(np.kron(hn, np.ones((int(th), int(tw)))), cmap="RdYlBu_r",
                  alpha=0.45, extent=[0, w, h, 0])
        for sel in s.get("selected", []):
            ax.add_patch(plt.Rectangle((sel["col"]*tw, sel["row"]*th), tw, th,
                                       fill=False, edgecolor="#e11d1d", lw=2.2))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s['slide_id']}  Stage A: "
                     f"{s['results'].get('stage_a',{}).get('grade_pred','')}")
        save_fig(fig, "stageA_riskmap.jpg")

    if save_patches:
        for (r, c), patch in s["patches"].items():
            p = fig_dir / f"patch_r{r}c{c}.jpg"
            patch.resize((2048, 2048), Image.BICUBIC).save(p, quality=90)
            saved.append(str(p))

    for (r, c), imp in s.get("occlusion", {}).items():
        patch = s["patches"][(r, c)]
        fig, ax = plt.subplots(figsize=(2048/300, 2048/300))
        ax.imshow(patch)
        rows, cols = GRID; w, h = patch.size; tw, th = w/cols, h/rows
        hn = (imp-imp.min())/(np.ptp(imp)+1e-8)
        ax.imshow(np.kron(hn, np.ones((int(th), int(tw)))), cmap="RdYlBu_r",
                  alpha=0.45, extent=[0, w, h, 0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Stage B occlusion  r{r}c{c}")
        save_fig(fig, f"stageB_occlusion_r{r}c{c}.jpg")

    sel_rank = {(sel["row"], sel["col"]): i + 1
                for i, sel in enumerate(s.get("selected", []))}
    for (r, c), info in s.get("occl_top", {}).items():
        rank = sel_rank.get((r, c), 0)
        sr, sc = info["cell"]
        # the slide_id is intentionally NOT repeated here: this file already
        # lives inside that slide's own folder, and including a full TCGA stem
        # (which embeds a long GUID) pushed some paths past Windows' 260-char
        # MAX_PATH once nested under a deep input tree.
        name = f"stageB_rank{rank:02d}_r{r}c{c}_top1_r{sr}c{sc}_zoom.jpg"
        info["zoom"].resize((2048, 2048), Image.BICUBIC).save(fig_dir / name, quality=90)
        saved.append(str(fig_dir / name))

    xlsx_path = rep_dir / "report.xlsx"
    _write_xlsx(xlsx_path, s)
    saved.append(str(xlsx_path))

    def grid_or_none(k):
        v = s.get(k)
        if v is None:
            return None
        n = (v - v.min()) / (np.ptp(v) + 1e-8)
        return [[round(float(x), 3) for x in row] for row in n]
    patch_records = []
    sel_rank = {(x["row"], x["col"]): i + 1
                for i, x in enumerate(s.get("selected", []))}
    has_occl = bool(s.get("occlusion"))
    for (r, c), patch in s["patches"].items():
        sel = next((x for x in s.get("selected", []) if x["row"] == r and x["col"] == c), {})
        bbox = s.get("patch_bbox", {}).get((r, c))
        occ_top = s.get("occl_top", {}).get((r, c))
        imp = s.get("occlusion", {}).get((r, c))
        occ_grid = None
        zoom_file = None
        if imp is not None:
            n = (imp - imp.min()) / (np.ptp(imp) + 1e-8)
            occ_grid = [[round(float(x), 3) for x in row] for row in n]
        if occ_top is not None:
            rank = sel_rank.get((r, c), 0)
            sr, sc = occ_top["cell"]
            zoom_file = f"stageB_rank{rank:02d}_r{r}c{c}_top1_r{sr}c{sc}_zoom.jpg"
        patch_records.append({
            "row": r, "col": c,
            "risk": sel.get("risk"), "tissue": sel.get("tissue"),
            "wsi_bbox": list(bbox) if bbox else None,
            "patch_file": (f"patch_r{r}c{c}.jpg" if save_patches else None),
            "occl_grid": occ_grid,
            "occl_top": ({"cell": list(occ_top["cell"]),
                          "importance": occ_top["importance"],
                          "zoom_file": zoom_file} if occ_top else None),
        })
    session_json = {
        "occlusion_available": has_occl,
        "slide_id": s["slide_id"], "slide_path": s["slide_path"],
        "device": DEVICE, "grid": list(GRID),
        "thumbnail_w": s["thumbnail"].size[0], "thumbnail_h": s["thumbnail"].size[1],
        "wsi_level0_dim": s["mapping"].wsi_level0_dim,
        "wsi_crop_origin": s["mapping"].wsi_crop_origin,
        "wsi_crop_extent": _wsi_crop_extent(s["mapping"]),
        "tissue_in_thumb": _tissue_in_thumb(s["mapping"]),
        "resample_ratio": (round(s["mapping"].target_mpp / s["mapping"].base_mpp, 3)
                           if s["mapping"].has_wsi_mapping() else None),
        "risk_grid": grid_or_none("risk_map"),
        "percentile": s.get("percentile"),
        "archA": s["results"].get("stage_a", {}).get("arch"),
        "archB": s["results"].get("stage_b", {}).get("arch"),
        "metricA": s["results"].get("stage_a", {}).get("metric"),
        "metricB": s["results"].get("stage_b", {}).get("metric"),
        "selected": s.get("selected", []),
        "patches": patch_records,
        "stage_a": s["results"].get("stage_a"),
        "stage_b": s["results"].get("stage_b"),
        "integrated": s["results"].get("integrated"),
        "thumbnail_file": Path(s["thumb_path"]).name,
        "patches_saved": save_patches,
    }
    (gs / "session.json").write_text(json.dumps(session_json, indent=2, default=str))
    saved.append(str(gs / "session.json"))

    (rep_dir / "settings.json").write_text(json.dumps({
        "slide_id": s["slide_id"], "slide_path": s["slide_path"],
        "device": DEVICE, "grid": GRID, "target_mpp": TARGET_MPP,
        "results": s["results"],
    }, indent=2, default=str))
    saved.append(str(rep_dir / "settings.json"))
    return saved


# ---- 8. BROWSE FOLDER + LOAD PREVIOUS RESULT ----
class LoadReq(BaseModel):
    folder: str


@app.post("/api/browse_folder")
def api_browse_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select a GS_ analysis folder")
        root.destroy()
        return {"folder": path or ""}
    except Exception as e:
        raise HTTPException(400, f"native dialog unavailable ({e}); type the path")


@app.post("/api/load")
def api_load(req: LoadReq):
    """Reconstruct a saved GS_<abbrev> folder into a viewable result."""
    folder = Path(req.folder)
    sj = folder / "session.json"
    if not sj.exists():
        raise HTTPException(400, f"no session.json in {folder}; "
                                 "was this saved by the dashboard?")
    data = json.loads(sj.read_text())

    def load_jpg_b64(p):
        pp = Path(p)
        if not pp.exists():
            return None
        return "data:image/jpeg;base64," + base64.b64encode(pp.read_bytes()).decode()

    # thumbnail
    thumb_file = folder / data.get("thumbnail_file", "")
    thumb_b64 = load_jpg_b64(thumb_file) if thumb_file.exists() else None

    # patches (from figures/)
    fig = folder / "figures"
    patches = []
    for pr in data.get("patches", []):
        pf = fig / pr["patch_file"] if pr.get("patch_file") else None
        rec = {**pr, "patch": load_jpg_b64(pf) if pf else None}
        # load the top-cell zoom image if occlusion was computed for this patch
        ot = pr.get("occl_top")
        if ot and ot.get("zoom_file"):
            rec["top_zoom"] = load_jpg_b64(fig / ot["zoom_file"])
        patches.append(rec)

    return {
        "loaded": True,
        "slide_id": data.get("slide_id"),
        "folder": str(folder),
        "thumbnail": thumb_b64,
        "thumbnail_w": data.get("thumbnail_w"), "thumbnail_h": data.get("thumbnail_h"),
        "wsi_level0_dim": data.get("wsi_level0_dim"),
        "wsi_crop_origin": data.get("wsi_crop_origin", [0, 0]),
        "wsi_crop_extent": data.get("wsi_crop_extent"),
        "resample_ratio": data.get("resample_ratio"),
        "grid": data.get("grid", [8, 8]),
        "risk_grid": data.get("risk_grid"),
        "tissue_in_thumb": data.get("tissue_in_thumb"),
        "occlusion_available": data.get("occlusion_available", False),
        "patches_saved": data.get("patches_saved", True),
        "percentile": data.get("percentile"),
        "archA": data.get("archA"), "archB": data.get("archB"),
        "metricA": data.get("metricA"), "metricB": data.get("metricB"),
        "selected": data.get("selected", []),
        "patches": patches,
        "stage_a": data.get("stage_a"),
        "stage_b": data.get("stage_b"),
        "integrated": data.get("integrated"),
    }


# ============================================================
# BATCH PROCESSING
# ============================================================

class BatchReq(BaseModel):
    input_folder: str
    output_location: str
    archA: str = "resnet50"
    archB: str = "resnet50"
    metricA: str = DEFAULT_METRIC
    metricB: str = DEFAULT_METRIC
    percentile: float = 95.0
    occlusion: bool = False
    recursive: bool = False
    save_patches: bool = True
    stain_profile: str | None = None


def _process_slide_headless(slide_path, archA, archB, percentile, occlusion,
                             on_thumbnail=None, on_riskmap=None, stain_profile=None,
                             metricA=DEFAULT_METRIC, metricB=DEFAULT_METRIC):
    """Run the full pipeline on one slide with no HTTP session, returning a
    session-like dict ready for _save_session_folder.

    on_thumbnail(thumbnail) and on_riskmap(thumbnail, risk_map) are optional
    callbacks fired as soon as those cheap artifacts exist, so a caller (batch
    processing) can show a live preview without waiting for Stage B."""
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="gs_batch_"))
    thumb_path = tmp_dir / f"{slide_path.stem}.jpg"
    thumbnail, tissue_frac, _ = create_wsi_thumbnail(
        slide_path, output_path=str(thumb_path),
        target_mpp=TARGET_MPP, output_size=THUMB_SIZE,
        stain_profile=stain_profile)
    mapping = load_mapping(thumb_path)
    if on_thumbnail:
        on_thumbnail(thumbnail)
    s = {"slide_path": str(slide_path), "slide_id": slide_path.stem,
         "thumb_path": str(thumb_path), "thumbnail": thumbnail, "mapping": mapping,
         "patches": {}, "patch_bbox": {}, "occlusion": {}, "occl_top": {},
         "stain_profile": stain_profile, "results": {}}

    # Stage A
    mA = get_model("stage_a", archA, metricA)
    _, foldA, _ = resolve_best_checkpoint("stage_a", archA, metricA)
    probsA, _ = forward_probs(mA, thumbnail)
    predA = int(np.argmax(probsA))
    s["stage_a"] = {"arch": archA, "metric": metricA, "probs": probsA, "pred_idx": predA}
    s["results"]["stage_a"] = {
        "arch": archA, "metric": metricA, "checkpoint_fold": foldA,
        "grade_pred": GRADE_CLASSES[predA],
        "prob_control": round(float(probsA[0]), 2),
        "prob_low_grade": round(float(probsA[1]), 2),
        "prob_high_grade": round(float(probsA[2]), 2),
        "confidence": entropy_conf(probsA)}

    # risk map + selection
    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img(), model=mA,
        transform=_transform, grid=GRID, target_class=predA,
        device=DEVICE, control_shuffle_seed=SEED, score="logit")
    s["risk_map"] = risk_map
    if on_riskmap:
        on_riskmap(thumbnail, risk_map)
    thresh = float(np.percentile(risk_map.flatten(), percentile))
    cand = []
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            if risk_map[r, c] >= thresh:
                tf = _tissue_fraction(thumbnail, r, c, GRID)
                if tf >= 0.25:
                    cand.append({"row": r, "col": c,
                                 "risk": round(float(risk_map[r, c]), 2),
                                 "tissue": round(tf, 2)})
    cand.sort(key=lambda d: -d["risk"])
    s["selected"] = cand
    s["percentile"] = percentile

    # extract + Stage B
    mB = get_model("stage_b", archB, metricB)
    _, foldB, _ = resolve_best_checkpoint("stage_b", archB, metricB)
    slide = openslide.OpenSlide(mapping.wsi_path)
    probs_stack = []
    try:
        for sel in cand:
            bbox_thumb = grid_cell_bbox(sel["row"], sel["col"], GRID[0], GRID[1],
                                        mapping.thumbnail_size)
            wsi_bbox = thumbnail_bbox_to_wsi(mapping, bbox_thumb)
            if wsi_bbox is None:
                continue
            wx, wy, ww, wh = wsi_bbox
            patch = slide.read_region((wx, wy), 0, (ww, wh)).convert("RGB")
            if patch.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                patch = patch.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
            if Path(slide_path).suffix.lower() == ".mrxs":
                from glioma_sparse.preprocessing.mrxs_tissue_crop import mask_black_artifacts
                patch, artifact_frac = mask_black_artifacts(patch)
                if artifact_frac > 0:
                    print(f"[_process_slide_headless] r{sel['row']}c{sel['col']}: "
                          f"cleaned {artifact_frac*100:.1f}% black scan-artifact pixels")
            patch = normalize_stage_b_patch(patch, stain_profile)
            s["patches"][(sel["row"], sel["col"])] = patch
            s["patch_bbox"][(sel["row"], sel["col"])] = (wx, wy, ww, wh)
            pv, _ = forward_probs(mB, patch)
            probs_stack.append(pv)
            if occlusion:
                pidx = int(np.argmax(pv))
                imp = occlusion_map(mB, patch, pidx)
                s["occlusion"][(sel["row"], sel["col"])] = imp
                flat = sorted(((imp[a, b], a, b) for a in range(GRID[0])
                               for b in range(GRID[1])), key=lambda t: -t[0])
                tv, tr, tc = flat[0]
                zoom = extract_subtile(slide, (wx, wy, ww, wh), tr, tc, OUTPUT_SIZE)
                if zoom is None:
                    tw, th = patch.size[0]//GRID[1], patch.size[1]//GRID[0]
                    zoom = patch.crop((tc*tw, tr*th, tc*tw+tw, tr*th+th)).resize(
                        (OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
                if Path(slide_path).suffix.lower() == ".mrxs":
                    from glioma_sparse.preprocessing.mrxs_tissue_crop import mask_black_artifacts
                    zoom, artifact_frac = mask_black_artifacts(zoom)
                    if artifact_frac > 0:
                        print(f"[_process_slide_headless] r{sel['row']}c{sel['col']} zoom: "
                              f"cleaned {artifact_frac*100:.1f}% black scan-artifact pixels")
                s["occl_top"][(sel["row"], sel["col"])] = {
                    "zoom": zoom, "cell": (int(tr), int(tc)),
                    "importance": round(float(tv), 2)}
    finally:
        slide.close()

    if probs_stack:
        slide_probs = np.mean(np.stack(probs_stack, 0), axis=0)
        predB = int(np.argmax(slide_probs))
        s["stage_b"] = {"arch": archB, "metric": metricB, "slide_probs": slide_probs, "pred_idx": predB}
        s["results"]["stage_b"] = {
            "arch": archB, "metric": metricB, "checkpoint_fold": foldB,
            "subtype_pred": SUBTYPE_CLASSES[predB],
            "prob_IDH_mt": round(float(slide_probs[0]), 2),
            "prob_IDH_mt_1p19q": round(float(slide_probs[1]), 2),
            "prob_IDH_wt": round(float(slide_probs[2]), 2),
            "confidence": entropy_conf(slide_probs),
            "n_patches": len(probs_stack)}
        gc = grade_class_short(GRADE_CLASSES[predA])
        st = SUBTYPE_CLASSES[predB]
        dx = "Control tissue" if gc == "control" else INTEGRATED.get((gc, st), "Unresolved")
        s["results"]["integrated"] = {
            "diagnosis": dx,
            "confidence": round(entropy_conf(probsA) * entropy_conf(slide_probs), 2)}
    return s


# ============================================================
# METHODOLOGY WALKTHROUGH  (real pipeline on a preset example)
# ============================================================

def _permuted_control(seed=SEED, grid=GRID):
    """Reproduce the exact permuted control the risk map uses. Returns
    (control_square_img, permuted_img, permuted_tiles_list, order)."""
    ctrl = control_img()
    ctrl_sq = ctrl.resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
    tiles = list(tile_image(ctrl_sq, grid))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))
    permuted = [tiles[i] for i in order]
    perm_img = reconstruct_image(permuted, grid, ctrl_sq.size)
    return ctrl_sq, perm_img, permuted, order


METHOD_CACHE_DIR = REPO_ROOT / "scripts" / "dashboard" / "methodology_cache"


def _injection_demo(thumbnail, model, target_class, top_cells, grid=GRID, seed=SEED):
    """Show the ACTUAL patch injection: for each of a few high-signal target
    tiles, replace one tile of the permuted control with that target tile and
    record how the model's target-class logit changes. Returns the base
    (no-injection) logit and a list of per-injection frames with before/after
    images and logits."""
    ctrl_sq, perm_img, permuted_tiles, _ = _permuted_control(seed, grid)
    rows, cols = grid
    # baseline: permuted control alone
    _, base_logits = forward_probs(model, perm_img)
    base_logit = float(base_logits[target_class])

    target_tiles = list(tile_image(thumbnail, grid))
    frames = []
    # inject into a fixed, visually central control slot so the eye can follow it
    slot = (rows // 2) * cols + (cols // 2)
    for (r, c) in top_cells[:4]:
        tgt_tile = target_tiles[r * cols + c]
        injected = list(permuted_tiles)
        injected[slot] = tgt_tile
        inj_img = reconstruct_image(injected, grid, ctrl_sq.size)
        _, inj_logits = forward_probs(model, inj_img)
        inj_logit = float(inj_logits[target_class])
        frames.append({
            "row": int(r), "col": int(c),
            "slot_row": slot // cols, "slot_col": slot % cols,
            "injected_img": b64(inj_img),
            "target_tile": b64(tgt_tile.resize((256, 256), Image.BILINEAR)),
            "logit": round(inj_logit, 3),
            "delta": round(inj_logit - base_logit, 3),
        })
    return {
        "permuted_img": b64(perm_img),
        "base_logit": round(base_logit, 3),
        "slot_row": slot // cols, "slot_col": slot % cols,
        "frames": frames,
    }


def _compute_methodology(save_dir):
    """Run the entire methodology pipeline ONCE on the preset slide and return a
    fully self-contained dict (all images as b64, all params). Also writes it to
    save_dir/methodology_cache.json so subsequent opens load instantly."""
    target = Path(METHODOLOGY_TARGET)
    if not target.exists():
        raise HTTPException(400, f"preset example slide not found: {target}")

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="gs_method_"))
    thumb_path = tmp_dir / f"{target.stem}.jpg"
    thumbnail, _, _ = create_wsi_thumbnail(
        target, output_path=str(thumb_path), target_mpp=TARGET_MPP, output_size=THUMB_SIZE,
        stain_profile=None)
    mapping = load_mapping(thumb_path)

    # real pipeline (RN50/RN50, p95, occlusion on) via the shared headless runner
    s = _process_slide_headless(target, "resnet50", "resnet50",
                                percentile=95.0, occlusion=True, stain_profile=None)

    mA = get_model("stage_a", "resnet50")
    rm = s["risk_map"]
    norm = (rm - rm.min()) / (np.ptp(rm) + 1e-8)

    # ordered high-signal cells for the injection demo
    flat = [(rm[r, c], r, c) for r in range(GRID[0]) for c in range(GRID[1])]
    flat.sort(key=lambda t: -t[0])
    top_cells = [(r, c) for _, r, c in flat]
    injection = _injection_demo(s["thumbnail"], mA, s["stage_a"]["pred_idx"], top_cells)

    # extracted patches + per-patch probs + occlusion
    mB = get_model("stage_b", "resnet50")
    patches = []
    for sel in s["selected"]:
        r, c = sel["row"], sel["col"]
        patch = s["patches"].get((r, c))
        if patch is None:
            continue
        pv, _ = forward_probs(mB, patch)
        occ = s["occlusion"].get((r, c))
        occ_top = s["occl_top"].get((r, c))
        patches.append({
            "row": r, "col": c, "risk": sel.get("risk"), "tissue": sel.get("tissue"),
            "patch": b64(patch),
            "wsi_bbox": list(s["patch_bbox"].get((r, c))) if s["patch_bbox"].get((r, c)) else None,
            "probs": {"IDH_mt": round(float(pv[0]), 3),
                      "IDH_mt_1p19q": round(float(pv[1]), 3),
                      "IDH_wt": round(float(pv[2]), 3)},
            "pred": SUBTYPE_CLASSES[int(np.argmax(pv))],
            "occl_grid": ([[round(float(x), 3) for x in row]
                           for row in ((occ - occ.min()) / (np.ptp(occ) + 1e-8))]
                          if occ is not None else None),
            "occl_top": ({"cell": list(occ_top["cell"]), "zoom": b64(occ_top["zoom"])}
                         if occ_top else None),
        })

    ctrl_sq, _, _, _ = _permuted_control()
    payload = {
        "slide_id": target.stem,
        "target_thumb": b64(s["thumbnail"]),
        "thumbnail_w": s["thumbnail"].size[0], "thumbnail_h": s["thumbnail"].size[1],
        "control_orig": b64(ctrl_sq), "control_permuted": injection["permuted_img"],
        "grid": list(GRID),
        "risk_grid": [[round(float(v), 3) for v in row] for row in norm],
        "raw_grid": [[round(float(v), 2) for v in row] for row in rm],
        "selected": s["selected"],
        "injection": injection,
        "patches": patches,
        "stage_a": s["results"].get("stage_a"),
        "stage_b": s["results"].get("stage_b"),
        "integrated": s["results"].get("integrated"),
        "percentile": 95.0, "archA": "resnet50", "archB": "resnet50",
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "methodology_cache.json").write_text(json.dumps(payload))
    return payload


@app.post("/api/methodology")
def api_methodology():
    """Return the fully-precomputed methodology walkthrough. Computes once
    (real pipeline + real injection demo) and caches to disk; every later call
    loads the cache instantly, so the walkthrough never recomputes."""
    cache = METHOD_CACHE_DIR / "methodology_cache.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass   # corrupt cache -> recompute
    return _compute_methodology(METHOD_CACHE_DIR)


@app.post("/api/methodology_rebuild")
def api_methodology_rebuild():
    """Force a recompute of the methodology cache (e.g. after changing the
    preset slide or the models)."""
    cache = METHOD_CACHE_DIR / "methodology_cache.json"
    if cache.exists():
        cache.unlink()
    return _compute_methodology(METHOD_CACHE_DIR)


@app.post("/api/browse_output")
def api_browse_output():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select where the batch run folder will be created")
        root.destroy()
        return {"folder": path or ""}
    except Exception as e:
        raise HTTPException(400, f"native dialog unavailable ({e}); type the path")


BATCH_PROGRESS = {}   # run_id -> {total, done, current, ok, errors, finished, run_dir, index, updates}

PREVIEW_SIZE = 112   # small + cheap: no extra inference, just a resize + colormap lookup


def _preview_thumb(thumbnail):
    im = thumbnail.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _preview_overlay(thumbnail, risk_map):
    """Cheap signal-map preview: colormap lookup + nearest-neighbour upscale,
    no matplotlib figure rendering (that would be too slow over hundreds of
    slides)."""
    import matplotlib
    norm = (risk_map - risk_map.min()) / (np.ptp(risk_map) + 1e-8)
    try:
        cmap = matplotlib.colormaps["RdYlBu_r"]      # matplotlib >= 3.7
    except AttributeError:
        import matplotlib.cm as mcm
        cmap = mcm.get_cmap("RdYlBu_r")              # older matplotlib
    colors = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colors).resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
    base = thumbnail.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.BILINEAR).convert("RGB")
    blended = Image.blend(base, heat, alpha=0.45)
    buf = io.BytesIO(); blended.save(buf, format="JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _batch_worker(slide_path_str, archA, archB, percentile, occlusion,
                  save_patches, run_dir_str, result_queue, stain_profile=None,
                  metricA=DEFAULT_METRIC, metricB=DEFAULT_METRIC):
    """Runs in its own OS process (see _run_batch). Processes one slide fully
    and puts a small, picklable result dict onto result_queue."""
    slide_path = Path(slide_path_str)
    run_dir = Path(run_dir_str)
    try:
        s = _process_slide_headless(slide_path, archA, archB, percentile, occlusion,
                                    stain_profile=stain_profile,
                                    metricA=metricA, metricB=metricB)

        base = _safe_slide_folder_name(slide_path.stem)
        gs = run_dir / base
        k = 2
        while gs.exists():
            gs = run_dir / f"{base}_{k}"; k += 1
        _save_session_folder(s, gs, save_patches=save_patches)

        entry = {
            "folder": str(gs),
            "diagnosis": s["results"].get("integrated", {}).get("diagnosis", ""),
            "grade": s["results"].get("stage_a", {}).get("grade_pred", ""),
            "subtype": s["results"].get("stage_b", {}).get("subtype_pred", ""),
            "confidence": s["results"].get("integrated", {}).get("confidence", ""),
        }
        thumb_preview = _preview_thumb(s["thumbnail"]) if "thumbnail" in s else None
        overlay_preview = (_preview_overlay(s["thumbnail"], s["risk_map"])
                          if "risk_map" in s else None)
        result_queue.put({"ok": True, "entry": entry,
                          "thumb_preview": thumb_preview,
                          "overlay_preview": overlay_preview})
    except Exception as e:
        result_queue.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _run_batch(run_id, slides, run_dir, req):
    prog = BATCH_PROGRESS[run_id]

    def emit(ev):
        prog["updates"].append(ev)

    index = {"run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "input_folder": req.input_folder, "archA": req.archA, "archB": req.archB,
             "metricA": req.metricA, "metricB": req.metricB,
             "percentile": req.percentile, "occlusion": req.occlusion,
             "save_patches": req.save_patches,
             "n_slides": len(slides), "slides": []}
    import multiprocessing
    import queue as _queue_mod
    ctx = multiprocessing.get_context("spawn")

    for i, sp in enumerate(slides):
        prog["current"] = sp.stem
        prog["done"] = i
        emit({"i": i, "slide_id": sp.stem, "field": "start"})
        entry = {"slide_id": sp.stem, "index": i + 1, "status": "ok"}

        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_batch_worker,
            args=(str(sp), req.archA, req.archB, req.percentile, req.occlusion,
                 req.save_patches, str(run_dir), result_queue, req.stain_profile,
                 req.metricA, req.metricB),
            daemon=True)
        proc.start()
        proc.join(timeout=SLIDE_TIMEOUT_SECONDS)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
            entry["status"] = "error"
            entry["error"] = (f"timed out after {SLIDE_TIMEOUT_SECONDS}s and was killed "
                             f"(slide likely hung reading a malformed region)")
            prog["errors"] += 1
            emit({"i": i, "field": "status", "value": "error", "error": entry["error"]})
        elif proc.exitcode != 0:
            entry["status"] = "error"
            entry["error"] = (f"worker process crashed (exit code {proc.exitcode}); "
                             f"see server console for the native crash record")
            prog["errors"] += 1
            emit({"i": i, "field": "status", "value": "error", "error": entry["error"]})
        else:
            try:
                result = result_queue.get_nowait()
            except _queue_mod.Empty:
                result = {"ok": False, "error": "worker exited cleanly but returned no result"}
            if result["ok"]:
                entry.update(result["entry"])
                prog["ok"] += 1
                if result.get("thumb_preview"):
                    emit({"i": i, "field": "thumb", "value": result["thumb_preview"]})
                if result.get("overlay_preview"):
                    emit({"i": i, "field": "overlay", "value": result["overlay_preview"]})
                emit({"i": i, "field": "status", "value": "done", "folder": entry["folder"],
                     "diagnosis": entry["diagnosis"]})
            else:
                entry["status"] = "error"; entry["error"] = result["error"]
                prog["errors"] += 1
                emit({"i": i, "field": "status", "value": "error", "error": entry["error"]})
        result_queue.close()
        index["slides"].append(entry)
        prog["done"] = i + 1
    (run_dir / "batch_index.json").write_text(json.dumps(index, indent=2, default=str))
    _write_batch_xlsx(run_dir / "batch_summary.xlsx", index)
    prog["finished"] = True
    prog["current"] = None
    prog["index"] = index


@app.post("/api/batch")
def api_batch(req: BatchReq):
    inp = Path(req.input_folder)
    if not inp.is_dir():
        raise HTTPException(400, f"not a folder: {inp}")
    out_root = Path(req.output_location)
    if not out_root.is_dir():
        raise HTTPException(400, f"output location not found: {out_root}")

    if req.recursive:
        slides = sorted([p for p in inp.rglob("*")
                         if p.suffix.lower() in SUPPORTED_EXTS_BATCH])
    else:
        slides = sorted([p for p in inp.iterdir()
                         if p.suffix.lower() in SUPPORTED_EXTS_BATCH])
    if not slides:
        raise HTTPException(400, f"no WSIs in {inp}")

    run_id = uuid.uuid4().hex[:8]
    run_dir = out_root / (f"GS_batchrun_{_arch_tag(req.archA, req.archB)}_"
                          f"{_metric_tag(req.metricA, req.metricB)}_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=False)

    BATCH_PROGRESS[run_id] = {
        "total": len(slides), "done": 0, "current": None,
        "ok": 0, "errors": 0, "finished": False,
        "run_dir": str(run_dir), "index": None, "updates": []}

    import threading
    t = threading.Thread(target=_run_batch, args=(run_id, slides, run_dir, req),
                         daemon=True)
    t.start()
    # return immediately; frontend polls /api/batch_progress with a cursor
    return {"run_id": run_id, "run_dir": str(run_dir), "total": len(slides),
            "started": True}


class BatchProgressReq(BaseModel):
    run_id: str
    cursor: int = 0


@app.post("/api/batch_progress")
def api_batch_progress(req: BatchProgressReq):
    """Returns only the update events since `cursor` (append-only log), so
    repeated polling over hundreds of slides stays cheap: unchanged/finished
    slides are never re-sent."""
    p = BATCH_PROGRESS.get(req.run_id)
    if p is None:
        raise HTTPException(404, "unknown run_id")
    updates = p["updates"][req.cursor:]
    return {"total": p["total"], "done": p["done"], "current": p["current"],
            "ok": p["ok"], "errors": p["errors"], "finished": p["finished"],
            "run_dir": p["run_dir"], "index": p["index"],
            "updates": updates, "next_cursor": len(p["updates"])}


@app.post("/api/batch_index")
def api_batch_index(req: LoadReq):
    """Load a batch run's index (req.folder = the *_GS_batchrun dir)."""
    idx = Path(req.folder) / "batch_index.json"
    if not idx.exists():
        raise HTTPException(400, f"no batch_index.json in {req.folder}")
    return json.loads(idx.read_text())


SUPPORTED_EXTS_BATCH = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")


def _write_batch_xlsx(path, index):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    HEAD = "305496"
    wb = Workbook(); ws = wb.active; ws.title = "batch"
    ws.append([f"Glioma-SPARSE batch run {index['run_id']}"])
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append(["created", index["created"], "models",
               f"A:{index['archA']}/{index.get('metricA','f1')} "
               f"B:{index['archB']}/{index.get('metricB','f1')}", "p", index["percentile"]])
    ws.append([])
    hdr = ["#", "slide_id", "grade", "subtype", "integrated diagnosis",
           "confidence", "status"]
    ws.append(hdr)
    for j in range(1, len(hdr) + 1):
        c = ws.cell(row=ws.max_row, column=j)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=HEAD)
    for e in index["slides"]:
        ws.append([e["index"], e["slide_id"], e.get("grade", ""),
                   e.get("subtype", ""), e.get("diagnosis", e.get("error", "")),
                   e.get("confidence", ""), e["status"]])
    for col in ws.columns:
        w = max((len(str(c.value)) for c in col if c.value is not None), default=10)
        ws.column_dimensions[col[0].column_letter].width = min(w + 2, 50)
    wb.save(path)


def _write_xlsx(path, s):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    HEAD = "305496"; GREEN = "C6EFCE"; AMBER = "FFEB9C"; RED = "FFC7CE"
    def fill(c): return PatternFill("solid", fgColor=c)
    def conf_fill(v):
        try: v = float(v)
        except: return None
        return GREEN if v >= 0.66 else AMBER if v >= 0.33 else RED

    wb = Workbook(); ws = wb.active; ws.title = "diagnosis"
    ws.append(["Glioma-SPARSE report"]); ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append([]); ws.append(["Slide", s["slide_id"]])
    R = s["results"]
    integ = R.get("integrated", {})
    rows = [
        ("INTEGRATED DIAGNOSIS", integ.get("diagnosis", "")),
        ("Integrated confidence", integ.get("confidence", "")),
        ("", ""),
        ("Stage A model", R.get("stage_a", {}).get("arch", "")),
        ("Stage A checkpoint", f"best {R.get('stage_a', {}).get('metric', '')} "
                               f"(fold {R.get('stage_a', {}).get('checkpoint_fold', '')})"),
        ("Stage A grade", R.get("stage_a", {}).get("grade_pred", "")),
        ("Stage A confidence", R.get("stage_a", {}).get("confidence", "")),
        ("Stage B model", R.get("stage_b", {}).get("arch", "")),
        ("Stage B checkpoint", f"best {R.get('stage_b', {}).get('metric', '')} "
                               f"(fold {R.get('stage_b', {}).get('checkpoint_fold', '')})"),
        ("Stage B subtype", R.get("stage_b", {}).get("subtype_pred", "")),
        ("Stage B confidence", R.get("stage_b", {}).get("confidence", "")),
        ("Patches used", R.get("stage_b", {}).get("n_patches", "")),
    ]
    for k, v in rows:
        ws.append([k, v])
        if "confidence" in str(k).lower():
            col = conf_fill(v)
            if col: ws.cell(row=ws.max_row, column=2).fill = fill(col)
        if k in ("INTEGRATED DIAGNOSIS", "Stage A grade", "Stage B subtype"):
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
            ws.cell(row=ws.max_row, column=2).font = Font(bold=True)
    ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 52

    # Stage A probs sheet
    a = wb.create_sheet("stage_a"); a.append(["class", "probability"])
    for j in range(1, 3):
        a.cell(row=1, column=j).font = Font(bold=True, color="FFFFFF"); a.cell(row=1, column=j).fill = fill(HEAD)
    for cls, key in zip(GRADE_CLASSES, ["prob_control", "prob_low_grade", "prob_high_grade"]):
        a.append([cls, R.get("stage_a", {}).get(key, "")])
    # Stage B probs sheet
    b = wb.create_sheet("stage_b"); b.append(["class", "probability"])
    for j in range(1, 3):
        b.cell(row=1, column=j).font = Font(bold=True, color="FFFFFF"); b.cell(row=1, column=j).fill = fill(HEAD)
    for cls, key in zip(SUBTYPE_CLASSES, ["prob_IDH_mt", "prob_IDH_mt_1p19q", "prob_IDH_wt"]):
        b.append([cls, R.get("stage_b", {}).get(key, "")])
    wb.save(path)
