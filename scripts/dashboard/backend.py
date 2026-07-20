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
import time
import uuid

import numpy as np
from PIL import Image
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- real pipeline imports (same as inference_report.py) ----
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
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
# CONFIG  (fold locked to split_02 for this demo, as requested)
# ============================================================

HERE = Path(__file__).resolve().parent

CHECKPOINTS = {
    "stage_a": {
        "resnet18": r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\ResNet18_output\training_output\20260703_0554_resnet18_cw_split_02\best_f1.pth",
        "resnet50": r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\ResNet50_output\training_output\20260714_0955_resnet50_cw_split_02\best_f1.pth",
    },
    "stage_b": {
        "resnet18": r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\ResNet18_output\training_output_stageB_F1\20260716_1932_resnet18_stageB_fold2_cw\best_f1.pth",
        "resnet50": r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\ResNet50_output\training_output_stageB_F1\20260716_0552_resnet50_stageB_fold2_cw\best_f1.pth",
    },
}
CONTROL_IMAGE = r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\control\86242943-7775-11eb-827d-001a7dda7111.jpg"
LOGO_PATH = r"C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\docs\logo.png"

GRADE_CLASSES = ["control", "low_grade", "high_grade"]
SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
GRID = (8, 8)
SEED = 42
TARGET_MPP = 4.0
THUMB_SIZE = 2048
OUTPUT_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
_models = {}   # (stage, arch) -> model


def control_img():
    global _control_img
    if _control_img is None:
        _control_img = Image.open(CONTROL_IMAGE).convert("RGB")
    return _control_img


def get_model(stage, arch):
    key = (stage, arch)
    if key not in _models:
        ckpt = CHECKPOINTS[stage][arch]
        if not Path(ckpt).exists():
            raise HTTPException(400, f"checkpoint not found: {ckpt}")
        n = len(GRADE_CLASSES) if stage == "stage_a" else len(SUBTYPE_CLASSES)
        m = build_model(arch, num_classes=n)
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


# ============================================================
# OCCLUSION (mirrors inference_end_to_end.occlusion_map)
# ============================================================

def occlusion_map(model, patch, pred_idx, grid=GRID):
    rows, cols = grid
    _, base_logits = forward_probs(model, patch)
    base = float(base_logits[pred_idx])
    tiles = tile_image(patch, grid)
    mean_col = tuple(int(v) for v in np.array(patch).reshape(-1, 3).mean(0))
    imp = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            occ = list(tiles)
            occ[r * cols + c] = Image.new("RGB", occ[r * cols + c].size, mean_col)
            occ_img = reconstruct_image(occ, grid, patch.size)
            _, zl = forward_probs(model, occ_img)
            imp[r, c] = base - float(zl[pred_idx])
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

SESSIONS = {}   # session_id -> dict


def sess(sid):
    if sid not in SESSIONS:
        raise HTTPException(400, "unknown session; import a slide first")
    return SESSIONS[sid]


# ============================================================
# API MODELS
# ============================================================

class ImportReq(BaseModel):
    slide_path: str

class StageAReq(BaseModel):
    session_id: str
    arch: str = "resnet50"

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

class OcclReq(BaseModel):
    session_id: str
    row: int
    col: int
    arch: str = "resnet50"

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


@app.get("/api/config")
def config():
    return {
        "device": DEVICE,
        "grade_classes": GRADE_CLASSES,
        "subtype_classes": SUBTYPE_CLASSES,
        "grid": GRID,
        "archs": ["resnet18", "resnet50"],
        "checkpoints_present": {
            f"{s}:{a}": Path(p).exists()
            for s, d in CHECKPOINTS.items() for a, p in d.items()
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
        target_mpp=TARGET_MPP, output_size=THUMB_SIZE)
    mapping = load_mapping(thumb_path)

    # planned (not yet created) analysis folder alongside the slide
    gs_dir = slide_path.parent / f"GS_{_abbrev(slide_path.stem)}"

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {
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
    }
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
        "tissue_fraction": round(float(tissue_frac), 4),
        "has_wsi_mapping": mapping.has_wsi_mapping(),
        "tissue_in_thumb": _tissue_in_thumb(mapping),
        "elapsed_sec": round(time.time() - t0, 2),
    }


# ---- 2. STAGE A: grade ----
@app.post("/api/stage_a")
def api_stage_a(req: StageAReq):
    s = sess(req.session_id)
    model = get_model("stage_a", req.arch)
    probs, logits = forward_probs(model, s["thumbnail"])
    pred_idx = int(np.argmax(probs))
    s["stage_a"] = {"arch": req.arch, "probs": probs, "pred_idx": pred_idx}
    s["results"]["stage_a"] = {
        "arch": req.arch,
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
    # reuse the cached risk map if the Stage A model has not changed; only the
    # percentile threshold is re-applied (fast). Recompute only on model change.
    if s.get("risk_map") is not None and s.get("risk_map_arch") == arch:
        risk_map = s["risk_map"]
    else:
        model = get_model("stage_a", arch)
        risk_map, target_class = compute_risk_map(
            target_img=s["thumbnail"], control_img=control_img(), model=model,
            transform=_transform, grid=GRID, target_class=s["stage_a"]["pred_idx"],
            device=DEVICE, control_shuffle_seed=SEED, score="logit")
        s["risk_map"] = risk_map
        s["risk_map_arch"] = arch

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
    s["patches"][(req.row, req.col)] = patch
    s.setdefault("patch_bbox", {})[(req.row, req.col)] = (wx, wy, ww, wh)
    return {
        "patch": b64(patch),
        "bbox_thumb": list(bbox_thumb),
        "wsi_bbox": [wx, wy, ww, wh],
        "wsi_level0_dim": mapping.wsi_level0_dim,
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
    model = get_model("stage_b", req.arch)
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
    s["stage_b"] = {"arch": req.arch, "slide_probs": slide_probs, "pred_idx": pred_idx}
    s["results"]["stage_b"] = {
        "arch": req.arch,
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
    }


# ---- 5b. STAGE B occlusion map (optional, off by default) ----
@app.post("/api/stage_b_occlusion")
def api_occlusion(req: OcclReq):
    s = sess(req.session_id)
    patch = s["patches"].get((req.row, req.col))
    if patch is None:
        raise HTTPException(400, "extract that patch first")
    model = get_model("stage_b", req.arch)
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


def _save_session_folder(s, gs):
    """Write the full analysis folder (figures, report, session.json) for a
    session into gs. Shared by single-slide save and batch processing."""
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
        name = (f"stageB_{s['slide_id']}_rank{rank:02d}_r{r}c{c}"
                f"_top1_r{sr}c{sc}_zoom.jpg")
        info["zoom"].resize((2048, 2048), Image.BICUBIC).save(fig_dir / name, quality=90)
        saved.append(str(fig_dir / name))

    xlsx_path = rep_dir / f"{s['slide_id']}_report.xlsx"
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
            zoom_file = (f"stageB_{s['slide_id']}_rank{rank:02d}_r{r}c{c}"
                         f"_top1_r{sr}c{sc}_zoom.jpg")
        patch_records.append({
            "row": r, "col": c,
            "risk": sel.get("risk"), "tissue": sel.get("tissue"),
            "wsi_bbox": list(bbox) if bbox else None,
            "patch_file": f"patch_r{r}c{c}.jpg",
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
        "tissue_in_thumb": _tissue_in_thumb(s["mapping"]),
        "resample_ratio": (round(s["mapping"].target_mpp / s["mapping"].base_mpp, 3)
                           if s["mapping"].has_wsi_mapping() else None),
        "risk_grid": grid_or_none("risk_map"),
        "selected": s.get("selected", []),
        "patches": patch_records,
        "stage_a": s["results"].get("stage_a"),
        "stage_b": s["results"].get("stage_b"),
        "integrated": s["results"].get("integrated"),
        "thumbnail_file": Path(s["thumb_path"]).name,
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
        pf = fig / pr["patch_file"]
        rec = {**pr, "patch": load_jpg_b64(pf)}
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
        "resample_ratio": data.get("resample_ratio"),
        "grid": data.get("grid", [8, 8]),
        "risk_grid": data.get("risk_grid"),
        "tissue_in_thumb": data.get("tissue_in_thumb"),
        "occlusion_available": data.get("occlusion_available", False),
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
    percentile: float = 95.0
    occlusion: bool = False


def _process_slide_headless(slide_path, archA, archB, percentile, occlusion):
    """Run the full pipeline on one slide with no HTTP session, returning a
    session-like dict ready for _save_session_folder."""
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="gs_batch_"))
    thumb_path = tmp_dir / f"{slide_path.stem}.jpg"
    thumbnail, tissue_frac, _ = create_wsi_thumbnail(
        slide_path, output_path=str(thumb_path),
        target_mpp=TARGET_MPP, output_size=THUMB_SIZE)
    mapping = load_mapping(thumb_path)
    s = {"slide_path": str(slide_path), "slide_id": slide_path.stem,
         "thumb_path": str(thumb_path), "thumbnail": thumbnail, "mapping": mapping,
         "patches": {}, "patch_bbox": {}, "occlusion": {}, "occl_top": {},
         "results": {}}

    # Stage A
    mA = get_model("stage_a", archA)
    probsA, _ = forward_probs(mA, thumbnail)
    predA = int(np.argmax(probsA))
    s["stage_a"] = {"arch": archA, "probs": probsA, "pred_idx": predA}
    s["results"]["stage_a"] = {
        "arch": archA, "grade_pred": GRADE_CLASSES[predA],
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

    # extract + Stage B
    mB = get_model("stage_b", archB)
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
                s["occl_top"][(sel["row"], sel["col"])] = {
                    "zoom": zoom, "cell": (int(tr), int(tc)),
                    "importance": round(float(tv), 2)}
    finally:
        slide.close()

    if probs_stack:
        slide_probs = np.mean(np.stack(probs_stack, 0), axis=0)
        predB = int(np.argmax(slide_probs))
        s["stage_b"] = {"arch": archB, "slide_probs": slide_probs, "pred_idx": predB}
        s["results"]["stage_b"] = {
            "arch": archB, "subtype_pred": SUBTYPE_CLASSES[predB],
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


BATCH_PROGRESS = {}   # run_id -> {total, done, current, ok, errors, finished, run_dir, index}


def _run_batch(run_id, slides, run_dir, req):
    prog = BATCH_PROGRESS[run_id]
    index = {"run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "input_folder": req.input_folder, "archA": req.archA, "archB": req.archB,
             "percentile": req.percentile, "occlusion": req.occlusion,
             "n_slides": len(slides), "slides": []}
    for i, sp in enumerate(slides):
        prog["current"] = sp.stem
        prog["done"] = i
        entry = {"slide_id": sp.stem, "index": i + 1, "status": "ok"}
        try:
            s = _process_slide_headless(sp, req.archA, req.archB,
                                        req.percentile, req.occlusion)
            gs = run_dir / f"GS_{_abbrev(sp.stem)}"
            _save_session_folder(s, gs)
            entry["folder"] = str(gs)
            entry["diagnosis"] = s["results"].get("integrated", {}).get("diagnosis", "")
            entry["grade"] = s["results"].get("stage_a", {}).get("grade_pred", "")
            entry["subtype"] = s["results"].get("stage_b", {}).get("subtype_pred", "")
            entry["confidence"] = s["results"].get("integrated", {}).get("confidence", "")
            prog["ok"] += 1
        except Exception as e:
            entry["status"] = "error"; entry["error"] = str(e)
            prog["errors"] += 1
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

    slides = sorted([p for p in inp.iterdir()
                     if p.suffix.lower() in SUPPORTED_EXTS_BATCH])
    if not slides:
        raise HTTPException(400, f"no WSIs in {inp}")

    run_id = uuid.uuid4().hex[:8]
    run_dir = out_root / f"{run_id}_GS_batchrun"
    run_dir.mkdir(parents=True, exist_ok=False)

    BATCH_PROGRESS[run_id] = {
        "total": len(slides), "done": 0, "current": None,
        "ok": 0, "errors": 0, "finished": False,
        "run_dir": str(run_dir), "index": None}

    import threading
    t = threading.Thread(target=_run_batch, args=(run_id, slides, run_dir, req),
                         daemon=True)
    t.start()
    # return immediately; frontend polls /api/batch_progress
    return {"run_id": run_id, "run_dir": str(run_dir), "total": len(slides),
            "started": True}


@app.post("/api/batch_progress")
def api_batch_progress(req: LoadReq):
    """req.folder carries the run_id here."""
    p = BATCH_PROGRESS.get(req.folder)
    if p is None:
        raise HTTPException(404, "unknown run_id")
    return p


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
               f"A:{index['archA']} B:{index['archB']}", "p", index["percentile"]])
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
        ("Stage A grade", R.get("stage_a", {}).get("grade_pred", "")),
        ("Stage A confidence", R.get("stage_a", {}).get("confidence", "")),
        ("Stage B model", R.get("stage_b", {}).get("arch", "")),
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


# mount static (index.html, etc.) if present
if (HERE).exists():
    app.mount("/static", StaticFiles(directory=str(HERE)), name="static")
