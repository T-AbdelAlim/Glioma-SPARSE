r"""
Glioma-SPARSE comprehensive inference report.

Runs the two-stage pipeline on a single slide or a folder of slides and writes a
structured, colour-coded Excel report plus a per-slide .npz evidence cache.

Two things distinguish this from inference_end_to_end.py:

  1. It CACHES THE PER-TILE EVIDENCE. For every candidate region it stores the
     Stage A risk value, tissue fraction, grid position, WSI bbox and the full
     Stage B softmax. The slide-level call is then computed FROM that cache.
     Nothing is thrown away at soft-vote time.

  2. It EXTRACTS WIDE BY DEFAULT (low percentile, high cap). The cache therefore
     contains a superset of any (percentile, cap) configuration you might want
     later, so tune_selection.py can replay every setting with no GPU work.

Deployment note: the reported diagnosis uses the DEPLOY_* settings below, which
default to the values Stage B was trained on (p95, cap 4). The wider extraction
only populates the cache; it does not change the headline call.

Outputs, under results/<run-name>/:

    report_<slide_id>.xlsx        one workbook per slide   (single mode)
    report_batch.xlsx             one workbook for all     (folder mode)
    <abbrev>/cache/tiles.npz      per-tile evidence cache
    <abbrev>/riskmap/*.jpg        Stage A risk map overlay

Excel sheets:
    diagnosis     the integrated call, both stages, uncertainty, flags
    stage_a       grade probabilities and the risk-map summary
    stage_b       per-tile softmax, risk, tissue, agreement with the slide call
    uncertainty   entropy/margin/dispersion and the propagated confidence
    (batch mode)  overview sheet, one row per slide, colour-coded

Run:
    python -m scripts.inference.inference_report \
        --input path/to/slide.ndpi \
        --stage-a-ckpt ResNet50_output/training_output/<run>/best_f1.pth \
        --stage-b-ckpt ResNet50_output/training_output_stageB_F1/<run>/best_f1.pth \
        --model resnet50 --run-name report_single

    python -m scripts.inference.inference_report \
        --input path/to/folder --run-name report_batch --model resnet50 \
        --stage-a-ckpt ... --stage-b-ckpt ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.models.factory import build_model
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import (
    load_mapping, thumbnail_bbox_to_wsi, grid_cell_bbox,
)
import openslide


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "run_name": "report_run",
    "input": "",
    "stage_a_ckpt": "",
    "stage_b_ckpt": "",
    "model": "resnet50",

    # --- testset mode (per-fold, leakage-safe; mirrors inference_end_to_end.py) ---
    "testset": False,
    "resume": True,         # skip slides whose tiles.npz already exists
    "splits_dir": "splits",
    "stage_a_glob": "",
    "stage_b_glob": "",
    "sidecar_root": "",
    "splits": "val,test",   # roles to cache; val is required for tuning

    # --- what the reported diagnosis uses (match Stage B training) ---
    "deploy_percentile": 95.0,
    "deploy_cap": 4,
    "deploy_aggregation": "mean",

    # --- what gets cached (wide superset for later tuning) ---
    "cache_percentile": 50.0,   # low -> many candidates
    "cache_cap": 32,            # high -> room for aggregation experiments
    "min_tissue": 0.25,

    "target_mpp": 4.0,
    "thumb_size": 2048,
    "output_size": 2048,
    "save_patches": False,      # cache is enough; patches cost disk
    "wsi_root": "",
    "control_image": "",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

GRADE_CLASSES = ["control", "low_grade", "high_grade"]
SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
MUTANT_SET = {"IDH_mt", "IDH_mt_1p19q"}
GRID = (8, 8)
SEED = 42
WSI_EXTS = {".ndpi", ".svs", ".mrxs", ".tif", ".tiff"}
RESULTS_ROOT = REPO_ROOT / "results"

INTEGRATED = {
    ("low", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 2",
    ("high", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 3-4",
    ("low", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 2",
    ("high", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 3",
    ("low", "IDH_wt"): "IDH-wildtype glioma (low-grade morphology)",
    ("high", "IDH_wt"): "Glioblastoma, IDH-wildtype, grade 4",
}

FIG_DPI, FIG_INCHES, JPG_QUALITY = 300, 2048 / 300, 90


# ============================================================
# SHARED HELPERS  (kept identical to inference_end_to_end.py)
# ============================================================

def _rel(p):
    if not p:
        return p
    q = Path(str(p).replace("\\", "/"))
    return str(q if q.is_absolute() else REPO_ROOT / q)


def load_model(ckpt, model_name, n_classes, device):
    model = build_model(model_name, num_classes=n_classes)
    state = torch.load(str(ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def forward_probs(model, transform, img, device):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, logits.cpu().numpy()[0]


def grade_to_class(g):
    return {"low_grade": "low", "high_grade": "high"}.get(g, "control")


def mutant_status(s):
    return "mutant" if s in MUTANT_SET else "wildtype"


def abbrev(slide_id):
    head = slide_id.split("-")[0]
    return head if len(head) >= 6 else slide_id[:12]


def _tissue_fraction(thumbnail, row, col, grid):
    rows, cols = grid
    w, h = thumbnail.size
    tw, th = w // cols, h // rows
    tile = thumbnail.crop((col * tw, row * th, col * tw + tw, row * th + th))
    arr = np.array(tile).astype(np.float32) / 255.0
    maxc, minc = arr.max(axis=2), arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-8)
    mask = (maxc < 0.9) & (sat > 0.05)
    return float(mask.sum()) / float(mask.size)


def extract_highres(slide, mapping, row, col, grid, out_size):
    bbox_thumb = grid_cell_bbox(row, col, grid[0], grid[1], mapping.thumbnail_size)
    wsi_bbox = thumbnail_bbox_to_wsi(mapping, bbox_thumb)
    if wsi_bbox is None:
        return None, None
    wx, wy, ww, wh = wsi_bbox
    patch = slide.read_region((wx, wy), 0, (ww, wh)).convert("RGB")
    if patch.size != (out_size, out_size):
        patch = patch.resize((out_size, out_size), Image.BICUBIC)
    return patch, wsi_bbox


def resolve_wsi(stored, wsi_root, explicit=None):
    if explicit and Path(explicit).exists():
        return explicit
    if stored and Path(stored).exists():
        return stored
    if wsi_root:
        hit = _wsi_index(wsi_root).get(Path(str(stored).replace("\\", "/")).name)
        if hit:
            return hit
    raise FileNotFoundError(f"WSI not found: {stored}")


_WSI_IDX = {}
def _wsi_index(root):
    if root not in _WSI_IDX:
        idx = {}
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in WSI_EXTS:
                idx.setdefault(p.name, str(p))
        _WSI_IDX[root] = idx
    return _WSI_IDX[root]


# ============================================================
# UNCERTAINTY
# ============================================================

def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p * np.log(p + 1e-12)))


def norm_entropy(p):
    return entropy(p) / np.log(len(p))


def describe(probs):
    """Point summary of a probability vector."""
    p = np.asarray(probs, dtype=float)
    order = np.argsort(p)[::-1]
    top, second = float(p[order[0]]), float(p[order[1]]) if len(p) > 1 else 0.0
    return {
        "pred_idx": int(order[0]),
        "top_prob": round(top, 4),
        "margin": round(top - second, 4),
        "entropy": round(entropy(p), 4),
        "confidence": round(1.0 - norm_entropy(p), 4),
    }


def tile_dispersion(tile_probs):
    """How much do the tiles disagree? Mean pairwise total-variation distance."""
    P = np.asarray(tile_probs, dtype=float)
    n = len(P)
    if n < 2:
        return 0.0
    d, k = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d += 0.5 * float(np.abs(P[i] - P[j]).sum())
            k += 1
    return round(d / k, 4)


def consensus(tile_probs, slide_idx):
    """Fraction of tiles whose own argmax matches the slide-level call."""
    P = np.asarray(tile_probs, dtype=float)
    if len(P) == 0:
        return 0.0
    return round(float(np.mean(np.argmax(P, axis=1) == slide_idx)), 4)


def flags(stage_a_probs, tile_probs, slide_probs, n_tiles):
    """Human-readable warnings. This is where the pipeline's weak points live."""
    out = []
    a = np.asarray(stage_a_probs, float)
    if abs(a[1] - a[2]) < 0.15 and np.argmax(a) != 0:
        out.append("grade borderline low/high")
    if n_tiles < 3:
        out.append(f"few regions (n={n_tiles})")
    if len(tile_probs):
        s = describe(slide_probs)
        if s["margin"] < 0.15:
            out.append("subtype margin narrow")
        if consensus(tile_probs, s["pred_idx"]) < 0.5:
            out.append("tiles disagree with slide call")
        if tile_dispersion(tile_probs) > 0.5:
            out.append("high tile dispersion")
    return "; ".join(out) if out else "none"


# ============================================================
# AGGREGATION RULES  (shared with tune_selection.py)
# ============================================================

def aggregate(tile_probs, risks, rule="mean", **kw):
    """Combine per-tile softmaxes into one slide-level distribution.

    tile_probs : (n_tiles, n_classes)
    risks      : (n_tiles,) Stage A risk value per tile
    """
    P = np.asarray(tile_probs, dtype=float)
    if len(P) == 0:
        return None
    R = np.asarray(risks, dtype=float)

    if rule == "mean":
        out = P.mean(axis=0)

    elif rule == "max":
        # a single strongly-positive tile can carry the call
        out = P.max(axis=0)
        out = out / out.sum()

    elif rule == "topq_mean":
        # mean over the most confident q-fraction of tiles, per class
        q = kw.get("q", 0.25)
        k = max(1, int(np.ceil(q * len(P))))
        out = np.zeros(P.shape[1])
        for c in range(P.shape[1]):
            out[c] = np.sort(P[:, c])[::-1][:k].mean()
        out = out / out.sum()

    elif rule == "risk_weighted":
        # weight each tile by its Stage A risk
        w = R - R.min()
        w = w / (w.sum() + 1e-12) if w.sum() > 0 else np.ones(len(P)) / len(P)
        out = (P * w[:, None]).sum(axis=0)

    elif rule == "logodds_mean":
        # mean in log-odds space; sharper than mean, less brittle than max
        eps = 1e-6
        L = np.log(np.clip(P, eps, 1 - eps) / np.clip(1 - P, eps, 1 - eps))
        m = L.mean(axis=0)
        out = 1.0 / (1.0 + np.exp(-m))
        out = out / out.sum()

    elif rule == "noisy_or":
        # P(class present anywhere) - rewards focal evidence
        out = 1.0 - np.prod(1.0 - P, axis=0)
        out = out / out.sum()

    elif rule == "trimmed_mean":
        # drop the most extreme tile per class
        if len(P) <= 2:
            out = P.mean(axis=0)
        else:
            out = np.array([np.sort(P[:, c])[1:-1].mean() for c in range(P.shape[1])])
            out = out / out.sum()

    else:
        raise ValueError(f"unknown aggregation rule: {rule}")

    return out


def apply_class_weights(probs, weights):
    """Rescale a slide-level distribution by per-class weights, then renormalise.

    weights = (w_IDH_mt, w_IDH_mt_1p19q, w_IDH_wt). All ones = no change.
    This is the knob for 'focal IDH_wt evidence should count more'.
    """
    p = np.asarray(probs, float) * np.asarray(weights, float)
    return p / p.sum()


AGGREGATION_RULES = ["mean", "max", "topq_mean", "risk_weighted",
                     "logodds_mean", "noisy_or", "trimmed_mean"]


# ============================================================
# SELECTION
# ============================================================

def rank_candidates(risk_map, thumbnail, grid, percentile, cap, min_tissue):
    """All cells >= percentile of R that pass the tissue filter, best risk first."""
    rows, cols = grid
    thresh = float(np.percentile(risk_map.flatten(), percentile))
    kept = []
    for r in range(rows):
        for c in range(cols):
            if risk_map[r, c] >= thresh:
                tf = _tissue_fraction(thumbnail, r, c, grid)
                if tf >= min_tissue:
                    kept.append({"row": r, "col": c,
                                 "risk": float(risk_map[r, c]),
                                 "tissue_fraction": round(tf, 4)})
    kept.sort(key=lambda d: -d["risk"])
    return kept[:cap], thresh


def subset_for(cache, percentile, cap):
    """Replay a (percentile, cap) setting from a cache, with no GPU work.

    The cache holds every candidate down to cache_percentile. A higher
    percentile is a strict subset, selected on the SAME risk values.
    """
    R = cache["risk"]
    if len(R) == 0:
        return np.array([], dtype=int)
    thresh = np.percentile(cache["risk_map"].flatten(), percentile)
    idx = np.where(R >= thresh)[0]
    idx = idx[np.argsort(-R[idx])][:cap]
    return idx


# ============================================================
# PER-SLIDE PIPELINE
# ============================================================

def run_slide(thumb_source, model_a, model_b, transform, control_img, cfg,
              slide_dir, wsi_path=None, generate_thumb=False):
    """Stage A + Stage B with full per-tile caching. Returns (result, cache)."""
    slide_id = Path(thumb_source).stem
    (slide_dir / "cache").mkdir(parents=True, exist_ok=True)
    (slide_dir / "riskmap").mkdir(parents=True, exist_ok=True)

    thumb_path = Path(thumb_source)
    if generate_thumb:
        thumb_path = slide_dir / f"{slide_id}.jpg"
        create_wsi_thumbnail(wsi_path, output_path=str(thumb_path),
                             target_mpp=cfg["target_mpp"],
                             output_size=cfg["thumb_size"])
    thumbnail = Image.open(thumb_path).convert("RGB")

    # ---- Stage A ----
    probsA, logitsA = forward_probs(model_a, transform, thumbnail, cfg["device"])
    dA = describe(probsA)
    grade_name = GRADE_CLASSES[dA["pred_idx"]]
    grade_class = grade_to_class(grade_name)

    risk_map, _ = compute_risk_map(
        target_img=thumbnail, control_img=control_img, model=model_a,
        transform=transform, grid=GRID, target_class=dA["pred_idx"],
        device=cfg["device"], control_shuffle_seed=SEED, score="logit")

    # ---- candidates: WIDE, for the cache ----
    cands, cache_thresh = rank_candidates(
        risk_map, thumbnail, GRID, cfg["cache_percentile"],
        cfg["cache_cap"], cfg["min_tissue"])

    result = {
        "slide_id": slide_id,
        "grade_pred": grade_name, "grade_class": grade_class,
        "prob_control": round(float(probsA[0]), 4),
        "prob_low_grade": round(float(probsA[1]), 4),
        "prob_high_grade": round(float(probsA[2]), 4),
        "stageA_top_prob": dA["top_prob"], "stageA_margin": dA["margin"],
        "stageA_entropy": dA["entropy"], "stageA_confidence": dA["confidence"],
        "n_candidates_cached": len(cands),
    }

    empty_cache = {"slide_id": slide_id, "risk_map": risk_map,
                   "risk": np.array([]), "tile_probs": np.zeros((0, 3)),
                   "row": np.array([]), "col": np.array([]),
                   "tissue": np.array([]), "stage_a_probs": probsA}

    if grade_name == "control":
        result.update({"subtype_pred": "NA",
                       "integrated_diagnosis": "Control tissue",
                       "stageB_confidence": "", "integrated_confidence": "",
                       "n_regions_used": 0, "flags": "none"})
        _save_riskmap(thumbnail, risk_map, [], slide_dir, slide_id, grade_name)
        np.savez_compressed(slide_dir / "cache" / "tiles.npz", **empty_cache)
        return result, empty_cache

    if not cands:
        result.update({"subtype_pred": "NA",
                       "integrated_diagnosis": "No region extracted",
                       "stageB_confidence": "", "integrated_confidence": "",
                       "n_regions_used": 0, "flags": "no candidate region"})
        _save_riskmap(thumbnail, risk_map, [], slide_dir, slide_id, grade_name)
        np.savez_compressed(slide_dir / "cache" / "tiles.npz", **empty_cache)
        return result, empty_cache

    # ---- Stage B on EVERY candidate (this is the expensive part, done once) ----
    mapping = load_mapping(thumb_path)
    if not mapping.has_wsi_mapping():
        raise RuntimeError(f"no WSI mapping for {slide_id}")
    wsi_real = resolve_wsi(mapping.wsi_path, cfg["wsi_root"], wsi_path)
    slide = openslide.OpenSlide(wsi_real)

    tile_probs, keep = [], []
    try:
        for rank, sel in enumerate(cands, 1):
            patch, bbox = extract_highres(slide, mapping, sel["row"], sel["col"],
                                          GRID, cfg["output_size"])
            if patch is None:
                continue
            pB, _ = forward_probs(model_b, transform, patch, cfg["device"])
            tile_probs.append(pB)
            keep.append(sel)
            if cfg["save_patches"]:
                pdir = slide_dir / "patches"; pdir.mkdir(exist_ok=True)
                patch.save(pdir / f"{slide_id}_r{sel['row']}c{sel['col']}.jpg",
                           quality=JPG_QUALITY)
    finally:
        slide.close()

    if not tile_probs:
        result.update({"subtype_pred": "NA",
                       "integrated_diagnosis": "No region extracted",
                       "stageB_confidence": "", "integrated_confidence": "",
                       "n_regions_used": 0, "flags": "extraction failed"})
        np.savez_compressed(slide_dir / "cache" / "tiles.npz", **empty_cache)
        return result, empty_cache

    tile_probs = np.asarray(tile_probs, dtype=float)
    cache = {
        "slide_id": slide_id,
        "risk_map": risk_map,
        "risk": np.array([k["risk"] for k in keep], dtype=float),
        "tissue": np.array([k["tissue_fraction"] for k in keep], dtype=float),
        "row": np.array([k["row"] for k in keep], dtype=int),
        "col": np.array([k["col"] for k in keep], dtype=int),
        "tile_probs": tile_probs,
        "stage_a_probs": probsA,
    }
    np.savez_compressed(slide_dir / "cache" / "tiles.npz", **cache)

    # ---- DEPLOY setting: subset the cache, then aggregate ----
    idx = subset_for(cache, cfg["deploy_percentile"], cfg["deploy_cap"])
    used_probs = tile_probs[idx] if len(idx) else tile_probs[:1]
    used_risks = cache["risk"][idx] if len(idx) else cache["risk"][:1]

    slide_probs = aggregate(used_probs, used_risks, rule=cfg["deploy_aggregation"])
    dB = describe(slide_probs)
    subtype = SUBTYPE_CLASSES[dB["pred_idx"]]
    integrated = INTEGRATED.get((grade_class, subtype), "Unresolved")

    result.update({
        "subtype_pred": subtype,
        "prob_IDH_mt": round(float(slide_probs[0]), 4),
        "prob_IDH_mt_1p19q": round(float(slide_probs[1]), 4),
        "prob_IDH_wt": round(float(slide_probs[2]), 4),
        "mutant_status": mutant_status(subtype),
        "stageB_top_prob": dB["top_prob"], "stageB_margin": dB["margin"],
        "stageB_entropy": dB["entropy"], "stageB_confidence": dB["confidence"],
        "tile_consensus": consensus(used_probs, dB["pred_idx"]),
        "tile_dispersion": tile_dispersion(used_probs),
        "n_regions_used": int(len(used_probs)),
        "integrated_diagnosis": integrated,
        # an integrated call is a CONJUNCTION of two calls: multiply.
        "integrated_confidence": round(dA["confidence"] * dB["confidence"], 4),
        "flags": flags(probsA, used_probs, slide_probs, len(used_probs)),
    })
    _save_riskmap(thumbnail, risk_map, [keep[i] for i in idx],
                  slide_dir, slide_id, grade_name)
    return result, cache


def _save_riskmap(thumbnail, risk_map, selected, slide_dir, slide_id, grade_name):
    fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
    ax.imshow(thumbnail)
    rows, cols = GRID
    w, h = thumbnail.size
    tw, th = w / cols, h / rows
    hn = (risk_map - risk_map.min()) / (np.ptp(risk_map) + 1e-8)
    ax.imshow(np.kron(hn, np.ones((int(th), int(tw)))), cmap="RdYlBu_r",
              alpha=0.45, extent=[0, w, h, 0])
    for s in selected:
        ax.add_patch(plt.Rectangle((s["col"] * tw, s["row"] * th), tw, th,
                                   fill=False, edgecolor="#e11d1d", lw=2.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{slide_id}  Stage A: {grade_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(slide_dir / "riskmap" / "stageA_riskmap.jpg", dpi=FIG_DPI,
                format="jpg", bbox_inches="tight", pil_kwargs={"quality": JPG_QUALITY})
    plt.close(fig)


# ============================================================
# EXCEL REPORT
# ============================================================

HEAD_FILL = "305496"
GREEN, AMBER, RED, GREY = "C6EFCE", "FFEB9C", "FFC7CE", "F2F2F2"


def _fill(color):
    from openpyxl.styles import PatternFill
    return PatternFill("solid", fgColor=color)


def _conf_color(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return GREEN if v >= 0.66 else AMBER if v >= 0.33 else RED


def _style_header(ws, ncols, row=1):
    from openpyxl.styles import Font, Alignment
    for j in range(1, ncols + 1):
        c = ws.cell(row=row, column=j)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = _fill(HEAD_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center")


def _autosize(ws, max_width=46):
    for col in ws.columns:
        letter = col[0].column_letter
        width = max((len(str(c.value)) for c in col if c.value is not None),
                    default=8)
        ws.column_dimensions[letter].width = min(width + 2, max_width)


def _sheet_diagnosis(wb, res):
    from openpyxl.styles import Font, Alignment
    ws = wb.create_sheet("diagnosis")
    ws.append(["Glioma-SPARSE integrated report"])
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append([])
    rows = [
        ("Slide", res["slide_id"]),
        ("", ""),
        ("INTEGRATED DIAGNOSIS", res["integrated_diagnosis"]),
        ("Integrated confidence", res.get("integrated_confidence", "")),
        ("Flags", res.get("flags", "")),
        ("", ""),
        ("Stage A - grade", res["grade_pred"]),
        ("Stage A confidence", res.get("stageA_confidence", "")),
        ("Stage B - subtype", res.get("subtype_pred", "NA")),
        ("Stage B confidence", res.get("stageB_confidence", "")),
        ("IDH status", res.get("mutant_status", "NA")),
        ("", ""),
        ("Regions used", res.get("n_regions_used", 0)),
        ("Regions cached", res.get("n_candidates_cached", 0)),
    ]
    for k, v in rows:
        ws.append([k, v])
    for r in range(1, ws.max_row + 1):
        label = ws.cell(row=r, column=1).value
        if label in ("INTEGRATED DIAGNOSIS", "Stage A - grade", "Stage B - subtype"):
            ws.cell(row=r, column=1).font = Font(bold=True)
            ws.cell(row=r, column=2).font = Font(bold=True)
        if label and "confidence" in str(label).lower():
            col = _conf_color(ws.cell(row=r, column=2).value)
            if col:
                ws.cell(row=r, column=2).fill = _fill(col)
        if label == "Flags" and ws.cell(row=r, column=2).value not in ("none", ""):
            ws.cell(row=r, column=2).fill = _fill(AMBER)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 58
    return ws


def _sheet_stage_a(wb, res, cache):
    ws = wb.create_sheet("stage_a")
    ws.append(["class", "probability"])
    for name, key in zip(GRADE_CLASSES,
                         ["prob_control", "prob_low_grade", "prob_high_grade"]):
        ws.append([name, res[key]])
    ws.append([])
    ws.append(["metric", "value"])
    ws.append(["prediction", res["grade_pred"]])
    ws.append(["top probability", res["stageA_top_prob"]])
    ws.append(["margin (top - 2nd)", res["stageA_margin"]])
    ws.append(["entropy (nats)", res["stageA_entropy"]])
    ws.append(["confidence (1 - norm. entropy)", res["stageA_confidence"]])
    rm = cache.get("risk_map")
    if rm is not None and np.size(rm):
        ws.append([])
        ws.append(["risk map", "value"])
        ws.append(["max R", round(float(np.max(rm)), 4)])
        ws.append(["mean R", round(float(np.mean(rm)), 4)])
        ws.append(["p95 of R", round(float(np.percentile(rm, 95)), 4)])
    _style_header(ws, 2, row=1)
    _autosize(ws)
    return ws


def _sheet_stage_b(wb, res, cache, cfg):
    from openpyxl.styles import Font
    ws = wb.create_sheet("stage_b")
    hdr = ["rank", "row", "col", "risk_R", "tissue_frac",
           "p_IDH_mt", "p_IDH_mt_1p19q", "p_IDH_wt",
           "tile_call", "tile_margin", "tile_entropy",
           "in_deploy_set", "agrees_with_slide"]
    ws.append(hdr)
    _style_header(ws, len(hdr))

    P = cache.get("tile_probs", np.zeros((0, 3)))
    if len(P):
        idx = set(subset_for(cache, cfg["deploy_percentile"], cfg["deploy_cap"]).tolist())
        slide_call = res.get("subtype_pred", "NA")
        order = np.argsort(-cache["risk"])
        for rank, i in enumerate(order, 1):
            p = P[i]
            d = describe(p)
            call = SUBTYPE_CLASSES[d["pred_idx"]]
            ws.append([rank, int(cache["row"][i]), int(cache["col"][i]),
                       round(float(cache["risk"][i]), 4),
                       round(float(cache["tissue"][i]), 4),
                       round(float(p[0]), 4), round(float(p[1]), 4),
                       round(float(p[2]), 4),
                       call, d["margin"], d["entropy"],
                       "yes" if i in idx else "no",
                       "yes" if call == slide_call else "no"])
        # colour the probability of the slide-level call, and the flags
        for r in range(2, ws.max_row + 1):
            for c in (6, 7, 8):
                v = ws.cell(row=r, column=c).value
                if isinstance(v, (int, float)):
                    ws.cell(row=r, column=c).fill = _fill(
                        GREEN if v >= 0.66 else AMBER if v >= 0.33 else GREY)
            if ws.cell(row=r, column=12).value == "no":
                for c in range(1, len(hdr) + 1):
                    if ws.cell(row=r, column=c).fill.fgColor.rgb in (None, "00000000"):
                        ws.cell(row=r, column=c).fill = _fill(GREY)
            if ws.cell(row=r, column=13).value == "no":
                ws.cell(row=r, column=13).fill = _fill(RED)

        ws.append([])
        ws.append(["slide-level (deploy setting)", "", "", "", "",
                   res.get("prob_IDH_mt", ""), res.get("prob_IDH_mt_1p19q", ""),
                   res.get("prob_IDH_wt", ""), res.get("subtype_pred", "")])
        for c in range(1, 10):
            ws.cell(row=ws.max_row, column=c).font = Font(bold=True)
    else:
        ws.append(["no Stage B tiles (control or no region extracted)"])
    ws.freeze_panes = "A2"
    _autosize(ws)
    return ws


def _sheet_uncertainty(wb, res, cache, cfg):
    ws = wb.create_sheet("uncertainty")
    ws.append(["quantity", "value", "interpretation"])
    _style_header(ws, 3)
    rows = [
        ("Stage A confidence", res.get("stageA_confidence", ""),
         "1 - normalised entropy of the grade softmax"),
        ("Stage A margin", res.get("stageA_margin", ""),
         "gap between the top two grade probabilities"),
        ("Stage B confidence", res.get("stageB_confidence", ""),
         "1 - normalised entropy of the aggregated subtype distribution"),
        ("Stage B margin", res.get("stageB_margin", ""),
         "gap between the top two subtype probabilities"),
        ("Tile consensus", res.get("tile_consensus", ""),
         "fraction of used tiles whose own call matches the slide call"),
        ("Tile dispersion", res.get("tile_dispersion", ""),
         "mean pairwise total-variation distance between tiles; high = tiles disagree"),
        ("Integrated confidence", res.get("integrated_confidence", ""),
         "product of the two stage confidences: an integrated call needs BOTH right"),
        ("Flags", res.get("flags", ""), "automatic warnings"),
    ]
    for k, v, note in rows:
        ws.append([k, v, note])
    for r in range(2, ws.max_row + 1):
        col = _conf_color(ws.cell(row=r, column=2).value)
        if col and "confidence" in str(ws.cell(row=r, column=1).value).lower():
            ws.cell(row=r, column=2).fill = _fill(col)
    ws.append([])
    ws.append(["Note", "The integrated confidence is a product, so it is always "
                       "at most the weaker stage. A slide can be confident on "
                       "grade and weak on subtype and still be an unreliable "
                       "integrated call."])
    _autosize(ws)
    return ws


def write_slide_report(res, cache, cfg, path):
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    _sheet_diagnosis(wb, res)
    _sheet_stage_a(wb, res, cache)
    _sheet_stage_b(wb, res, cache, cfg)
    _sheet_uncertainty(wb, res, cache, cfg)
    wb.save(path)


BATCH_COLS = ["slide_id", "grade_pred", "prob_control", "prob_low_grade",
              "prob_high_grade", "stageA_confidence", "subtype_pred",
              "prob_IDH_mt", "prob_IDH_mt_1p19q", "prob_IDH_wt",
              "mutant_status", "stageB_confidence", "tile_consensus",
              "tile_dispersion", "n_regions_used", "n_candidates_cached",
              "integrated_diagnosis", "integrated_confidence", "flags"]


def write_batch_report(rows, caches, cfg, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "overview"
    cols = [c for c in BATCH_COLS if any(c in r for r in rows)]
    ws.append(cols)
    _style_header(ws, len(cols))
    conf_cols = {"stageA_confidence", "stageB_confidence", "integrated_confidence"}
    for i, r in enumerate(rows, 2):
        for j, col in enumerate(cols, 1):
            cell = ws.cell(row=i, column=j, value=r.get(col, ""))
            if col in conf_cols:
                c = _conf_color(r.get(col))
                if c:
                    cell.fill = _fill(c)
            if col == "flags" and r.get(col) not in ("none", "", None):
                cell.fill = _fill(AMBER)
    ws.freeze_panes = "A2"
    _autosize(ws)

    # distribution sheet
    ds = wb.create_sheet("distribution")
    ds.append(["integrated_diagnosis", "n", "mean_integrated_confidence"])
    _style_header(ds, 3)
    seen = {}
    for r in rows:
        d = r.get("integrated_diagnosis", "?")
        seen.setdefault(d, []).append(r.get("integrated_confidence") or 0)
    for d, vs in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        vals = [float(v) for v in vs if isinstance(v, (int, float))]
        ds.append([d, len(vs), round(float(np.mean(vals)), 4) if vals else ""])
    _autosize(ds)

    # per-tile long sheet, all slides
    ts = wb.create_sheet("all_tiles")
    hdr = ["slide_id", "rank", "row", "col", "risk_R", "tissue_frac",
           "p_IDH_mt", "p_IDH_mt_1p19q", "p_IDH_wt", "tile_call"]
    ts.append(hdr)
    _style_header(ts, len(hdr))
    for sid, cache in caches.items():
        P = cache.get("tile_probs", np.zeros((0, 3)))
        if not len(P):
            continue
        for rank, i in enumerate(np.argsort(-cache["risk"]), 1):
            p = P[i]
            ts.append([sid, rank, int(cache["row"][i]), int(cache["col"][i]),
                       round(float(cache["risk"][i]), 4),
                       round(float(cache["tissue"][i]), 4),
                       round(float(p[0]), 4), round(float(p[1]), 4),
                       round(float(p[2]), 4),
                       SUBTYPE_CLASSES[int(np.argmax(p))]])
    ts.freeze_panes = "A2"
    _autosize(ts)

    # settings sheet: provenance
    st = wb.create_sheet("settings")
    st.append(["setting", "value"])
    _style_header(st, 2)
    for k in ["model", "stage_a_ckpt", "stage_b_ckpt", "deploy_percentile",
              "deploy_cap", "deploy_aggregation", "cache_percentile",
              "cache_cap", "min_tissue", "output_size", "device"]:
        st.append([k, str(cfg.get(k, ""))])
    _autosize(st)
    wb.save(path)


# ============================================================
# DRIVER
# ============================================================

def list_inputs(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(q for q in p.rglob("*") if q.suffix.lower() in WSI_EXTS)


# ---------- testset mode ----------

def _fold_of(path):
    """split_01.csv -> 1;  *_split_01/best_f1.pth -> 1;  *_fold1_*/... -> 1."""
    import re
    s = str(path).replace("\\", "/")
    m = re.search(r"split[_-]?0*(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"fold[_-]?0*(\d+)", s)
    return int(m.group(1)) if m else None


def _pick(ckpts, fold):
    for c in ckpts:
        if _fold_of(c) == fold:
            return c
    return None


def _sidecar_for(stem, root):
    for jpg in Path(root).rglob(f"{stem}.jpg"):
        if jpg.with_suffix(".json").exists():
            return jpg
    return None


def _true_labels(sc_thumb):
    """True labels from the ebrains subtype folder, matching build_stageB_cohort.

    Sidecar layout is <sidecar-root>/<subtype>/included/<slide>.jpg, so the
    subtype folder (e.g. astro_IDHmt_G2, GBM_IDHwt, oligo_IDHmt_1p19qdel_G3)
    is two levels up from the sidecar file. The split CSV carries only the
    train/val/test role, never the molecular label.
    """
    folder = Path(sc_thumb).parent.parent.name
    n = folder.lower()
    if "gbm" in n or "idhwt" in n:
        mutation, grade_default = "IDH_wt", "high"
    elif "oligo" in n and "1p19q" in n:
        mutation, grade_default = "IDH_mt_1p19q", None
    elif "astro" in n and "idhmt" in n:
        mutation, grade_default = "IDH_mt", None
    else:
        return {"grade_class": "unknown", "mutation": "unknown",
                "subtype_folder": folder}
    import re as _re
    m = _re.search(r"_g(\d)", n)
    if m:
        grade_class = "low" if int(m.group(1)) == 2 else "high"
    else:
        grade_class = grade_default or "unknown"
    return {"grade_class": grade_class, "mutation": mutation,
            "subtype_folder": folder}


def _result_from_cache(npz_path, cfg):
    """Rebuild a batch-report row from an existing tiles.npz, no model needed.

    Applies the SAME deploy-setting subset + aggregation as run_slide, so a
    resumed slide is scored identically to a freshly computed one.
    """
    d = np.load(npz_path, allow_pickle=True)
    sid = str(d["slide_id"])
    probsA = d["stage_a_probs"]
    dA = describe(probsA)
    grade_name = GRADE_CLASSES[dA["pred_idx"]]
    grade_class = grade_to_class(grade_name)
    P = d["tile_probs"]
    res = {
        "slide_id": sid, "grade_pred": grade_name, "grade_class": grade_class,
        "prob_control": round(float(probsA[0]), 4),
        "prob_low_grade": round(float(probsA[1]), 4),
        "prob_high_grade": round(float(probsA[2]), 4),
        "stageA_top_prob": dA["top_prob"], "stageA_margin": dA["margin"],
        "stageA_entropy": dA["entropy"], "stageA_confidence": dA["confidence"],
        "n_candidates_cached": int(len(P)),
    }
    if grade_name == "control" or len(P) == 0:
        res.update({"subtype_pred": "NA",
                    "integrated_diagnosis": "Control tissue" if grade_name == "control"
                    else "No region extracted",
                    "stageB_confidence": "", "integrated_confidence": "",
                    "n_regions_used": 0, "flags": "resumed from cache"})
        return res
    cache = {"risk": d["risk"], "risk_map": d["risk_map"], "tile_probs": P}
    idx = subset_for(cache, cfg["deploy_percentile"], cfg["deploy_cap"])
    used = P[idx] if len(idx) else P[:1]
    used_r = d["risk"][idx] if len(idx) else d["risk"][:1]
    slide_probs = aggregate(used, used_r, rule=cfg["deploy_aggregation"])
    dB = describe(slide_probs)
    subtype = SUBTYPE_CLASSES[dB["pred_idx"]]
    res.update({
        "subtype_pred": subtype,
        "prob_IDH_mt": round(float(slide_probs[0]), 4),
        "prob_IDH_mt_1p19q": round(float(slide_probs[1]), 4),
        "prob_IDH_wt": round(float(slide_probs[2]), 4),
        "mutant_status": mutant_status(subtype),
        "stageB_top_prob": dB["top_prob"], "stageB_margin": dB["margin"],
        "stageB_entropy": dB["entropy"], "stageB_confidence": dB["confidence"],
        "tile_consensus": consensus(used, dB["pred_idx"]),
        "tile_dispersion": tile_dispersion(used),
        "n_regions_used": int(len(used)),
        "integrated_diagnosis": INTEGRATED.get((grade_class, subtype), "Unresolved"),
        "integrated_confidence": round(dA["confidence"] * dB["confidence"], 4),
        "flags": "resumed from cache",
    })
    return res, {"risk": d["risk"], "tile_probs": P, "row": d["row"],
                 "col": d["col"], "tissue": d["tissue"], "risk_map": d["risk_map"],
                 "stage_a_probs": probsA}


def run_testset(cfg):
    """Cache each fold's val+test tumour slides using that fold's own models.

    Controls are skipped: Stage B never runs on them, so they contribute no
    tiles to the cache and nothing to the tuning sweep.
    """
    import glob as _glob

    transform = build_eval_transform()
    control_img = Image.open(_rel(cfg["control_image"])).convert("RGB")
    a_ckpts = sorted(_glob.glob(_rel(cfg["stage_a_glob"])))
    b_ckpts = sorted(_glob.glob(_rel(cfg["stage_b_glob"])))
    if not a_ckpts or not b_ckpts:
        sys.exit(f"no checkpoints matched\n  A: {cfg['stage_a_glob']}\n  B: {cfg['stage_b_glob']}")
    print(f"Stage A checkpoints: {len(a_ckpts)}, Stage B: {len(b_ckpts)}")

    want = {s.strip() for s in str(cfg["splits"]).split(",") if s.strip()}
    run_dir = RESULTS_ROOT / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows, all_caches = [], {}
    for split_csv in sorted(Path(_rel(cfg["splits_dir"])).glob("split_0*.csv")):
        fold = _fold_of(split_csv)
        a_ck, b_ck = _pick(a_ckpts, fold), _pick(b_ckpts, fold)
        if not a_ck or not b_ck:
            print(f"fold {fold}: missing checkpoint, skipped")
            continue
        print(f"\n=== fold {fold} ===")
        print(f"  A: {Path(a_ck).parent.name}/{Path(a_ck).name}")
        print(f"  B: {Path(b_ck).parent.name}/{Path(b_ck).name}")
        fold_dir = run_dir / f"fold_{fold}"
        model_a = model_b = None       # loaded lazily on first uncached slide

        import csv as _csv
        with open(split_csv, newline="") as f:
            rows = [r for r in _csv.DictReader(f) if r["split"] in want]
        print(f"  {len(rows)} slides ({', '.join(sorted(want))})")

        for row in rows:
            split_path = Path(str(row["path"]).replace("\\", "/"))
            stem = split_path.stem
            if split_path.parent.name == "control":
                continue                      # Stage B never runs: no tiles to cache
            sc = _sidecar_for(stem, _rel(cfg["sidecar_root"]))
            if sc is None:
                print(f"    no sidecar for {stem}")
                continue

            # --- resume: reuse an existing cache instead of recomputing ---
            cache_npz = fold_dir / abbrev(stem) / "cache" / "tiles.npz"
            if cfg["resume"] and cache_npz.exists():
                try:
                    out = _result_from_cache(cache_npz, cfg)
                    res, cache = out if isinstance(out, tuple) else (out, None)
                    true = _true_labels(sc)
                    res["fold"] = fold
                    res["split"] = row["split"]
                    res["true_mutation"] = true["mutation"]
                    res["true_grade_class"] = true["grade_class"]
                    res["subtype_correct"] = int(res.get("subtype_pred") == true["mutation"])
                    all_rows.append(res)
                    if cache is not None:
                        all_caches[stem] = cache
                    print(f"    {stem[:14]} [{row['split']:4s}] skip (cached, "
                          f"{res.get('n_candidates_cached', 0)} tiles)")
                    continue
                except Exception as e:
                    print(f"    {stem[:14]} cache unreadable, recomputing: {e}")

            if model_a is None:           # first uncached slide in this fold
                model_a = load_model(a_ck, cfg["model"], len(GRADE_CLASSES), cfg["device"])
                model_b = load_model(b_ck, cfg["model"], len(SUBTYPE_CLASSES), cfg["device"])

            try:
                res, cache = run_slide(sc, model_a, model_b, transform,
                                       control_img, cfg, fold_dir / abbrev(stem))
                true = _true_labels(sc)
                res["fold"] = fold
                res["split"] = row["split"]
                res["true_mutation"] = true["mutation"]
                res["true_grade_class"] = true["grade_class"]
                res["subtype_correct"] = int(res.get("subtype_pred") == true["mutation"])
                all_rows.append(res)
                all_caches[stem] = cache
                print(f"    {stem[:14]} [{row['split']:4s}] {res.get('subtype_pred','NA'):14s} "
                      f"({res.get('n_candidates_cached',0)} tiles cached)")
            except Exception as e:
                print(f"    FAILED {stem}: {e}")

    if all_rows:
        out = run_dir / "report_batch.xlsx"
        write_batch_report(all_rows, all_caches, cfg, out)

        # one authoritative label file for tune_selection.py: slide -> fold,
        # split, true mutation. Derived here so the sweep never re-guesses.
        import csv as _csv
        with open(run_dir / "labels.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["slide_id", "fold", "split", "true_mutation",
                        "true_grade_class"])
            for r in all_rows:
                w.writerow([r["slide_id"], r.get("fold", ""), r.get("split", ""),
                            r.get("true_mutation", ""), r.get("true_grade_class", "")])

        n = [r.get("n_candidates_cached", 0) for r in all_rows]
        n_lab = sum(1 for r in all_rows if r.get("true_mutation") not in
                    ("unknown", "", None))
        print(f"\n{len(all_rows)} slides cached, {n_lab} with a molecular label. "
              f"Tiles per slide: median {int(np.median(n))}, range {min(n)}-{max(n)}")
        if n_lab == 0:
            print("  WARNING: no molecular labels resolved. Check that "
                  "--sidecar-root points at the <subtype>/included/ tree.")
        print(f"Batch report: {out}")
        print(f"Labels:       {run_dir / 'labels.csv'}  (feed to tune_selection)")
        print(f"Caches:       {run_dir}/fold_*/*/cache/tiles.npz")
    else:
        print("\nnothing cached")


def main():
    cfg = dict(CONFIG)
    ap = argparse.ArgumentParser()
    for k, v in CONFIG.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:  # default True: add a --no-<flag> to turn it off
                ap.add_argument(f"--no-{k.replace('_', '-')}", dest=k,
                                action="store_false", default=True)
            else:
                ap.add_argument(flag, action="store_true", default=False)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            ap.add_argument(flag, type=type(v), default=v)
        else:
            ap.add_argument(flag, type=str, default=v)
    args = ap.parse_args()
    cfg.update(vars(args))

    if cfg["testset"]:
        run_testset(cfg)
        return

    transform = build_eval_transform()
    control_img = Image.open(_rel(cfg["control_image"])).convert("RGB")
    model_a = load_model(_rel(cfg["stage_a_ckpt"]), cfg["model"],
                         len(GRADE_CLASSES), cfg["device"])
    model_b = load_model(_rel(cfg["stage_b_ckpt"]), cfg["model"],
                         len(SUBTYPE_CLASSES), cfg["device"])

    run_dir = RESULTS_ROOT / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    inputs = list_inputs(_rel(cfg["input"]))
    if not inputs:
        sys.exit(f"no slides found under {cfg['input']}")
    single = len(inputs) == 1

    print(f"Running {len(inputs)} slide(s). Deploy: p{cfg['deploy_percentile']} "
          f"cap {cfg['deploy_cap']} / {cfg['deploy_aggregation']}. "
          f"Cache: p{cfg['cache_percentile']} cap {cfg['cache_cap']}.")

    rows, caches = [], {}
    for wsi in inputs:
        sid = wsi.stem
        slide_dir = run_dir / abbrev(sid)
        try:
            res, cache = run_slide(wsi, model_a, model_b, transform, control_img,
                                   cfg, slide_dir, wsi_path=str(wsi),
                                   generate_thumb=True)
            rows.append(res)
            caches[sid] = cache
            print(f"  {sid}: {res['integrated_diagnosis']} "
                  f"(conf {res.get('integrated_confidence','')}) "
                  f"[{res.get('flags','')}]")
            if single:
                out = run_dir / f"report_{sid}.xlsx"
                write_slide_report(res, cache, cfg, out)
                print(f"\nReport: {out}")
        except Exception as e:
            print(f"  FAILED {sid}: {e}")

    if not single and rows:
        out = run_dir / "report_batch.xlsx"
        write_batch_report(rows, caches, cfg, out)
        print(f"\nBatch report: {out}")
    print(f"Caches under {run_dir}/*/cache/tiles.npz "
          f"-> feed these to tune_selection.py")


if __name__ == "__main__":
    main()
