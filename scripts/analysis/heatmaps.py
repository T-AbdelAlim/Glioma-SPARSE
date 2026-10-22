"""
Whole-slide heatmaps in one shared coordinate frame (level-0 pixels), for the paper figures.

Per slide, up to four panels:
  1. H&E overview (openslide thumbnail)
  2. Stage A patch-injection map: the 8 x 8 grid cells of the 2048 px thumbnail mapped back to the slide,
     coloured by the injection score R (min-max normalised within the slide); the (up to 4) selected regions
     are outlined
  3. CLAM attention: gated-attention score of every 20x / 256 px tile (TRIDENT coordinates), converted to
     within-slide percentiles (CLAM convention) and blended onto the overview
  4. UNI2-h at Stage B: the selected regions and, inside them, the per-tile contribution to the predicted class of the
     UNI2-h logistic head (the head is linear, so the slide logit is the mean of the tile contributions)

CLAM attention is class-agnostic (CLAM-SB has one attention branch). Attention of the five fold models is
averaged after within-slide percentile ranking for external slides; for EBRAINS slides the model of the fold
in which the slide is a test slide is used.
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from PIL import Image
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
CLAM_DIR = REPO.parent / "mahmoodlab" / "CLAM"
sys.path.insert(0, str(CLAM_DIR))
from models.model_clam import CLAM_SB  # noqa: E402
from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, load_mapping, thumbnail_bbox_to_wsi  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = CLAM_DIR / "results" / "glioma_subtype_clam_s1"
CMAP = plt.get_cmap("turbo")


# ------------------------------------------------------------------ CLAM
_MODELS = {}


def clam_model(fold):
    """fold = 0..4 (CLAM numbering = split_01..05)."""
    if fold not in _MODELS:
        m = CLAM_SB(gate=True, size_arg="small", dropout=0.25, k_sample=8, n_classes=6, embed_dim=1536)
        sd = torch.load(RES / f"s_{fold}_checkpoint.pt", map_location="cpu")
        sd = {k.replace(".module", ""): v for k, v in sd.items() if "instance_loss_fn" not in k}
        m.load_state_dict(sd, strict=True)
        _MODELS[fold] = m.to(DEV).eval()
    return _MODELS[fold]


@torch.no_grad()
def attention_percentiles(feats, folds):
    """Mean over `folds` of the within-slide percentile rank of the raw gated-attention score."""
    f = torch.as_tensor(feats).float().to(DEV)
    out = []
    for k in folds:
        a = clam_model(k)(f, attention_only=True).squeeze(0).cpu().numpy()
        out.append((rankdata(a) - 0.5) / len(a))
    return np.mean(out, axis=0)


def load_tiles(h5_path):
    with h5py.File(h5_path) as h:
        return h["coords"][:], int(h["coords"].attrs["patch_size_level0"]), h["features"][:]


# -------------------------------------------------------------- rendering
class Canvas:
    """Overview of one slide plus helpers to paint level-0 rectangles onto it."""

    def __init__(self, slide_path, long_side=1400):
        self.slide = openslide.OpenSlide(str(slide_path))
        w0, h0 = self.slide.dimensions
        self.scale = long_side / max(w0, h0)
        self.w, self.h = int(round(w0 * self.scale)), int(round(h0 * self.scale))
        self.base = np.asarray(self.slide.get_thumbnail((self.w, self.h)).convert("RGB")).copy()

    def blank(self):
        return self.base.copy()

    def rect(self, img, x, y, size_w, size_h, color, alpha=0.6):
        x0, y0 = int(round(x * self.scale)), int(round(y * self.scale))
        x1, y1 = max(x0 + 1, int(round((x + size_w) * self.scale))), max(y0 + 1, int(round((y + size_h) * self.scale)))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        reg = img[y0:y1, x0:x1].astype(float)
        img[y0:y1, x0:x1] = (reg * (1 - alpha) + np.asarray(color[:3]) * 255 * alpha).astype(np.uint8)

    def outline(self, img, x, y, size_w, size_h, color=(255, 0, 0), t=3):
        x0, y0 = int(round(x * self.scale)), int(round(y * self.scale))
        x1, y1 = int(round((x + size_w) * self.scale)), int(round((y + size_h) * self.scale))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w - 1), min(y1, self.h - 1)
        for tt in range(t):
            img[y0:y1, min(x0 + tt, self.w - 1)] = color
            img[y0:y1, max(x1 - tt, 0)] = color
            img[min(y0 + tt, self.h - 1), x0:x1] = color
            img[max(y1 - tt, 0), x0:x1] = color


def tile_heat(canvas, coords, ps, values01, alpha=0.6, mask=None):
    img = canvas.blank()
    for i, ((x, y), v) in enumerate(zip(coords, values01)):
        if mask is not None and not mask[i]:
            continue
        canvas.rect(img, x, y, ps, ps, CMAP(float(v)), alpha)
    return img


def injection_panel(canvas, thumb_path, risk_map, selected_boxes):
    """8 x 8 patch-injection cells mapped to the slide; selected regions outlined."""
    mapping = load_mapping(thumb_path)
    R = np.asarray(risk_map, float)
    lo, hi = R.min(), R.max()
    Rn = (R - lo) / (hi - lo + 1e-9)
    img = canvas.blank()
    rows, cols = R.shape
    for r in range(rows):
        for c in range(cols):
            b = thumbnail_bbox_to_wsi(mapping, grid_cell_bbox(r, c, rows, cols, mapping.thumbnail_size))
            if b is None:
                continue
            canvas.rect(img, b[0], b[1], b[2], b[3], CMAP(float(Rn[r, c])), 0.55)
    for b in selected_boxes:
        canvas.outline(img, b[0], b[1], b[2], b[3])
    return img


def resuni_panel(canvas, coords, ps, contrib, boxes):
    """Per-tile contribution of the UNI2-h head to the predicted class, inside the selected regions."""
    inside = np.zeros(len(coords), bool)
    cx, cy = coords[:, 0] + ps // 2, coords[:, 1] + ps // 2
    for bx, by, bw, bh in boxes:
        inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
    v = np.zeros(len(coords))
    if inside.any():
        r = rankdata(contrib[inside])
        v[inside] = (r - 0.5) / inside.sum()
    img = tile_heat(canvas, coords, ps, v, 0.65, mask=inside)
    for b in boxes:
        canvas.outline(img, b[0], b[1], b[2], b[3])
    return img


def slide_figure(path, title, panels, subtitles):
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.9))
    axs = np.atleast_1d(axs)
    for ax, im, st in zip(axs, panels, subtitles):
        ax.imshow(im)
        ax.set_title(st, fontsize=9, loc="left")
        ax.axis("off")
        ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP)
    fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.025, pad=0.02,
                 label="low  <  within-slide score (percentile / normalised)  >  high")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- per-slide driver
def tile_contributions(feats, lr_npz, class_idx):
    """Per-tile contribution to the logit of `class_idx` of the standardised linear UNI2-h head (slide logit = mean)."""
    z = np.load(lr_npz)
    x = (feats.astype(np.float64) - z["mean"]) / z["scale"]
    return x @ z["coef"][class_idx]


def render_slide(out_path, title, wsi_path, thumb_path, h5_path, risk_map, boxes, clam_folds,
                 lr_npz=None, pred_class_idx=None, long_side=1400):
    """H&E | Stage A patch-injection | CLAM attention | UNI2-h at Stage B tile evidence (if lr_npz is given)."""
    coords, ps, feats = load_tiles(h5_path)
    canvas = Canvas(wsi_path, long_side)
    att = attention_percentiles(feats, clam_folds)
    panels = [canvas.blank()]
    subs = ["H&E"]
    if risk_map is not None:
        panels.append(injection_panel(canvas, thumb_path, risk_map, boxes))
        subs.append("Stage A patch injection (selected regions in red)")
    panels.append(tile_heat(canvas, coords, ps, att, 0.6))
    subs.append(f"CLAM attention (percentile; fold model(s) {[k + 1 for k in clam_folds]})")
    if lr_npz is not None and boxes:
        contrib = tile_contributions(feats, lr_npz, pred_class_idx)
        panels.append(resuni_panel(canvas, coords, ps, contrib, boxes))
        subs.append("UNI2-h at Stage B: UNI2-h evidence for the predicted subtype, inside the selected regions")
    slide_figure(Path(out_path), title, panels, subs)
    return att
