#!/usr/bin/env python3
r"""
Glioma-SPARSE: publication-quality methodology figures.

Generates one 300-DPI figure per conceptual step of the two-stage
weakly-supervised pipeline, for use in the manuscript and slides. Figures are
schematic (illustrative synthetic data), not tied to any real slide, so they
communicate the *method* cleanly.

Steps:
  1  patch-permutation during training  (position-invariant Stage A)
  2  tile injection into a permuted control (one tile at a time)
  3  signal map builds up cell-by-cell (injection logit -> risk)
  4  p95 high-signal tile selection + high-resolution extraction
  5  Stage B molecular classification of the extracted patches
  6  occlusion-based signal map + integrated diagnosis

OUTPUT (in --out, default ./methodology_figures)
  step1_patch_permutation.jpg ... step6_occlusion_diagnosis.jpg   (300 DPI, white bg)
  *_transparent.png versions too (transparent background, for slides/overlays)
  methodology_overview.jpg    all six panels on one sheet

USAGE
  python generate_methodology_figures.py --out figures/methodology
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ---- palette (mirrors the dashboard) ----
COL_CONTROL = "#c9cba3"
COL_LOW = "#ffe1a8"
COL_HIGH = "#e26d5c"
COL_MT = "#a8dadc"
COL_COD = "#f1faee"
COL_WT = "#457b9d"
INK = "#1d2733"
MUT = "#6b7686"
RISK_CMAP = "RdYlBu_r"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0,
    "figure.dpi": 300,
})

RNG = np.random.default_rng(42)
GRID = 8


# ---------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------

def _tissue_texture(n=8, seed=0):
    """A soft synthetic 'tissue' field for illustrative patches."""
    rng = np.random.default_rng(seed)
    base = rng.random((n, n))
    # smooth it a little
    from scipy.ndimage import gaussian_filter
    try:
        base = gaussian_filter(base, 0.8)
    except Exception:
        pass
    return base


def _draw_grid_patches(ax, colors, x0, y0, w, h, grid=GRID, lw=0.6,
                       edge="#ffffff", boxes=None, labels=None):
    """Draw a grid of colored cells inside [x0,x0+w]x[y0,y0+h]."""
    cw, ch = w / grid, h / grid
    for r in range(grid):
        for c in range(grid):
            col = colors[r][c] if isinstance(colors, (list, np.ndarray)) else colors
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (grid - 1 - r) * ch), cw, ch,
                                    facecolor=col, edgecolor=edge, linewidth=lw))
    if boxes:
        for (r, c) in boxes:
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (grid - 1 - r) * ch), cw, ch,
                                    facecolor="none", edgecolor="#e11d1d", linewidth=2.2))


def _cmap_colors(values):
    cmap = matplotlib.colormaps[RISK_CMAP]
    v = np.asarray(values, dtype=float)
    v = (v - v.min()) / (np.ptp(v) + 1e-9)
    return [[cmap(v[r, c]) for c in range(v.shape[1])] for r in range(v.shape[0])]


def _arrow(ax, xy1, xy2, color=INK, lw=2, style="-|>", mut=14, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=mut,
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def _title(ax, step, text, x=0.02, y=0.97):
    ax.text(x, y, f"Step {step}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#457b9d", va="top")
    ax.text(x, y - 0.055, text, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", color=INK, va="top")


def _clean(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------
# STEP 1 - patch permutation during training
# ---------------------------------------------------------------

def step1(ax):
    _clean(ax)
    _title(ax, 1, "Patch-permutation training \u2192 position-invariant Stage A")
    tex = _tissue_texture(GRID, seed=3)
    cmap = matplotlib.colormaps["BuPu"]
    ordered = [[cmap(0.25 + 0.5 * tex[r, c]) for c in range(GRID)] for r in range(GRID)]
    # left: ordered tissue
    _draw_grid_patches(ax, ordered, 0.6, 2.6, 3.4, 3.4)
    ax.text(2.3, 6.25, "tissue thumbnail", ha="center", fontsize=10.5, color=INK)
    ax.text(2.3, 1.9, "(8\u00d78 grid of patches)", ha="center", fontsize=9, color=MUT)
    # permute
    idx = RNG.permutation(GRID * GRID)
    flat = [ordered[r][c] for r in range(GRID) for c in range(GRID)]
    perm = [[flat[idx[r * GRID + c]] for c in range(GRID)] for r in range(GRID)]
    _draw_grid_patches(ax, perm, 6.0, 2.6, 3.4, 3.4)
    ax.text(7.7, 6.25, "patches permuted", ha="center", fontsize=10.5, color=INK)
    ax.text(7.7, 1.9, "spatial layout destroyed", ha="center", fontsize=9, color=MUT)
    _arrow(ax, (4.2, 4.3), (5.8, 4.3))
    ax.text(5.0, 4.65, "shuffle", ha="center", fontsize=9.5, color=MUT, style="italic")
    ax.text(5.0, 0.75,
            "Training on permuted patches forces the grader to judge each patch on its\n"
            "own content, not its position \u2014 so a single patch's effect can later be\n"
            "measured by injecting it into a neutral (permuted) reference.",
            ha="center", fontsize=9.3, color=INK)


# ---------------------------------------------------------------
# STEP 2 - tile injection into a permuted control
# ---------------------------------------------------------------

def step2(ax):
    _clean(ax)
    _title(ax, 2, "Inject each target tile into a permuted control, one at a time")
    # permuted control (neutral greys)
    grey = [[(0.82 - 0.12 * RNG.random(),) * 3 for _ in range(GRID)] for _ in range(GRID)]
    _draw_grid_patches(ax, grey, 0.6, 3.0, 3.3, 3.3)
    ax.text(2.25, 6.55, "permuted control", ha="center", fontsize=10.5, color=INK)
    ax.text(2.25, 2.35, "neutral reference", ha="center", fontsize=9, color=MUT)
    # a target tile
    ax.add_patch(Rectangle((4.55, 4.35), 0.75, 0.75, facecolor=COL_HIGH,
                            edgecolor="#ffffff", linewidth=1))
    ax.text(4.92, 5.35, "target\ntile", ha="center", fontsize=8.5, color=INK)
    # inject arrow into a control cell
    _arrow(ax, (5.4, 4.7), (3.0, 4.7), color="#e11d1d", rad=-0.15)
    ax.text(4.2, 5.15, "inject", ha="center", fontsize=9, color="#e11d1d", style="italic")
    # logit meter
    mx, my, mw, mh = 6.4, 3.2, 2.9, 2.9
    ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#f5f7fa", edgecolor="#d9dee5", linewidth=1))
    ax.text(mx + mw / 2, my + mh + 0.28, "Stage A logit shift", ha="center",
            fontsize=10, color=INK)
    # gauge bar
    ax.add_patch(Rectangle((mx + 0.4, my + 0.6), mw - 0.8, 0.42, facecolor="#e6e9ee",
                           edgecolor="none"))
    ax.add_patch(Rectangle((mx + 0.4, my + 0.6), (mw - 0.8) * 0.78, 0.42,
                           facecolor=COL_HIGH, edgecolor="none"))
    ax.text(mx + mw / 2, my + 1.5, "control \u2192 high-grade", ha="center",
            fontsize=9.5, color=INK)
    ax.text(mx + mw / 2, my + 1.0, "large shift = high-signal tile", ha="center",
            fontsize=8.6, color=MUT, style="italic")
    ax.text(5.0, 1.35,
            "The shift in the grader's logit caused by injecting one tile is that tile's\n"
            "signal score. Repeating for every tile builds the signal map.",
            ha="center", fontsize=9.3, color=INK)


# ---------------------------------------------------------------
# STEP 3 - signal map builds up
# ---------------------------------------------------------------

def _demo_riskfield():
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    field = (np.exp(-(((xx - 5.2) ** 2 + (yy - 2.4) ** 2) / 5.5))
             + 0.7 * np.exp(-(((xx - 2.0) ** 2 + (yy - 5.5) ** 2) / 6.0)))
    field += 0.12 * RNG.random((GRID, GRID))
    return field


def step3(ax, field=None):
    _clean(ax)
    _title(ax, 3, "Signal map: every tile scored, colored by injection response")
    if field is None:
        field = _demo_riskfield()
    colors = _cmap_colors(field)
    _draw_grid_patches(ax, colors, 3.0, 2.3, 4.0, 4.0, lw=0.5)
    ax.text(5.0, 6.6, "Stage A signal map", ha="center", fontsize=11, color=INK)
    # colorbar
    cax = ax.inset_axes([0.80, 0.30, 0.03, 0.42])
    grad = np.linspace(0, 1, 100).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=RISK_CMAP, origin="lower")
    cax.set_xticks([]); cax.set_yticks([0, 99]); cax.set_yticklabels(["low", "high"], fontsize=8)
    cax.set_title("signal", fontsize=8.5, pad=3)
    ax.text(5.0, 1.4,
            "Warm cells = tissue that pushes the grade toward tumour. This map localises\n"
            "diagnostic signal without any pixel-level annotation (weakly supervised).",
            ha="center", fontsize=9.3, color=INK)
    return field


# ---------------------------------------------------------------
# STEP 4 - p95 selection + extraction
# ---------------------------------------------------------------

def _top_cells(field, pct=95):
    thr = np.percentile(field, pct)
    cells = [(r, c) for r in range(GRID) for c in range(GRID) if field[r, c] >= thr]
    cells.sort(key=lambda rc: -field[rc[0], rc[1]])
    return cells


def step4(ax, field=None):
    _clean(ax)
    _title(ax, 4, "Select p95 high-signal tiles \u2192 extract at full resolution")
    if field is None:
        field = _demo_riskfield()
    colors = _cmap_colors(field)
    cells = _top_cells(field, 95)
    _draw_grid_patches(ax, colors, 0.5, 2.6, 3.6, 3.6, lw=0.4, boxes=cells)
    ax.text(2.3, 6.5, "signal map (p95 boxed)", ha="center", fontsize=10, color=INK)
    # extracted high-res patches
    px0 = 6.2
    for i, (r, c) in enumerate(cells[:4]):
        py = 5.6 - i * 1.25
        tex = _tissue_texture(6, seed=10 + i)
        cmap = matplotlib.colormaps["PuRd"]
        cols = [[cmap(0.2 + 0.6 * tex[a, b]) for b in range(6)] for a in range(6)]
        _draw_grid_patches(ax, cols, px0, py, 1.05, 1.05, grid=6, lw=0.3, edge="#ffffff")
        ax.add_patch(Rectangle((px0, py), 1.05, 1.05, facecolor="none",
                               edgecolor="#3b82f6", linewidth=1.4))
        _arrow(ax, (4.2, 4.4 - i * 0.05), (px0 - 0.15, py + 0.5),
               color="#3b82f6", lw=1.3, mut=10, rad=0.12)
    ax.text(px0 + 0.5, 6.0, "level-0 patches", ha="center", fontsize=9.5, color=INK)
    ax.text(5.0, 1.5,
            "Only the few tiles above the 95th percentile (\u2264 4 on an 8\u00d78 grid) are read\n"
            "back from the whole-slide image at full resolution and passed to Stage B.",
            ha="center", fontsize=9.3, color=INK)
    return cells


# ---------------------------------------------------------------
# STEP 5 - Stage B molecular classification
# ---------------------------------------------------------------

def step5(ax):
    _clean(ax)
    _title(ax, 5, "Stage B: molecular subtype from the high-resolution patches")
    # patches stack
    for i in range(4):
        py = 5.4 - i * 1.15
        tex = _tissue_texture(6, seed=20 + i)
        cmap = matplotlib.colormaps["PuRd"]
        cols = [[cmap(0.2 + 0.6 * tex[a, b]) for b in range(6)] for a in range(6)]
        _draw_grid_patches(ax, cols, 0.6, py, 0.95, 0.95, grid=6, lw=0.3)
        ax.add_patch(Rectangle((0.6, py), 0.95, 0.95, facecolor="none",
                               edgecolor="#3b82f6", linewidth=1.2))
    ax.text(1.08, 5.75, "patches", ha="center", fontsize=9.5, color=INK)
    # resnet block
    ax.add_patch(FancyBboxPatch((2.7, 3.6), 1.7, 1.6,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#1c232d", edgecolor="#6366f1", linewidth=1.6))
    ax.text(3.55, 4.55, "Stage B", ha="center", fontsize=11, color="#ffffff", fontweight="bold")
    ax.text(3.55, 4.15, "ResNet", ha="center", fontsize=9.5, color="#a9b2c0")
    ax.text(3.55, 3.85, "(soft-voted)", ha="center", fontsize=8, color="#8b949e")
    _arrow(ax, (1.7, 4.4), (2.6, 4.4), color="#3b82f6")
    _arrow(ax, (4.5, 4.4), (5.4, 4.4), color="#6366f1")
    # probability bars
    subs = [("IDH-mut (astro)", 0.24, COL_MT),
            ("IDH-mut 1p/19q", 0.62, COL_COD),
            ("IDH-wildtype", 0.14, COL_WT)]
    bx, bw = 5.6, 3.6
    by = 5.55; bar_h = 0.42; spacing = 1.05
    for i, (name, p, col) in enumerate(subs):
        yy = by - i * spacing
        ax.add_patch(Rectangle((bx, yy), bw, bar_h, facecolor="#eef1f4", edgecolor="none"))
        ax.add_patch(Rectangle((bx, yy), bw * p, bar_h, facecolor=col, edgecolor="#c9ced6",
                               linewidth=0.5))
        ax.text(bx, yy + bar_h + 0.12, name, fontsize=8.8, color=INK)
        ax.text(bx + bw + 0.12, yy + bar_h / 2, f"{p:.2f}", fontsize=9, color=INK, va="center")
    ax.text(bx + bw / 2, 6.45, "molecular probabilities", ha="center", fontsize=10, color=INK)
    ax.text(5.0, 1.5,
            "Each selected patch is classified into three molecular subtypes; the per-patch\n"
            "predictions are soft-voted into one slide-level molecular call.",
            ha="center", fontsize=9.3, color=INK)


# ---------------------------------------------------------------
# STEP 6 - occlusion + integrated diagnosis
# ---------------------------------------------------------------

def step6(ax):
    _clean(ax)
    _title(ax, 6, "Occlusion signal map + integrated WHO diagnosis")
    # patch with occlusion overlay
    g = 8
    tex = _tissue_texture(g, seed=31)
    cmap = matplotlib.colormaps["PuRd"]
    base = [[cmap(0.2 + 0.6 * tex[a, b]) for b in range(g)] for a in range(g)]
    _draw_grid_patches(ax, base, 0.8, 3.0, 3.4, 3.4, grid=g, lw=0.2, edge="#ffffff")
    # occlusion heat overlay (semi-transparent)
    yy, xx = np.mgrid[0:g, 0:g]
    imp = np.exp(-(((xx - 5) ** 2 + (yy - 3) ** 2) / 4.0)) + 0.1 * RNG.random((g, g))
    impn = (imp - imp.min()) / (np.ptp(imp) + 1e-9)
    cw = 3.4 / g
    ocmap = matplotlib.colormaps[RISK_CMAP]
    for r in range(g):
        for c in range(g):
            ax.add_patch(Rectangle((0.8 + c * cw, 3.0 + (g - 1 - r) * cw), cw, cw,
                                    facecolor=ocmap(impn[r, c]), edgecolor="none", alpha=0.45))
    # top cell box
    tr, tc = np.unravel_index(np.argmax(impn), impn.shape)
    ax.add_patch(Rectangle((0.8 + tc * cw, 3.0 + (g - 1 - tr) * cw), cw, cw,
                           facecolor="none", edgecolor="#fde047", linewidth=2.2))
    ax.text(2.5, 6.6, "occlusion signal map", ha="center", fontsize=10, color=INK)
    ax.text(2.5, 2.4, "which regions drove the call", ha="center", fontsize=8.6, color=MUT)
    _arrow(ax, (4.4, 4.7), (5.5, 4.7), color=INK)
    # integrated diagnosis card
    ax.add_patch(FancyBboxPatch((5.7, 3.9), 3.7, 1.7,
                                boxstyle="round,pad=0.05,rounding_size=0.14",
                                facecolor="#132029", edgecolor="#457b9d", linewidth=1.6))
    ax.text(7.55, 5.15, "Integrated diagnosis", ha="center", fontsize=9, color="#8fb4c9")
    ax.text(7.55, 4.55, "Oligodendroglioma,", ha="center", fontsize=11,
            color="#ffffff", fontweight="bold")
    ax.text(7.55, 4.18, "IDH-mutant 1p/19q-codeleted, gr. 2", ha="center",
            fontsize=8.6, color="#e6edf3")
    ax.text(5.0, 1.6,
            "Grade (Stage A) and molecular subtype (Stage B) combine into the WHO-2021\n"
            "diagnosis, with an occlusion map showing the evidence behind the call.",
            ha="center", fontsize=9.3, color=INK)


# ---------------------------------------------------------------
# driver
# ---------------------------------------------------------------

STEPS = [
    ("step1_patch_permutation", step1),
    ("step2_tile_injection", step2),
    ("step3_signal_map", step3),
    ("step4_p95_extraction", step4),
    ("step5_stageB_molecular", step5),
    ("step6_occlusion_diagnosis", step6),
]


def save_step(name, fn, out_dir, shared_field):
    for transparent in (False, True):
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        # pass the shared risk field to steps 3/4 so they're visually consistent
        if fn is step3:
            fn(ax, shared_field)
        elif fn is step4:
            fn(ax, shared_field)
        else:
            fn(ax)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        if transparent:
            p = out_dir / f"{name}_transparent.png"
            fig.savefig(p, dpi=300, transparent=True)
        else:
            p = out_dir / f"{name}.jpg"
            fig.savefig(p, dpi=300, facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)


def save_overview(out_dir, shared_field):
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    for (name, fn), ax in zip(STEPS, axes.ravel()):
        if fn in (step3, step4):
            fn(ax, shared_field)
        else:
            fn(ax)
    fig.suptitle("Glioma-SPARSE: two-stage weakly-supervised pipeline",
                 fontsize=16, fontweight="bold", color=INK, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, hspace=0.08, wspace=0.04)
    fig.savefig(out_dir / "methodology_overview.jpg", dpi=300,
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="methodology_figures")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    shared_field = _demo_riskfield()   # steps 3 & 4 share one signal map

    for name, fn in STEPS:
        save_step(name, fn, out_dir, shared_field)
        print(f"  {name}.jpg  (+ _transparent.png)")
    save_overview(out_dir, shared_field)
    print(f"  methodology_overview.jpg")
    print(f"\nAll figures (300 DPI) in: {out_dir}")


if __name__ == "__main__":
    main()
