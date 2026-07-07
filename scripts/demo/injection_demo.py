"""
Demo: how Stage A turns patch injection into a risk map.

This produces a single explainer figure that walks through the mechanism a
reader needs to understand in one glance:

  1. the target tumour thumbnail (what we want to explain),
  2. the control slide, shuffled once with a fixed seed (the baseline canvas),
  3. a handful of single-tile injections: one target tile dropped into the
     shuffled control, with the resulting change in the model's probability for
     the predicted class printed under each, so a positive change reads as
     "this tile pushes the model toward the tumour class",
  4. the full 8x8 risk map R, where every cell is that probability change, with
     the p95 strongest-signal tiles outlined.

The point the figure should make immediately: injecting a tile that carries
tumour morphology raises the predicted probability a lot (large R), injecting
bland tissue barely moves it (R near zero), and the map of these changes is the
class-associated signal used to pick regions for Stage B.

Run (from repo root):
    python -m scripts.demo.injection_demo \
        --checkpoint training_output/<fold>/best_auc.pth \
        --model resnet18 \
        --control-image data/included/control/<control_id>.jpg \
        --target-image  data/included/high_grade/<tumour_id>.jpg \
        --out injection_demo.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from glioma_sparse.models.factory import build_model
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import (
    compute_risk_map, tile_image, reconstruct_image, shuffle_control_once,
)

CLASS_ORDER = ["control", "low_grade", "high_grade"]
GRID = (8, 8)
SEED = 42
FIXED_SITE = (2, 3)         # default control slot for fixed-site injection

# ---- visual style (calm, publication-friendly) ----
COL_STRONG = "#3b6ea5"      # muted blue for informative tiles
COL_BLAND = "#c7ccd1"       # light grey for the near-zero tile
COL_OUTLINE = "#e11d1d"     # red box outline for the p95 tiles
COL_OUTLINE_BLAND = "#9aa0a6"
RISK_CMAP = "RdYlBu_r"      # diverging red-yellow-blue: red high, blue low

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})


def load_model(ckpt, model_name, device):
    model = build_model(model_name, num_classes=len(CLASS_ORDER))
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def prob_for_class(model, transform, img, cls_idx, device):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return float(p[cls_idx]), p


def logit_for_class(model, transform, img, cls_idx, device):
    """Raw logit for one class. R is measured on this scale, so the demo
    bars must use it too, otherwise a confident model saturates to 0."""
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(x).cpu().numpy()[0]
    return float(z[cls_idx])


def inject_at(control_tiles, target_tiles, src_idx, dst_idx, grid, size):
    """Place target tile src_idx into control slot dst_idx and rebuild."""
    tiles = list(control_tiles)
    tiles[dst_idx] = target_tiles[src_idx]
    return reconstruct_image(tiles, grid, size)


def inject_tile(control_tiles, target_tiles, idx, grid, size):
    # same-position injection (kept for the --same-position path)
    return inject_at(control_tiles, target_tiles, idx, idx, grid, size)


def fixed_site_risk_map(model, transform, target_img, control_img, site_rc,
                        pred_idx, device):
    """Inject every target tile into ONE fixed control slot (site_rc).

    R for target tile (r,c) is stored at (r,c), so the map stays aligned with
    the target slide. Because the displaced control tile is identical for every
    injection, R reflects the injected tile's content, not its position.
    """
    rows, cols = GRID
    control_shuffled = shuffle_control_once(control_img, GRID, SEED)
    control_tiles = tile_image(control_shuffled, GRID)
    target_tiles = tile_image(target_img, GRID)
    site_idx = site_rc[0] * cols + site_rc[1]

    base = logit_for_class(model, transform, control_shuffled, pred_idx, device)
    R = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            img = inject_at(control_tiles, target_tiles, r * cols + c, site_idx,
                            GRID, control_img.size)
            R[r, c] = logit_for_class(model, transform, img, pred_idx, device) - base
    return R, control_shuffled, base


def pick_demo_tiles(risk_map, percentile=95, k_weak=1):
    """Choose the p95 tiles (same rule as the risk-map panel) plus one
    near-zero tile to contrast. Keeps both demo panels consistent."""
    thresh = np.percentile(risk_map.flatten(), percentile)
    flat = [(risk_map[r, c], r, c)
            for r in range(risk_map.shape[0])
            for c in range(risk_map.shape[1])]
    strong = sorted([t for t in flat if t[0] >= thresh], key=lambda t: -t[0])
    weak = min(flat, key=lambda t: abs(t[0]))
    return strong, weak


def build_figure(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()

    target_img = Image.open(args.target_image).convert("RGB")
    control_img = Image.open(args.control_image).convert("RGB")
    model = load_model(args.checkpoint, args.model, device)

    # predicted class for the target
    _, probs = prob_for_class(model, transform, target_img, 0, device)
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_ORDER[pred_idx]

    site_rc = (args.site_row, args.site_col)
    site_idx = site_rc[0] * GRID[1] + site_rc[1]

    # Build the risk map. Default is fixed-site injection (every tile into one
    # control slot); --same-position uses the original same-position map.
    if args.same_position:
        risk_map, _ = compute_risk_map(
            target_img=target_img, control_img=control_img, model=model,
            transform=transform, grid=GRID, target_class=pred_idx, device=device,
            control_shuffle_seed=SEED,
        )
        control_shuffled = shuffle_control_once(control_img, GRID, SEED)
        base_logit = logit_for_class(model, transform, control_shuffled,
                                     pred_idx, device)
        mode_label = "same-position injection"
    else:
        risk_map, control_shuffled, base_logit = fixed_site_risk_map(
            model, transform, target_img, control_img, site_rc, pred_idx, device)
        mode_label = f"fixed-site injection at (r{site_rc[0]},c{site_rc[1]})"

    target_tiles = tile_image(target_img, GRID)
    control_tiles = tile_image(control_shuffled, GRID)

    strong, weak = pick_demo_tiles(risk_map)
    demo_cells = list(strong) + [weak]

    # ---- figure layout ----
    n_inj = len(demo_cells)
    fig = plt.figure(figsize=(4 + 3 * n_inj, 8))
    gs = fig.add_gridspec(2, 1 + n_inj, height_ratios=[1, 1], hspace=0.35,
                          wspace=0.25)

    # top-left: target with grid + strong tiles outlined
    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.imshow(target_img)
    _draw_grid(ax_t, target_img.size, GRID)
    for _, r, c in strong:
        _outline_cell(ax_t, r, c, target_img.size, GRID, color=COL_OUTLINE)
    ax_t.set_title(f"Target thumbnail\npredicted: {pred_name} "
                   f"(p={probs[pred_idx]:.2f})", fontsize=10)
    ax_t.axis("off")

    # bottom-left: shuffled control baseline
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.imshow(control_shuffled)
    _draw_grid(ax_c, control_shuffled.size, GRID)
    ax_c.set_title(f"Control, shuffled (seed {SEED})\n"
                   f"baseline logit({pred_name}) = {base_logit:.2f}", fontsize=10)
    ax_c.axis("off")

    # per injection: the injected composite (top) and a delta bar (bottom)
    for j, (rscore, r, c) in enumerate(demo_cells):
        src_idx = r * GRID[1] + c
        dst_idx = src_idx if args.same_position else site_idx
        dst_r, dst_c = divmod(dst_idx, GRID[1])
        injected = inject_at(control_tiles, target_tiles, src_idx, dst_idx,
                             GRID, control_img.size)
        inj_logit = logit_for_class(model, transform, injected, pred_idx, device)
        delta = inj_logit - base_logit

        ax_i = fig.add_subplot(gs[0, 1 + j])
        ax_i.imshow(injected)
        is_strong = (rscore, r, c) in strong
        _outline_cell(ax_i, dst_r, dst_c, injected.size, GRID,
                      color=COL_OUTLINE if is_strong else COL_OUTLINE_BLAND)
        kind = "strong" if is_strong else "bland"
        ax_i.set_title(f"Inject tile (r{r},c{c}) [{kind}]\n"
                       f"logit={inj_logit:.2f}", fontsize=9)
        ax_i.axis("off")

        ax_d = fig.add_subplot(gs[1, 1 + j])
        ax_d.bar([0], [delta], color=COL_STRONG if is_strong else COL_BLAND,
                 width=0.6, edgecolor="none")
        ax_d.axhline(0, color="#444444", linewidth=0.8)
        pad = 0.1 * max(abs(risk_map.min()), abs(risk_map.max()), 1e-3)
        ax_d.set_ylim(min(risk_map.min(), 0) - pad, max(risk_map.max(), 0) + pad)
        ax_d.set_xticks([])
        for side in ("top", "right"):
            ax_d.spines[side].set_visible(False)
        ax_d.set_title(f"R = {delta:+.3f}", fontsize=9)
        if j == 0:
            ax_d.set_ylabel("R (shift in class logit)")

    fig.suptitle(
        f"Patch injection ({mode_label}): each target tile is dropped into the "
        f"shuffled control, and the shift in the {pred_name} logit is R",
        fontsize=12, y=0.99)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # second, compact figure: the full risk map with p95 tiles outlined
    _save_riskmap_panel(target_img, risk_map, pred_name,
                        Path(args.out).with_name(
                            Path(args.out).stem + "_riskmap.png"))
    print(f"Saved {args.out}")
    print(f"Saved {Path(args.out).with_name(Path(args.out).stem + '_riskmap.png')}")


def _draw_grid(ax, size, grid):
    w, h = size
    rows, cols = grid
    for i in range(1, rows):
        ax.axhline(i * h / rows, color="white", linewidth=0.5, alpha=0.6,
                   zorder=1)
    for j in range(1, cols):
        ax.axvline(j * w / cols, color="white", linewidth=0.5, alpha=0.6,
                   zorder=1)


def _outline_cell(ax, r, c, size, grid, color="red"):
    w, h = size
    rows, cols = grid
    cw, ch = w / cols, h / rows
    ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, fill=False,
                           edgecolor=color, linewidth=1.75, zorder=5))


def _save_riskmap_panel(target_img, risk_map, pred_name, out_path):
    thresh = np.percentile(risk_map.flatten(), 95)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(target_img)
    _draw_grid(axes[0], target_img.size, GRID)
    rows, cols = GRID
    for r in range(rows):
        for c in range(cols):
            if risk_map[r, c] >= thresh:
                _outline_cell(axes[0], r, c, target_img.size, GRID, COL_OUTLINE)
    axes[0].set_title("Target with p95 strongest-signal tiles", fontsize=11)
    axes[0].axis("off")

    lim = max(abs(risk_map.min()), abs(risk_map.max()), 1e-3)
    im = axes[1].imshow(risk_map, cmap=RISK_CMAP, vmin=-lim, vmax=lim)
    axes[1].set_title(f"Risk map R  (shift in {pred_name} logit)", fontsize=11)
    axes[1].set_xticks(range(cols))
    axes[1].set_yticks(range(rows))
    for r in range(rows):
        for c in range(cols):
            axes[1].text(c, r, f"{risk_map[r, c]:.2f}", ha="center",
                         va="center", fontsize=6, color="#222222")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--control-image", required=True)
    p.add_argument("--target-image", required=True)
    p.add_argument("--same-position", action="store_true",
                   help="Use same-position injection (each tile into its own "
                        "control slot). Default is fixed-site injection.")
    p.add_argument("--site-row", type=int, default=FIXED_SITE[0],
                   help="Fixed control slot row (default %d)." % FIXED_SITE[0])
    p.add_argument("--site-col", type=int, default=FIXED_SITE[1],
                   help="Fixed control slot column (default %d)." % FIXED_SITE[1])
    p.add_argument("--out", default="injection_demo.png")
    return p.parse_args()


if __name__ == "__main__":
    build_figure(parse_args())

    # python - m
    # scripts.demo.injection_demo - -checkpoint
    # training_output\20260702_1722
    # _resnet18_cw_split_01\best_auc.pth - -model
    # resnet18 - -control - image
    # "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\control\86242943-7775-11eb-827d-001a7dda7111.jpg" - -target - image
    # "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\high_grade\a195bae9-357f-11eb-b01c-001a7dda7111.jpg" - -out
    # injection_demo.png 