"""
Ablation: does the risk score R depend on WHERE a tile is injected?

The interpretability claim rests on the model being position-invariant, so that
R reflects the CONTENT of an injected tile rather than the control slot it lands
in. This script tests that directly. It takes one informative tumour tile (by
default the highest-R tile of a chosen target slide) and injects it into every
one of the 64 control positions, recording R at each site.

If the model is position-invariant, R is nearly constant across sites, so a
small spread (low std, small max-min) is evidence that the score reflects tile
content. A large spread would mean the injection position drives R, which would
weaken the interpretation.

Outputs:
  site_ablation_<slide>.csv     R at every injection site, for the chosen tile
  site_ablation_<slide>.png     heatmap of R across sites + summary stats

Run (from repo root):
    python -m scripts.stage_a.injection_site_ablation \
        --checkpoint training_output/<fold>/best_auc.pth \
        --model resnet18 \
        --control-image data/included/control/<id>.jpg \
        --target-image  data/included/high_grade/<id>.jpg \
        --out-dir ablation_out
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
RISK_CMAP = "RdYlBu_r"


def load_model(ckpt, model_name, device):
    model = build_model(model_name, num_classes=len(CLASS_ORDER))
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def class_score(model, transform, img, cls_idx, device, score="logit"):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if score == "prob":
            out = torch.softmax(out, dim=1)
        return float(out.cpu().numpy()[0][cls_idx])


def pick_source_tile(risk_map, tile_rc):
    """Return (r, c) of the tile to test: explicit if given, else the max-R tile."""
    if tile_rc is not None:
        return tile_rc
    idx = int(np.argmax(risk_map))
    return idx // risk_map.shape[1], idx % risk_map.shape[1]


def run_site_ablation(model, transform, target_img, control_img, tile_rc,
                      pred_idx, device, score="logit"):
    """Inject one target tile into every control site; return R per site."""
    rows, cols = GRID
    control_shuffled = shuffle_control_once(control_img, GRID, SEED)
    control_tiles = tile_image(control_shuffled, GRID)
    target_tiles = tile_image(target_img, GRID)

    src_idx = tile_rc[0] * cols + tile_rc[1]
    source_tile = target_tiles[src_idx]

    baseline = class_score(model, transform, control_shuffled, pred_idx, device,
                           score)

    site_R = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tiles = list(control_tiles)
            tiles[r * cols + c] = source_tile
            injected = reconstruct_image(tiles, GRID, control_img.size)
            s = class_score(model, transform, injected, pred_idx, device, score)
            site_R[r, c] = s - baseline
    return site_R, baseline


def summarize(site_R):
    v = site_R.flatten()
    mean = float(v.mean())
    std = float(v.std(ddof=1))
    return {
        "mean_R": round(mean, 4),
        "std_R": round(std, 4),
        "min_R": round(float(v.min()), 4),
        "max_R": round(float(v.max()), 4),
        "range_R": round(float(v.max() - v.min()), 4),
        # coefficient of variation: spread relative to the signal size
        "cv": round(std / abs(mean), 4) if abs(mean) > 1e-6 else float("nan"),
    }


def make_figure(site_R, stats, tile_rc, slide_id, pred_name, out_path):
    lim = max(abs(site_R.min()), abs(site_R.max()), 1e-3)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(site_R, cmap=RISK_CMAP, vmin=-lim, vmax=lim)
    ax.set_title(f"R of tile (r{tile_rc[0]},c{tile_rc[1]}) injected at each site\n"
                 f"{slide_id}  |  class {pred_name}", fontsize=10)
    ax.set_xlabel("injection column")
    ax.set_ylabel("injection row")
    ax.set_xticks(range(GRID[1]))
    ax.set_yticks(range(GRID[0]))
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            ax.text(c, r, f"{site_R[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    txt = (f"mean {stats['mean_R']:.3f}   std {stats['std_R']:.3f}\n"
           f"range {stats['range_R']:.3f}   CV {stats['cv']}")
    ax.text(0.5, -0.16, txt, transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    target_img = Image.open(args.target_image).convert("RGB")
    control_img = Image.open(args.control_image).convert("RGB")
    model = load_model(args.checkpoint, args.model, device)

    # predicted class of the target
    x = transform(target_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = int(torch.softmax(model(x), dim=1).argmax().item())
    pred_name = CLASS_ORDER[pred_idx]

    # risk map to locate the important tile (unless one is given)
    risk_map, _ = compute_risk_map(
        target_img=target_img, control_img=control_img, model=model,
        transform=transform, grid=GRID, target_class=pred_idx, device=device,
        control_shuffle_seed=SEED, score=args.score,
    )
    tile_rc = None
    if args.tile_row is not None and args.tile_col is not None:
        tile_rc = (args.tile_row, args.tile_col)
    tile_rc = pick_source_tile(risk_map, tile_rc)

    site_R, baseline = run_site_ablation(
        model, transform, target_img, control_img, tile_rc, pred_idx, device,
        score=args.score)
    stats = summarize(site_R)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_id = Path(args.target_image).stem

    # CSV
    csv_path = out_dir / f"site_ablation_{slide_id}.csv"
    with open(csv_path, "w") as f:
        f.write("inject_row,inject_col,R\n")
        for r in range(GRID[0]):
            for c in range(GRID[1]):
                f.write(f"{r},{c},{site_R[r, c]:.6f}\n")

    make_figure(site_R, stats, tile_rc, slide_id, pred_name,
                out_dir / f"site_ablation_{slide_id}.png")

    print(f"Tile tested: (r{tile_rc[0]}, c{tile_rc[1]})  class {pred_name}")
    print(f"baseline {args.score}: {baseline:.4f}")
    print(f"R across 64 sites -> mean {stats['mean_R']:.3f}  "
          f"std {stats['std_R']:.3f}  range {stats['range_R']:.3f}  "
          f"CV {stats['cv']}")
    print(f"Wrote {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--control-image", required=True)
    p.add_argument("--target-image", required=True)
    p.add_argument("--tile-row", type=int, default=None,
                   help="Row of the tile to test. Default: the max-R tile.")
    p.add_argument("--tile-col", type=int, default=None,
                   help="Column of the tile to test. Default: the max-R tile.")
    p.add_argument("--score", default="logit", choices=["logit", "prob"])
    p.add_argument("--out-dir", default="ablation_out")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
