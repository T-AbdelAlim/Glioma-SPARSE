"""
Ablation: fixed-site injection.

Every target tile is injected one-by-one into a SINGLE fixed control slot
(default (2,3)), and R is recorded for each. Because the displaced control tile
is identical for every injection, the variation in R comes only from the content
of the injected target tile, not from which control tile was removed. This
removes the position confound that same-position injection carries.

The value for target tile (r,c) is stored back at (r,c), so the resulting map is
spatially aligned with the target slide and can be compared directly to the
standard same-position risk map.

Outputs:
  fixedsite_<slide>.csv    R per target tile, injected at the fixed site
  fixedsite_<slide>.png    map of R aligned to the target, p95 tiles outlined

Run (from repo root):
    python -m scripts.stage_a.fixed_site_injection \
        --checkpoint training_output/<fold>/best_auc.pth \
        --model resnet18 \
        --control-image data/included/control/<id>.jpg \
        --target-image  data/included/high_grade/<id>.jpg \
        --site-row 2 --site-col 3 \
        --out-dir ablation_out
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
    tile_image, reconstruct_image, shuffle_control_once,
)

CLASS_ORDER = ["control", "low_grade", "high_grade"]
GRID = (8, 8)
SEED = 42
RISK_CMAP = "RdYlBu_r"
COL_OUTLINE = "#e11d1d"


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


def fixed_site_map(model, transform, target_img, control_img, site_rc,
                   pred_idx, device, score="logit", batch_size=8):
    """Inject every target tile at the fixed control slot site_rc.

    Returns a (rows, cols) map where cell (r,c) is R for target tile (r,c)
    injected at site_rc.
    """
    rows, cols = GRID
    control_shuffled = shuffle_control_once(control_img, GRID, SEED)
    control_tiles = tile_image(control_shuffled, GRID)
    target_tiles = tile_image(target_img, GRID)
    site_idx = site_rc[0] * cols + site_rc[1]

    baseline = class_score(model, transform, control_shuffled, pred_idx, device,
                           score)

    # build one injected image per target tile, all injected at site_idx
    variants, positions = [], []
    for r in range(rows):
        for c in range(cols):
            tiles = list(control_tiles)
            tiles[site_idx] = target_tiles[r * cols + c]
            variants.append(reconstruct_image(tiles, GRID, control_img.size))
            positions.append((r, c))

    R = np.zeros((rows, cols), dtype=np.float32)
    for start in range(0, len(variants), batch_size):
        batch = variants[start:start + batch_size]
        x = torch.stack([transform(im) for im in batch]).to(device)
        with torch.no_grad():
            out = model(x)
            if score == "prob":
                out = torch.softmax(out, dim=1)
            out = out.cpu().numpy()
        for i, (r, c) in enumerate(positions[start:start + batch_size]):
            R[r, c] = out[i, pred_idx] - baseline
    return R, baseline


def summarize(R):
    v = R.flatten()
    mean = float(v.mean())
    std = float(v.std(ddof=1))
    return {"mean_R": round(mean, 4), "std_R": round(std, 4),
            "min_R": round(float(v.min()), 4), "max_R": round(float(v.max()), 4),
            "range_R": round(float(v.max() - v.min()), 4)}


def make_figure(R, site_rc, slide_id, pred_name, out_path):
    thresh = np.percentile(R.flatten(), 95)
    lim = max(abs(R.min()), abs(R.max()), 1e-3)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(R, cmap=RISK_CMAP, vmin=-lim, vmax=lim)
    ax.set_title(f"R per target tile, all injected at site "
                 f"(r{site_rc[0]},c{site_rc[1]})\n{slide_id}  |  class {pred_name}",
                 fontsize=10)
    ax.set_xlabel("target column")
    ax.set_ylabel("target row")
    ax.set_xticks(range(GRID[1]))
    ax.set_yticks(range(GRID[0]))
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            ax.text(c, r, f"{R[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="#222222", zorder=3)
            if R[r, c] >= thresh:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor=COL_OUTLINE, linewidth=2.2,
                                       zorder=5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_eval_transform()
    target_img = Image.open(args.target_image).convert("RGB")
    control_img = Image.open(args.control_image).convert("RGB")
    model = load_model(args.checkpoint, args.model, device)

    x = transform(target_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = int(torch.softmax(model(x), dim=1).argmax().item())
    pred_name = CLASS_ORDER[pred_idx]

    site_rc = (args.site_row, args.site_col)
    R, baseline = fixed_site_map(model, transform, target_img, control_img,
                                 site_rc, pred_idx, device, score=args.score)
    stats = summarize(R)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_id = Path(args.target_image).stem

    csv_path = out_dir / f"fixedsite_{slide_id}.csv"
    with open(csv_path, "w") as f:
        f.write("target_row,target_col,R,injected_at_row,injected_at_col\n")
        for r in range(GRID[0]):
            for c in range(GRID[1]):
                f.write(f"{r},{c},{R[r, c]:.6f},{site_rc[0]},{site_rc[1]}\n")

    make_figure(R, site_rc, slide_id, pred_name,
                out_dir / f"fixedsite_{slide_id}.png")

    print(f"Fixed injection site: (r{site_rc[0]}, c{site_rc[1]})  class {pred_name}")
    print(f"baseline {args.score}: {baseline:.4f}")
    print(f"R over target tiles -> mean {stats['mean_R']:.3f}  "
          f"std {stats['std_R']:.3f}  range {stats['range_R']:.3f}")
    print(f"Wrote {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--control-image", required=True)
    p.add_argument("--target-image", required=True)
    p.add_argument("--site-row", type=int, default=2)
    p.add_argument("--site-col", type=int, default=3)
    p.add_argument("--score", default="logit", choices=["logit", "prob"])
    p.add_argument("--out-dir", default="ablation_out")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())


# python -m scripts.stage_a.fixed_site_injection --checkpoint training_output\20260702_1722_resnet18_cw_split_01\best_auc.pth --model resnet18 --control-image "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\control\86242943-7775-11eb-827d-001a7dda7111.jpg" --target-image "C:\Users\TAbde\PycharmProjects\Glioma-SPARSE\data\included\high_grade\a195bae9-357f-11eb-b01c-001a7dda7111.jpg" --site-row 2 --site-col 3 --out-dir ablation_out