"""
Patch-injection risk map.

For a target slide that the Stage A model predicts as class C, this module
estimates which tiles of the target thumbnail are most indicative of class C
by injecting each target tile into a control slide and measuring how much
the predicted probability for class C shifts.

The control image is shuffled once with a fixed seed before injection so
that the control's own tile arrangement carries no positional information,
which makes the per-position risk score reflect only the injected tile's
content (the model is also trained to be shuffle-invariant).
"""

import numpy as np
import torch
from PIL import Image


# ============================================================
# TILE HELPERS
# ============================================================

def tile_image(img, grid):
    """Split a PIL image into grid[0] * grid[1] tiles, row-major."""
    rows, cols = grid
    w, h = img.size
    tile_w = w // cols
    tile_h = h // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


def reconstruct_image(tiles, grid, image_size):
    """Reassemble tiles into a single image, row-major."""
    rows, cols = grid
    W, H = image_size
    tile_w = W // cols
    tile_h = H // rows

    out = Image.new("RGB", (W, H))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            out.paste(tiles[idx], (c * tile_w, r * tile_h))
            idx += 1
    return out


def shuffle_control_once(control_img, grid, seed):
    """Shuffle the control image's tiles once with a fixed seed."""
    tiles = tile_image(control_img, grid)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))
    shuffled = [tiles[int(i)] for i in order]
    return reconstruct_image(shuffled, grid, control_img.size)


# ============================================================
# RISK MAP
# ============================================================

def compute_risk_map(
    target_img,
    control_img,
    model,
    transform,
    grid=(8, 8),
    target_class=None,
    device="cuda",
    control_shuffle_seed=42,
    batch_size=8,
    score="logit",
):
    """
    Compute a (rows, cols) risk map by patch injection.

    Args:
        target_img: PIL image, the slide to interpret
        control_img: PIL image, the reference slide
        model: trained Stage A classifier (already on `device`)
        transform: eval transform (same as inference)
        grid: (rows, cols) tile grid
        target_class: class index for which to measure the shift. If None, uses
                      argmax of the model's prediction on target_img.
        device: 'cuda' or 'cpu'
        control_shuffle_seed: fixed seed for the once-only control shuffle
        batch_size: how many injected variants to forward in one batch
        score: 'logit' (default) or 'prob'. The risk score R is the shift in the
               model's output for `target_class` when a tile is injected. On a
               confident model the softmax probability saturates near 0 or 1, so
               a single injected tile barely moves it and R collapses to ~0. The
               logit keeps that contrast, so it is the default. Use 'prob' only
               when you explicitly want the probability-scale shift.

    Returns:
        risk_map: float32 ndarray of shape (rows, cols).
                  risk_map[r, c] = score_injected[target_class]
                                   - score_baseline[target_class]
        target_class: the class index that was used (useful when caller passed None)
    """
    if score not in ("logit", "prob"):
        raise ValueError("score must be 'logit' or 'prob'")

    def read_score(logits_tensor):
        """Return the per-sample score matrix for the chosen scale."""
        if score == "prob":
            return torch.softmax(logits_tensor, dim=1).cpu().numpy()
        return logits_tensor.cpu().numpy()
    rows, cols = grid
    model.eval()

    # 1. Shuffle the control once
    control_shuffled = shuffle_control_once(control_img, grid, control_shuffle_seed)

    # 2. Tile target and (shuffled) control
    target_tiles = tile_image(target_img, grid)
    control_tiles = tile_image(control_shuffled, grid)

    # 3. Resolve target_class if not provided
    if target_class is None:
        with torch.no_grad():
            x = transform(target_img).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)
            target_class = int(probs.argmax(dim=1).item())

    # 4. Baseline: shuffled control with no injection
    with torch.no_grad():
        x = transform(control_shuffled).unsqueeze(0).to(device)
        baseline_scores = read_score(model(x))[0]
        baseline = float(baseline_scores[target_class])

    # 5. Build all injected variants (one per tile position)
    injected_variants = []
    positions = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            injected_tiles = list(control_tiles)
            injected_tiles[idx] = target_tiles[idx]
            injected_img = reconstruct_image(injected_tiles, grid, control_img.size)
            injected_variants.append(injected_img)
            positions.append((r, c))

    # 6. Forward in batches, fill risk map
    risk_map = np.zeros((rows, cols), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(injected_variants), batch_size):
            batch_imgs = injected_variants[start:start + batch_size]
            batch_x = torch.stack([transform(im) for im in batch_imgs]).to(device)
            batch_scores = read_score(model(batch_x))

            for i, (r, c) in enumerate(positions[start:start + batch_size]):
                injected = float(batch_scores[i, target_class])
                risk_map[r, c] = injected - baseline

    return risk_map, target_class