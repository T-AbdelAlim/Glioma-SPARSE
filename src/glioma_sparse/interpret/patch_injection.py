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

method="crop" gives the same risk map with far less computation for
torchvision ResNets: a last-layer feature only sees a window of the image
(427 px for ResNet-50), so injecting one tile only changes the features near
that tile. The control is passed once; for each tile only a crop around it is
passed and its centre features replace those of the control before pooling.
"""

import math

import numpy as np
import torch
import torch.nn as nn
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
    method="full",
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
        method: 'full' (default) passes every injected image whole. 'crop' passes
                the control once and only a crop around each injected tile; same
                result up to float rounding, much faster. Needs a torchvision
                ResNet (falls back to 'full' otherwise).

    Returns:
        risk_map: float32 ndarray of shape (rows, cols).
                  risk_map[r, c] = score_injected[target_class]
                                   - score_baseline[target_class]
        target_class: the class index that was used (useful when caller passed None)
    """
    if score not in ("logit", "prob"):
        raise ValueError("score must be 'logit' or 'prob'")
    if method not in ("full", "crop"):
        raise ValueError("method must be 'full' or 'crop'")

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

    if method == "crop":
        risk_map = _risk_map_crop(target_tiles, control_shuffled, model, transform,
                                  grid, target_class, device, batch_size, read_score)
        if risk_map is not None:
            return risk_map, target_class

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


# ============================================================
# CROPPED INJECTION
# ============================================================

def _receptive_field(model):
    """Receptive field (pixels) and stride of the last feature map of a torchvision ResNet."""
    rf, stride = 1, 1
    for name, m in model.named_modules():
        if "downsample" in name:
            continue  # shortcut path, never wider than the main path
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            k, s, d = (v if isinstance(v, int) else v[0] for v in (m.kernel_size, m.stride, m.dilation))
            rf += (k - 1) * d * stride
            stride *= s
    return rf, stride


def _crop_margins(model, image_size, grid):
    """(stride, keep, margin) for cropped injection, or None if the model or grid does not allow it.

    keep: feature positions kept on each side of the tile; margin: pixels added on
    each side of the tile, so that the kept features never see the crop border.
    """
    from torchvision.models import ResNet
    if not isinstance(model, ResNet):
        return None
    rf, stride = _receptive_field(model)
    (W, H), (rows, cols) = image_size, grid
    if W % cols or H % rows or (W // cols) % stride or (H // rows) % stride:
        return None
    radius = (rf - 1) // 2
    keep = math.ceil(radius / stride)
    margin = math.ceil((keep * stride + radius) / stride) * stride
    return stride, keep, margin


def _features(model, x):
    """Last feature map of a torchvision ResNet (before pooling)."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


def _risk_map_crop(target_tiles, control_shuffled, model, transform, grid,
                   target_class, device, batch_size, read_score):
    """Risk map from one pass of the control and a crop around each injected tile.
    Returns None when cropping does not apply (the caller then uses the full passes)."""
    margins = _crop_margins(model, control_shuffled.size, grid)
    if margins is None:
        return None
    stride, keep, margin = margins
    rows, cols = grid
    W, H = control_shuffled.size
    tw, th = W // cols, H // rows

    with torch.no_grad():
        x = transform(control_shuffled)
        if tuple(x.shape[-2:]) != (H, W):
            return None  # transform resizes: crops would not match the full image
        base = _features(model, x.unsqueeze(0).to(device))
        fh, fw = base.shape[-2:]
        baseline = float(read_score(model.fc(torch.flatten(model.avgpool(base), 1)))[0, target_class])

        # group tiles by crop size so that every batch has one shape
        jobs = {}
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * tw, r * th
                box = (max(0, x0 - margin), max(0, y0 - margin),
                       min(W, x0 + tw + margin), min(H, y0 + th + margin))
                kept = (max(0, x0 // stride - keep), max(0, y0 // stride - keep),
                        min(fw, (x0 + tw) // stride + keep), min(fh, (y0 + th) // stride + keep))
                jobs.setdefault((box[2] - box[0], box[3] - box[1]), []).append((r, c, box, kept))

        risk_map = np.zeros((rows, cols), dtype=np.float32)
        for items in jobs.values():
            for start in range(0, len(items), batch_size):
                part = items[start:start + batch_size]
                crops = []
                for r, c, box, _ in part:
                    crop = control_shuffled.crop(box)
                    crop.paste(target_tiles[r * cols + c], (c * tw - box[0], r * th - box[1]))
                    crops.append(transform(crop))
                feats = _features(model, torch.stack(crops).to(device))
                for i, (r, c, box, (kx0, ky0, kx1, ky1)) in enumerate(part):
                    f = base.clone()
                    ox, oy = box[0] // stride, box[1] // stride
                    f[0, :, ky0:ky1, kx0:kx1] = feats[i, :, ky0 - oy:ky1 - oy, kx0 - ox:kx1 - ox]
                    injected = float(read_score(model.fc(torch.flatten(model.avgpool(f), 1)))[0, target_class])
                    risk_map[r, c] = injected - baseline
    return risk_map