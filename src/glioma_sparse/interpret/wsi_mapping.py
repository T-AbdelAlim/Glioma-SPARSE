"""
WSI <-> thumbnail coordinate mapping.

Each thumbnail produced by create_wsi_thumbnail.py has a sibling .json
sidecar that records every parameter needed to map a thumbnail-pixel
bounding box back to WSI level-0 pixel coordinates. This module loads
that sidecar and performs the mapping.

The coordinate chain (mirroring create_wsi_thumbnail.py) is:

    WSI level 0 (W0, H0) at base_mpp
        |   resample by r = target_mpp / base_mpp
        v
    Tissue image (Wt, Ht) at target_mpp
        |   pad to S = max(Wt, Ht), tissue placed at (off_x, off_y)
        v
    Square canvas (S, S)
        |   resize by f = T / S
        v
    Thumbnail (T, T)
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple, List
import json


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class ThumbMapping:
    wsi_path: str
    wsi_level0_dim: List[int]                # [W0, H0] - the TRUE full-slide
                                              # level-0 dimensions, always,
                                              # regardless of whether a crop
                                              # was used (see wsi_crop_origin)
    base_mpp: Optional[float]
    target_mpp: Optional[float]
    tissue_image_dim: List[int]              # [Wt, Ht]
    canvas_size: int                         # S = max(Wt, Ht)
    tissue_offset_in_canvas: List[int]       # [off_x, off_y]
    thumbnail_size: int                      # T (typically 2048)
    # Additive field. [0, 0] (the default) reproduces the exact original
    # behavior: tissue_image_dim was read starting at WSI level-0 (0, 0), as
    # create_wsi_thumbnail always does for .svs/.ndpi/etc. A non-zero value
    # means the region was cropped starting at this level-0 pixel location
    # BEFORE resampling (the .mrxs tissue-crop fix), so it must be added back
    # after the tissue-image-local -> WSI transform to recover true absolute
    # WSI level-0 coordinates. Old sidecars (no such key) load with [0, 0]
    # via .get(), so every existing .svs/.ndpi thumbnail needs no changes.
    wsi_crop_origin: List[int] = field(default_factory=lambda: [0, 0])

    def has_wsi_mapping(self):
        return self.base_mpp is not None and self.target_mpp is not None


# ============================================================
# LOAD / SAVE
# ============================================================

def save_mapping(mapping, thumbnail_path):
    thumbnail_path = Path(thumbnail_path)
    sidecar_path = thumbnail_path.with_suffix(".json")
    with open(sidecar_path, "w") as f:
        json.dump(asdict(mapping), f, indent=2)
    return sidecar_path


def load_mapping(thumbnail_path):
    thumbnail_path = Path(thumbnail_path)
    sidecar_path = thumbnail_path.with_suffix(".json")

    if not sidecar_path.exists():
        raise FileNotFoundError(
            "No mapping sidecar found at {}. The thumbnail may have been generated "
            "before sidecar support was added; re-run preprocessing for this slide."
            .format(sidecar_path)
        )

    with open(sidecar_path, "r") as f:
        data = json.load(f)

    return ThumbMapping(
        wsi_path=data["wsi_path"],
        wsi_level0_dim=list(data["wsi_level0_dim"]),
        base_mpp=data["base_mpp"],
        target_mpp=data["target_mpp"],
        tissue_image_dim=list(data["tissue_image_dim"]),
        canvas_size=int(data["canvas_size"]),
        tissue_offset_in_canvas=list(data["tissue_offset_in_canvas"]),
        thumbnail_size=int(data["thumbnail_size"]),
        wsi_crop_origin=list(data.get("wsi_crop_origin", [0, 0])),
    )


# ============================================================
# COORDINATE TRANSFORM
# ============================================================

def thumbnail_bbox_to_wsi(mapping, bbox_thumb):
    """
    Convert a thumbnail-pixel bounding box to WSI level-0 pixel coordinates.

    Args:
        mapping: ThumbMapping for the slide
        bbox_thumb: (px_left, py_top, px_right, py_bottom) in thumbnail coords [0, T]

    Returns:
        (wx, wy, ww, wh) at WSI level 0, suitable for slide.read_region((wx, wy), 0, (ww, wh)),
        or None if the bbox falls entirely in the padding region (no tissue).
    """
    if not mapping.has_wsi_mapping():
        return None

    px_l, py_t, px_r, py_b = bbox_thumb

    S = float(mapping.canvas_size)
    T = float(mapping.thumbnail_size)
    Wt, Ht = mapping.tissue_image_dim
    off_x, off_y = mapping.tissue_offset_in_canvas

    # Step 1: thumbnail -> canvas
    cx_l = px_l * S / T
    cy_t = py_t * S / T
    cx_r = px_r * S / T
    cy_b = py_b * S / T

    # Step 2: canvas -> tissue (subtract padding offset)
    tx_l = cx_l - off_x
    ty_t = cy_t - off_y
    tx_r = cx_r - off_x
    ty_b = cy_b - off_y

    # Step 3: clip to actual tissue bounds; empty intersection -> all padding
    tx_l = max(0.0, min(tx_l, Wt))
    tx_r = max(0.0, min(tx_r, Wt))
    ty_t = max(0.0, min(ty_t, Ht))
    ty_b = max(0.0, min(ty_b, Ht))

    if tx_r <= tx_l or ty_b <= ty_t:
        return None

    # Step 4: tissue -> WSI level 0 (still relative to wherever the read
    # started - (0,0) for the original code, or a crop origin for a
    # tissue-cropped .mrxs thumbnail)
    r = mapping.target_mpp / mapping.base_mpp
    wx = int(round(tx_l * r))
    wy = int(round(ty_t * r))
    ww = int(round((tx_r - tx_l) * r))
    wh = int(round((ty_b - ty_t) * r))

    # Step 5: add back the crop origin to get TRUE absolute WSI level-0
    # coordinates. [0, 0] for the original (uncropped) code path, a no-op.
    off_x0, off_y0 = mapping.wsi_crop_origin
    wx += off_x0
    wy += off_y0

    # Final safety clamp to level-0 bounds (always the TRUE full-slide
    # dimensions, regardless of whether a crop was used)
    W0, H0 = mapping.wsi_level0_dim
    wx = max(0, min(wx, W0 - 1))
    wy = max(0, min(wy, H0 - 1))
    ww = max(1, min(ww, W0 - wx))
    wh = max(1, min(wh, H0 - wy))

    return (wx, wy, ww, wh)


def grid_cell_bbox(row, col, n_rows, n_cols, thumbnail_size):
    """
    Return the thumbnail-pixel bounding box of grid cell (row, col).

    For an 8x8 grid on a 2048x2048 thumbnail, cell (0, 0) is (0, 0, 256, 256).
    """
    tile_w = thumbnail_size // n_cols
    tile_h = thumbnail_size // n_rows
    px_l = col * tile_w
    py_t = row * tile_h
    px_r = px_l + tile_w
    py_b = py_t + tile_h
    return (px_l, py_t, px_r, py_b)
