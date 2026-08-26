"""
Top-k high-resolution patch extraction.

Given a Stage A risk map and the thumbnail's mapping sidecar, this module
picks the top-k most informative tiles, maps each one back to the WSI
level-0 coordinates, calls openslide.read_region, and saves each patch
at a fixed output resolution (default 2048x2048, ~8x zoom relative to the
thumbnail tile).

Tiles whose thumbnail content is mostly padding/background are filtered
out before ranking, so that white-padding artefacts cannot win the top-k.

Optionally re-stains each patch (Macenko, see preprocessing.stain_normalization)
against a named profile's Stage-B reference; stain_profile=None is a no-op.
"""

from pathlib import Path
import numpy as np
import openslide
from PIL import Image

from glioma_sparse.interpret.wsi_mapping import (
    load_mapping,
    thumbnail_bbox_to_wsi,
    grid_cell_bbox,
)
from glioma_sparse.preprocessing.stain_normalization import normalize_with_profile


# ============================================================
# TISSUE FILTER (per tile)
# ============================================================

def _tile_tissue_fraction(thumbnail, row, col, grid):
    """Fraction of tissue pixels inside grid cell (row, col) of the thumbnail."""
    rows, cols = grid
    w, h = thumbnail.size
    tile_w = w // cols
    tile_h = h // rows

    left = col * tile_w
    upper = row * tile_h
    tile = thumbnail.crop((left, upper, left + tile_w, upper + tile_h))

    arr = np.array(tile).astype(np.float32) / 255.0
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    saturation = (maxc - minc) / (maxc + 1e-8)
    value = maxc
    mask = (value < 0.9) & (saturation > 0.05)
    return float(mask.sum()) / float(mask.size)


# ============================================================
# MAIN
# ============================================================

def extract_topk_patches(
    thumbnail_path,
    risk_map,
    k=5,
    output_size=2048,
    output_dir=None,
    grid=None,
    min_tissue_fraction=0.5,
    slide_id=None,
    label=None,
    stain_profile=None,
):
    """
    Extract the top-k highest-risk tiles from the WSI at high resolution.

    Args:
        thumbnail_path: path to the thumbnail JPG (with a sibling .json sidecar)
        risk_map: 2D ndarray (rows, cols) with importance scores
        k: number of patches to extract
        output_size: every high-res patch is resized to (output_size, output_size)
        output_dir: where to save patches; defaults to thumbnail_path.parent / "highres"
        grid: (rows, cols) override; defaults to risk_map.shape
        min_tissue_fraction: tiles with thumbnail tissue fraction below this
                             are excluded from selection (prevents picking padding)
        slide_id: identifier used in output filenames; defaults to thumbnail stem
        label: optional class label embedded in the filename (for Stage B labelling)
        stain_profile: name of a fitted profile (see stain_normalization.list_profiles())
                       to normalize each patch against; None applies no correction

    Returns:
        list of dicts, one per extracted patch, with keys:
            rank, row, col, risk, tissue_fraction, wsi_bbox, output_path
    """
    thumbnail_path = Path(thumbnail_path)
    risk_map = np.asarray(risk_map)

    if grid is None:
        grid = (int(risk_map.shape[0]), int(risk_map.shape[1]))
    rows, cols = grid

    # Load the sidecar
    mapping = load_mapping(thumbnail_path)
    if not mapping.has_wsi_mapping():
        raise RuntimeError(
            "Thumbnail {} has no WSI mapping (base_mpp was None at preprocessing). "
            "Cannot extract high-resolution patches.".format(thumbnail_path)
        )

    # Load thumbnail for tissue filtering
    thumbnail = Image.open(thumbnail_path).convert("RGB")

    # Output directory
    if output_dir is None:
        output_dir = thumbnail_path.parent / "highres"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if slide_id is None:
        slide_id = thumbnail_path.stem

    # Score every tile, drop tiles that are mostly padding/background
    candidates = []
    for r in range(rows):
        for c in range(cols):
            tissue_frac = _tile_tissue_fraction(thumbnail, r, c, grid)
            if tissue_frac < min_tissue_fraction:
                continue
            candidates.append((r, c, float(risk_map[r, c]), tissue_frac))

    # Sort by risk descending
    candidates.sort(key=lambda x: -x[2])
    selected = candidates[:k]

    if len(selected) == 0:
        print("WARNING: no tiles met the tissue threshold for {}".format(thumbnail_path))
        return []

    # Open the WSI once
    slide = openslide.OpenSlide(mapping.wsi_path)

    results = []
    try:
        for rank, (r, c, risk, tissue_frac) in enumerate(selected):
            # Grid cell -> thumbnail bbox
            bbox_thumb = grid_cell_bbox(r, c, rows, cols, mapping.thumbnail_size)

            # Thumbnail bbox -> WSI level-0 bbox
            wsi_bbox = thumbnail_bbox_to_wsi(mapping, bbox_thumb)
            if wsi_bbox is None:
                # Entirely in padding; should be rare after tissue filter
                continue

            wx, wy, ww, wh = wsi_bbox
            patch = slide.read_region((wx, wy), 0, (ww, wh)).convert("RGB")

            # Standardise output size
            if patch.size != (output_size, output_size):
                patch = patch.resize((output_size, output_size), Image.BICUBIC)

            # Stain normalization (named profile's Stage-B reference, if given)
            if stain_profile:
                patch = Image.fromarray(
                    normalize_with_profile(np.array(patch), stain_profile, stage="B"))

            # Save
            label_part = "_{}".format(label) if label is not None else ""
            out_name = "{}_rank{:02d}_r{}c{}{}.jpg".format(
                slide_id, rank + 1, r, c, label_part
            )
            out_path = output_dir / out_name
            patch.save(out_path, quality=90)

            results.append({
                "rank": rank + 1,
                "row": r,
                "col": c,
                "risk": risk,
                "tissue_fraction": tissue_frac,
                "wsi_bbox": list(wsi_bbox),
                "output_path": str(out_path),
            })
    finally:
        slide.close()

    return results
