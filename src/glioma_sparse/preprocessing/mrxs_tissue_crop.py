#!/usr/bin/env python3
r"""
.mrxs-only fix for the low tissue/background ratio problem, plus black
scan-artifact cleanup for high-resolution patches.

WHY THIS HAPPENS
    .mrxs (MIRAX) files store the ENTIRE scan bed as their level-0 extent,
    including the large blank slide-holder margin around the tissue. .svs
    (Aperio) and .ndpi (Hamamatsu) files are pre-cropped much closer to the
    tissue. Thumbnailing the full slide extent at a fixed target_mpp/output
    size means .mrxs tissue only fills a small corner of the canvas, while
    .svs/.ndpi tissue fills most of it.

WHAT THIS FILE PROVIDES
    tissue_bbox_from_overview(img)   - core tissue-bbox detection, no
                                        OpenSlide dependency, independently
                                        testable.
    detect_tissue_bbox_wsi(path)     - OpenSlide wrapper: fast low-res
                                        overview -> bbox in level-0 pixels.
    create_wsi_thumbnail_smart(...)  - drop-in replacement for
                                        create_wsi_thumbnail() that crops to
                                        the detected bbox first, reusing the
                                        ORIGINAL function's own helpers
                                        (get_base_mpp, pick_level,
                                        create_tissue_mask,
                                        pad_with_background_color,
                                        write_mapping_sidecar) so behavior is
                                        identical except for WHERE it reads.
    mask_black_artifacts(img)        - detect near-pure-black scan-artifact
                                        pixels and replace them with a fixed
                                        background color (not a per-tile
                                        median - see its own docstring).

THIS FILE NEVER RUNS FOR .svs/.ndpi/etc
    Dispatch happens in thumbnail_auto.py, purely on file extension. This
    module's functions are only ever CALLED when the slide is .mrxs; nothing
    here executes for other formats, so create_wsi_thumbnail.py's existing
    behavior for .svs/.ndpi is completely unaffected.

DESIGN NOTES CARRIED FORWARD FROM EARLIER DEBUGGING (so they aren't
re-broken by a future edit)
  1. Otsu on the saturation channel can degenerate when a tile has (almost)
     only one distinct saturation value - the ordinary "find the split that
     maximizes between-class variance" loop can end with an uninitialized
     threshold, misclassifying an all-background or all-tissue tile. Handled
     explicitly below.
  2. A single min()/max() bounding box over "everything above the saturation
     threshold" is defenseless against a THIN, elongated mark (a scan
     registration line, a ruler tick) - it's not small, just thin, so a
     simple speck-removal filter doesn't catch it, and even one such mark
     near an edge silently drags the whole bbox out to include it. Fixed
     with connected-component + fill-ratio filtering: real tissue is a
     solid blob (high fill ratio within its own local bounding box); a thin
     mark has a bounding box much larger than its actual pixel count.
  3. pad_with_background_color() (in the ORIGINAL create_wsi_thumbnail.py)
     estimates background color by sampling the image's four corners,
     assuming corners are always background. That's safe for a full,
     uncropped slide read, but NOT for a tightly-cropped region: a crop
     chosen to tightly bound tissue can have tissue right at its own
     corners, pulling the estimate toward tissue color instead of true
     background. Fixed by passing an explicit, reliable near-white color
     instead of relying on that fallback for cropped reads.
  4. Padding must run BEFORE black-artifact cleanup, not after: padding
     detects true out-of-bounds regions by looking for exact pure black.
     Cleaning artifacts first would consume that same signal, leaving
     padding with nothing to whiten.
  5. mask_black_artifacts() must fill with a FIXED background color, not the
     median of the tile's own non-artifact pixels - for a high-res Stage B
     patch (selected because it's mostly tissue), that median is itself a
     tissue color, producing a fabricated-tissue-looking fill instead of a
     clean "no data here" region.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# CORE TISSUE-BBOX DETECTION (no OpenSlide dependency - independently
# testable)
# ============================================================

def _otsu_threshold(values, bins=256):
    """Standard Otsu threshold on a float array in [0, 1]."""
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.1
    sum_all = np.sum(hist * np.arange(bins))
    sum_bg, w_bg, best_thr, best_var = 0.0, 0.0, 0.0, -1.0
    for t in range(bins):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        # >= not >: when there's a gap of empty bins between the background
        # and tissue clusters, every t in that gap ties for the same
        # var_between. With strict >, the FIRST t in the tie wins, landing
        # the threshold at the edge of the background cluster itself. With
        # >=, the LAST t wins, correctly pushing the threshold to just
        # before the tissue cluster begins.
        if var_between >= best_var:
            best_var = var_between
            best_thr = t

    if best_var < 0:
        # DEGENERATE CASE: every pixel has (almost) the same saturation, so
        # there is no second cluster to split against (the w_fg==0 break
        # fires on the very first populated bin). Real scans almost always
        # have enough per-pixel noise to avoid this, but a uniform test
        # image or an unusually flat slide can hit it. Fall back to a fixed
        # absolute reference: true background saturation is reliably very
        # low regardless of scanner, so 0.05 cleanly separates "uniformly
        # background" from "uniformly tissue" without needing a real split.
        return 0.05

    margin = 1.5 / (bins - 1)   # small safety margin above the raw split
    return best_thr / (bins - 1) + margin


def tissue_bbox_from_overview(img, pad_frac=0.03, min_area_frac=0.0008,
                              sat_thresh=None, min_fill_ratio=0.25,
                              return_debug=False):
    """Find the tissue bounding box in a low-res overview image (PIL RGB).

    Uses HSV saturation, not grayscale: true glass/background is close to
    zero saturation regardless of brightness, while stained tissue (H&E or
    otherwise) has real saturation - robust across scanner formats.

    Returns (x0, y0, x1, y1) as FRACTIONS of the overview's (width, height).
    Returns None if no tissue is found (fully blank overview).
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mx = arr.max(axis=2); mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0.0)

    if sat_thresh is None:
        sat_thresh = _otsu_threshold(sat)

    mask = sat > sat_thresh

    # remove small specks (dust, single-pixel artifacts)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = np.asarray(mask_img) > 127

    total_px = mask.size
    if mask.sum() < min_area_frac * total_px:
        # Otsu found almost nothing above threshold. Usually genuinely
        # blank, but can also mean the WHOLE frame is tissue with no
        # background contrast at all (no split possible either way).
        # Distinguish via the raw median saturation across the whole frame.
        if np.median(sat) > 0.06:
            bbox_frac = (0.0, 0.0, 1.0, 1.0)
            return bbox_frac if not return_debug else (bbox_frac, np.ones_like(mask), sat_thresh)
        return None if not return_debug else (None, mask, sat_thresh)

    # Connected-component filtering: exclude thin marks (registration lines,
    # ruler ticks) that survive speck removal because they're thin, not
    # small, by judging SHAPE (fill ratio), not just size. Real tissue
    # fragments are solid blobs; thin marks have low fill ratio within their
    # own local bounding box. Components are unioned if they pass both a
    # minimum size AND the fill-ratio test, so legitimately separate tissue
    # fragments (multiple biopsy pieces) are still all included.
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        kept_boxes = []
        min_component_px = max(8, int(min_area_frac * total_px))
        for lbl in range(1, n + 1):
            ys, xs = np.where(labeled == lbl)
            area = len(ys)
            if area < min_component_px:
                continue
            cy0, cy1 = ys.min(), ys.max() + 1
            cx0, cx1 = xs.min(), xs.max() + 1
            local_bbox_area = (cy1 - cy0) * (cx1 - cx0)
            fill_ratio = area / max(1, local_bbox_area)
            if fill_ratio >= min_fill_ratio:
                kept_boxes.append((cy0, cy1, cx0, cx1))
        if kept_boxes:
            y0 = min(b[0] for b in kept_boxes); y1 = max(b[1] for b in kept_boxes)
            x0 = min(b[2] for b in kept_boxes); x1 = max(b[3] for b in kept_boxes)
        else:
            # nothing passed the fill-ratio test (everything was thin marks)
            ys, xs = np.where(mask)
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
    except ImportError:
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1

    h, w = mask.shape
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = max(0, x0 - pw); x1 = min(w, x1 + pw)
    y0 = max(0, y0 - ph); y1 = min(h, y1 + ph)

    bbox_frac = (x0 / w, y0 / h, x1 / w, y1 / h)
    return bbox_frac if not return_debug else (bbox_frac, mask, sat_thresh)


# ============================================================
# OPENSLIDE WRAPPER
# ============================================================

_BBOX_CACHE = {}   # slide_path (str) -> (x0,y0,w,h) in level-0 px, or None.
                   # Speed: probing is already cheap (small overview only),
                   # but this avoids repeating it if the same slide is
                   # touched more than once in a process (e.g. import, then
                   # later a methodology-walkthrough rebuild on the same
                   # preset slide).

def detect_tissue_bbox_wsi(slide_path, probe_size=1024, use_cache=True, **kwargs):
    """Open the slide, grab a fast low-res overview (OpenSlide's own
    get_thumbnail, efficient for every format), detect the tissue bbox, and
    map back to level-0 pixel coordinates.

    Returns (x0, y0, w, h) in level-0 pixels, or None if no tissue detected.
    """
    key = str(Path(slide_path).resolve())
    if use_cache and key in _BBOX_CACHE:
        return _BBOX_CACHE[key]

    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    try:
        level0_w, level0_h = slide.dimensions
        overview = slide.get_thumbnail((probe_size, probe_size))
        bbox_frac = tissue_bbox_from_overview(overview, **kwargs)
        if bbox_frac is None:
            result = None
        else:
            fx0, fy0, fx1, fy1 = bbox_frac
            x0 = int(fx0 * level0_w); y0 = int(fy0 * level0_h)
            x1 = int(fx1 * level0_w); y1 = int(fy1 * level0_h)
            result = (x0, y0, x1 - x0, y1 - y0)
    finally:
        slide.close()

    if use_cache:
        _BBOX_CACHE[key] = result
    return result


# ============================================================
# BLACK SCAN-ARTIFACT CLEANUP
# ============================================================

def mask_black_artifacts(img, black_thresh=10, fill_color=(245, 245, 245),
                         return_fraction=True):
    """Detect near-pure-black scan-artifact pixels and replace them with a
    consistent background color.

    Args:
        img: PIL RGB image (a tile, patch, or full thumbnail)
        black_thresh: a pixel is "artifact" if ALL THREE channels are below
            this value (0-255). Kept low (default 10) so genuine dark
            hematoxylin-stained tissue - which always has real channel
            separation (e.g. dark purple, not achromatic near-zero) - is
            never misclassified as artifact. Only a hard digital drop-out
            (typically exactly (0,0,0) or very close) triggers this.
        fill_color: fixed color artifact pixels are replaced with, near-white
            by default. NOT computed as a median of the tile's own
            non-artifact pixels: for a high-res Stage B patch (selected
            because it's mostly tissue), that median is itself a tissue
            color, so the artifact region would be replaced with
            fabricated-looking tissue rather than reading clearly as
            "no data here." Pass an explicit color if a better-known
            background estimate is available for this slide.
        return_fraction: if True, also return the fraction of the image that
            was artifact (0.0 if none found).

    Returns:
        (cleaned_img, artifact_fraction) if return_fraction else cleaned_img.
        If no artifact pixels are found, returns the image UNCHANGED (same
        object) - a true no-op when there's nothing to fix.
    """
    arr = np.asarray(img.convert("RGB"))
    is_artifact = np.all(arr < black_thresh, axis=2)
    frac = float(is_artifact.mean())

    if frac == 0.0:
        return (img, 0.0) if return_fraction else img

    clean = arr.copy()
    clean[is_artifact] = np.array(fill_color, dtype=np.uint8)
    cleaned_img = Image.fromarray(clean, mode="RGB")
    return (cleaned_img, frac) if return_fraction else cleaned_img


# ============================================================
# MAIN THUMBNAIL FUNCTION
# ============================================================

def create_wsi_thumbnail_smart(slide_path, output_path=None, target_mpp=4.0,
                               output_size=2048, pad_frac=0.03, save_mask=False,
                               clean_artifacts=True, verbose=True):
    """Drop-in replacement for create_wsi_thumbnail() that reads only the
    detected tissue bounding box instead of the whole slide from (0,0).

    Mirrors create_wsi_thumbnail.py's ACTUAL structure exactly (reusing its
    own helpers: get_base_mpp, pick_level, create_tissue_mask,
    estimate_fraction, pad_with_background_color, write_mapping_sidecar), so
    behavior matches it precisely except for WHERE the read starts. Writes a
    sidecar compatible with wsi_mapping.load_mapping()/thumbnail_bbox_to_wsi(),
    including wsi_crop_origin (the true level-0 offset of the crop) so
    downstream patch extraction reads from the correct absolute location.

    Same return signature as the original: (thumbnail, tissue_fraction,
    effective_fraction).
    """
    import openslide
    from glioma_sparse.preprocessing.create_wsi_thumbnail import (
        get_base_mpp, get_background_color_from_props, pick_level,
        create_tissue_mask, estimate_fraction, pad_with_background_color,
        write_mapping_sidecar,
    )

    slide_path = Path(slide_path)
    if not slide_path.exists():
        raise FileNotFoundError(slide_path)

    slide = openslide.OpenSlide(str(slide_path))
    props = slide.properties
    base_mpp = get_base_mpp(props)
    bg_hint = get_background_color_from_props(props)
    wsi_level0_dim = list(slide.level_dimensions[0])

    bbox = detect_tissue_bbox_wsi(slide_path, pad_frac=pad_frac)
    if bbox is None:
        # no tissue detected at all: fall back to the full slide from (0,0),
        # matching the original function's behavior in that case
        x0, y0, bw, bh = 0, 0, wsi_level0_dim[0], wsi_level0_dim[1]
        if verbose:
            print(f"[mrxs_tissue_crop] {slide_path.name}: no tissue detected "
                  f"by the crop probe; using the full slide extent.")
    else:
        x0, y0, bw, bh = bbox
        if verbose:
            frac = (bw * bh) / (wsi_level0_dim[0] * wsi_level0_dim[1])
            print(f"[mrxs_tissue_crop] {slide_path.name}: cropping to "
                  f"tissue bbox ({bw}x{bh} px, {frac*100:.1f}% of the "
                  f"level-0 canvas) instead of the full slide.")

    # --------------------------------------------------------
    # READ IMAGE (only the tissue bbox, not the whole slide from (0,0))
    # --------------------------------------------------------
    if base_mpp is None:
        level = min(2, slide.level_count - 1)
        ds_level = slide.level_downsamples[level]
        read_w = max(1, int(bw / ds_level))
        read_h = max(1, int(bh / ds_level))
        img = slide.read_region((x0, y0), level, (read_w, read_h))
        recorded_target_mpp = None
    else:
        target_ds = target_mpp / base_mpp
        level = pick_level(slide, target_ds)
        ds = slide.level_downsamples[level]
        scale = target_ds / ds

        read_w = max(1, int(bw / ds))
        read_h = max(1, int(bh / ds))
        img = slide.read_region((x0, y0), level, (read_w, read_h))

        if abs(scale - 1.0) > 0.01:
            new_w = max(1, int(read_w / scale))
            new_h = max(1, int(read_h / scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)

        recorded_target_mpp = float(target_mpp)

    img = img.convert("RGB")

    tissue_w, tissue_h = img.size
    canvas_size = max(tissue_w, tissue_h)
    tissue_offset = ((canvas_size - tissue_w) // 2, (canvas_size - tissue_h) // 2)

    # --------------------------------------------------------
    # TISSUE FRACTION (BEFORE PADDING) - identical to the original
    # --------------------------------------------------------
    mask_before = create_tissue_mask(img)
    tissue_fraction = estimate_fraction(mask_before)

    # --------------------------------------------------------
    # POSTPROCESSING - identical to the original, EXCEPT the background
    # color hint. pad_with_background_color falls back to sampling the
    # image's four corners when no bg_color_hint is given, assuming corners
    # are safely background - true for a full, uncropped slide read, but NOT
    # for our tightly-cropped region: a crop chosen to tightly bound tissue
    # can very plausibly have tissue right at its own corners. When that
    # happens the corner estimate gets pulled toward tissue color instead of
    # true background, and that wrong color then fills both the real
    # out-of-bounds padding and the square-canvas letterbox bars. H&E slide
    # backgrounds (glass + mounting medium) are reliably near-white
    # regardless of scanner, so default to that explicitly here.
    # --------------------------------------------------------
    bg_hint_safe = bg_hint if bg_hint is not None else (245, 245, 245)
    img = pad_with_background_color(img, bg_color_hint=bg_hint_safe)

    # Black scan-artifact cleanup runs AFTER padding, not before: padding
    # detects true out-of-bounds regions by looking for EXACT pure black.
    # Cleaning artifacts first would consume that same signal, leaving
    # padding with nothing left to whiten.
    if clean_artifacts:
        img, artifact_frac = mask_black_artifacts(img)
        if artifact_frac > 0 and verbose:
            print(f"[mrxs_tissue_crop] {slide_path.name}: cleaned "
                  f"{artifact_frac*100:.1f}% black scan-artifact pixels "
                  f"from the thumbnail")

    img = img.resize((output_size, output_size), Image.BICUBIC)

    # --------------------------------------------------------
    # EFFECTIVE FRACTION (AFTER RESIZE) - identical to the original
    # --------------------------------------------------------
    mask_after = create_tissue_mask(img)
    effective_fraction = estimate_fraction(mask_after)

    # --------------------------------------------------------
    # SAVE OUTPUT (+ MAPPING SIDECAR, with the crop origin recorded)
    # --------------------------------------------------------
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=90)

        if save_mask:
            mask_img = Image.fromarray((mask_after * 255).astype(np.uint8))
            mask_img.save(output_path.with_name("{}_mask.png".format(output_path.stem)))

        sidecar = {
            "wsi_path": str(slide_path.resolve()),
            "wsi_level0_dim": wsi_level0_dim,          # TRUE full-slide dims
            "base_mpp": float(base_mpp) if base_mpp is not None else None,
            "target_mpp": recorded_target_mpp,
            "tissue_image_dim": [int(tissue_w), int(tissue_h)],
            "canvas_size": int(canvas_size),
            "tissue_offset_in_canvas": [int(tissue_offset[0]), int(tissue_offset[1])],
            "thumbnail_size": int(output_size),
            "wsi_crop_origin": [int(x0), int(y0)],     # where the crop
                                                        # started, true
                                                        # level-0 pixels
        }
        write_mapping_sidecar(output_path, sidecar)

    slide.close()
    return img, tissue_fraction, effective_fraction
