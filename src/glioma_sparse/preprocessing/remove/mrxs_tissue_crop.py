#!/usr/bin/env python3
r"""
Fix for the .mrxs low tissue/white ratio problem.

WHY THIS HAPPENS (confirmed from your example thumbnails)
    .mrxs (MIRAX) files store the ENTIRE scan bed as their level-0 extent,
    including the large blank slide-holder margin around the tissue. .svs
    (Aperio) and .ndpi (Hamamatsu) files are typically pre-cropped much closer
    to the tissue during scanning or export. Your thumbnail step resizes the
    FULL slide bounding box down to a fixed output_size x output_size canvas at
    a fixed target_mpp, so for .mrxs slides the real tissue only fills a small
    corner of that canvas (exactly what your GL_S01 example shows). Most 8x8
    grid cells then land on pure background, fail the min_tissue threshold at
    extraction, and if too few cells pass, there is nothing left to reach
    Stage B, or Stage B receives an unrepresentative handful of patches.

THE FIX
    Detect the tissue bounding box BEFORE thumbnailing (format-agnostic, works
    the same for .mrxs/.svs/.ndpi), then thumbnail only that region. This is
    independent of scanner format because it works on pixel content
    (background is white/pale, low saturation, regardless of which scanner
    produced the file), not on file-format metadata.

IMPORTANT: THIS PRESERVES TARGET_MPP, IT DOES NOT CHANGE SCALE
    The crop only removes excess non-tissue margin. The cropped region is
    still thumbnailed at the SAME target_mpp your models were trained with,
    then centered (not stretched) onto the output_size x output_size canvas.
    So this is a preprocessing fix, not a scale change, and should not require
    retraining. Still, re-validate Stage A/B on a handful of .mrxs slides after
    switching, since this is new inference-time code path.

WHAT THIS FILE PROVIDES
    tissue_bbox_from_overview(img)       - core detection, works on any PIL
                                            image, no OpenSlide dependency,
                                            independently testable.
    detect_tissue_bbox_wsi(slide_path)   - OpenSlide wrapper: fast low-res
                                            overview -> bbox in level-0 pixels.
    create_wsi_thumbnail_smart(...)      - drop-in replacement for
                                            create_wsi_thumbnail() that crops
                                            to the detected tissue bbox first.

INTEGRATION
    In backend.py / your training preprocessing, replace:
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
        thumbnail, tissue_frac, eff_frac = create_wsi_thumbnail(slide_path, ...)
    with:
        from mrxs_tissue_crop import create_wsi_thumbnail_smart
        thumbnail, tissue_frac, eff_frac = create_wsi_thumbnail_smart(slide_path, ...)
    Same call signature and return shape, so nothing downstream needs to change.
"""

import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

try:
    from .scan_artifacts import mask_black_artifacts
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scan_artifacts import mask_black_artifacts


# ============================================================
# CORE DETECTION (no OpenSlide dependency - independently testable)
# ============================================================

def tissue_bbox_from_overview(img, pad_frac=0.03, min_area_frac=0.0008,
                              sat_thresh=None, min_fill_ratio=0.25,
                              return_debug=False):
    """Find the tissue bounding box in a low-res overview image (PIL RGB).

    Uses HSV saturation, not grayscale, because it is robust to both very
    pale/washed-out tissue and to grayish scan-bed backgrounds: true glass/
    background is close to zero saturation regardless of its brightness,
    while stained tissue (H&E pink/purple, or anything else) has real
    saturation. This is why it works the same for any scanner format.

    Returns (x0, y0, x1, y1) as FRACTIONS of the overview's (width, height),
    i.e. in [0, 1], so the caller can map to level-0 pixels at any scale.
    Returns None if no tissue is found (fully blank overview).
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mx = arr.max(axis=2); mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0.0)

    if sat_thresh is None:
        # Otsu on the saturation channel. Background clusters tightly near 0,
        # tissue has a broad higher-saturation distribution, so Otsu on this
        # single channel separates them reliably without per-slide tuning.
        sat_thresh = _otsu_threshold(sat)

    mask = sat > sat_thresh

    # remove small specks (dust, single-pixel artifacts) that would otherwise
    # pull the bounding box out toward noise far from the tissue
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    mask = np.asarray(mask_img) > 127

    total_px = mask.size
    if mask.sum() < min_area_frac * total_px:
        # Otsu found almost nothing above threshold. This is normally a
        # genuinely blank/background-only overview, BUT it can also happen
        # in the opposite degenerate case: the WHOLE frame is tissue with no
        # background at all (e.g. a needle biopsy scanned edge-to-edge, or a
        # synthetic test image). With no contrast anywhere, Otsu's threshold
        # sits just above the single uniform cluster, so "sat > thresh" ends
        # up excluding everything, misreporting "no tissue" for a frame that
        # is actually ALL tissue. Distinguish the two: if the median
        # saturation across the WHOLE overview is already clearly non-
        # background-like, treat the whole frame as tissue instead of
        # reporting none.
        if np.median(sat) > 0.06:
            bbox_frac = (0.0, 0.0, 1.0, 1.0)
            return bbox_frac if not return_debug else (bbox_frac, np.ones_like(mask), sat_thresh)
        return None if not return_debug else (None, mask, sat_thresh)

    # Connected-component filtering: a single mask.min()/max() bbox is
    # defenseless against a THIN mark (a scan registration line, a pen mark,
    # a ruler tick) that survives the speck filter above because it's thin
    # and elongated rather than small, not solid. Even one such mark near an
    # image edge silently drags the whole bbox out to include it, producing a
    # crop that's wide but not tight - exactly the "tissue looks small again"
    # symptom this whole module exists to fix, plus a much slower read since
    # far more of the slide gets processed than necessary.
    #
    # The fix: label connected components and keep only ones that are
    # actually solid blobs (real tissue), judged by fill ratio = pixel count
    # / the component's own local bounding-box area. A real tissue fragment
    # is a fairly solid shape (high fill ratio within its own bbox); a thin
    # line spanning many columns has a bbox area much larger than its actual
    # pixel count (low fill ratio). Components are combined (their bboxes
    # unioned) if they pass BOTH a minimum size and the fill-ratio test, so
    # legitimately separate tissue fragments (e.g. multiple biopsy pieces)
    # are still all included, while thin marks are excluded regardless of
    # how far they reach across the image.
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
            # nothing passed the fill-ratio test (everything detected was
            # thin marks, not solid tissue) - fall back to the plain mask
            # bbox rather than reporting no tissue at all
            ys, xs = np.where(mask)
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
    except ImportError:
        # scipy not available - fall back to the plain (less robust) bbox
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1

    h, w = mask.shape

    # pad a little so we don't clip tissue right at the edge
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = max(0, x0 - pw); x1 = min(w, x1 + pw)
    y0 = max(0, y0 - ph); y1 = min(h, y1 + ph)

    bbox_frac = (x0 / w, y0 / h, x1 / w, y1 / h)
    return bbox_frac if not return_debug else (bbox_frac, mask, sat_thresh)


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
        # NOTE: >= not > . When there's a gap of empty bins between the
        # background and tissue clusters (common, since real tissue/background
        # sit at clearly different saturations), every t in that gap ties for
        # the same var_between (nothing changes as empty bins are crossed).
        # With strict >, the FIRST t in that tie wins, which lands the
        # threshold right at the edge of the background cluster itself rather
        # than between the two clusters, misclassifying background as tissue.
        # With >=, the LAST t in the tie wins, pushing the threshold to just
        # before the tissue cluster begins, which is the correct split point.
        if var_between >= best_var:
            best_var = var_between
            best_thr = t

    if best_var < 0:
        # DEGENERATE CASE: every pixel has (essentially) the exact same
        # saturation value, so there is no second cluster to split against at
        # all (the w_fg==0 break above fires on the very first populated
        # bin). This happens for perfectly uniform test fills; real scans
        # almost always have enough per-pixel noise to avoid it, but it must
        # still be handled correctly rather than silently returning the
        # uninitialized default of 0 (which would misclassify a uniform
        # BACKGROUND-only image as 100% tissue). Fall back to a fixed
        # absolute reference: background saturation is reliably very low
        # (near 0) regardless of scanner, so 0.05 cleanly separates "uniformly
        # background" from "uniformly tissue" without needing a real split.
        return 0.05

    # small safety margin above the raw split, so real (noisy) background
    # pixels that scatter slightly above the ideal threshold still don't
    # get misclassified as tissue
    margin = 1.5 / (bins - 1)
    return best_thr / (bins - 1) + margin


# ============================================================
# OPENSLIDE WRAPPER (format-agnostic: works for .mrxs/.svs/.ndpi alike)
# ============================================================

def detect_tissue_bbox_wsi(slide_path, probe_size=1024, **kwargs):
    """Open the slide, grab a fast low-res overview (OpenSlide's own
    get_thumbnail, which is efficient for every format including .mrxs),
    detect the tissue bbox on it, and map back to level-0 pixel coordinates.

    Returns (x0, y0, w, h) in level-0 pixels, or None if no tissue detected.
    """
    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    try:
        level0_w, level0_h = slide.dimensions
        overview = slide.get_thumbnail((probe_size, probe_size))
        bbox_frac = tissue_bbox_from_overview(overview, **kwargs)
        if bbox_frac is None:
            return None
        fx0, fy0, fx1, fy1 = bbox_frac
        x0 = int(fx0 * level0_w); y0 = int(fy0 * level0_h)
        x1 = int(fx1 * level0_w); y1 = int(fy1 * level0_h)
        return (x0, y0, x1 - x0, y1 - y0)
    finally:
        slide.close()


def create_wsi_thumbnail_smart(slide_path, output_path=None, target_mpp=4.0,
                               output_size=2048, pad_frac=0.03, save_mask=False):
    """Drop-in replacement for create_wsi_thumbnail() that reads only the
    detected tissue bounding box instead of the whole slide from (0,0).

    This mirrors create_wsi_thumbnail.py's ACTUAL structure exactly (reusing
    its own helper functions: get_base_mpp, pick_level, create_tissue_mask,
    estimate_fraction, pad_with_background_color, write_mapping_sidecar), so
    behavior matches it precisely except for WHERE the read starts. The
    sidecar it writes is fully compatible with wsi_mapping.load_mapping() /
    thumbnail_bbox_to_wsi(), including the new wsi_crop_origin field that
    records the true level-0 pixel offset of the crop, so patch extraction
    downstream reads from the correct absolute WSI location, not (0,0) plus
    the offset.

    Same return signature as the original: (thumbnail, tissue_fraction,
    effective_fraction).
    """
    from PIL import Image
    import numpy as np
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
        # exactly matching the original function's behavior in that case
        x0, y0, bw, bh = 0, 0, wsi_level0_dim[0], wsi_level0_dim[1]
    else:
        x0, y0, bw, bh = bbox

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
    # POSTPROCESSING - identical to the original
    # --------------------------------------------------------
    # pad_with_background_color falls back to sampling the image's four
    # corners when no bg_color_hint is given, assuming corners are safely
    # background - true for a full, uncropped slide read, but NOT for our
    # tightly-cropped region: a crop chosen to tightly bound tissue can very
    # plausibly have tissue right at or near its own corners. When that
    # happens the corner estimate gets pulled toward the tissue color instead
    # of true background, and that wrong color then fills both the real
    # out-of-bounds padding and the square-canvas letterbox bars, producing
    # blocky, wrongly-tinted patches instead of clean white. H&E slide
    # backgrounds (glass + mounting medium) are reliably near-white regardless
    # of scanner, so default to that explicitly here rather than letting
    # corner-sampling run on a crop where its core assumption doesn't hold.
    bg_hint_safe = bg_hint if bg_hint is not None else (245, 245, 245)
    img = pad_with_background_color(img, bg_color_hint=bg_hint_safe)

    # clean up .mrxs black scan-artifact regions AFTER padding, not before.
    # pad_with_background_color detects true out-of-bounds padding by looking
    # for EXACT pure black and whitens it - running artifact-cleaning first
    # (as an earlier version of this function did) replaced that same pure
    # black with a tissue-colored median fill, so padding's own black-
    # detection found nothing left to whiten, leaving blocky, wrongly-
    # colored patches where clean white background should be. Running this
    # after padding means the true padding is already white by the time this
    # runs (so it's not touched), and only genuine WITHIN-TISSUE scan
    # artifacts (which padding's corner/exact-zero logic does not target)
    # remain to be cleaned.
    img, artifact_frac = mask_black_artifacts(img)
    if artifact_frac > 0:
        print(f"[mrxs_tissue_crop] {slide_path.name}: cleaned "
              f"{artifact_frac*100:.1f}% black scan-artifact pixels from the thumbnail")

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
            "wsi_crop_origin": [int(x0), int(y0)],     # NEW: where the crop
                                                        # started, in true
                                                        # level-0 pixels
        }
        write_mapping_sidecar(output_path, sidecar)

    slide.close()
    return img, tissue_fraction, effective_fraction
