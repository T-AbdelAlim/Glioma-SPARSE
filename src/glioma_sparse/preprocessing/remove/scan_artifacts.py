#!/usr/bin/env python3
r"""
Fix for .mrxs black scan-artifact tiles.

WHAT'S HAPPENING (confirmed from your example images)
    Some .mrxs slides contain rectangular regions that are pure/near-pure
    black - a scanner drop-out where a field of view failed to capture and
    the format defaults to black rather than white. These are NOT part of
    the tissue-crop problem (that fix is working correctly, per your
    feedback) - this is a separate issue: real tissue is never anywhere near
    pure black in an H&E scan (even the darkest hematoxylin-dense nuclei are
    a dark PURPLE/BLUE with real channel separation, not achromatic near-zero
    RGB), so a solid black rectangle is wildly out of the distribution the
    model was ever trained on. Feeding it a tile that contains one produces
    an oversized, spurious activation that has nothing to do with genuine
    histological signal, but:
      - inflates that tile's patch-injection risk score (Stage A), causing it
        to be wrongly selected as "high signal" for p95 extraction
      - dominates the Stage B classification and occlusion importance for
        that patch, since the black region is far more "salient" to the
        network than the real tissue sharing the same tile
    Both of these are visible directly in your uploaded images: the risk map
    ranks the artifact-containing cells among the highest-signal tiles, and
    the occlusion map assigns its largest importance to the black band, not
    the tissue.

THE FIX
    Detect near-pure-black pixels (all channels below `black_thresh`, default
    10/255) and replace them with the image's own median non-black color,
    so the model never sees a jarring out-of-distribution black region. This
    is a content-based fix, not a structural one: it only ever touches pixels
    that are actually black, so it cannot alter genuine tissue, however dark.

WHERE THIS IS APPLIED (per your standing requirement: .mrxs only)
    - Already applied automatically inside create_wsi_thumbnail_smart's
      output, since that function is only ever called for .mrxs slides (via
      thumbnail_auto's extension check) - covers Stage A grading and the
      patch-injection risk map, both of which operate on the thumbnail.
    - Needs to be added at the high-resolution patch extraction step for
      Stage B, in TWO places, each gated by the SAME .mrxs extension check
      (see the two patches below, for backend.py and inference_end_to_end.py):
        1. backend.py's /api/extract endpoint
        2. inference_end_to_end.py's extract_highres()
    Both are content-based AND extension-gated, so .svs/.ndpi patches are
    never touched, by construction, twice over.
"""

import numpy as np
from PIL import Image


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
            (typically exactly (0,0,0) or very close) will trigger this.
        fill_color: the color artifact pixels are replaced with. Fixed and
            near-white by default (matching the same background color used
            for padding elsewhere), NOT computed as a median of the tile's
            own non-artifact pixels. Using this tile's own median was the
            earlier approach, but for a high-res Stage B patch - which is
            selected specifically because it's mostly tissue - that median
            is itself a tissue color (pink/purple), so the artifact region
            got replaced with fabricated tissue-looking content rather than
            reading clearly as "no data here." A fixed background color is
            simpler and semantically correct in every context this is used.
            Pass an explicit color (e.g. a slide's own detected background
            hint) if a better-known one is available.
        return_fraction: if True, also return the fraction of the image that
            was artifact (0.0 if none found), useful for logging/QC.

    Returns:
        (cleaned_img, artifact_fraction) if return_fraction else cleaned_img.
        If no artifact pixels are found, returns the image UNCHANGED (same
        object), so this is a true no-op when there's nothing to fix.
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
