#!/usr/bin/env python3
r"""
Drop-in, .mrxs-only replacement for create_wsi_thumbnail(), with an optional
Stage-A stain-normalization correction (by named, dataset-specific PROFILE)
applied afterwards.

Dispatch is a PURE file-extension check, nothing else:
  - slide_path ends in .mrxs (case-insensitive)
        -> apply the tissue-crop + artifact-cleanup fix
           (mrxs_tissue_crop.create_wsi_thumbnail_smart)
  - anything else (.svs, .ndpi, .tif, .tiff, ...)
        -> call the EXISTING, unmodified create_wsi_thumbnail() directly,
           unconditionally. No tissue probing, no measurement, no other
           logic runs at all for these files - guaranteeing identical
           behavior and identical performance to calling
           create_wsi_thumbnail() yourself for every non-.mrxs format.

STAIN NORMALIZATION (by profile)
    After either path produces the thumbnail, a Macenko stain-normalization
    step (glioma_sparse.preprocessing.stain_normalization) can be applied,
    re-staining the thumbnail to match a reference fitted from the EBRAINS
    training data for ONE named external dataset -- e.g. "RD-mrxs" (Radboud)
    or "TCGA-Glioma" (TCGA). Pass `stain_profile=None` (the default) for no
    normalization: this is correct for the EBRAINS training/test data itself,
    since there is no distribution to correct there, and it is the only sane
    default now that more than one external dataset has its own profile --
    there is no single "always on" correction to fall back to.

    Background: the Radboud and TCGA external-validation sets were each found
    to carry systematically less apparent hematoxylin and a paler, washed-out
    appearance than the training distribution, by different amounts. See
    glioma_sparse.preprocessing.stain_normalization and the stain_profiles/
    directory for the fitted numbers and how to add a new profile.

WHERE TO SAVE THIS
    Alongside mrxs_tissue_crop.py and stain_normalization.py, inside your
    glioma_sparse package, e.g.:
        glioma_sparse/preprocessing/mrxs_tissue_crop.py
        glioma_sparse/preprocessing/stain_normalization.py
        glioma_sparse/preprocessing/stain_profiles/*.json
        glioma_sparse/preprocessing/thumbnail_auto.py   (this file)

HOW TO WIRE IT IN (one line, at each entry point)
    Wherever you currently have:
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
    change it to:
        from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto as create_wsi_thumbnail
    Every existing call site keeps working unchanged (stain_profile defaults
    to None, i.e. no normalization, matching the original behavior); pass
    stain_profile="RD-mrxs" / "TCGA-Glioma" / etc. wherever you know which
    external dataset you're processing.

CLI DIAGNOSTIC MODE (no pipeline changes needed to just check a slide)
    python thumbnail_auto.py /path/to/slide1 /path/to/slide2 ...
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart
    from .stain_normalization import normalize_with_profile
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart
    from stain_normalization import normalize_with_profile


def is_mrxs(slide_path):
    return Path(slide_path).suffix.lower() == ".mrxs"


def measure_tissue_fraction(slide_path, probe_size=1024):
    """Cheap probe: fraction of the level-0 canvas that is tissue, or None if
    no tissue detected. Only used for the CLI diagnostic's context, NOT for
    dispatch (dispatch is a pure extension check)."""
    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    try:
        level0_w, level0_h = slide.dimensions
        bbox = detect_tissue_bbox_wsi(slide_path, probe_size=probe_size)
        if bbox is None:
            return None
        _, _, bw, bh = bbox
        return (bw * bh) / (level0_w * level0_h)
    finally:
        slide.close()


def _apply_stain_normalization(img, output_path, stain_profile):
    """Re-stain `img` (PIL, RGB) against `stain_profile`'s Stage-A reference,
    and re-save it over output_path if one was given (the underlying
    thumbnail functions already wrote the un-normalized version there)."""
    rgb = np.array(img)
    normalized = normalize_with_profile(rgb, stain_profile, stage="A")
    normalized_img = Image.fromarray(normalized)

    if output_path is not None:
        output_path = Path(output_path)
        normalized_img.save(output_path, quality=90)

    return normalized_img


def create_wsi_thumbnail_auto(slide_path, output_path=None, target_mpp=4.0,
                              output_size=2048, save_mask=False, verbose=True,
                              stain_profile=None):
    """.mrxs-only auto-dispatching replacement for create_wsi_thumbnail(), with
    an optional Stage-A stain-normalization correction applied afterwards.
    Same signature/return shape as the original, plus:
        stain_profile (str, optional; default None): name of a fitted profile
            (see stain_normalization.list_profiles()) to normalize this
            thumbnail against, e.g. "RD-mrxs" or "TCGA-Glioma". None (the
            default) applies no correction -- the right choice for EBRAINS
            data itself, since there is no distribution to correct there.
    """
    slide_path = Path(slide_path)

    if not is_mrxs(slide_path):
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
        img, tissue_fraction, effective_fraction = create_wsi_thumbnail(
            slide_path, output_path=output_path, target_mpp=target_mpp,
            output_size=output_size, save_mask=save_mask)
    else:
        img, tissue_fraction, effective_fraction = create_wsi_thumbnail_smart(
            slide_path, output_path=output_path, target_mpp=target_mpp,
            output_size=output_size, save_mask=save_mask, verbose=verbose)

    if stain_profile:
        img = _apply_stain_normalization(img, output_path, stain_profile)

    return img, tissue_fraction, effective_fraction


def _cli():
    import sys
    if len(sys.argv) < 2:
        print("usage: python thumbnail_auto.py slide1 [slide2 ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        p = Path(p)
        if not is_mrxs(p):
            print(f"{p}: not .mrxs -> passed straight through, unchanged (pass "
                  f"stain_profile=... to create_wsi_thumbnail_auto for stain correction)")
            continue
        frac = measure_tissue_fraction(p)
        if frac is None:
            print(f"{p}: .mrxs, no tissue detected -> crop fix falls back to full slide")
        else:
            print(f"{p}: .mrxs, tissue = {frac*100:.1f}% of level-0 canvas -> crop fix applied")


if __name__ == "__main__":
    _cli()
