#!/usr/bin/env python3
r"""
Drop-in, .mrxs-only replacement for create_wsi_thumbnail().

WHAT IT DOES
    create_wsi_thumbnail_auto(slide_path, ...) has the EXACT same signature
    and return shape as the original create_wsi_thumbnail(slide_path, ...):
    (thumbnail, tissue_frac, eff_frac).

    Dispatch is a PURE file-extension check, nothing else:
      - slide_path ends in .mrxs (case-insensitive)
            -> apply the tissue-crop fix (mrxs_tissue_crop.create_wsi_thumbnail_smart)
      - anything else (.svs, .ndpi, .tif, .tiff, ...)
            -> call your EXISTING, unmodified create_wsi_thumbnail() directly,
               unconditionally. No tissue probing, no measurement, no other
               logic runs at all for these files. This guarantees byte-for-
               byte identical behavior and identical performance to calling
               create_wsi_thumbnail() yourself for every non-.mrxs format,
               since that is genuinely the only thing that happens.

    An earlier version of this wrapper decided which path to use by measuring
    tissue coverage (so it would help automatically with any future format
    that had the same problem, not just .mrxs). That is no longer how this
    works: dispatch is .mrxs-only, by design, so there is zero risk of it
    ever touching a non-.mrxs file for any reason.

WHERE THIS FILE LIVES
    Save this alongside mrxs_tissue_crop.py, inside your glioma_sparse
    package, e.g.:
        glioma_sparse/preprocessing/mrxs_tissue_crop.py
        glioma_sparse/preprocessing/thumbnail_auto.py   (this file)

HOW TO WIRE IT IN (both entry points, one line each)
    Wherever you currently have:
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
    change it to:
        from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto as create_wsi_thumbnail
    Every existing call site keeps working unchanged, since the wrapper's
    signature and return shape match exactly.

CLI DIAGNOSTIC MODE (no pipeline changes needed to just check a slide)
    python thumbnail_auto.py /path/to/slide1 /path/to/slide2 ...
    prints, for each slide, whether it is .mrxs (and would get the crop fix)
    or not (and would be passed straight through, unchanged), plus - for
    .mrxs files only - the measured tissue fraction as extra context. Writes
    no files.
"""

from pathlib import Path

try:
    # normal case: imported as part of the glioma_sparse package
    from .mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart
except ImportError:
    # fallback for direct script execution (e.g. the CLI diagnostic mode),
    # where relative imports don't work because the file has no package
    # context when run this way
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart


def is_mrxs(slide_path):
    return Path(slide_path).suffix.lower() == ".mrxs"


def measure_tissue_fraction(slide_path, probe_size=1024):
    """Cheap probe: fraction of the level-0 canvas that is tissue, or None if
    no tissue detected at all. Only used for the CLI diagnostic's extra
    context on .mrxs files - NOT used to decide dispatch (dispatch is a pure
    extension check, see module docstring)."""
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


def create_wsi_thumbnail_auto(slide_path, output_path=None, target_mpp=4.0,
                              output_size=2048, save_mask=False, verbose=True):
    """.mrxs-only auto-dispatching replacement for create_wsi_thumbnail().
    See module docstring. Same signature/return shape as the original."""
    slide_path = Path(slide_path)

    if not is_mrxs(slide_path):
        # every non-.mrxs format: call the ORIGINAL function directly and
        # unconditionally. Nothing else runs - no bbox probing, no
        # measurement - so behavior and performance are identical to calling
        # create_wsi_thumbnail() yourself.
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
        return create_wsi_thumbnail(
            slide_path, output_path=output_path, target_mpp=target_mpp,
            output_size=output_size, save_mask=save_mask)

    if verbose:
        print(f"[thumbnail_auto] {slide_path.name}: .mrxs detected, "
              f"applying tissue-crop fix.")
    return create_wsi_thumbnail_smart(
        slide_path, output_path=output_path, target_mpp=target_mpp,
        output_size=output_size, save_mask=save_mask)


def _cli():
    import sys
    if len(sys.argv) < 2:
        print("usage: python thumbnail_auto.py slide1 [slide2 ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        p = Path(p)
        if not is_mrxs(p):
            print(f"{p}: not .mrxs -> passed straight through, unchanged")
            continue
        frac = measure_tissue_fraction(p)
        if frac is None:
            print(f"{p}: .mrxs, no tissue detected -> CROP FIX APPLIED "
                  f"(falls back to full slide)")
        else:
            print(f"{p}: .mrxs, tissue = {frac*100:.1f}% of level-0 canvas "
                  f"-> CROP FIX APPLIED")


if __name__ == "__main__":
    _cli()
