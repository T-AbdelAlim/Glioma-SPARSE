#!/usr/bin/env python3
r"""
Drop-in, .mrxs-only replacement for create_wsi_thumbnail().

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

WHERE TO SAVE THIS
    Alongside mrxs_tissue_crop.py, inside your glioma_sparse package, e.g.:
        glioma_sparse/preprocessing/mrxs_tissue_crop.py
        glioma_sparse/preprocessing/thumbnail_auto.py   (this file)

HOW TO WIRE IT IN (one line, at each entry point)
    Wherever you currently have:
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
    change it to:
        from glioma_sparse.preprocessing.thumbnail_auto import create_wsi_thumbnail_auto as create_wsi_thumbnail
    Every existing call site keeps working unchanged.

CLI DIAGNOSTIC MODE (no pipeline changes needed to just check a slide)
    python thumbnail_auto.py /path/to/slide1 /path/to/slide2 ...
"""

from pathlib import Path

try:
    from .mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mrxs_tissue_crop import detect_tissue_bbox_wsi, create_wsi_thumbnail_smart


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


def create_wsi_thumbnail_auto(slide_path, output_path=None, target_mpp=4.0,
                              output_size=2048, save_mask=False, verbose=True):
    """.mrxs-only auto-dispatching replacement for create_wsi_thumbnail().
    Same signature/return shape as the original."""
    slide_path = Path(slide_path)

    if not is_mrxs(slide_path):
        from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail
        return create_wsi_thumbnail(
            slide_path, output_path=output_path, target_mpp=target_mpp,
            output_size=output_size, save_mask=save_mask)

    return create_wsi_thumbnail_smart(
        slide_path, output_path=output_path, target_mpp=target_mpp,
        output_size=output_size, save_mask=save_mask, verbose=verbose)


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
            print(f"{p}: .mrxs, no tissue detected -> crop fix falls back to full slide")
        else:
            print(f"{p}: .mrxs, tissue = {frac*100:.1f}% of level-0 canvas -> crop fix applied")


if __name__ == "__main__":
    _cli()
