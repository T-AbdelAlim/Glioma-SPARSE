"""
Molecular-subtype labels for Stage A thumbnails.

Stage A thumbnails live in grade-organised folders (control/low_grade/high_grade)
that carry no molecular information. The subtype of every tumour slide is known
from the EBRAINS tree (`<sidecar_root>/<subtype_folder>/included/<slide_id>.jpg`),
whose folder name encodes entity, IDH and 1p/19q status, e.g.
`astro_IDHmt_G2`, `oligo_IDHmt_1p19qdel_G3`, `GBM_IDHwt`.

This module turns a slide id into the Stage B subtype index (same class order as
Stage B) so Stage A can be trained with an auxiliary subtype head. Control
slides have no subtype and get IGNORE_INDEX, which nn.CrossEntropyLoss skips.
"""

from pathlib import Path

SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]   # == Stage B class order
SUBTYPE_TO_IDX = {c: i for i, c in enumerate(SUBTYPE_CLASSES)}
IGNORE_INDEX = -100


def subtype_from_folder(folder_name):
    """Subtype class name for an EBRAINS subtype folder name, or None."""
    n = folder_name.lower()
    if "gbm" in n or "idhwt" in n:
        return "IDH_wt"
    if "oligo" in n and "1p19q" in n:
        return "IDH_mt_1p19q"
    if "astro" in n and "idhmt" in n:
        return "IDH_mt"
    return None


def build_subtype_index(sidecar_root):
    """Map slide id (stem) -> subtype class name, from the EBRAINS thumbnail tree."""
    index = {}
    for jpg in Path(sidecar_root).rglob("*.jpg"):
        # <root>/<subtype_folder>/included/<stem>.jpg
        sub = subtype_from_folder(jpg.parent.parent.name)
        if sub is not None:
            index.setdefault(jpg.stem, sub)
    return index


def subtype_label(stem, index):
    """Subtype index for a slide id; IGNORE_INDEX for control / unknown slides."""
    name = index.get(stem)
    return SUBTYPE_TO_IDX[name] if name is not None else IGNORE_INDEX
