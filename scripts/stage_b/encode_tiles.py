"""
Encode the 20x tiles inside the patch-regions with DistillPath, for fitting the Stage B heads.

The tiles are those of TRIDENT's tiling (20x, 256 px, non-overlapping; https://github.com/mahmoodlab/TRIDENT):
run TRIDENT's segmentation and tiling steps on the slides first (no feature extraction needed), e.g.

    python run_batch_of_slides.py --task seg --wsi_dir <slides> --job_dir <trident job> --segmenter hest
    python run_batch_of_slides.py --task coords --wsi_dir <slides> --job_dir <trident job> \
        --mag 20 --patch_size 256 --overlap 0

Of every slide, the tiles whose centre lies inside any of its patch-regions (union over folds and splits of
select_regions.py) are read with TRIDENT and encoded with the frozen DistillPath-KS16-UNI2h, with the same
weights and input pipeline as at inference (glioma_sparse.models.resuni). The output keeps TRIDENT's tile
order, so fit_heads.py draws the same tiles as from TRIDENT's own feature files.

Output: <out>/<slide_id>.h5 with 'coords' (the kept tiles, with TRIDENT's attributes), 'features' (n, 384)
and 'idx' (row in TRIDENT's tiling). Resumable per slide. Needs TRIDENT installed (or cloned next to this
repository as ../mahmoodlab/TRIDENT) and preferably a GPU.

    python -m scripts.stage_b.encode_tiles --regions data/regions_rn50_f1.csv ^
        --trident-dir <trident job> --wsi-root <slides> --out data/tile_features/distillpath
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO.parent / "mahmoodlab" / "TRIDENT"))
from trident import load_wsi  # noqa: E402

from glioma_sparse.models import resuni  # noqa: E402

WSI_EXT = {".ndpi", ".svs", ".tif", ".tiff", ".mrxs"}


def region_tiles(coords, patch_size_level0, boxes):
    """Rows of `coords` whose tile centre lies inside any of `boxes` (level-0 [x, y, w, h])."""
    cx, cy = coords[:, 0] + patch_size_level0 // 2, coords[:, 1] + patch_size_level0 // 2
    keep = np.zeros(len(coords), bool)
    for bx, by, bw, bh in boxes:
        keep |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
    return np.flatnonzero(keep)


@torch.no_grad()
def encode_slide(wsi_path, attrs, coords, model, transform, device, batch_size):
    wsi = load_wsi(slide_path=str(wsi_path), lazy_init=False)
    patcher = wsi.create_patcher(patch_size=int(attrs["patch_size"]), src_mag=int(attrs["level0_magnification"]),
                                 dst_mag=int(attrs["target_magnification"]), custom_coords=coords,
                                 coords_only=False, pil=True)
    feats, batch = [], []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    for i in range(len(coords)):
        batch.append(transform(patcher[i][0]))
        if len(batch) == batch_size or i == len(coords) - 1:
            with autocast:
                feats.append(model(torch.stack(batch).to(device)).float().cpu().numpy())
            batch = []
    return np.concatenate(feats) if feats else np.zeros((0, resuni.ENCODERS["distillpath"]["dim"]), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True, help="CSV of select_regions.py")
    ap.add_argument("--trident-dir", required=True, help="TRIDENT job folder with <slide_id>_patches.h5 files")
    ap.add_argument("--wsi-root", required=True, help="folder with the whole-slide images (searched recursively)")
    ap.add_argument("--out", default="data/tile_features/distillpath")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="encode at most this many slides")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(REPO / a.regions)
    boxes = {}
    for sid, s in zip(reg.slide_id, reg.bboxes):
        boxes.setdefault(sid, []).extend(json.loads(s))
    boxes = {s: b for s, b in boxes.items() if b}
    tiling = {p.name[:-len("_patches.h5")]: p for p in Path(a.trident_dir).rglob("*_patches.h5")}
    wsis = {p.stem: p for p in Path(a.wsi_root).rglob("*") if p.suffix.lower() in WSI_EXT}
    todo = [s for s in sorted(boxes) if not (out / f"{s}.h5").exists()]
    lacking = [s for s in todo if s not in tiling or s not in wsis]
    if lacking:
        print(f"WARNING: {len(lacking)} slides without a TRIDENT tiling or slide file are skipped, e.g. {lacking[:3]}")
    todo = [s for s in todo if s not in lacking][:a.limit]
    model, transform = resuni.get_distillpath_encoder(device)
    print(f"{len(boxes)} slides with patch-regions, {len(todo)} to encode on {device}", flush=True)

    t0 = time.time()
    for i, sid in enumerate(todo, 1):
        with h5py.File(tiling[sid]) as h:
            coords, attrs = h["coords"][:], dict(h["coords"].attrs)
        sel = region_tiles(coords, int(attrs["patch_size_level0"]), boxes[sid])
        F = encode_slide(wsis[sid], attrs, coords[sel], model, transform, device, a.batch_size)
        tmp = out / f"{sid}.h5.tmp"
        with h5py.File(tmp, "w") as h:
            d = h.create_dataset("coords", data=coords[sel])
            for k, v in attrs.items():
                d.attrs[k] = v
            h.create_dataset("features", data=F).attrs["encoder"] = "DistillPath-KS16-UNI2h"
            h.create_dataset("idx", data=sel)
        tmp.replace(out / f"{sid}.h5")
        print(f"  {i}/{len(todo)} {sid}: {len(sel)} tiles ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
