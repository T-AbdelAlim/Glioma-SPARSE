"""
Experiment E7b/E7c (docs/experiments/EXPERIMENT_LOG.md): DistillPath-KS16-UNI2h features of tiles drawn at random from
the WHOLE slide (no Stage A regions), with the tile draws of E6 (single_stage_ntiles.py): per slide and draw d,
perm_d = default_rng([d, crc32(slide_id)]).permutation(number of TRIDENT tiles), n tiles = perm_d[:n].

  pass A  perm_d[:64] for d = 0, 1, 2                           (E7b: Stage A grade + Stage B on 64 random tiles;
                                                                  E7c at n = 64)
  pass all  every TRIDENT tile of the slide                     (E7c: n = 64 ... 1024 with three draws, and all tiles)

Slides: every slide of the five internal splits (controls included) and the 180 TCGA slides. Tiles and reads as in
tiny_encoder_features.py (TRIDENT coordinates of features_uni_v2, same transform, bf16). Output per slide:
<out>/random/<cohort>/dp_uni2h/<slide_id>_<pass>.h5 with 'idx' (rows of the TRIDENT file) and 'features'.
    python -m scripts.analysis.tiny_random_tiles --cohort internal --pass A        (mahmoodlab venv; GPU)
"""
import argparse, glob, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scripts.analysis import tiny_encoder_features as T

ENC = "dp_uni2h"
OUT = T.OUT / "random"
DRAWS = (0, 1, 2)
PASS_N = {"A": {0: 64, 1: 64, 2: 64}}


def feature_files(cohort):
    """Same feature files as single_stage_ntiles.py (E6)."""
    if cohort == "tcga":
        return {Path(p).stem: p for p in glob.glob(str(T.BRAIN / "mahmoodlab" / "trident_tcga_val" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}
    f = {Path(p).stem: p for p in glob.glob(str(T.BRAIN / "mahmoodlab" / "trident_output" / "*" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}
    ids = set()
    for k in range(1, 6):
        ids |= {Path(p).stem for p in pd.read_csv(T.REPO / "splits" / f"split_0{k}.csv").path}
    return {s: f[s] for s in sorted(ids) if s in f}


def perms(sid, m):
    return {d: np.random.default_rng([d, zlib.crc32(sid.encode())]).permutation(m) for d in DRAWS}


def tile_rows(sid, m, which):
    if which == "all":
        return np.arange(m, dtype=np.int64)
    P = perms(sid, m)
    return np.array(sorted(set().union(*[set(P[d][:PASS_N[which][d]].tolist()) for d in DRAWS])), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["internal", "tcga"], required=True)
    ap.add_argument("--pass", dest="which", choices=["A", "all"], required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import timm
    feats, P = feature_files(a.cohort), T.wsi_paths(a.cohort)
    out = OUT / a.cohort / ENC; out.mkdir(parents=True, exist_ok=True)
    miss = [s for s in feats if s not in P or not Path(P[s]).exists()]
    todo = [s for s in feats if s not in miss and not (out / f"{s}_{a.which}.h5").exists()]
    print(f"{a.cohort} pass {a.which}: {len(feats)} slides, {len(miss)} without WSI {miss[:5]}, {len(todo)} to do", flush=True)
    model = timm.create_model(f"hf_hub:{T.DP[ENC]}", pretrained=True, num_classes=0).eval().to("cuda")
    transform = T.encoder_factory("kaiko-vits16").eval_transforms
    jobs, sels = [], []
    for sid in todo:
        with h5py.File(feats[sid]) as h:
            coords = h["coords"][:]; attrs = dict(h["coords"].attrs)
        sel = tile_rows(sid, len(coords), a.which)
        jobs.append((P[sid], attrs, coords[sel])); sels.append(sel)
    total = sum(map(len, sels)); print(f"{total} tiles to encode", flush=True)
    t0, nt = time.time(), 0
    for i, (j, f) in enumerate(T.encode(T.Tiles(jobs, transform), {ENC: model}, "cuda", a.batch, a.workers), 1):
        sid, sel = todo[j], sels[j]
        assert len(f[ENC]) == len(sel), sid
        tmp = out / f"{sid}_{a.which}.h5.tmp"
        with h5py.File(tmp, "w") as h:
            h.create_dataset("idx", data=sel); h.create_dataset("features", data=f[ENC]).attrs["encoder"] = ENC
        tmp.replace(out / f"{sid}_{a.which}.h5")
        nt += len(sel); el = time.time() - t0
        print(f"  {i}/{len(todo)} {sid}: {len(sel)} tiles | {nt / el:.0f} tiles/s | eta {el / nt * (total - nt) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
