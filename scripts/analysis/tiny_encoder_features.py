"""
Experiment E7: tiny pathology ViTs as Stage B encoder instead of UNI2-h (681M).
Re-encodes exactly the TRIDENT tiles (20x, 256 px, same coordinates as features_uni_v2) whose centre lies inside any
Stage A region of any fold, with four ViT-S/16 encoders (22M) that share one input pipeline (224 px, mean = std = 0.5):

  kaiko_vits16      kaiko ViT-S/16 (TRIDENT 'kaiko-vits16'), the student initialisation of the three below
  dp_uni2h          DistillPath-KS16-UNI2h     (hf_hub:RamonK/..., distilled from UNI2-h)
  dp_hopt0          DistillPath-KS16-HOpt0     (distilled from H-optimus-0)
  dp_virchow2       DistillPath-KS16-Virchow2  (distilled from Virchow2)

Regions: training/validation slides from the Stage B cohort manifests (wsi_bbox), test slides from
results/e2e_regions_*.csv, external slides from results/e2e_external_*/per_slide.csv (bboxes), i.e. the same tiles
resuni_nsweep.py selects. Output per encoder: <out>/<encoder>/<slide_id>.h5 with 'coords' (the kept TRIDENT coordinates,
original order, original attributes), 'features' and 'idx' (row in the TRIDENT file), so resuni_nsweep.py
--features-root reproduces the same tile draws. Resumable per slide.
    python -m scripts.analysis.tiny_encoder_features --cohort internal          (mahmoodlab venv; GPU)
    python -m scripts.analysis.tiny_encoder_features --check                    (UNI2-h re-encoding check)
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
REPO = Path(__file__).resolve().parents[2]; BRAIN = REPO.parent
sys.path[:0] = [str(REPO), str(REPO / "src"), str(BRAIN / "mahmoodlab" / "TRIDENT")]
from scripts.analysis.region_foundation_features import TRIDENT, index_features, load_manifests
from trident import load_wsi
from trident.patch_encoder_models import encoder_factory
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset

OUT = BRAIN / "mahmoodlab" / "tiny_encoder_features"
DP = {"dp_uni2h": "RamonK/DistillPath-KS16-UNI2h", "dp_hopt0": "RamonK/DistillPath-KS16-HOpt0",
      "dp_virchow2": "RamonK/DistillPath-KS16-Virchow2"}
ENCODERS = ["kaiko_vits16", *DP]
EBRAINS_WSI = Path(r"C:\Users\TAbde\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data")
COHORTS = {
    "internal": dict(cohort="data/stage_b_cohort_RN50_F1", test="results/e2e_regions_rn50_f1.csv", external=None),
    "tcga": dict(cohort=None, test=None, external="results/e2e_external_rn50_f1_tcga/per_slide.csv",
                 features=BRAIN / "mahmoodlab" / "trident_tcga_val" / "20x_256px_0px_overlap" / "features_uni_v2",
                 labels="results/external_TCGA_val/labels.csv"),
}


def boxes_per_slide(cfg):
    """slide_id -> list of level-0 boxes (union over folds and splits)."""
    B = {}
    if cfg.get("cohort"):
        for sid, b in zip(*[load_manifests(REPO / cfg["cohort"])[c] for c in ("slide_id", "bbox")]):
            B.setdefault(sid, []).append(list(b))
    for key, col in (("test", "bboxes"), ("external", "bboxes")):
        if cfg.get(key):
            r = pd.read_csv(REPO / cfg[key])
            for sid, s in zip(r.slide_id, r[col]):
                if isinstance(s, str):          # no regions: empty cell
                    B.setdefault(sid, []).extend(json.loads(s))
    return {s: b for s, b in B.items() if b}


def wsi_paths(name):
    if name == "internal":
        return {Path(p).stem: p for p in glob.glob(str(EBRAINS_WSI / "**" / "*.ndpi"), recursive=True)}
    lab = pd.read_csv(REPO / COHORTS["tcga"]["labels"])
    # labels.csv lists the backup drive as D:; it may be mounted under another letter
    drive = next((d for d in "DEFGH" if Path(f"{d}{lab.wsi.iloc[0][1:]}").exists()), "D")
    return {s: f"{drive}{w[1:]}" for s, w in zip(lab.slide_id, lab.wsi)}


def load_encoders(dev):
    import timm
    kaiko = encoder_factory("kaiko-vits16")
    models = {"kaiko_vits16": kaiko.model}
    for k, hub in DP.items():
        models[k] = timm.create_model(f"hf_hub:{hub}", pretrained=True, num_classes=0)
    for m in models.values():
        m.eval().to(dev)
    return models, kaiko.eval_transforms


def inside_idx(coords, ps, boxes):
    cx, cy = coords[:, 0] + ps // 2, coords[:, 1] + ps // 2
    keep = np.zeros(len(coords), bool)
    for bx, by, bw, bh in boxes:
        keep |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
    return np.flatnonzero(keep)


def patcher_for(wsi_path, attrs, coords):
    wsi = load_wsi(slide_path=str(wsi_path), lazy_init=False)
    return wsi.create_patcher(patch_size=int(attrs["patch_size"]), src_mag=int(attrs["level0_magnification"]),
                              dst_mag=int(attrs["target_magnification"]), custom_coords=coords, coords_only=False, pil=True)


class Tiles(torch.utils.data.Dataset):
    """All region tiles of a list of slides, in slide order. Same reads as WSIPatcherDataset, but each worker opens
    the slides itself (slide handles do not pickle on Windows), so one persistent worker pool serves every slide."""
    def __init__(self, jobs, transform):
        self.jobs, self.transform, self.cache = jobs, transform, {}
        self.items = [(j, i) for j, (_, _, c) in enumerate(jobs) for i in range(len(c))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        j, i = self.items[k]
        if j not in self.cache:
            self.cache = {j: patcher_for(*self.jobs[j])}
        return self.transform(self.cache[j][i][0]), j


@torch.no_grad()
def encode(tiles, models, dev, bs, workers):
    """Yields (job index, {encoder: features}) as soon as all tiles of a slide are encoded."""
    dl = torch.utils.data.DataLoader(tiles, batch_size=bs, num_workers=workers, persistent_workers=workers > 0,
                                     prefetch_factor=4 if workers else None)
    buf, cur = {k: [] for k in models}, None
    for x, j in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = {k: m(x).float().cpu().numpy() for k, m in models.items()}
        j = j.numpy()
        for jj in np.unique(j):
            if cur is not None and jj != cur:
                yield cur, {k: np.concatenate(v) for k, v in buf.items()}
                buf = {k: [] for k in models}
            cur = jj
            for k in models:
                buf[k].append(f[k][j == jj])
    if cur is not None:
        yield cur, {k: np.concatenate(v) for k, v in buf.items()}


def check(dev, n=64):
    """Re-encode n region tiles of one slide with TRIDENT's UNI2-h and compare with the stored features."""
    idx = index_features(TRIDENT); B = boxes_per_slide(COHORTS["internal"]); P = wsi_paths("internal")
    sid = next(s for s in sorted(B) if s in idx and s in P)
    with h5py.File(idx[sid]) as h:
        coords = h["coords"][:]; attrs = dict(h["coords"].attrs); F = h["features"][:]
    sel = inside_idx(coords, int(attrs["patch_size_level0"]), B[sid])[:n]
    enc = encoder_factory("uni_v2").eval().to(dev)
    dl = torch.utils.data.DataLoader(WSIPatcherDataset(patcher_for(P[sid], attrs, coords[sel]), enc.eval_transforms), batch_size=n)
    x, _ = next(iter(dl))
    with torch.no_grad(), torch.autocast("cuda", dtype=enc.precision, enabled=enc.precision != torch.float32):
        f = enc(x.to(dev)).float().cpu().numpy()
    cos = (f * F[sel]).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(F[sel], axis=1)
    print(f"{sid}: {len(sel)} tiles, cosine to stored UNI2-h features min {cos.min():.4f} mean {cos.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=list(COHORTS), default="internal")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process at most this many new slides")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda"
    if a.check:
        return check(dev)
    cfg = COHORTS[a.cohort]
    idx = ({p.stem: p for p in Path(cfg["features"]).glob("*.h5")} if cfg.get("features") else index_features(TRIDENT))
    B, P = boxes_per_slide(cfg), wsi_paths(a.cohort)
    todo = [s for s in sorted(B) if s in idx]
    miss = [s for s in todo if s not in P or not Path(P[s]).exists()]
    print(f"{a.cohort}: {len(B)} slides with regions, {len(todo)} with TRIDENT features, {len(miss)} without a WSI file", flush=True)
    if miss:
        print("  missing WSI:", miss[:10])
    todo = [s for s in todo if s not in miss and not all((OUT / a.cohort / e / f"{s}.h5").exists() for e in ENCODERS)]
    if a.limit:
        todo = todo[:a.limit]
    models, transform = load_encoders(dev)
    for e in ENCODERS:
        (OUT / a.cohort / e).mkdir(parents=True, exist_ok=True)
    jobs, sels = [], []
    for sid in todo:
        with h5py.File(idx[sid]) as h:
            coords = h["coords"][:]; attrs = dict(h["coords"].attrs)
        sel = inside_idx(coords, int(attrs["patch_size_level0"]), B[sid])
        jobs.append((P[sid], attrs, coords[sel])); sels.append(sel)
    print(f"{sum(map(len, sels))} tiles to encode", flush=True)
    t0, nt = time.time(), 0
    for i, (j, feats) in enumerate(encode(Tiles(jobs, transform), models, dev, a.batch, a.workers), 1):
        sid, (_, attrs, crd), sel = todo[j], jobs[j], sels[j]
        assert all(len(F) == len(sel) for F in feats.values()), sid
        for e, F in feats.items():
            tmp = OUT / a.cohort / e / f"{sid}.h5.tmp"
            with h5py.File(tmp, "w") as h:
                d = h.create_dataset("coords", data=crd)
                for k, v in attrs.items():
                    d.attrs[k] = v
                h.create_dataset("features", data=F).attrs["encoder"] = e
                h.create_dataset("idx", data=sel)
            tmp.replace(OUT / a.cohort / e / f"{sid}.h5")
        nt += len(sel)
        el = time.time() - t0
        print(f"  {i}/{len(todo)} {sid}: {len(sel)} tiles | {nt / el:.0f} tiles/s | eta {el / nt * (sum(map(len, sels)) - nt) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
