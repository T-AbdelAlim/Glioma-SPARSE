"""
How many UNI2-h tiles per slide does the region-based subtype classifier need?

Same protocol as region_foundation_features.py (fold train/val/test slides, class-balanced
logistic regression, C chosen on validation), but the slide embedding is the mean over a random
subsample of n tiles drawn from the tiles inside that fold's Stage A regions (n = 8 ... all).
Averaged over several random draws. Reports pooled test accuracy / macro-AUC over the five folds
(same 182 tumour test slides as before), so the cost (tiles to embed per slide) can be read off.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.analysis.region_foundation_features import (
    REPO, SUB, TRIDENT, fit_predict, index_features, load_manifests,
)

NS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, None]     # None = all tiles in the regions
SEEDS = [0, 1, 2]


def main():
    cohort = REPO / "data/stage_b_cohort_RN18_F1"
    out = REPO / "results/region_foundation"
    idx = index_features(TRIDENT)
    m = load_manifests(cohort)
    roles = m.drop_duplicates(["fold", "slide_id"])[["fold", "slide_id", "split", "true_mutation"]].copy()
    roles["y"] = roles.true_mutation.map({k: i for i, k in enumerate(SUB)})
    cache = out / "subsample_embeddings.npz"
    if cache.exists():
        emb = np.load(cache, allow_pickle=True)["emb"].item()
    else:
        emb = {}
        slides = sorted(m.slide_id.unique())
        for i, sid in enumerate(slides, 1):
            with h5py.File(idx[sid]) as h:
                xy = h["coords"][:]
                ps = int(h["coords"].attrs["patch_size_level0"])
                F = h["features"][:]
            cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
            for fold, g in m[m.slide_id == sid].groupby("fold"):
                inside = np.zeros(len(F), bool)
                for bx, by, bw, bh in g.bbox:
                    inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
                if not inside.any():
                    continue
                Fi = F[inside]
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + int(fold))
                    perm = rng.permutation(len(Fi))
                    for n in NS:
                        sel = perm if n is None else perm[:n]
                        emb[(int(fold), sid, seed, n)] = Fi[sel].mean(0)
            if i % 50 == 0:
                print(f"  {i}/{len(slides)}", flush=True)
        np.savez(cache, emb=np.array(emb, dtype=object))

    rows = []
    for n in NS:
        for seed in SEEDS:
            ok, aucs, N = 0, [], 0
            P, Y = [], []
            for fold in range(1, 6):
                r = roles[(roles.fold == fold).to_numpy() & np.array([(fold, s, seed, n) in emb for s in roles.slide_id])]
                X = {sp: np.stack([emb[(fold, s, seed, n)] for s in r[r.split == sp].slide_id])
                     for sp in ("train", "val", "test")}
                y = {sp: r[r.split == sp].y.to_numpy() for sp in ("train", "val", "test")}
                p, _ = fit_predict(X["train"], y["train"], X["val"], y["val"], X["test"])
                P.append(p); Y.append(y["test"])
            P, Y = np.concatenate(P), np.concatenate(Y)
            rows.append({"tiles_per_slide": "all" if n is None else n, "seed": seed,
                         "acc": (P.argmax(1) == Y).mean(),
                         "macro_auc": roc_auc_score(Y, P, multi_class="ovr"), "n_test": len(Y)})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    s = d.groupby("tiles_per_slide", sort=False).agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                                                     macro_auc=("macro_auc", "mean")).reset_index()
    s.to_csv(out / "subsample_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
