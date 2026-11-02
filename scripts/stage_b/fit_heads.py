"""
Fit the Stage B heads: a logistic regression on the mean of n tile features from the patch-regions.

For each fold, every slide's embedding is the mean of the frozen-encoder features of n tiles drawn at random
among the 20x tiles whose centre lies inside that fold's patch-regions of the slide (select_regions.py),
n = 8, 16, ..., 1,024 or all tiles. A class-balanced multinomial logistic regression is fitted on the
training slides, with C in {1e-3, ..., 10} chosen on the validation slides by log-loss, and applied to the
test slides. Tile draws use seeds 0, 1 and 2; the seed-0 head of every n is saved for inference
(the dashboard's tile-count option).

Features: one <slide_id>.h5 per slide with 'coords' (level-0 top-left corners, attribute patch_size_level0)
and 'features', in the same tile order as TRIDENT's tiling, e.g. the output of encode_tiles.py (DistillPath)
or TRIDENT's own features_uni_v2 files (UNI2-h). Searched recursively under --features-dir.

Output: <out-dir>/<config>/lr_n<n>_fold<k>.npz (scaler and head) and predictions_test.csv.

    python -m scripts.stage_b.fit_heads --regions data/regions_rn50_f1.csv ^
        --features-dir data/tile_features/distillpath --config rn50_f1_dp_uni2h
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
NS = [8, 16, 32, 64, 128, 256, 512, 1024, None]     # None: all tiles in the regions
SEEDS = [0, 1, 2]


def fit_model(Xtr, ytr, Xva, yva, Cs=(1e-3, 1e-2, 1e-1, 1, 10)):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        ll = log_loss(yva, clf.predict_proba(sc.transform(Xva)), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, C, clf)
    return sc, best[2], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True, help="CSV of select_regions.py")
    ap.add_argument("--features-dir", required=True, help="folder with <slide_id>.h5 tile features")
    ap.add_argument("--config", required=True, help="head name, e.g. rn50_f1_dp_uni2h (DistillPath) or rn50_f1 (UNI2-h)")
    ap.add_argument("--out-dir", default="data/resuni_heads")
    a = ap.parse_args()
    out = REPO / a.out_dir / a.config
    out.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(REPO / a.regions)
    reg = reg[(reg.n_regions > 0) & reg.subtype.isin(SUB)]
    files = {}
    for p in Path(a.features_dir).rglob("*.h5"):
        files.setdefault(p.stem, p)
    missing = sorted(set(reg.slide_id) - set(files))
    if missing:
        print(f"WARNING: {len(missing)} slides without a feature file are left out, e.g. {missing[:3]}")
    reg = reg[reg.slide_id.isin(files)]

    # embeddings: (fold, slide, seed, n) -> mean of n tile features inside that fold's regions
    emb = {}
    for i, (sid, g) in enumerate(reg.groupby("slide_id"), 1):
        with h5py.File(files[sid]) as h:
            xy = h["coords"][:]
            ps = int(h["coords"].attrs["patch_size_level0"])
            F = h["features"][:]
        cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
        for fold, s in zip(g.fold, g.bboxes):
            inside = np.zeros(len(F), bool)
            for bx, by, bw, bh in json.loads(s):
                inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
            if not inside.any():
                continue
            Fi = F[inside]
            for seed in SEEDS:
                perm = np.random.default_rng(1000 * seed + int(fold)).permutation(len(Fi))
                for n in NS:
                    emb[(int(fold), sid, seed, n)] = Fi[perm if n is None else perm[:n]].mean(0)
        if i % 100 == 0:
            print(f"  {i}/{reg.slide_id.nunique()} slides", flush=True)

    rows = []
    for n in NS:
        for seed in SEEDS:
            for fold in sorted(reg.fold.unique()):
                d = {sp: [(sid, SUB.index(y)) for sid, y in zip(g.slide_id, g.subtype) if (fold, sid, seed, n) in emb]
                     for sp, g in reg[reg.fold == fold].groupby("split")}
                X = {sp: np.stack([emb[(fold, sid, seed, n)] for sid, _ in v]) for sp, v in d.items() if v}
                y = {sp: np.array([l for _, l in v]) for sp, v in d.items()}
                sc, clf, C = fit_model(X["train"], y["train"], X["val"], y["val"])
                if seed == 0:
                    np.savez(out / f"lr_n{'all' if n is None else n}_fold{fold}.npz", mean=sc.mean_, scale=sc.scale_,
                             coef=clf.coef_, intercept=clf.intercept_, C=C, classes=np.array(SUB))
                if "test" in X:
                    p = clf.predict_proba(sc.transform(X["test"]))
                    rows.append(pd.DataFrame({"n": "all" if n is None else n, "seed": seed, "fold": fold,
                                              "slide_id": [s for s, _ in d["test"]], "y": y["test"],
                                              **{f"p_{k}": p[:, i] for i, k in enumerate(SUB)}, "C": C}))
        if rows:
            t = pd.concat([r for r in rows if r.n.iloc[0] == ("all" if n is None else n)])
            bacc = np.mean([balanced_accuracy_score(g.y, g[[f"p_{k}" for k in SUB]].values.argmax(1))
                            for _, g in t.groupby(["seed", "fold"])])
            print(f"n={'all' if n is None else n}: test balanced accuracy {bacc:.3f} (mean over folds and seeds)", flush=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "predictions_test.csv", index=False)
    print("heads in", out)


if __name__ == "__main__":
    main()
