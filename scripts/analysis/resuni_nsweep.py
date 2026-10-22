"""
Extended tile-count sweep (2026-10-22, experiment E2 in docs/experiments/EXPERIMENT_LOG.md).
Identical to resuni_predictions.py (same regions, folds, draws, head fitting) except:
  * NS = 8, 16, 32, 64, 128, 256, 512, 1024, all  (internal AND external, so both curves share the same grid);
  * a head is saved for EVERY n (seed 0): lr_n{n}_fold{k}.npz / lr_nall_fold{k}.npz (used by the dashboard's tile-count option).
Run with --config rn50_f1_nsweep so nothing in results/resuni/rn50_f1 is overwritten.
"""
"""
ResUNI: ResNet Stage A (region selection) + UNI2-h subtype classifier on n tiles per slide.

For one Stage A configuration (RN18/RN50, best-F1/best-AUC checkpoints) and each n:

  * training and validation slides: the regions that fold's Stage A selected on them
    (Stage B cohort manifest, `wsi_bbox`);
  * test slides: the regions the end-to-end pipeline selects at test time
    (results/e2e_regions_*.csv from e2e_regions.py; p95, cap 4, tissue filter 0.25), i.e.
    exactly the regions the ResNet Stage B was given in the archived end-to-end evaluation;
  * EXTERNAL slides (optional, --external-*): the regions each fold's Stage A selects on them
    (results/e2e_external_*/per_slide.csv, column `bboxes`), with UNI2-h features from TRIDENT run on the
    external slides. The fold's EBRAINS-trained head is applied unchanged (no re-fitting, no tuning);
  * slide embedding = mean of the frozen UNI2-h features (TRIDENT, 20x, 256 px tiles) of n tiles
    drawn at random from the tiles whose centre lies inside the regions (n = "all" uses every tile);
    n tiles are drawn per (fold, slide, seed) for train, validation, test and external slides alike;
  * class-balanced multinomial logistic regression, C in {1e-3..10} chosen on the validation
    slides by log-loss.

Output: results/resuni/<config>/predictions.csv (EBRAINS test slides; unless --external-only),
        results/resuni/<config>/predictions_external.csv (n, seed, fold, slide_id, y, p_*),
        results/resuni/<config>/lr_n64_fold{k}.npz (scaler + head of the n=64, seed-0 model, for heatmaps).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from glioma_sparse.data_utils.subtype_labels import build_subtype_index
from scripts.analysis.region_foundation_features import (
    REPO, SUB, TRIDENT, index_features, load_manifests,
)

NS = [8, 16, 32, 64, 128, 256, 512, 1024, None]
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
    ap.add_argument("--config", required=True, help="name, e.g. rn18_f1")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--e2e-regions", default=None, help="EBRAINS test regions (e2e_regions_*.csv)")
    ap.add_argument("--external-regions", default=None,
                    help="per_slide.csv of an external e2e_multi run (columns fold, slide_id, bboxes)")
    ap.add_argument("--external-features-dir", default=None, help="TRIDENT features_uni_v2 folder of the external slides")
    ap.add_argument("--external-labels", default=None, help="labels.csv of the external cohort")
    ap.add_argument("--external-tag", default="none", help="suffix for the external output (stain variant)")
    ap.add_argument("--external-only", action="store_true")
    a = ap.parse_args()
    out = REPO / "results" / "resuni" / a.config
    out.mkdir(parents=True, exist_ok=True)

    idx = dict(index_features(TRIDENT))
    m = load_manifests(REPO / a.cohort)
    trv = m[m.split.isin(["train", "val"])]
    sub_index = build_subtype_index(REPO / "data/ebrains_thumbnails")

    # (fold, slide) -> (split, boxes, true subtype)
    jobs = {}
    for (fold, sid), g in trv.groupby(["fold", "slide_id"]):
        jobs[(int(fold), sid)] = (g.split.iloc[0], [list(b) for b in g.bbox], g.true_mutation.iloc[0])
    if a.e2e_regions and not a.external_only:
        reg = pd.read_csv(REPO / a.e2e_regions)
        for _, r in reg[reg.n_regions > 0].iterrows():
            if r.slide_id in sub_index:
                jobs[(int(r.fold), r.slide_id)] = ("test", json.loads(r.bboxes), sub_index[r.slide_id])
    if a.external_regions:
        ext = pd.read_csv(REPO / a.external_regions)
        lab = pd.read_csv(REPO / a.external_labels).set_index("slide_id")
        fdir = Path(a.external_features_dir)
        ext_files = {p.stem: p for p in fdir.glob("*.h5")}
        idx.update(ext_files)
        n_missing = 0
        for _, r in ext[ext.n_regions > 0].iterrows():
            if r.slide_id not in ext_files:
                n_missing += 1
                continue
            jobs[(int(r.fold), r.slide_id)] = ("external", json.loads(r.bboxes), lab.loc[r.slide_id, "true_mutation"])
        if n_missing:
            print(f"WARNING: {n_missing} external (fold, slide) pairs without a TRIDENT feature file")
    by_slide = {}
    for (fold, sid), v in jobs.items():
        by_slide.setdefault(sid, []).append((fold, v))

    emb = {}
    for i, (sid, items) in enumerate(sorted(by_slide.items()), 1):
        with h5py.File(idx[sid]) as h:
            xy = h["coords"][:]
            ps = int(h["coords"].attrs["patch_size_level0"])
            F = h["features"][:]
        cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
        for fold, (split, boxes, y) in items:
            inside = np.zeros(len(F), bool)
            for bx, by, bw, bh in boxes:
                inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
            if not inside.any():
                continue
            Fi = F[inside]
            for seed in SEEDS:
                perm = np.random.default_rng(1000 * seed + fold).permutation(len(Fi))
                for n in NS:
                    emb[(fold, sid, seed, n)] = Fi[perm if n is None else perm[:n]].mean(0)
        if i % 50 == 0:
            print(f"  {i}/{len(by_slide)} slides", flush=True)

    rows_t, rows_e = [], []
    for n in NS:
        for seed in SEEDS:
            for fold in range(1, 6):
                d = {sp: [(sid, SUB.index(y)) for (f, sid), (s, _, y) in jobs.items()
                          if f == fold and s == sp and (f, sid, seed, n) in emb]
                     for sp in ("train", "val", "test", "external")}
                X = {sp: (np.stack([emb[(fold, sid, seed, n)] for sid, _ in d[sp]]) if d[sp] else None) for sp in d}
                sc, clf, C = fit_model(X["train"], np.array([l for _, l in d["train"]]),
                                       X["val"], np.array([l for _, l in d["val"]]))
                if seed == 0:
                    np.savez(out / f"lr_n{'all' if n is None else n}_fold{fold}.npz", mean=sc.mean_, scale=sc.scale_, coef=clf.coef_,
                             intercept=clf.intercept_, C=C, classes=np.array(SUB))
                for sp, rows in (("test", rows_t), ("external", rows_e)):
                    if not d[sp] or (sp == "test" and a.external_only):
                        continue
                    p = clf.predict_proba(sc.transform(X[sp]))
                    rows.append(pd.DataFrame({"n": "all" if n is None else n, "seed": seed, "fold": fold,
                                              "slide_id": [s for s, _ in d[sp]],
                                              "y": [l for _, l in d[sp]],
                                              **{f"p_{k}": p[:, i] for i, k in enumerate(SUB)}, "C": C}))
        print(f"n={n} done", flush=True)
    if rows_t:
        pd.concat(rows_t, ignore_index=True).to_csv(out / "predictions.csv", index=False)
        print("wrote", out / "predictions.csv")
    if rows_e:
        pd.concat(rows_e, ignore_index=True).to_csv(out / f"predictions_external_{a.external_tag}.csv", index=False)
        print("wrote", out / f"predictions_external_{a.external_tag}.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
