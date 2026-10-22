"""
Foundation-model features on the regions Stage A selects (a "sparse foundation" Stage B).

Idea: keep Stage A's cheap thumbnail-level region selection, but replace the ResNet-18
Stage B (which sees a ~14,000 px region squeezed to 2048 px) with a pathology foundation
model applied at native 20x resolution inside the same regions.

This script needs no new slide reading: TRIDENT + UNI2-h features already exist for every
slide (256 px tiles at 20x = 512 px at level 0, coordinates in level-0 pixels, the same frame
as the Stage B cohort's `wsi_bbox`). For each fold it

  1. takes the regions that fold's Stage A selected (cohort manifest, `wsi_bbox`),
  2. pools (mean) the UNI2-h features of all tiles whose centre lies in those regions
     -> one 1536-d embedding per slide,
  3. trains a class-balanced multinomial logistic regression on the fold's train slides,
     chooses the regularisation on the fold's validation slides, evaluates on its test slides.

Variants: `regions` (Stage A regions only; the deployable sparse setting) and `whole`
(mean over all tissue tiles of the slide; a reference for what mean-pooled UNI2-h can do
when nothing is selected).

Outputs (results/region_foundation/): embeddings cache, per-fold predictions, a summary and
paired comparison with the ResNet Stage B on the same test slides.
"""

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
TRIDENT = Path(r"C:\Users\TAbde\PycharmProjects\brain\mahmoodlab\trident_output")


def index_features(root):
    idx = {}
    for p in Path(root).glob("*/20x_256px_0px_overlap/features_uni_v2/*.h5"):
        idx.setdefault(p.stem, p)
    return idx


def load_manifests(cohort):
    rows = []
    for k in range(1, 6):
        m = pd.read_csv(Path(cohort) / "manifests" / f"stageB_fold{k}.csv")
        rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    m["bbox"] = m.wsi_bbox.apply(lambda s: json.loads(s))
    return m[m.true_mutation.isin(SUB)]


def embed_slides(cohort, feat_idx, cache):
    """Per (fold, slide): mean UNI2-h feature of tiles inside that fold's regions; per slide:
    mean of all tiles. Reads each feature file once."""
    if Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return z["reg"].item(), z["whole"].item(), z["ntiles"].item()
    m = load_manifests(cohort)
    reg, whole, ntiles = {}, {}, {}
    slides = sorted(m.slide_id.unique())
    for i, sid in enumerate(slides, 1):
        if sid not in feat_idx:
            continue
        with h5py.File(feat_idx[sid]) as h:
            xy = h["coords"][:]
            ps = int(h["coords"].attrs["patch_size_level0"])
            F = h["features"][:]
        cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
        whole[sid] = F.mean(0)
        for fold, g in m[m.slide_id == sid].groupby("fold"):
            inside = np.zeros(len(F), bool)
            for bx, by, bw, bh in g.bbox:
                inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
            if inside.any():
                reg[(int(fold), sid)] = F[inside].mean(0)
                ntiles[(int(fold), sid)] = int(inside.sum())
        if i % 50 == 0:
            print(f"  embedded {i}/{len(slides)} slides", flush=True)
    np.savez(cache, reg=np.array(reg, dtype=object), whole=np.array(whole, dtype=object),
             ntiles=np.array(ntiles, dtype=object))
    return reg, whole, ntiles


def fit_predict(Xtr, ytr, Xva, yva, Xte, Cs=(1e-3, 1e-2, 1e-1, 1, 10)):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        ll = log_loss(yva, clf.predict_proba(sc.transform(Xva)), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, C, clf)
    _, C, clf = best
    return clf.predict_proba(sc.transform(Xte)), C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/stage_b_cohort_RN18_F1")
    ap.add_argument("--out", default="results/region_foundation")
    ap.add_argument("--features-root", default=str(TRIDENT))
    ap.add_argument("--archived-b", default="ResNet18_output/training_output_stageB_F1/*fold{fold}_*/test_predictions.npz",
                    help="glob (with {fold}) of the archived ResNet Stage B slide-level test predictions to compare with")
    ap.add_argument("--archived-name", default="ResNet18 Stage B (archived)")
    ap.add_argument("--no-continued", action="store_true")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    feat_idx = index_features(a.features_root)
    print(f"{len(feat_idx)} feature files indexed")
    reg, whole, ntiles = embed_slides(REPO / a.cohort, feat_idx, out / "embeddings.npz")
    m = load_manifests(REPO / a.cohort)
    roles = m.drop_duplicates(["fold", "slide_id"])[["fold", "slide_id", "split", "true_mutation"]]
    roles["y"] = roles.true_mutation.map({k: i for i, k in enumerate(SUB)})

    allp = []
    for variant in ("regions", "whole"):
        for fold in range(1, 6):
            r = roles[roles.fold == fold]
            get = (lambda s: reg.get((fold, s))) if variant == "regions" else (lambda s: whole.get(s))
            r = r[[get(s) is not None for s in r.slide_id]]
            X = {sp: np.stack([get(s) for s in r[r.split == sp].slide_id]) for sp in ("train", "val", "test")}
            y = {sp: r[r.split == sp].y.to_numpy() for sp in ("train", "val", "test")}
            p, C = fit_predict(X["train"], y["train"], X["val"], y["val"], X["test"])
            te = r[r.split == "test"]
            allp.append(pd.DataFrame({"variant": variant, "fold": fold, "slide_id": te.slide_id.values,
                                      "y": y["test"], **{f"p_{k}": p[:, i] for i, k in enumerate(SUB)},
                                      "C": C}))
            acc = (p.argmax(1) == y["test"]).mean()
            print(f"{variant:8s} fold {fold}: test slides {len(te)}  acc {acc:.3f}  C={C}", flush=True)
    P = pd.concat(allp, ignore_index=True)
    P.to_csv(out / "test_predictions.csv", index=False)

    # ResNet Stage B on the same fold-test slides (cohort-level: all tumour test slides)
    def resnet(dirpat, name):
        rows = []
        for fold in range(1, 6):
            f = glob.glob(str(REPO / dirpat.format(fold=fold)))[0]
            z = np.load(f)
            rows.append(pd.DataFrame({"variant": name, "fold": fold, "slide_id": z["slide_ids"].astype(str),
                                      "y": z["y_true"], **{f"p_{k}": z["y_prob"][:, i] for i, k in enumerate(SUB)}}))
        return pd.concat(rows, ignore_index=True)
    R = [resnet(a.archived_b, a.archived_name)]
    if not a.no_continued:
        R.append(resnet("ResNet18_output/training_output_stageB_ft_continue/rn18_A0_stageB_continue_fold{fold}/test_predictions.npz",
                        "ResNet18 Stage B (continued)"))
    P = pd.concat([P] + R, ignore_index=True)

    def metrics(d):
        pr = d[[f"p_{k}" for k in SUB]].to_numpy()
        return (pr.argmax(1) == d.y.to_numpy()).mean(), roc_auc_score(d.y, pr, multi_class="ovr")
    lines = ["| Subtype classifier (test slides of each fold) | n | accuracy pooled | fold mean ± SD | macro-AUC pooled |",
             "|---|---|---|---|---|"]
    for v, d in P.groupby("variant", sort=False):
        pa = [(g[[f"p_{k}" for k in SUB]].to_numpy().argmax(1) == g.y.to_numpy()).mean() for _, g in d.groupby("fold")]
        acc, auc = metrics(d)
        lines.append(f"| {v} | {len(d)} | {acc:.3f} | {np.mean(pa):.3f} ± {np.std(pa, ddof=1):.3f} | {auc:.3f} |")
    # paired bootstrap over unique slides vs archived ResNet
    ref = P[P.variant == a.archived_name].set_index(["fold", "slide_id"])
    lines += ["", "| Candidate vs archived ResNet Stage B | Δ accuracy [95% CI, bootstrap over slides] |", "|---|---|"]
    rng = np.random.default_rng(0)
    for v in P.variant.unique():
        if v == a.archived_name:
            continue
        c = P[P.variant == v].set_index(["fold", "slide_id"])
        idx = ref.index.intersection(c.index)
        ok_r = (ref.loc[idx, [f"p_{k}" for k in SUB]].to_numpy().argmax(1) == ref.loc[idx, "y"].to_numpy())
        ok_c = (c.loc[idx, [f"p_{k}" for k in SUB]].to_numpy().argmax(1) == c.loc[idx, "y"].to_numpy())
        sid = np.array([i[1] for i in idx])
        ids, inv = np.unique(sid, return_inverse=True)
        d = (ok_c.astype(float) - ok_r.astype(float))
        s, n = np.bincount(inv, weights=d, minlength=len(ids)), np.bincount(inv, minlength=len(ids)).astype(float)
        dr = rng.integers(0, len(ids), size=(10000, len(ids)))
        b = s[dr].sum(1) / n[dr].sum(1)
        lines.append(f"| {v} | {d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] |")
    txt = "\n".join(lines)
    (out / "summary.md").write_text(txt, encoding="utf8")
    print("\n" + txt)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
