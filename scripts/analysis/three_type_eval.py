"""
Experiment E9 (docs/experiments/EXPERIMENT_LOG.md): the six-class predictions collapsed to the three WHO 2021 types
(astrocytoma, IDH-mutant; oligodendroglioma, IDH-mutant and 1p/19q-codeleted; glioblastoma, IDH-wildtype), for comparison
with published work that predicts type without grade. Tumour slides only; a tumour slide predicted as control (or without a
patch-region) counts as an error. Type scores: sums of the six-class scores per type, renormalised over the three types.
Also IDH status (mutant vs wildtype) AUC.
  Reference: TRIDENT + UNI2-h + CLAM-SB. Glioma-SPARSE: ResNet-50 best-F1 + UNI2-h, 64 tiles, tile draws 0-2.
  Internal: pooled test predictions of the five folds. TCGA: ensemble of the five fold models.
Writes results/revision_oct22/e9_three_type.json.
    python -m scripts.analysis.three_type_eval
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT

OUT = REPO / "results" / "revision_oct22"
TYPES = ["astro", "oligo", "GBM"]
TYPE_OF = {"astro_low": "astro", "astro_high": "astro", "oligo_low": "oligo", "oligo_high": "oligo", "GBM": "GBM", "wt_low_grade": "GBM"}


def three(t):
    t = t[t.true6 != "control"].reset_index(drop=True)
    y = np.array([TYPE_OF[c] for c in t.true6])
    pred = np.array([TYPE_OF.get(c, "none") for c in t.six_pred])
    S = np.stack(t.six_score.to_numpy())
    T = np.stack([S[:, 1] + S[:, 2], S[:, 3] + S[:, 4], S[:, 5]], 1); T = T / T.sum(1, keepdims=True)
    ok = pred == y
    r = {"n": int(len(t)), "acc": float(ok.mean()),
         "balacc": float(np.mean([ok[y == c].mean() for c in TYPES])),
         "recall": {c: float(ok[y == c].mean()) for c in TYPES},
         "auc": {c: float(roc_auc_score(y == c, T[:, i])) for i, c in enumerate(TYPES)},
         "idh_auc": float(roc_auc_score(y == "GBM", T[:, 2])),
         "pred_control_or_none": float(np.isin(pred, ["none"]).mean())}
    r["macro_auc"] = float(np.mean(list(r["auc"].values())))
    key = t.slide_id.astype(str) if t.fold.nunique() == 1 else t.fold.astype(str) + "_" + t.slide_id.astype(str)
    return r, pd.Series(ok, index=key.to_numpy())


def paired(okr, okg):
    a, b = okr.align(okg, join="inner"); a, b = a.to_numpy(bool), b.to_numpy(bool)
    return {"n": int(len(a)), "diff": float(b.mean() - a.mean()), "mcnemar_p": MC.mcnemar(int((a & ~b).sum()), int((~a & b).sum()))}


def boot(ok, n=10000, seed=0):
    ok = np.asarray(ok, bool); rng = np.random.default_rng(seed); b = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mean_draws(rs):
    keys = ("acc", "balacc", "macro_auc", "idh_auc")
    return {k: float(np.mean([r[k] for r in rs])) for k in keys} | {"auc": {c: float(np.mean([r["auc"][c] for r in rs])) for c in TYPES},
                                                                     "recall": {c: float(np.mean([r["recall"][c] for r in rs])) for c in TYPES},
                                                                     "acc_draws": [r["acc"] for r in rs]}


def main():
    res = {}
    # internal
    systems, _, _, _, tabs = MC.build_systems()
    S = {n: t for g, n, t, c, _ in systems}
    bd = tabs["ResNet50 best-F1"][4]
    add6 = lambda t: t.assign(true6=[MC.six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)])
    ref, okr = three(add6(S["TRIDENT + UNI2-h + CLAM-SB"]))
    gs = [three(add6(MC.cascade_table(bd, MC.resuni("rn50_f1", 64, sd, bd)))) for sd in (0, 1, 2)]
    six = [MC.metrics(MC.cascade_table(bd, MC.resuni("rn50_f1", 64, sd, bd)))[0]["e2e_acc_pooled"] for sd in (0, 1, 2)]
    okg = gs[0][1]
    res["internal"] = {"reference": ref | {"acc_ci": boot(okr)}, "glioma_sparse": mean_draws([g[0] for g in gs]) | {"acc_ci_draw0": boot(okg)},
                       "glioma_sparse_vs_reference_draw0": paired(okr, okg),
                       "six_class_check_glioma_sparse": six}
    # TCGA
    lab = XT.labels()
    d = XT.cascade_ext()["rn50_f1"]; d = d[d.role == "external"].reset_index(drop=True)
    sysx = {n: t for g, n, t, _ in XT.build_external_systems(ns=(64,))}
    rx = sysx["TRIDENT + UNI2-h + CLAM-SB"]
    refx, okrx = three(XT.ensemble(rx if "true6" in rx else add6(rx)))
    gsx = []
    for sd in (0, 1, 2):
        t = add6(MC.cascade_table(d, XT.resuni_ext("rn50_f1", 64, d, sd)))
        gsx.append(three(XT.ensemble(t)))
    okgx = gsx[0][1]
    res["tcga"] = {"reference": refx | {"acc_ci": boot(okrx)}, "glioma_sparse": mean_draws([g[0] for g in gsx]) | {"acc_ci_draw0": boot(okgx)},
                   "glioma_sparse_vs_reference_draw0": paired(okrx, okgx)}
    json.dump(res, open(OUT / "e9_three_type.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
