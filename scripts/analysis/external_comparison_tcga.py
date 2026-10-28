"""
External validation (TCGA, 180 slides, subtype-balanced: 60 GBM / 60 astro / 60 oligo): all systems,
one metric code path (same helpers as master_comparison.py).

Systems (EBRAINS-trained models applied unchanged; no tuning on the external data):
  * ResNet cascades: RN18 / RN50 x best-F1 / best-AUC checkpoints (results/e2e_external_<cfg>_tcga/per_slide.csv)
  * UNI2-h at Stage B: results/resuni/<cfg>/predictions_external_tcga.csv
  * UNI2-h + CLAM-SB full-slide benchmark: results/external_TCGA_val/clam_predictions.csv

The cohort is subtype-balanced by design (no majority-class shortcut); the majority baseline is still
reported for comparability.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.master_comparison import (
    BRAIN, GRADES, SIX, SUB, cascade_table, mcnemar, metrics, six_true,
)

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "results" / "external_TCGA_val"
CFG = {"rn18_f1": "ResNet18 best-F1 (S0)", "rn18_auc": "ResNet18 best-AUC",
       "rn50_f1": "ResNet50 best-F1", "rn50_auc": "ResNet50 best-AUC"}
RES_CFG = {"rn18_f1": "ResNet18 best-F1", "rn50_f1": "ResNet50 best-F1", "rn50_auc": "ResNet50 best-AUC"}
CLAM_RES = BRAIN / "mahmoodlab/CLAM/results/glioma_subtype_clam_s1"
TAG = "tcga"


def labels():
    return pd.read_csv(EXT / "labels.csv").set_index("slide_id")


def cascade_ext():
    out = {}
    for cfg in CFG:
        p = REPO / "results" / f"e2e_external_{cfg}_{TAG}" / "per_slide.csv"
        if p.exists():
            out[cfg] = pd.read_csv(p)
    return out


def clam_class_columns():
    ds = pd.read_csv(BRAIN / "clam_workspace/dataset_csv/dpwsi_6class.csv").set_index("slide_id").label
    pairs = {}
    for k in range(5):
        r = pickle.load(open(CLAM_RES / f"split_{k}_results.pkl", "rb"))
        for sid, v in r.items():
            pairs[ds[str(sid)]] = int(v["label"])
    name = {"control": "control", "low_grade_astro": "astro_low", "high_grade_astro": "astro_high",
            "low_grade_oligo": "oligo_low", "high_grade_oligo": "oligo_high", "GBM": "GBM"}
    return {name[l]: f"p_{i}" for l, i in pairs.items()}


def clam_table_ext(lab):
    d = pd.read_csv(EXT / "clam_predictions.csv")
    col = clam_class_columns()
    S = d[[col[c] for c in SIX]].to_numpy(float)
    t = pd.DataFrame({"fold": d.fold, "slide_id": d.slide_id})
    t["true6"] = lab.loc[d.slide_id, "true6"].to_numpy()
    t["true_grade_class"] = lab.loc[d.slide_id, "true_grade_class"].to_numpy()
    t["true_mutation"] = lab.loc[d.slide_id, "true_mutation"].to_numpy()
    t["six_score"] = list(S)
    t["six_pred"] = np.array(SIX)[S.argmax(1)]
    G = np.stack([S[:, 0], S[:, 1] + S[:, 3], S[:, 2] + S[:, 4] + S[:, 5]], 1)
    t["grade_prob"] = list(G)
    t["grade_pred"] = np.array(GRADES)[G.argmax(1)]
    T = np.stack([S[:, 1] + S[:, 2], S[:, 3] + S[:, 4], S[:, 5]], 1)
    T = T / T.sum(1, keepdims=True).clip(1e-12)
    t["sub_prob"] = list(T)
    t["sub_pred"] = np.array(SUB)[T.argmax(1)]
    t["has_b"] = True
    return t


def resuni_ext(cfg, n, base_df, seed=0):
    p = REPO / "results" / "resuni" / cfg / f"predictions_external_{TAG}.csv"
    P = pd.read_csv(p)
    P = P[(P.n.astype(str) == str(n)) & (P.seed == seed)].set_index(["fold", "slide_id"])
    pb = np.full((len(base_df), 3), np.nan)
    for i, (f, s) in enumerate(zip(base_df.fold, base_df.slide_id)):
        if (f, s) in P.index:
            pb[i] = P.loc[(f, s), [f"p_{k}" for k in SUB]].to_numpy(float)
    return pb


def ensemble(t):
    rows = []
    for sid, g in t.groupby("slide_id", sort=False):
        S = np.mean(np.stack(g.six_score.to_numpy()), 0)
        G = np.mean(np.stack(g.grade_prob.to_numpy()), 0)
        Pb = np.nanmean(np.stack(g.sub_prob.to_numpy()), 0) if g.has_b.any() else np.full(3, np.nan)
        r = g.iloc[0].to_dict()
        r.update({"fold": 1, "six_score": S, "grade_prob": G, "sub_prob": Pb, "has_b": bool(g.has_b.any()),
                  "six_pred": SIX[int(S.argmax())], "grade_pred": GRADES[int(G.argmax())],
                  "sub_pred": SUB[int(np.nan_to_num(Pb).argmax())] if g.has_b.any() else "NA"})
        rows.append(r)
    return pd.DataFrame(rows)


def build_external_systems(ns=(64, 8, 32, "all")):
    lab = labels()
    systems = []
    cas = cascade_ext()
    tabs = {}
    for cfg, d in cas.items():
        d = d[d.role == "external"].reset_index(drop=True)
        pb = d[[f"B0__p_{k}" for k in SUB]].to_numpy(float)
        t = cascade_table(d, pb)
        t["true6"] = [six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)]
        tabs[cfg] = (d, t)
        systems.append(("ResNet cascades (archived)", CFG[cfg], t, d))
    for cfg, name in RES_CFG.items():
        if cfg not in tabs or not (REPO / "results" / "resuni" / cfg / f"predictions_external_{TAG}.csv").exists():
            continue
        d = tabs[cfg][0]
        for n in ns:
            t = cascade_table(d, resuni_ext(cfg, n, d))
            t["true6"] = [six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)]
            systems.append(("UNI2-h at Stage B (sparse)", f"{name} Stage A + UNI2-h, {n} tiles", t, d))
    if (EXT / "clam_predictions.csv").exists():
        systems.append(("Benchmark (full slide)", "TRIDENT + UNI2-h + CLAM-SB", clam_table_ext(lab), None))
    return systems


def per_class_recall(t):
    return {c: float(((t.true6 == c) & (t.six_pred == c)).sum() / max((t.true6 == c).sum(), 1)) for c in SIX if (t.true6 == c).any()}


def boot_ci(ok, ids, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=ok.astype(float), minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u)))
    return np.percentile(s[dr].sum(1) / c[dr].sum(1), [2.5, 97.5])


def main():
    lab = labels()
    systems = build_external_systems()
    rows, oks = [], {}
    maj = float((lab.true6 == lab.true6.value_counts().index[0]).mean())
    for group, name, t, _ in systems:
        r, ok = metrics(t)
        e = ensemble(t)
        re_, oke = metrics(e)
        lo, hi = boot_ci(ok, t.slide_id.to_numpy())
        elo, ehi = boot_ci(oke, e.slide_id.to_numpy())
        rc = per_class_recall(e)
        oks[name] = (t, ok, e, oke)
        draws = (np.nan, np.nan)
        if group == "UNI2-h at Stage B (sparse)":
            cfg = [c for c, nm in RES_CFG.items() if name.startswith(nm + " Stage A")][0]
            n_ = name.split(", ")[1].split(" ")[0]
            base_d = [g[3] for g in systems if g[1] == CFG[cfg]][0]
            accs = [metrics(cascade_table(base_d, resuni_ext(cfg, n_, base_d, sd)).assign(
                true6=lambda x: [six_true(g_, m_) for g_, m_ in zip(x.true_grade_class, x.true_mutation)]))[0]["e2e_acc_pooled"]
                for sd in (0, 1, 2)]
            draws = (float(np.mean(accs)), float(np.std(accs, ddof=1)))
        rows.append({"group": group, "system": name, "n_slides": e.slide_id.nunique(),
                     "e2e_acc_pooled_over_tile_draws_mean": draws[0], "e2e_acc_pooled_over_tile_draws_sd": draws[1],
                     **{f"fm_{k}": v for k, v in r.items()},
                     "fm_acc_ci_lo": lo, "fm_acc_ci_hi": hi,
                     "ens_acc": re_["e2e_acc_pooled"], "ens_acc_ci_lo": elo, "ens_acc_ci_hi": ehi,
                     "ens_balacc": re_["e2e_balacc_mean"], "ens_macroF1": re_["e2e_macroF1_pooled"],
                     "ens_macroAUC": re_["e2e_macroAUC_pooled"],
                     **{f"recall_{c}": v for c, v in rc.items()},
                     "majority_baseline_acc": maj})
    R = pd.DataFrame(rows)
    R.to_csv(EXT / "external_master_tcga.csv", index=False)

    pairs = []
    def pair(a_name, b_name):
        ea, eb = oks[a_name][2], oks[b_name][2]
        ea = ea.set_index("slide_id"); eb = eb.set_index("slide_id")
        ids = ea.index.intersection(eb.index)
        oka = (ea.loc[ids, "true6"] == ea.loc[ids, "six_pred"]).to_numpy()
        okb = (eb.loc[ids, "true6"] == eb.loc[ids, "six_pred"]).to_numpy()
        d = okb.astype(float) - oka.astype(float)
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
        n01, n10 = int(((oka == 1) & (okb == 0)).sum()), int(((oka == 0) & (okb == 1)).sum())
        pairs.append({"A": a_name, "B": b_name, "acc_A": oka.mean(), "acc_B": okb.mean(), "delta_B_minus_A": d.mean(),
                      "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "mcnemar_p": mcnemar(n01, n10), "n": len(ids)})
    names = list(R.system)
    clam = "TRIDENT + UNI2-h + CLAM-SB"
    for n_ in names:
        if n_ != clam and clam in names:
            pair(n_, clam)
    for cfg, name in RES_CFG.items():
        base = CFG[cfg]
        for n_ in ("64", "8", "32", "all"):
            u = f"{name} Stage A + UNI2-h, {n_} tiles"
            if u in names and base in names:
                pair(base, u)
    pd.DataFrame(pairs).to_csv(EXT / "external_paired_tcga.csv", index=False)
    show = R[["system", "n_slides", "fm_e2e_acc_pooled", "fm_e2e_balacc_mean", "fm_e2e_macroF1_mean", "fm_e2e_macroAUC_mean",
              "ens_acc", "ens_balacc", "ens_macroF1", "ens_macroAUC"]].round(3)
    print(show.to_string())
    print("majority-class baseline accuracy:", round(maj, 3))
    print("\nper-class recall (ensemble):")
    rc_cols = [c for c in R.columns if c.startswith("recall_")]
    print(R[["system"] + rc_cols].round(2).to_string())
    print("\npaired comparisons vs CLAM benchmark and vs matching cascade:")
    print(pd.DataFrame(pairs).round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
