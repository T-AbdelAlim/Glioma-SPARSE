"""
Evaluation of experiments E1 (Stage-A selection rules) and E2 (tile-count sweep, internal + external on the same grid),
docs/experiments/EXPERIMENT_LOG.md. Uses the metric code of master_comparison.py / external_comparison_tcga.py.
    python -m scripts.analysis.revision_oct22_eval e1|e2      (Glioma-SPARSE venv)
Outputs: results/revision_oct22/e1_selection_rules.csv|json, e2_tile_sweep.csv|json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
OUT = REPO / "results" / "revision_oct22"; OUT.mkdir(parents=True, exist_ok=True)
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]


def _base():
    systems, _, _, _, tabs = MC.build_systems()
    S = {n: t for g, n, t, c, _ in systems}
    return tabs["ResNet50 best-F1"][4], S


def _pb(bd, P):
    P = P.set_index(["fold", "slide_id"]); pb = np.full((len(bd), 3), np.nan)
    for i, (f, s) in enumerate(zip(bd.fold, bd.slide_id)):
        if (f, s) in P.index: pb[i] = P.loc[(f, s), [f"p_{k}" for k in SUB]].to_numpy(float)
    return pb


def _summ(t, okref=None, okbm=None):
    r, ok = MC.metrics(t)
    d = {"acc": r["e2e_acc_pooled"], "macroAUC": r["e2e_macroAUC_mean"], "balacc": r["e2e_balacc_mean"],
         "type_acc": r["subtype_acc"], "type_macroAUC": r["subtype_macroAUC"], "no_region": r["wrong_noStageB"]}
    if okref is not None: d["mcnemar_vs_main"] = MC.mcnemar(int((okref & ~ok).sum()), int((~okref & ok).sum()))
    if okbm is not None: d["mcnemar_vs_benchmark"] = MC.mcnemar(int((okbm & ~ok).sum()), int((~okbm & ok).sum()))
    return d, ok


def e1():
    bd, S = _base()
    okbm = MC.metrics(S["TRIDENT + UNI2-h + CLAM-SB"])[1]
    okmain = MC.metrics(S["ResNet50 best-F1 Stage A + UNI2-h, 64 tiles"])[1]
    P = pd.read_csv(REPO / "results" / "region_selection_variants" / "predictions.csv")
    rows = []
    for (v, d), g in P.groupby(["variant", "draw"]):
        t = MC.cascade_table(bd, _pb(bd, g))
        m, ok = _summ(t, okmain, okbm)
        for gr in ("low", "high"):   # six-class accuracy within each Stage-A grade call (correctly graded tumours)
            i = ((t.grade_pred == gr) & (t.true_grade_class == gr)).to_numpy()
            m[f"acc_{gr}"] = float(ok[i].mean()); m[f"n_{gr}"] = int(i.sum())
        rows.append({"variant": v, "draw": int(d), **m})
    R = pd.DataFrame(rows); R.to_csv(OUT / "e1_selection_rules.csv", index=False)
    agg = R.groupby("variant").agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"), auc_mean=("macroAUC", "mean"),
                                    type_acc_mean=("type_acc", "mean"), type_auc_mean=("type_macroAUC", "mean"),
                                    low_mean=("acc_low", "mean"), high_mean=("acc_high", "mean"))
    regs = pd.read_csv(REPO / "results" / "region_selection_variants" / "regions.csv")
    main = regs[regs.variant == "main"].set_index(["fold", "slide_id"]).bboxes
    def jac(row):
        a = {tuple(b) for b in json.loads(main.get((row.fold, row.slide_id), "[]"))}; b = {tuple(x) for x in json.loads(row.bboxes)}
        return len(a & b) / len(a | b) if (a | b) else np.nan
    regs["jaccard_vs_main"] = regs.apply(jac, axis=1)
    ov = regs.groupby("variant").jaccard_vs_main.median().to_dict()
    json.dump({"per_draw": rows, "aggregate": agg.reset_index().to_dict(orient="records"), "median_jaccard_vs_main": ov},
              open(OUT / "e1_selection_rules.json", "w"), indent=1, default=float)
    print(R.round(3).to_string()); print(agg.round(3).to_string()); print("median Jaccard vs main:", {k: round(v, 2) for k, v in ov.items()})


def e1b(nd=100):
    """E1 repeated with nd random draws per variant (tile draws for main/control_drop, region+tile draws for random_cells,
    tile draws for random_slide); distribution of six-class accuracy over draws."""
    bd, _ = _base()
    P = pd.read_csv(REPO / "results" / "region_selection_variants" / f"predictions_{nd}draws.csv")
    rows = []
    for (v, d), g in P.groupby(["variant", "draw"]):
        r, _ = MC.metrics(MC.cascade_table(bd, _pb(bd, g)))
        rows.append({"variant": v, "draw": int(d), "acc": r["e2e_acc_pooled"], "macroAUC": r["e2e_macroAUC_mean"], "type_acc": r["subtype_acc"]})
    R = pd.DataFrame(rows); R.to_csv(OUT / f"e1b_selection_rules_{nd}draws.csv", index=False)
    W = R.pivot(index="draw", columns="variant", values="acc")
    summ = {v: {"mean": float(W[v].mean()), "sd": float(W[v].std()), "p2.5": float(W[v].quantile(0.025)), "p97.5": float(W[v].quantile(0.975)),
                "min": float(W[v].min()), "max": float(W[v].max()),
                "frac_draws_above_main_mean": float((W[v] > W["main"].mean()).mean()),
                "frac_draws_beating_main_same_draw": float((W[v] > W["main"]).mean()),
                "frac_draws_equal_main_same_draw": float((W[v] == W["main"]).mean())} for v in W.columns}
    json.dump({"n_draws": nd, "summary": summ}, open(OUT / f"e1b_selection_rules_{nd}draws.json", "w"), indent=1)
    print(pd.DataFrame(summ).T.round(3).to_string())


def e2(cfg="rn50_f1_nsweep"):
    bd, _ = _base()
    Pint = pd.read_csv(REPO / "results" / "resuni" / cfg / "predictions.csv")
    Pext = pd.read_csv(REPO / "results" / "resuni" / cfg / "predictions_external_tcga.csv")
    lab = XT.labels(); dext = XT.cascade_ext()["rn50_f1"]; dext = dext[dext.role == "external"].reset_index(drop=True)
    rows = []
    for n in sorted(Pint.n.astype(str).unique(), key=lambda v: 10 ** 9 if v == "all" else int(v)):
        for seed in sorted(Pint.seed.unique()):
            gi = Pint[(Pint.n.astype(str) == n) & (Pint.seed == seed)]
            ti = MC.cascade_table(bd, _pb(bd, gi)); ri, _ = MC.metrics(ti)
            ge = Pext[(Pext.n.astype(str) == n) & (Pext.seed == seed)]
            te = MC.cascade_table(dext, _pb(dext, ge)); te["true6"] = [MC.six_true(g, m) for g, m in zip(te.true_grade_class, te.true_mutation)]
            e = XT.ensemble(te); oke = (e.six_pred == e.true6).to_numpy()
            ci = XT.boot_ci(oke, e.slide_id.to_numpy())
            rows.append({"n": n, "seed": int(seed), "int_acc": ri["e2e_acc_pooled"], "int_macroAUC": ri["e2e_macroAUC_mean"],
                         "int_type_acc": ri["subtype_acc"], "ext_acc": float(oke.mean()), "ext_ci_lo": float(ci[0]), "ext_ci_hi": float(ci[1])})
    R = pd.DataFrame(rows); R.to_csv(OUT / "e2_tile_sweep.csv", index=False)
    agg = R.groupby("n", sort=False).agg(int_mean=("int_acc", "mean"), int_sd=("int_acc", "std"), ext_mean=("ext_acc", "mean"),
                                         ext_sd=("ext_acc", "std"), ext_ci_lo_draw0=("ext_ci_lo", "first"), ext_ci_hi_draw0=("ext_ci_hi", "first"))
    json.dump({"per_draw": rows, "aggregate": agg.reset_index().to_dict(orient="records")}, open(OUT / "e2_tile_sweep.json", "w"), indent=1)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    {"e1": e1, "e1b": e1b, "e2": e2}[sys.argv[1]]()
