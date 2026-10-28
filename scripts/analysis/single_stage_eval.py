"""
Evaluation of experiment E6 (single_stage_ntiles.py): single-stage six-class prediction from n random whole-slide
UNI2-h tiles (A: benchmark CLAM-SB at test time, B: mean + logistic regression trained per n) versus the two-stage
Glioma-SPARSE (ResNet-50 best-F1 Stage A + UNI2-h on n tiles from the Stage-A regions, results/resuni/rn50_f1_nsweep).
Internal: pooled accuracy over the 220 test predictions, balanced accuracy and macro-AUC (mean over folds);
TCGA: ensemble of the five fold models. Mean ± SD over three tile draws. Paired comparisons (bootstrap over slides,
McNemar) at n = 64 (same number of UNI2-h tiles) and against single-stage n = 256 (more compute than Glioma-SPARSE at 64).
    python -m scripts.analysis.single_stage_eval        (Glioma-SPARSE venv)
Output: results/revision_oct22/e6_single_stage.json, e6_single_stage_summary.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT

OUT = REPO / "results" / "revision_oct22"
SIX = MC.SIX; NS = ["64", "128", "256", "512", "1024", "all"]
GF_UNI, GF_A = 0.360585363456, 65 * 0.683302465536          # TFLOPs per UNI2-h tile, Stage A per slide (ResNet-50)
TE = json.load(open(OUT / "e2_tiles_encoded.json"))
BENCH_TILES = {"internal": pd.read_csv(REPO / "results/master_comparison/compute_tiles.csv").n_tissue_tiles_20x.mean(), "tcga": 11367.2}


def scores(g):
    P = g[[f"p_{c}" for c in SIX]].to_numpy(float); y = g.true6.map({c: i for i, c in enumerate(SIX)}).to_numpy()
    return P, y


def internal_metrics(g):
    P, y = scores(g); ok = P.argmax(1) == y
    per_fold_auc = [MC.macro_auc(y[(g.fold == f).to_numpy()], P[(g.fold == f).to_numpy()], 6) for f in sorted(g.fold.unique())]
    bal = np.mean([(P.argmax(1)[y == c] == c).mean() for c in range(6) if (y == c).any()])
    return {"acc": float(ok.mean()), "balacc": float(bal), "macroAUC": float(np.nanmean(per_fold_auc))}, ok


def ensemble(g):
    e = g.groupby("slide_id").agg({**{f"p_{c}": "mean" for c in SIX}, "true6": "first"}).reset_index()
    P, y = scores(e); ok = P.argmax(1) == y
    bal = np.mean([(P.argmax(1)[y == c] == c).mean() for c in range(6) if (y == c).any()])
    return {"acc": float(ok.mean()), "balacc": float(bal), "macroAUC": float(MC.macro_auc(y, P, 6))}, ok, e.slide_id.to_numpy()


def glioma_sparse():
    """Two-stage results per n and draw from the E2 sweep (same metric code as the manuscript)."""
    systems, *_, tabs = MC.build_systems(); bd = tabs["ResNet50 best-F1"][4]
    Pi = pd.read_csv(REPO / "results/resuni/rn50_f1_nsweep/predictions.csv")
    Pe = pd.read_csv(REPO / "results/resuni/rn50_f1_nsweep/predictions_external_tcga.csv")
    dext = XT.cascade_ext()["rn50_f1"]; dext = dext[dext.role == "external"].reset_index(drop=True)
    out, oks = {}, {}
    for n in NS:
        for d in ((0,) if n == "all" else (0, 1, 2)):
            def pb(P, base):
                P = P[(P.n.astype(str) == n) & (P.seed == d)].set_index(["fold", "slide_id"]); a = np.full((len(base), 3), np.nan)
                for i, (f, s) in enumerate(zip(base.fold, base.slide_id)):
                    if (f, s) in P.index: a[i] = P.loc[(f, s), [f"p_{k}" for k in MC.SUB]].to_numpy(float)
                return a
            ti = MC.cascade_table(bd, pb(Pi, bd)); ri, oki = MC.metrics(ti)
            te = MC.cascade_table(dext, pb(Pe, dext)); te["true6"] = [MC.six_true(g, m) for g, m in zip(te.true_grade_class, te.true_mutation)]
            e = XT.ensemble(te); oke = (e.six_pred == e.true6).to_numpy()
            out[(n, d)] = {"int": {"acc": ri["e2e_acc_pooled"], "balacc": ri["e2e_balacc_mean"], "macroAUC": ri["e2e_macroAUC_mean"]},
                           "ext": {"acc": float(oke.mean())}}
            oks[(n, d)] = (pd.Series(oki, index=pd.MultiIndex.from_arrays([ti.fold, ti.slide_id])),
                           pd.Series(oke, index=e.slide_id.to_numpy()))
    return out, oks


def paired(a, b, ids, n=10000, seed=0):
    d = a.astype(float) - b.astype(float); rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u))); bs = s[dr].sum(1) / c[dr].sum(1)
    res = {"two_stage": float(a.mean()), "single_stage": float(b.mean()), "diff": float(d.mean()),
           "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}
    if set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}:
        res["mcnemar_p"] = MC.mcnemar(int((a.astype(bool) & ~b.astype(bool)).sum()), int((~a.astype(bool) & b.astype(bool)).sum()))
    return res


def main():
    D = pd.read_csv(OUT / "e6_single_stage_predictions.csv"); D["n"] = D.n.astype(str)
    rows, OK = [], {}
    for (v, n, d), g in D.groupby(["variant", "n", "draw"]):
        mi, oki = internal_metrics(g[g.cohort == "internal"])
        me, oke, ide = ensemble(g[g.cohort == "tcga"])
        gi = g[g.cohort == "internal"]
        OK[(v, n, d)] = (pd.Series(oki, index=pd.MultiIndex.from_arrays([gi.fold, gi.slide_id])), pd.Series(oke, index=ide))
        rows.append({"system": v, "n": n, "draw": d, **{f"int_{k}": x for k, x in mi.items()}, **{f"ext_{k}": x for k, x in me.items()}})
    gs, gok = glioma_sparse()
    for (n, d), r in gs.items():
        rows.append({"system": "two_stage_glioma_sparse", "n": n, "draw": d, **{f"int_{k}": x for k, x in r["int"].items()},
                     **{f"ext_{k}": x for k, x in r["ext"].items()}})
    R = pd.DataFrame(rows)
    agg = R.groupby(["system", "n"], sort=False).agg(**{f"{c}_{s}": (c, s) for c in ("int_acc", "int_balacc", "int_macroAUC", "ext_acc") for s in ("mean", "std")}).reset_index()
    def tflops(row):
        if row.system == "two_stage_glioma_sparse":
            return GF_A + TE["internal"]["tiles_encoded_mean"][row.n] * GF_UNI
        return (BENCH_TILES["internal"] if row.n == "all" else int(row.n)) * GF_UNI
    agg["tflops_internal"] = agg.apply(tflops, axis=1)
    agg.to_csv(OUT / "e6_single_stage_summary.csv", index=False)
    # paired: draw-averaged correctness, same predictions
    def avg(key_fn, draws):
        s = [key_fn(d) for d in draws]; return pd.concat(s, axis=1).mean(1)
    res = {"summary": agg.to_dict(orient="records"), "paired": {}}
    for v in ("A_clam", "B_meanLR"):
        for n_ss in ("64", "256"):
            gi = avg(lambda d: gok[("64", d)][0], (0, 1, 2)); si = avg(lambda d: OK[(v, n_ss, d)][0], (0, 1, 2))
            ge = avg(lambda d: gok[("64", d)][1], (0, 1, 2)); se = avg(lambda d: OK[(v, n_ss, d)][1], (0, 1, 2))
            ki = gi.index.intersection(si.index); ke = ge.index.intersection(se.index)
            res["paired"][f"two_stage64_vs_{v}_{n_ss}"] = {
                "internal": paired(gi[ki].to_numpy(), si[ki].to_numpy(), np.array([k[1] for k in ki])),
                "tcga": paired(ge[ke].to_numpy(), se[ke].to_numpy(), np.array(list(ke)))}
    json.dump(res, open(OUT / "e6_single_stage.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(agg[["system", "n", "int_acc_mean", "int_acc_std", "int_balacc_mean", "int_macroAUC_mean", "ext_acc_mean", "ext_acc_std", "tflops_internal"]].round(3).to_string())
    for k, v in res["paired"].items():
        print(k, {c: {a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in r.items()} for c, r in v.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
