"""
Experiment E4 (docs/experiments/EXPERIMENT_LOG.md): ResNet-18 vs ResNet-50 as Stage A of Glioma-SPARSE
(best-F1 checkpoints, Stage B = UNI2-h + logistic head trained on the regions of the same Stage A).

Paired comparison on the same 220 internal test predictions (5 EBRAINS folds) and the same 180 TCGA slides
(ensemble of the five fold models), for
  - Stage A alone (three-grade decision: accuracy, macro-F1, macro-AUC; tumour slides without a selected region),
  - the complete pipeline with 64 tiles (six-class accuracy, macro-AUC, balanced accuracy, subtype accuracy),
    for each of the 3 tile draws and averaged over draws,
  - accuracy versus number of tiles (8 ... 1024, all; results/resuni/rn{18,50}_f1_nsweep),
  - overlap of the selected regions (same test slides),
  - compute (GFLOPs, parameters, measured Stage-A latency from compute_components.csv).
Paired differences (ResNet-18 minus ResNet-50): bootstrap over slides (10,000 resamples, clustered by slide),
95% and 90% CI, exact McNemar test. Equivalence (pre-specified 22 Oct 2026): the 90% CI of the draw-averaged
six-class accuracy difference lies within +-5 percentage points (two one-sided tests at alpha = 0.05), internally
and externally.
    python -m scripts.analysis.rn18_vs_rn50        (Glioma-SPARSE venv)
Outputs: results/revision_oct22/e4_rn18_vs_rn50.json, e4_tile_sweep.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
OUT = REPO / "results" / "revision_oct22"; OUT.mkdir(parents=True, exist_ok=True)
SUB = MC.SUB; MARGIN = 0.05; DRAWS = (0, 1, 2)
ARCH = {"rn18": ("ResNet18 best-F1 (S0)", "rn18_f1"), "rn50": ("ResNet50 best-F1", "rn50_f1")}


def pb_from(P, bd):
    P = P.set_index(["fold", "slide_id"]); pb = np.full((len(bd), 3), np.nan)
    for i, (f, s) in enumerate(zip(bd.fold, bd.slide_id)):
        if (f, s) in P.index: pb[i] = P.loc[(f, s), [f"p_{k}" for k in SUB]].to_numpy(float)
    return pb


def paired(a, b, ids, seed=0, n=10000):
    """a, b: per-row scores (correct = 1) of ResNet-18 and ResNet-50 on the same rows. Clustered bootstrap of mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u))); bs = s[dr].sum(1) / c[dr].sum(1)
    out = {"rn18": float(np.mean(a)), "rn50": float(np.mean(b)), "diff": float(d.mean()),
           "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "ci90": np.percentile(bs, [5, 95]).tolist()}
    if set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}:
        a_, b_ = np.asarray(a, bool), np.asarray(b, bool)
        out["mcnemar_p"] = MC.mcnemar(int((a_ & ~b_).sum()), int((~a_ & b_).sum()))
        out["only_rn18_correct"], out["only_rn50_correct"] = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
    out["equivalent_5pp"] = bool(-MARGIN < out["ci90"][0] and out["ci90"][1] < MARGIN)
    return out


def summary(t):
    r, ok = MC.metrics(t)
    keep = ("e2e_acc_pooled", "e2e_acc_mean", "e2e_acc_sd", "e2e_macroAUC_mean", "e2e_macroAUC_pooled", "e2e_balacc_mean",
            "e2e_macroF1_pooled", "grade_acc", "grade_macroF1", "grade_macroAUC", "subtype_n", "subtype_acc", "subtype_macroAUC",
            "wrong_grade", "wrong_noStageB", "wrong_subtype")
    return {k: float(r[k]) for k in keep}, ok


def internal():
    systems, *_ , tabs = MC.build_systems()
    bd = {a: tabs[ARCH[a][0]][4].sort_values(["fold", "slide_id"]).reset_index(drop=True) for a in ARCH}
    assert (bd["rn18"][["fold", "slide_id"]].values == bd["rn50"][["fold", "slide_id"]].values).all()
    ids = bd["rn18"].slide_id.to_numpy()
    res = {"n_rows": len(ids), "n_slides": int(len(set(ids)))}
    T = {a: {d: MC.cascade_table(bd[a], MC.resuni(ARCH[a][1], 64, d, bd[a])) for d in DRAWS} for a in ARCH}
    S = {a: {d: summary(T[a][d]) for d in DRAWS} for a in ARCH}
    res["per_arch"] = {a: {"draw0": S[a][0][0], "acc_draws": [S[a][d][0]["e2e_acc_pooled"] for d in DRAWS],
                           "auc_draws": [S[a][d][0]["e2e_macroAUC_mean"] for d in DRAWS],
                           "subtype_acc_draws": [S[a][d][0]["subtype_acc"] for d in DRAWS]} for a in ARCH}
    # Stage A grade decision (identical across draws)
    g = {a: (T[a][0].grade_pred == T[a][0].true_grade_class).to_numpy() for a in ARCH}
    res["stageA_grade_acc"] = paired(g["rn18"], g["rn50"], ids)
    res["stageA_grade_per_fold"] = {a: [float(g[a][(bd[a].fold == f).to_numpy()].mean()) for f in range(1, 6)] for a in ARCH}
    res["stageA_same_grade_call"] = float((T["rn18"][0].grade_pred.values == T["rn50"][0].grade_pred.values).mean())
    # complete pipeline: per draw and averaged over draws
    res["e2e_per_draw"] = [paired(S["rn18"][d][1], S["rn50"][d][1], ids) for d in DRAWS]
    okm = {a: np.mean([S[a][d][1] for d in DRAWS], 0) for a in ARCH}
    res["e2e_draw_averaged"] = paired(okm["rn18"], okm["rn50"], ids)
    res["e2e_per_fold_draw_avg"] = {a: [float(okm[a][(bd[a].fold == f).to_numpy()].mean()) for f in range(1, 6)] for a in ARCH}
    # subtype step only, on rows where both Stage A models call the correct tumour grade and both selected regions
    both = (g["rn18"] & g["rn50"] & (T["rn18"][0].true_grade_class != "control").to_numpy()
            & T["rn18"][0].has_b.to_numpy() & T["rn50"][0].has_b.to_numpy())
    st = {a: np.mean([(T[a][d].sub_pred == T[a][d].true_mutation).to_numpy() for d in DRAWS], 0)[both] for a in ARCH}
    res["subtype_common_rows"] = {"n": int(both.sum()), **paired(st["rn18"], st["rn50"], ids[both])}
    # region overlap (cell sets in WSI coordinates; identical cells give identical boxes)
    reg = {a: pd.read_csv(REPO / "results" / f"e2e_regions_{ARCH[a][1]}.csv").set_index(["fold", "slide_id"]).bboxes for a in ARCH}
    jac, top, cnt = [], [], {a: [] for a in ARCH}
    for k in reg["rn18"].index.intersection(reg["rn50"].index):
        A = [tuple(b) for b in json.loads(reg["rn18"][k])]; B = [tuple(b) for b in json.loads(reg["rn50"][k])]
        cnt["rn18"].append(len(A)); cnt["rn50"].append(len(B))
        if A or B: jac.append(len(set(A) & set(B)) / len(set(A) | set(B)))
        if A and B: top.append(A[0] == B[0])
    res["regions"] = {"n_slides_both_tumour": len(jac), "median_jaccard": float(np.median(jac)), "mean_jaccard": float(np.mean(jac)),
                      "top_region_identical": float(np.mean(top)), "mean_n_regions": {a: float(np.mean(cnt[a])) for a in ARCH}}
    return res


def external():
    cas = XT.cascade_ext()
    base = {a: cas[ARCH[a][1]][lambda d: d.role == "external"].sort_values(["fold", "slide_id"]).reset_index(drop=True) for a in ARCH}
    res, E, F = {}, {}, {}
    for a in ARCH:
        E[a], F[a] = {}, {}
        for d in DRAWS:
            t = MC.cascade_table(base[a], XT.resuni_ext(ARCH[a][1], 64, base[a], d))
            t["true6"] = [MC.six_true(g_, m_) for g_, m_ in zip(t.true_grade_class, t.true_mutation)]
            F[a][d] = summary(t)
            E[a][d] = XT.ensemble(t).sort_values("slide_id").reset_index(drop=True)
    ids = E["rn18"][0].slide_id.to_numpy(); assert (ids == E["rn50"][0].slide_id.to_numpy()).all()
    ok = {a: {d: (E[a][d].six_pred == E[a][d].true6).to_numpy() for d in DRAWS} for a in ARCH}
    g = {a: (E[a][0].grade_pred == E[a][0].true_grade_class).to_numpy() for a in ARCH}
    res["n_slides"] = len(ids)
    res["per_arch"] = {a: {"ens_acc_draws": [float(ok[a][d].mean()) for d in DRAWS],
                           "ens_macroAUC_draw0": float(MC.metrics(E[a][0])[0]["e2e_macroAUC_pooled"]),
                           "ens_subtype_acc_draw0": float(MC.metrics(E[a][0])[0]["subtype_acc"]),
                           "ens_grade_acc": float(g[a].mean()),
                           "fold_model_acc_draws": [F[a][d][0]["e2e_acc_pooled"] for d in DRAWS],
                           "fold_model_grade_acc": F[a][0][0]["grade_acc"]} for a in ARCH}
    res["stageA_grade_acc_ensemble"] = paired(g["rn18"], g["rn50"], ids)
    res["e2e_per_draw"] = [paired(ok["rn18"][d], ok["rn50"][d], ids) for d in DRAWS]
    res["e2e_draw_averaged"] = paired(np.mean([ok["rn18"][d] for d in DRAWS], 0), np.mean([ok["rn50"][d] for d in DRAWS], 0), ids)
    # individual fold models (5 x 180 rows), draw-averaged
    assert (base["rn18"][["fold", "slide_id"]].values == base["rn50"][["fold", "slide_id"]].values).all()
    fk = {a: np.mean([F[a][d][1] for d in DRAWS], 0) for a in ARCH}
    res["e2e_fold_models_draw_averaged"] = paired(fk["rn18"], fk["rn50"], base["rn18"].slide_id.to_numpy())
    return res


def sweep():
    """Accuracy vs number of tiles for both Stage A models (mean and SD over 3 draws; heads retrained per n and draw)."""
    rows = []
    systems, *_ , tabs = MC.build_systems()
    cas = XT.cascade_ext()
    for a, (bname, cfg) in ARCH.items():
        bd = tabs[bname][4]
        dext = cas[cfg][lambda d: d.role == "external"].reset_index(drop=True)
        Pi = pd.read_csv(REPO / "results" / "resuni" / f"{cfg}_nsweep" / "predictions.csv")
        Pe = pd.read_csv(REPO / "results" / "resuni" / f"{cfg}_nsweep" / "predictions_external_tcga.csv")
        for n in sorted(Pi.n.astype(str).unique(), key=lambda v: 10 ** 9 if v == "all" else int(v)):
            for sd in DRAWS:
                ti = MC.cascade_table(bd, pb_from(Pi[(Pi.n.astype(str) == n) & (Pi.seed == sd)], bd))
                te = MC.cascade_table(dext, pb_from(Pe[(Pe.n.astype(str) == n) & (Pe.seed == sd)], dext))
                te["true6"] = [MC.six_true(g_, m_) for g_, m_ in zip(te.true_grade_class, te.true_mutation)]
                e = XT.ensemble(te)
                rows.append({"arch": a, "n": n, "seed": sd, "int_acc": MC.metrics(ti)[0]["e2e_acc_pooled"],
                             "ext_acc": float((e.six_pred == e.true6).mean())})
    R = pd.DataFrame(rows); R.to_csv(OUT / "e4_tile_sweep.csv", index=False)
    return R.groupby(["arch", "n"], sort=False).agg(int_mean=("int_acc", "mean"), int_sd=("int_acc", "std"),
                                                    ext_mean=("ext_acc", "mean"), ext_sd=("ext_acc", "std")).reset_index()


def compute():
    c = pd.read_csv(REPO / "results" / "master_comparison" / "compute_components.csv").set_index("component")
    r18, r50, uni = (c.loc[k] for k in ("ResNet-18 (3-class head), 2048x2048 input", "ResNet-50 (3-class head), 2048x2048 input",
                                         "UNI2-h ViT (frozen encoder), 224x224 tile"))
    out = {}
    for a, r in (("rn18", r18), ("rn50", r50)):
        out[a] = {"params": int(r.params_total), "gflops_forward": float(r.gflops_per_forward),
                  "stageA_tflops_per_slide": 65 * float(r.gflops_per_forward) / 1e3,
                  "total_tflops_64_tiles": (65 * float(r.gflops_per_forward) + 64 * float(uni.gflops_per_forward)) / 1e3,
                  "stageA_latency_s_fp32": 65 * float(r.latency_s_per_forward_fp32_RTX5080)}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"internal": internal(), "external": external(), "compute": compute()}
    sw = sweep(); res["tile_sweep"] = sw.to_dict(orient="records")
    json.dump(res, open(OUT / "e4_rn18_vs_rn50.json", "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("internal", "external", "compute")}, indent=1, default=float))
    print(sw.round(3).to_string())
