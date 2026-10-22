"""
Master comparison of every end-to-end system evaluated in this project, on identical test rows.

All systems are scored from per-slide probabilities on the five held-out test folds (5 x 44 = 220
rows, 190 unique slides), with one metric code path, so numbers are directly comparable:

  End-to-end (6-way integrated diagnosis: control / astro_low / astro_high / oligo_low / oligo_high / GBM)
    accuracy (pipeline rule), balanced accuracy, macro-F1, macro one-vs-rest AUC
    (fold mean +/- SD over 5 folds; accuracy also pooled)
  Grade (control / low / high):     accuracy, macro-F1, macro-AUC            (all 220 rows)
  Molecular subtype (3-way):        accuracy, macro-F1, macro-AUC, IDH mutant-vs-wildtype AUC
                                    (tumour slides that reach the subtype step)

For two-stage cascades the six-class AUC score is P(grade) x P(subtype) (control: P(control));
for the CLAM benchmark it is its own six-class softmax. Grade and subtype metrics of the
single-stage CLAM model are obtained by collapsing its six classes. Decision rule for cascades: a
tumour slide is correct only if the grade class is right AND a region reached the subtype step AND
the subtype is right (same as inference_end_to_end.score_row).

Compute columns come from results/master_comparison/compute_components.csv (measured parameters and
FLOPs) and the TRIDENT tile counts; see compute_costs.py.
"""

import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
BRAIN = REPO.parent
OUT = REPO / "results" / "master_comparison"
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
SIX = ["control", "astro_low", "astro_high", "oligo_low", "oligo_high", "GBM"]
GRADES = ["control", "low", "high"]


# ------------------------------------------------------------------ helpers
def six_true(g, m):
    if g == "control":
        return "control"
    return {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high",
            ("low", "IDH_mt_1p19q"): "oligo_low", ("high", "IDH_mt_1p19q"): "oligo_high",
            ("high", "IDH_wt"): "GBM"}.get((g, m))


def six_pred(gc, sub, has_b):
    if gc == "control":
        return "control"
    if not has_b:
        return "no_stageB"
    return {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high",
            ("low", "IDH_mt_1p19q"): "oligo_low", ("high", "IDH_mt_1p19q"): "oligo_high",
            ("high", "IDH_wt"): "GBM"}.get((gc, sub), "wt_low_grade")


def macro_auc(y_idx, S, k):
    a = [roc_auc_score((y_idx == c).astype(int), S[:, c]) for c in range(k) if 0 < (y_idx == c).sum() < len(y_idx)]
    return float(np.mean(a)) if a else float("nan")


def fold_ms(vals):
    v = np.array(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cluster_boot(d, ids, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u)))
    b = s[dr].sum(1) / c[dr].sum(1)
    return np.percentile(b, [2.5, 97.5])


# ------------------------------------------------------- system score tables
def cascade_table(df, pb):
    """Uniform table for a two-stage system. df has fold, slide_id, true_*, grade_class, prob_control/low/high."""
    t = df[["fold", "slide_id", "true_grade_class", "true_mutation"]].copy()
    P = df[["prob_control", "prob_low_grade", "prob_high_grade"]].to_numpy(float)
    pg = np.nan_to_num(P)
    has = ~np.isnan(pb).any(1)
    gclass = np.array(GRADES)[pg.argmax(1)]
    sub = np.array([SUB[i] if h else "NA" for i, h in zip(np.nan_to_num(pb).argmax(1), has)])
    pbf = np.where(np.isnan(pb), 1 / 3, pb)
    S = np.stack([pg[:, 0], pg[:, 1] * pbf[:, 0], pg[:, 2] * pbf[:, 0], pg[:, 1] * pbf[:, 1],
                  pg[:, 2] * pbf[:, 1], pg[:, 2] * pbf[:, 2]], 1)
    S = S / S.sum(1, keepdims=True)
    t["grade_prob"] = list(pg / pg.sum(1, keepdims=True))
    t["six_score"] = list(S)
    t["sub_prob"] = list(pb)
    t["has_b"] = has
    t["grade_pred"] = gclass
    t["sub_pred"] = sub
    t["six_pred"] = [six_pred(g, s, h) for g, s, h in zip(gclass, sub, has)]
    return t


def clam_table(results_dir):
    """CLAM-SB test predictions of the matched folds: split_{k}_results.pkl (k = 0..4 = our split_01..05;
    the CLAM training splits' test sets equal ours slide for slide). NOT clam_workspace/eval_results, which
    is a different, unmatched evaluation."""
    import pickle
    ds = pd.read_csv(BRAIN / "clam_workspace/dataset_csv/dpwsi_6class.csv").set_index("slide_id").label
    parts = []
    for k in range(5):
        r = pickle.load(open(Path(results_dir) / f"split_{k}_results.pkl", "rb"))
        ids = list(r.keys())
        d = pd.DataFrame(np.stack([np.asarray(r[i]["prob"]).reshape(-1) for i in ids]),
                         columns=[f"p_{j}" for j in range(6)])
        d["slide_id"] = [str(i) for i in ids]
        d["Y"] = [int(r[i]["label"]) for i in ids]
        d["fold"] = k + 1
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lab = ds.reindex(d.slide_id).to_numpy()
    idx_by_label = pd.Series(d.Y.astype(int).to_numpy(), index=lab).groupby(level=0).first().to_dict()
    name = {"control": "control", "low_grade_astro": "astro_low", "high_grade_astro": "astro_high",
            "low_grade_oligo": "oligo_low", "high_grade_oligo": "oligo_high", "GBM": "GBM"}
    col_of = {name[l]: f"p_{i}" for l, i in idx_by_label.items()}
    S = d[[col_of[c] for c in SIX]].to_numpy(float)
    t = pd.DataFrame({"fold": d.fold, "slide_id": d.slide_id})
    true6 = [name[l] for l in lab]
    t["true6"] = true6
    t["six_score"] = list(S)
    t["six_pred"] = np.array(SIX)[S.argmax(1)]
    # collapse
    t["true_grade_class"] = ["control" if c == "control" else ("low" if c.endswith("low") else "high") for c in true6]
    t["true_mutation"] = [{"control": np.nan, "astro_low": "IDH_mt", "astro_high": "IDH_mt", "oligo_low": "IDH_mt_1p19q",
                           "oligo_high": "IDH_mt_1p19q", "GBM": "IDH_wt"}[c] for c in true6]
    G = np.stack([S[:, 0], S[:, 1] + S[:, 3], S[:, 2] + S[:, 4] + S[:, 5]], 1)
    t["grade_prob"] = list(G)
    t["grade_pred"] = np.array(GRADES)[G.argmax(1)]
    T = np.stack([S[:, 1] + S[:, 2], S[:, 3] + S[:, 4], S[:, 5]], 1)
    T = T / T.sum(1, keepdims=True).clip(1e-12)
    t["sub_prob"] = list(T)
    t["sub_pred"] = np.array(SUB)[T.argmax(1)]
    t["has_b"] = True
    return t


def metrics(t):
    """Everything from a uniform table. Returns a dict."""
    if "true6" not in t:
        t = t.assign(true6=[six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)])
    ok = (t.true6 == t.six_pred).to_numpy()
    tum = (t.true_grade_class != "control").to_numpy()
    y6 = t.true6.map({c: i for i, c in enumerate(SIX)}).to_numpy()
    S = np.stack(t.six_score.to_numpy())
    r = {"n_rows": len(t), "n_unique_slides": t.slide_id.nunique(), "e2e_correct": int(ok.sum())}
    r["e2e_acc_pooled"] = ok.mean()
    per = {"acc": [], "bal": [], "f1": [], "auc": []}
    for f in sorted(t.fold.unique()):
        i = (t.fold == f).to_numpy()
        per["acc"].append(ok[i].mean())
        per["bal"].append(balanced_accuracy_score(t.true6[i], t.six_pred[i]) if False else
                          np.mean([(t.six_pred[i][(t.true6[i] == c).to_numpy()] == c).mean()
                                   for c in SIX if (t.true6[i] == c).any()]))
        per["f1"].append(f1_score(t.true6[i], t.six_pred[i], labels=SIX, average="macro", zero_division=0))
        per["auc"].append(macro_auc(y6[i], S[i], 6))
    for k, name in (("acc", "e2e_acc"), ("bal", "e2e_balacc"), ("f1", "e2e_macroF1"), ("auc", "e2e_macroAUC")):
        r[f"{name}_mean"], r[f"{name}_sd"] = fold_ms(per[k])
    # pooled
    r["e2e_macroF1_pooled"] = f1_score(t.true6, t.six_pred, labels=SIX, average="macro", zero_division=0)
    r["e2e_macroAUC_pooled"] = macro_auc(y6, S, 6)
    # grade
    gy = t.true_grade_class.map({c: i for i, c in enumerate(GRADES)}).to_numpy()
    G = np.stack(t.grade_prob.to_numpy())
    gp = G.argmax(1)
    r["grade_acc"] = (gp == gy).mean()
    r["grade_macroF1"] = f1_score(gy, gp, average="macro", labels=[0, 1, 2], zero_division=0)
    r["grade_macroAUC"] = macro_auc(gy, G, 3)
    # subtype among tumour slides that reach the subtype step
    m = tum & t.has_b.to_numpy()
    sy = t.true_mutation[m].map({k: i for i, k in enumerate(SUB)}).to_numpy()
    Pb = np.stack(t.sub_prob.to_numpy())[m]
    sp = Pb.argmax(1)
    r["subtype_n"] = int(m.sum())
    r["subtype_acc"] = (sp == sy).mean()
    r["subtype_macroF1"] = f1_score(sy, sp, average="macro", labels=[0, 1, 2], zero_division=0)
    r["subtype_macroAUC"] = macro_auc(sy, Pb, 3)
    r["IDH_mut_vs_wt_AUC"] = roc_auc_score((sy == 2).astype(int), Pb[:, 2])
    # error decomposition
    wrong = ~ok
    gok = (t.grade_pred == t.true_grade_class).to_numpy()
    r["wrong_grade"] = int((wrong & ~gok).sum())
    r["wrong_noStageB"] = int((wrong & gok & tum & ~t.has_b.to_numpy()).sum())
    r["wrong_subtype"] = int((wrong & gok & tum & t.has_b.to_numpy()).sum())
    return r, ok


# ------------------------------------------------------------------- sources
def archived(path_or_glob, per_fold=False):
    if per_fold:
        d = pd.concat([pd.read_excel(p, sheet_name="per_slide") for p in sorted(glob.glob(str(REPO / path_or_glob)))],
                      ignore_index=True)
    else:
        d = pd.read_excel(REPO / path_or_glob, sheet_name="per_slide")
    return d


def sub_probs_archived(d):
    return d[[f"prob_IDH_{k[4:]}" if False else f"prob_{k}" for k in SUB]].to_numpy(float)


def multi(run_csv, prefix, tta=False, w=None):
    d = pd.read_csv(REPO / run_csv)
    d = d[d.role == "test"].reset_index(drop=True)
    pre = f"{prefix}__tta_p_" if tta else f"{prefix}__p_"
    pb = d[[pre + k for k in SUB]].to_numpy(float)
    if w:
        pa = d[[f"stageA_sub_p_{k}" for k in SUB]].to_numpy(float)
        pb = (1 - w) * pb + w * pa
    return d, pb


def resuni(config, n, seed, base_df):
    """base_df: archived end-to-end table of the same Stage A (grade decisions)."""
    P = pd.read_csv(REPO / "results" / "resuni" / config / "predictions.csv")
    P = P[(P.n.astype(str) == str(n)) & (P.seed == seed)].set_index(["fold", "slide_id"])
    pb = np.full((len(base_df), 3), np.nan)
    for i, (f, s) in enumerate(zip(base_df.fold, base_df.slide_id)):
        if (f, s) in P.index:
            pb[i] = P.loc[(f, s), [f"p_{k}" for k in SUB]].to_numpy(float)
    return pb


def build_systems():
    """All systems as (group, name, uniform score table, cost dict, base table)."""
    OUT.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(OUT / "compute_components.csv").set_index("component")
    gf = {k: float(v) for k, v in {
        "rn18": comp.loc["ResNet-18 (3-class head), 2048x2048 input", "gflops_per_forward"],
        "rn50": comp.loc["ResNet-50 (3-class head), 2048x2048 input", "gflops_per_forward"],
        "uni": comp.loc["UNI2-h ViT (frozen encoder), 224x224 tile", "gflops_per_forward"]}.items()}
    P = {"rn18": 11178051, "rn50": 23514179, "uni": 681394176, "lr": 4611, "clam": 1059091}
    tiles_full = pd.read_csv(OUT / "compute_tiles.csv").n_tissue_tiles_20x.mean()
    full_flops = tiles_full * gf["uni"]
    A0 = pd.read_csv(REPO / "results/e2e_multi_rn18_A0/per_slide.csv")
    mean_regions = A0[(A0.role == "test") & (A0.n_regions > 0)].n_regions.mean()

    systems = []      # (group, name, table, cost dict)

    def cost_cascade(arch, regions):
        f = gf[arch]
        return {"stageA": arch, "stageB": arch, "uni_tiles": 0,
                "params_total": 2 * P[arch], "params_trained": 2 * P[arch], "params_frozen": 0,
                "gflops_per_slide": (1 + 64) * f + regions * f, "encoder_tile_forwards": 0}

    def cost_resuni(arch, n, ntiles_all):
        nt = n if n != "all" else ntiles_all
        return {"stageA": arch, "stageB": "UNI2-h + LR", "uni_tiles": nt,
                "params_total": P[arch] + P["uni"] + P["lr"], "params_trained": P[arch] + P["lr"],
                "params_frozen": P["uni"], "gflops_per_slide": (1 + 64) * gf[arch] + nt * gf["uni"],
                "encoder_tile_forwards": nt}

    # ---- archived ResNet cascades
    for name, d, arch in (
        ("ResNet18 best-F1 (S0)", archived("results/e2e_testset_rn18_f1/e2e_testset_report_RN18_f1.xlsx"), "rn18"),
        ("ResNet18 best-AUC", archived("results/e2e_testset_rn18/e2e_testset_fold_*/e2e_testset_report.xlsx", True), "rn18"),
        ("ResNet50 best-F1", archived("results/e2e_testset_rn50_f1/e2e_testset_report_RN50_f1.xlsx"), "rn50"),
        ("ResNet50 best-AUC", archived("results/e2e_testset_rn50/e2e_testset_report.xlsx"), "rn50"),
    ):
        systems.append(("ResNet cascades (archived)", name, cascade_table(d, sub_probs_archived(d)),
                        cost_cascade(arch, mean_regions), d))

    # ---- RN18 refinement study
    fw = json.loads((REPO / "results/cascade_refinement_rn18/fusion_weights.json").read_text())
    for name, run, pre, tta, fuse in (
        ("RN18 + continued Stage B (S0-c)", "results/e2e_multi_rn18_A0/per_slide.csv", "Bc", False, None),
        ("RN18 + refined Stage B (S1)", "results/e2e_multi_rn18_A0/per_slide.csv", "Br", False, None),
        ("RN18 grade-only Stage A ft (A-ctrl)", "results/e2e_multi_rn18_Actrl/per_slide.csv", "B0", False, None),
        ("RN18 multitask Stage A (A-mt)", "results/e2e_multi_rn18_Amt/per_slide.csv", "B0", False, None),
        ("RN18 multitask + continued B (S2)", "results/e2e_multi_rn18_Amt/per_slide.csv", "Bc", False, None),
        ("RN18 multitask + refined B (S12)", "results/e2e_multi_rn18_Amt/per_slide.csv", "Br", False, None),
        ("RN18 S2 + subtype-head fusion", "results/e2e_multi_rn18_Amt/per_slide.csv", "Bc", False, "S2+fuse"),
        ("RN18 S12 + TTA + fusion", "results/e2e_multi_rn18_Amt/per_slide.csv", "Br", True, "S12+TTA+fuse"),
    ):
        w = fw[fuse]["w"] if fuse else None
        d, pb = multi(run, pre, tta, w)
        systems.append(("ResNet-18 refinement study", name, cascade_table(d, pb), cost_cascade("rn18", mean_regions), d))

    # ---- UNI2-h at Stage B
    base = {"rn18_f1": ("ResNet18 best-F1 (S0)", "rn18"), "rn50_f1": ("ResNet50 best-F1", "rn50"),
            "rn50_auc": ("ResNet50 best-AUC", "rn50")}
    tabs = {s[1]: s for s in systems}
    for cfg, (bname, arch) in base.items():
        if not (REPO / "results" / "resuni" / cfg / "predictions.csv").exists():
            continue
        bd = tabs[bname][4]
        for n in (64, 32, 8, "all"):
            pb = resuni(cfg, n, 0, bd)
            ntiles_all = 2900.0
            label = f"{bname.split(' (')[0]} Stage A + UNI2-h, {n} tiles"
            systems.append(("UNI2-h at Stage B (sparse)", label, cascade_table(bd, pb), cost_resuni(arch, n, ntiles_all), bd))

    # ---- CLAM benchmark
    clam = clam_table(BRAIN / "mahmoodlab/CLAM/results/glioma_subtype_clam_s1")
    systems.append(("Benchmark (full slide)", "TRIDENT + UNI2-h + CLAM-SB", clam,
                    {"stageA": "-", "stageB": "UNI2-h + CLAM-SB", "uni_tiles": tiles_full,
                     "params_total": P["uni"] + P["clam"], "params_trained": P["clam"], "params_frozen": P["uni"],
                     "gflops_per_slide": tiles_full * (gf["uni"]), "encoder_tile_forwards": tiles_full}, None))

    return systems, tiles_full, full_flops, base, tabs


def main():
    systems, tiles_full, full_flops, base, tabs = build_systems()
    rows, oks, keys = [], {}, {}
    for group, name, t, cost, _ in systems:
        r, ok = metrics(t)
        oks[name] = ok
        keys[name] = t[["fold", "slide_id"]].reset_index(drop=True)
        seeds_acc = None
        if group == "UNI2-h at Stage B (sparse)":
            cfg = [c for c, (b, _a) in base.items() if name.startswith(b.split(' (')[0])][0]
            n = name.split(", ")[1].split(" ")[0]
            accs = []
            for sd in (0, 1, 2):
                bd = tabs[base[cfg][0]][4]
                tt = cascade_table(bd, resuni(cfg, n, sd, bd))
                accs.append(metrics(tt)[0]["e2e_acc_pooled"])
            seeds_acc = (np.mean(accs), np.std(accs, ddof=1))
        rows.append({"group": group, "system": name, **r, **cost,
                     "e2e_acc_pooled_over_tile_draws_mean": seeds_acc[0] if seeds_acc else np.nan,
                     "e2e_acc_pooled_over_tile_draws_sd": seeds_acc[1] if seeds_acc else np.nan,
                     "pct_full_UNI_tiles": 100 * cost["encoder_tile_forwards"] / tiles_full if cost["uni_tiles"] else np.nan,
                     "pct_full_UNI_gflops": 100 * cost["gflops_per_slide"] / full_flops})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "master_table.csv", index=False)

    # paired comparisons to the benchmark and to the matching ResNet cascade
    pairs = []
    ck = keys["TRIDENT + UNI2-h + CLAM-SB"]
    for name in R.system:
        if name == "TRIDENT + UNI2-h + CLAM-SB" or not name.startswith("ResNet"):
            continue
        if not ((keys[name].fold.values == ck.fold.values) & (keys[name].slide_id.values == ck.slide_id.values)).all():
            merged = keys[name].merge(ck.reset_index(), on=["fold", "slide_id"])
            ia = merged.index.to_numpy()
            ib = merged["index"].to_numpy()
            a_, b_ = oks[name], oks["TRIDENT + UNI2-h + CLAM-SB"]
            a_, b_ = a_[ia], b_[ib]
            ids = keys[name].slide_id.values[ia]
        else:
            a_, b_, ids = oks[name], oks["TRIDENT + UNI2-h + CLAM-SB"], ck.slide_id.values
        d = b_.astype(float) - a_.astype(float)
        lo, hi = cluster_boot(d, ids)
        n01, n10 = int(((a_ == 1) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ == 1)).sum())
        pairs.append({"A": name, "B": "UNI2-h + CLAM benchmark", "acc_A": a_.mean(), "acc_B": b_.mean(),
                      "delta_B_minus_A": d.mean(), "ci_lo": lo, "ci_hi": hi, "mcnemar_p": mcnemar(n01, n10),
                      "n_rows": len(a_)})
    for name in R.system[R.group == "UNI2-h at Stage B (sparse)"]:
        b = [x for x in base.values() if name.startswith(x[0].split(' (')[0])][0][0]
        a_, b_ = oks[b], oks[name]
        d = b_.astype(float) - a_.astype(float)
        lo, hi = cluster_boot(d, keys[b].slide_id.values)
        pairs.append({"A": b, "B": name, "acc_A": a_.mean(), "acc_B": b_.mean(),
                      "delta_B_minus_A": d.mean(), "ci_lo": lo, "ci_hi": hi,
                      "mcnemar_p": mcnemar(int(((a_ == 1) & (b_ == 0)).sum()), int(((a_ == 0) & (b_ == 1)).sum())),
                      "n_rows": len(a_)})
    pd.DataFrame(pairs).to_csv(OUT / "paired_comparisons.csv", index=False)
    print(R[["group", "system", "e2e_acc_pooled", "e2e_macroF1_mean", "e2e_macroAUC_mean", "pct_full_UNI_gflops"]]
          .round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
