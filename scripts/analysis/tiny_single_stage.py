"""
Experiment E7c (docs/experiments/EXPERIMENT_LOG.md): E6 variant B with DistillPath-KS16-UNI2h instead of UNI2-h.
Single-stage six-class prediction from n tiles drawn at random from the WHOLE slide (no Stage A): mean of the n tile
features -> StandardScaler -> class-balanced logistic regression over the six classes, per fold and n, C on the
validation slides; n = 64, 128, 256, 512, 1024 (three draws) and all. Tile draws, folds, test sets and head recipe are
those of single_stage_ntiles.py, features from tiny_random_tiles.py --pass all.
Compared with: E6 variant B (UNI2-h, same tiles), the two-stage Glioma-SPARSE with UNI2-h (rn50_f1_nsweep) and with
DistillPath (rn50_f1_dp_uni2h), and the TRIDENT + UNI2-h + CLAM benchmark (all tiles).
    python -m scripts.analysis.tiny_single_stage        (mahmoodlab venv)
Output: results/revision_oct22/e7c_single_stage_predictions.csv, e7c_single_stage.json, e7c_single_stage_summary.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scripts.analysis import single_stage_ntiles as S6
from scripts.analysis import single_stage_eval as SE
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
from scripts.analysis.tiny_random_tiles import OUT as RND

OUT = SE.OUT
GF_DP = 8.48166912e-3          # TFLOPs per ViT-S/16 tile


def embeddings():
    emb = {}
    slides = [(s, "internal") for s in sorted(set().union(*[set(S6.SPLIT[k]) for k in S6.SPLIT])) if s in S6.feat_int] + \
             [(s, "tcga") for s in sorted(S6.tcga) if s in S6.feat_ext]
    for i, (sid, coh) in enumerate(slides):
        with h5py.File(RND / coh / "dp_uni2h" / f"{sid}_all.h5") as h:
            F = h["features"][:].astype(np.float32); assert (h["idx"][:] == np.arange(len(F))).all()
        for key, idx in S6.subsets(sid, len(F)).items():
            emb[(sid, *key)] = F[idx].mean(0)
        if i % 100 == 0: print(f"  {i}/{len(slides)}", flush=True)
    return emb


def two_stage(cfg):
    """Two-stage results per n and draw for one Stage B config (as SE.glioma_sparse, any config)."""
    systems, *_, tabs = MC.build_systems(); bd = tabs["ResNet50 best-F1"][4]
    Pi = pd.read_csv(SE.REPO / "results" / "resuni" / cfg / "predictions.csv")
    Pe = pd.read_csv(SE.REPO / "results" / "resuni" / cfg / "predictions_external_tcga.csv")
    dext = XT.cascade_ext()["rn50_f1"]; dext = dext[dext.role == "external"].reset_index(drop=True)
    rows, oks = [], {}
    for n in SE.NS:
        for d in ((0,) if n == "all" else (0, 1, 2)):
            def pb(P, base):
                P = P[(P.n.astype(str) == n) & (P.seed == d)].set_index(["fold", "slide_id"]); a = np.full((len(base), 3), np.nan)
                for i, (f, s) in enumerate(zip(base.fold, base.slide_id)):
                    if (f, s) in P.index: a[i] = P.loc[(f, s), [f"p_{k}" for k in MC.SUB]].to_numpy(float)
                return a
            ti = MC.cascade_table(bd, pb(Pi, bd)); ri, oki = MC.metrics(ti)
            te = MC.cascade_table(dext, pb(Pe, dext)); te["true6"] = [MC.six_true(g, m) for g, m in zip(te.true_grade_class, te.true_mutation)]
            e = XT.ensemble(te); oke = (e.six_pred == e.true6).to_numpy()
            rows.append({"n": n, "draw": d, "int_acc": ri["e2e_acc_pooled"], "int_balacc": ri["e2e_balacc_mean"],
                         "int_macroAUC": ri["e2e_macroAUC_mean"], "ext_acc": float(oke.mean())})
            oks[(n, d)] = (pd.Series(oki, index=pd.MultiIndex.from_arrays([ti.fold, ti.slide_id])), pd.Series(oke, index=e.slide_id.to_numpy()))
    return rows, oks


def main():
    pred = OUT / "e7c_single_stage_predictions.csv"
    if not pred.exists():
        rows = S6.single_stage(embeddings())
        D = pd.DataFrame(rows); D["variant"] = "B_meanLR_distillpath"
        D["true6"] = [S6.lab_int[s] if c == "internal" else S6.tcga[s] for s, c in zip(D.slide_id, D.cohort)]
        D.to_csv(pred, index=False)
    D = pd.concat([pd.read_csv(pred), pd.read_csv(OUT / "e6_single_stage_predictions.csv").query("variant == 'B_meanLR'")])
    D["n"] = D.n.astype(str)
    rows, OK = [], {}
    for (v, n, d), g in D.groupby(["variant", "n", "draw"]):
        mi, oki = SE.internal_metrics(g[g.cohort == "internal"]); me, oke, ide = SE.ensemble(g[g.cohort == "tcga"])
        gi = g[g.cohort == "internal"]
        OK[(v, n, d)] = (pd.Series(oki, index=pd.MultiIndex.from_arrays([gi.fold, gi.slide_id])), pd.Series(oke, index=ide))
        rows.append({"system": v, "n": n, "draw": d, **{f"int_{k}": x for k, x in mi.items()}, "ext_acc": me["acc"]})
    for cfg, name in (("rn50_f1_nsweep", "two_stage_uni2h"), ("rn50_f1_dp_uni2h", "two_stage_distillpath")):
        r, oks = two_stage(cfg)
        rows += [{"system": name, **x} for x in r]
        OK.update({(name, n, d): v for (n, d), v in oks.items()})
    R = pd.DataFrame(rows)
    agg = R.groupby(["system", "n"], sort=False).agg(**{f"{c}_{s}": (c, s) for c in ("int_acc", "int_balacc", "int_macroAUC", "ext_acc") for s in ("mean", "std")}).reset_index()
    tiles = lambda n: SE.BENCH_TILES["internal"] if n == "all" else int(n)
    gf = {"B_meanLR_distillpath": lambda n: tiles(n) * GF_DP, "B_meanLR": lambda n: tiles(n) * SE.GF_UNI,
          "two_stage_uni2h": lambda n: SE.GF_A + SE.TE["internal"]["tiles_encoded_mean"][n] * SE.GF_UNI,
          "two_stage_distillpath": lambda n: SE.GF_A + SE.TE["internal"]["tiles_encoded_mean"][n] * GF_DP}
    agg["tflops_internal"] = [gf[s](n) for s, n in zip(agg.system, agg.n)]
    agg.to_csv(OUT / "e7c_single_stage_summary.csv", index=False)
    avg = lambda key, part: pd.concat([OK[(*key, d)][part] for d in (0, 1, 2)], axis=1).mean(1)
    res = {"summary": agg.to_dict(orient="records"), "paired": {}}
    for n_ss in ("64", "256"):
        for ref in ("two_stage_distillpath", "two_stage_uni2h"):
            out = {}
            for part, nm in ((0, "internal"), (1, "tcga")):
                a, b = avg((ref, "64"), part), avg(("B_meanLR_distillpath", n_ss), part)
                k = a.index.intersection(b.index)
                ids = np.array([x[1] for x in k]) if part == 0 else np.array(list(k))
                out[nm] = SE.paired(a[k].to_numpy(), b[k].to_numpy(), ids)
            res["paired"][f"{ref}64_vs_single_stage_distillpath_{n_ss}"] = out
    json.dump(res, open(OUT / "e7c_single_stage.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(agg[["system", "n", "int_acc_mean", "int_acc_std", "int_balacc_mean", "int_macroAUC_mean", "ext_acc_mean", "ext_acc_std", "tflops_internal"]].round(3).to_string())
    for k, v in res["paired"].items():
        print(k, {c: {a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b]) for a, b in r.items()} for c, r in v.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
