"""
Experiment E7 (docs/experiments/EXPERIMENT_LOG.md): tiny ViT-S/16 encoders (22M, 8.5 GFLOPs per tile) in Stage B
instead of UNI2-h (681M, 360.6 GFLOPs per tile). Stage A (ResNet-50 best-F1), regions, tile draws, head recipe and folds
are identical; only the tile features differ (tiny_encoder_features.py, heads by resuni_nsweep.py --features-root,
configs rn50_f1_<encoder>; reference rn50_f1_nsweep).

Decision rule, fixed 27 Oct 2026 before computing: an encoder may replace UNI2-h only if it is non-inferior, i.e. the
lower bound of the 90% CI of the draw-averaged six-class accuracy difference (encoder minus UNI2-h, 64 tiles) lies above
-0.05, internally (220 test predictions) AND on TCGA (180 slides, ensemble of the five fold models). Bootstrap over
slides (10,000), exact McNemar. Also reported: subtype accuracy on the rows Stage A grades correctly (the Stage B step
alone), macro-AUC, and accuracy versus number of tiles.
    python -m scripts.analysis.tiny_encoder_eval        (Glioma-SPARSE venv)
Output: results/revision_oct22/e7_tiny_encoders.json, e7_tile_sweep.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
from scripts.analysis.rn18_vs_rn50 import pb_from, summary

OUT = REPO / "results" / "revision_oct22"
MARGIN, DRAWS, N = 0.05, (0, 1, 2), 64
REF = "rn50_f1_nsweep"
ENC = {"kaiko_vits16": "kaiko ViT-S/16", "dp_uni2h": "DistillPath-KS16-UNI2h",
       "dp_hopt0": "DistillPath-KS16-HOpt0", "dp_virchow2": "DistillPath-KS16-Virchow2"}
GF = {"uni2h": 360.585363456, "vits16": 8.48166912, "stageA": 65 * 683.302465536}   # GFLOPs (torch flop counter)


def cfg(e):
    return REF if e == "uni2h" else f"rn50_f1_{e}"


def has_ext(e):
    return (REPO / "results" / "resuni" / cfg(e) / f"predictions_external_{XT.TAG}.csv").exists()


def paired(a, b, ids, seed=0, n=10000):
    """a: encoder, b: UNI2-h, per-row scores on the same rows. Clustered bootstrap of mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u))); bs = s[dr].sum(1) / c[dr].sum(1)
    out = {"encoder": float(np.mean(a)), "uni2h": float(np.mean(b)), "diff": float(d.mean()),
           "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "ci90": np.percentile(bs, [5, 95]).tolist()}
    if set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}:
        a_, b_ = np.asarray(a, bool), np.asarray(b, bool)
        out["mcnemar_p"] = MC.mcnemar(int((a_ & ~b_).sum()), int((~a_ & b_).sum()))
    out["noninferior_5pp"] = bool(out["ci90"][0] > -MARGIN)
    return out


def internal(encs):
    systems, *_, tabs = MC.build_systems()
    bd = tabs["ResNet50 best-F1"][4].sort_values(["fold", "slide_id"]).reset_index(drop=True); ids = bd.slide_id.to_numpy()
    T = {e: {d: MC.cascade_table(bd, MC.resuni(cfg(e), N, d, bd)) for d in DRAWS} for e in ["uni2h", *encs]}
    S = {e: {d: summary(T[e][d]) for d in DRAWS} for e in T}
    ok = {e: np.mean([S[e][d][1] for d in DRAWS], 0) for e in T}
    g = (T["uni2h"][0].grade_pred == T["uni2h"][0].true_grade_class).to_numpy()
    sel = g & (T["uni2h"][0].true_grade_class != "control").to_numpy() & T["uni2h"][0].has_b.to_numpy()
    sub = {e: np.mean([(T[e][d].sub_pred == T[e][d].true_mutation).to_numpy() for d in DRAWS], 0) for e in T}
    res = {e: {"acc_draws": [S[e][d][0]["e2e_acc_pooled"] for d in DRAWS],
               "auc_draws": [S[e][d][0]["e2e_macroAUC_mean"] for d in DRAWS],
               "balacc_draws": [S[e][d][0]["e2e_balacc_mean"] for d in DRAWS],
               "subtype_acc_correct_grade": float(sub[e][sel].mean())} for e in T}
    for e in encs:
        res[e]["vs_uni2h"] = paired(ok[e], ok["uni2h"], ids)
        res[e]["subtype_vs_uni2h"] = paired(sub[e][sel], sub["uni2h"][sel], ids[sel])
    return res


def external(encs):
    base = XT.cascade_ext()["rn50_f1"]; base = base[base.role == "external"].sort_values(["fold", "slide_id"]).reset_index(drop=True)
    E = {}
    for e in ["uni2h", *encs]:
        E[e] = {}
        for d in DRAWS:
            t = MC.cascade_table(base, XT.resuni_ext(cfg(e), N, base, d))
            t["true6"] = [MC.six_true(g_, m_) for g_, m_ in zip(t.true_grade_class, t.true_mutation)]
            E[e][d] = XT.ensemble(t).sort_values("slide_id").reset_index(drop=True)
    ids = E["uni2h"][0].slide_id.to_numpy()
    ok = {e: np.mean([(E[e][d].six_pred == E[e][d].true6).to_numpy() for d in DRAWS], 0) for e in E}
    res = {e: {"ens_acc_draws": [float((E[e][d].six_pred == E[e][d].true6).mean()) for d in DRAWS],
               "ens_macroAUC_draw0": float(MC.metrics(E[e][0])[0]["e2e_macroAUC_pooled"])} for e in E}
    for e in encs:
        assert (E[e][0].slide_id.to_numpy() == ids).all()
        res[e]["vs_uni2h"] = paired(ok[e], ok["uni2h"], ids)
    return res


def sweep(encs):
    systems, *_, tabs = MC.build_systems(); bd = tabs["ResNet50 best-F1"][4]
    dext = XT.cascade_ext()["rn50_f1"]; dext = dext[dext.role == "external"].reset_index(drop=True)
    rows = []
    for e in ["uni2h", *encs]:
        Pi = pd.read_csv(REPO / "results" / "resuni" / cfg(e) / "predictions.csv")
        Pe = pd.read_csv(REPO / "results" / "resuni" / cfg(e) / f"predictions_external_{XT.TAG}.csv") if has_ext(e) else None
        for n in sorted(Pi.n.astype(str).unique(), key=lambda v: 10 ** 9 if v == "all" else int(v)):
            for sd in (DRAWS if n != "all" else (0,)):
                r = {"encoder": e, "n": n, "seed": sd,
                     "int_acc": MC.metrics(MC.cascade_table(bd, pb_from(Pi[(Pi.n.astype(str) == n) & (Pi.seed == sd)], bd)))[0]["e2e_acc_pooled"]}
                if Pe is not None:
                    te = MC.cascade_table(dext, pb_from(Pe[(Pe.n.astype(str) == n) & (Pe.seed == sd)], dext))
                    te["true6"] = [MC.six_true(g_, m_) for g_, m_ in zip(te.true_grade_class, te.true_mutation)]
                    en = XT.ensemble(te); r["ext_acc"] = float((en.six_pred == en.true6).mean())
                rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "e7_tile_sweep.csv", index=False)
    return R.groupby(["encoder", "n"], sort=False).agg(**{f"{c}_{s}": (c, s) for c in ("int_acc", "ext_acc") if c in R for s in ("mean", "std")}).reset_index()


def compute():
    return {e: {"gflops_per_tile": GF[k], "stageB_tflops_64": N * GF[k] / 1e3, "total_tflops_64": (GF["stageA"] + N * GF[k]) / 1e3}
            for e, k in (("uni2h", "uni2h"), ("vits16", "vits16"))}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    encs = [e for e in ENC if (REPO / "results" / "resuni" / cfg(e) / "predictions.csv").exists()]
    res = {"encoders": {e: ENC[e] for e in encs}, "internal": internal(encs), "compute": compute()}
    ext = [e for e in encs if has_ext(e)]
    if ext:
        res["external"] = external(ext)
    sw = sweep(encs); res["tile_sweep"] = sw.to_dict(orient="records")
    json.dump(res, open(OUT / "e7_tiny_encoders.json", "w"), indent=1, default=float)
    for part in ("internal", "external"):
        for e, r in res.get(part, {}).items():
            v = r.get("vs_uni2h")
            print(part, e, "acc draws", [round(x, 3) for x in r.get("acc_draws", r.get("ens_acc_draws"))],
                  "" if v is None else f"| diff {v['diff']:+.3f} 95% CI {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} "
                                       f"90% CI lower {v['ci90'][0]:+.3f} non-inferior {v['noninferior_5pp']} p {v.get('mcnemar_p', float('nan')):.2f}",
                  f"| subtype (correct grade) {r['subtype_acc_correct_grade']:.3f}" if "subtype_acc_correct_grade" in r else "")
    print(sw.round(3).to_string())
