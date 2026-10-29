"""
Experiment E7b: does Stage B still need the Stage A regions? Stage A (ResNet-50
best-F1) grades the slide as usual; Stage B gets 64 tiles drawn at random from the WHOLE slide instead of from the
regions, with the unchanged n = 64 heads (fitted on region tiles of the training slides; seed-0 heads of
rn50_f1_nsweep / rn50_f1_dp_uni2h), as in E1 'random_slide'. Tile draws: E6 rule (tiny_random_tiles.py, pass A),
identical for both encoders. Compared with the region-based two-stage results of the same encoder (64 tiles, draws 0-2).
Internal: 220 test predictions; TCGA: ensemble of the five fold models. Paired bootstrap over slides on the
draw-averaged six-class correctness, exact McNemar on draw 0.
    python -m scripts.analysis.tiny_random_eval        (mahmoodlab venv: h5py)
Output: results/revision_oct22/e7b_random_slide.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scripts.analysis import tiny_encoder_features as T
from scripts.analysis.tiny_random_tiles import OUT as RND, feature_files, perms
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
from scripts.analysis.rn18_vs_rn50 import summary

OUT = T.REPO / "results" / "revision_oct22"
CFG = {"uni2h": "rn50_f1_nsweep", "dp_uni2h": "rn50_f1_dp_uni2h"}
DRAWS, N = (0, 1, 2), 64


def heads(cfg):
    return {k: np.load(T.REPO / "results" / "resuni" / cfg / f"lr_n{N}_fold{k}.npz") for k in range(1, 6)}


def head_probs(z, F):
    l = ((F.mean(0) - z["mean"]) / z["scale"]) @ z["coef"].T + z["intercept"]
    p = np.exp(l - l.max()); return p / p.sum()


def random_features(cohort, sids):
    """{encoder: {slide: {draw: (64, d) features}}} on the same tiles."""
    ff = feature_files(cohort); out = {"uni2h": {}, "dp_uni2h": {}}
    for s in sorted(set(sids)):
        with h5py.File(ff[s]) as h:
            m = len(h["coords"]); P = perms(s, m)
            U = h["features"]; rows = {d: P[d][:N] for d in DRAWS}
            out["uni2h"][s] = {d: U[np.sort(r)][np.argsort(np.argsort(r))] for d, r in rows.items()}
        with h5py.File(RND / cohort / "dp_uni2h" / f"{s}_A.h5") as h:
            pos = {int(i): j for j, i in enumerate(h["idx"][:])}; Fd = h["features"][:]
        out["dp_uni2h"][s] = {d: Fd[[pos[int(i)] for i in r]] for d, r in rows.items()}
    return out


def pb_random(base, F, H):
    pb = np.full((len(base), 3), np.nan)
    return {d: np.stack([head_probs(H[f], F[s][d]) for f, s in zip(base.fold, base.slide_id)]) for d in DRAWS}


def paired(a, b, ids, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u))); bs = s[dr].sum(1) / c[dr].sum(1)
    return {"a": float(np.mean(a)), "b": float(np.mean(b)), "diff": float(d.mean()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}


def run(base, tables, ids, finish):
    """tables: {system: {draw: cascade table}}; finish(t) -> (per-row correctness, metrics dict)."""
    res, ok = {}, {}
    for k, T_ in tables.items():
        r = {d: finish(T_[d]) for d in DRAWS}
        ok[k] = np.mean([r[d][0] for d in DRAWS], 0)
        res[k] = {"acc_draws": [float(r[d][0].mean()) for d in DRAWS], **{m: float(np.mean([r[d][1][m] for d in DRAWS])) for m in r[0][1]}}
    pairs = [("dp_uni2h_random", "dp_uni2h_regions"), ("uni2h_random", "uni2h_regions"),
             ("dp_uni2h_random", "uni2h_random"), ("dp_uni2h_random", "uni2h_regions")]
    res["paired"] = {f"{a}_vs_{b}": paired(ok[a], ok[b], ids) for a, b in pairs if a in ok and b in ok}
    return res


def internal():
    systems, *_, tabs = MC.build_systems()
    bd = tabs["ResNet50 best-F1"][4].sort_values(["fold", "slide_id"]).reset_index(drop=True)
    F = random_features("internal", bd.slide_id)
    tables = {}
    for e, cfg in CFG.items():
        pr = pb_random(bd, F[e], heads(cfg))
        tables[f"{e}_random"] = {d: MC.cascade_table(bd, pr[d]) for d in DRAWS}
        tables[f"{e}_regions"] = {d: MC.cascade_table(bd, MC.resuni(cfg, N, d, bd)) for d in DRAWS}
    def finish(t):
        s, ok = summary(t)
        return ok, {"auc": s["e2e_macroAUC_mean"], "balacc": s["e2e_balacc_mean"], "subtype_acc": s["subtype_acc"]}
    return run(bd, tables, bd.slide_id.to_numpy(), finish)


def external():
    base = XT.cascade_ext()["rn50_f1"]; base = base[base.role == "external"].sort_values(["fold", "slide_id"]).reset_index(drop=True)
    F = random_features("tcga", base.slide_id)
    tables = {}
    for e, cfg in CFG.items():
        pr = pb_random(base, F[e], heads(cfg))
        tables[f"{e}_random"] = {d: MC.cascade_table(base, pr[d]) for d in DRAWS}
        if (T.REPO / "results" / "resuni" / cfg / f"predictions_external_{XT.TAG}.csv").exists():
            tables[f"{e}_regions"] = {d: MC.cascade_table(base, XT.resuni_ext(cfg, N, base, d)) for d in DRAWS}
    ids = None
    def finish(t):
        nonlocal ids
        t["true6"] = [MC.six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)]
        e = XT.ensemble(t).sort_values("slide_id").reset_index(drop=True)
        ids = e.slide_id.to_numpy() if ids is None else ids
        assert (e.slide_id.to_numpy() == ids).all()
        return (e.six_pred == e.true6).to_numpy(), {"auc": float(MC.metrics(e)[0]["e2e_macroAUC_pooled"])}
    for k in tables:          # fixes ids before pairing
        finish(tables[k][0]); break
    return run(base, tables, ids, finish)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"internal": internal(), "tcga": external()}
    json.dump(res, open(OUT / "e7b_random_slide.json", "w"), indent=1, default=float)
    for part, r in res.items():
        for k, v in r.items():
            if k == "paired":
                for pk, pv in v.items():
                    print(f"{part:8s} {pk:38s} diff {pv['diff']:+.3f} (95% CI {pv['ci95'][0]:+.3f} to {pv['ci95'][1]:+.3f})")
            else:
                print(f"{part:8s} {k:18s} acc {np.mean(v['acc_draws']):.3f} draws {[round(x, 3) for x in v['acc_draws']]}",
                      {m: round(x, 3) for m, x in v.items() if m != 'acc_draws'})
