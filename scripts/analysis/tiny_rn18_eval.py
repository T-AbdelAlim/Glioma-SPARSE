"""
Experiment E7d (docs/experiments/EXPERIMENT_LOG.md): ResNet-18 Stage A + DistillPath-KS16-UNI2h Stage B end to end.
Both Stage A models (best-F1) get a DistillPath Stage B fitted on their own regions, from the same all-tile DistillPath
features (tiny_random_tiles.py --pass all), so the comparison is paired and tile-for-tile consistent:
  --convert   writes TRIDENT-format files (TRIDENT coords + DistillPath features) to <out>/full/<cohort>/dp_uni2h/
  (then)      resuni_nsweep.py --config rn{18,50}_f1_dp_uni2h_full --features-root ... (see the experiment log)
  (default)   evaluation: six-class accuracy at 64 tiles (3 draws), internal (220 test predictions) and TCGA
              (ensemble of the five fold models), for RN18/RN50 x UNI2-h/DistillPath; paired bootstrap over slides.
Decision rule as E7: RN18 + DistillPath may replace the current pipeline (RN50 + UNI2-h) only if the lower bound of the
90% CI of the draw-averaged accuracy difference lies above -0.05, internally and on TCGA.
    python -m scripts.analysis.tiny_rn18_eval [--convert]        (mahmoodlab venv)
Output: results/revision_oct22/e7d_rn18_distillpath.json
"""
import argparse, json, sys
import numpy as np, h5py
from scripts.analysis import tiny_encoder_features as T
from scripts.analysis.tiny_random_tiles import OUT as RND, feature_files
from scripts.analysis.tiny_encoder_eval import paired
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT

OUT = T.REPO / "results" / "revision_oct22"
FULL = T.OUT / "full"
DRAWS, N = (0, 1, 2), 64
SYS = {"rn50_uni2h": ("ResNet50 best-F1", "rn50_f1", "rn50_f1_nsweep"),
       "rn50_distillpath": ("ResNet50 best-F1", "rn50_f1", "rn50_f1_dp_uni2h_full"),
       "rn18_uni2h": ("ResNet18 best-F1 (S0)", "rn18_f1", "rn18_f1_nsweep"),
       "rn18_distillpath": ("ResNet18 best-F1 (S0)", "rn18_f1", "rn18_f1_dp_uni2h_full")}
GF = {"rn50": 65 * 683.302465536, "rn18": 65 * 303.2, "uni2h": 360.585363456, "distillpath": 8.48166912}


def convert():
    for coh in ("internal", "tcga"):
        out = FULL / coh / "dp_uni2h"; out.mkdir(parents=True, exist_ok=True)
        for sid, p in feature_files(coh).items():
            if (out / f"{sid}.h5").exists():
                continue
            with h5py.File(p) as h, h5py.File(RND / coh / "dp_uni2h" / f"{sid}_all.h5") as g, h5py.File(out / f"{sid}.h5.tmp", "w") as o:
                assert len(g["features"]) == len(h["coords"]), sid
                d = o.create_dataset("coords", data=h["coords"][:])
                for k, v in h["coords"].attrs.items():
                    d.attrs[k] = v
                o.create_dataset("features", data=g["features"][:])
            (out / f"{sid}.h5.tmp").replace(out / f"{sid}.h5")
        print(coh, "converted")


def evaluate():
    systems, *_, tabs = MC.build_systems(); cas = XT.cascade_ext()
    ok_i, ok_e, res = {}, {}, {}
    for k, (bname, acfg, cfg) in SYS.items():
        bd = tabs[bname][4].sort_values(["fold", "slide_id"]).reset_index(drop=True)
        ti = {d: MC.cascade_table(bd, MC.resuni(cfg, N, d, bd)) for d in DRAWS}
        ok_i[k] = (bd.slide_id.to_numpy(), np.mean([MC.metrics(ti[d])[1] for d in DRAWS], 0))
        base = cas[acfg][lambda x: x.role == "external"].sort_values(["fold", "slide_id"]).reset_index(drop=True)
        E = []
        for d in DRAWS:
            t = MC.cascade_table(base, XT.resuni_ext(cfg, N, base, d))
            t["true6"] = [MC.six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)]
            E.append(XT.ensemble(t).sort_values("slide_id").reset_index(drop=True))
        ok_e[k] = (E[0].slide_id.to_numpy(), np.mean([(e.six_pred == e.true6).to_numpy() for e in E], 0))
        a, b = k.split("_")
        res[k] = {"int_acc_draws": [float(MC.metrics(ti[d])[0]["e2e_acc_pooled"]) for d in DRAWS],
                  "ext_acc_draws": [float((e.six_pred == e.true6).mean()) for e in E],
                  "int_grade_acc": float(MC.metrics(ti[0])[0]["grade_acc"]),
                  "tflops_64": (GF[a] + N * GF[b]) / 1e3}
    res["paired"] = {}
    for a, b in (("rn18_distillpath", "rn50_uni2h"), ("rn18_distillpath", "rn50_distillpath"), ("rn18_distillpath", "rn18_uni2h"),
                 ("rn50_distillpath", "rn50_uni2h")):
        r = {}
        for nm, ok in (("internal", ok_i), ("tcga", ok_e)):
            assert (ok[a][0] == ok[b][0]).all()
            r[nm] = paired(ok[a][1], ok[b][1], ok[a][0])
        res["paired"][f"{a}_vs_{b}"] = r
    json.dump(res, open(OUT / "e7d_rn18_distillpath.json", "w"), indent=1, default=float)
    for k in SYS:
        r = res[k]
        print(f"{k:18s} internal {np.mean(r['int_acc_draws']):.3f}  TCGA {np.mean(r['ext_acc_draws']):.3f}  grade {r['int_grade_acc']:.3f}  {r['tflops_64']:.1f} TFLOPs")
    for k, v in res["paired"].items():
        print(k, {nm: f"{x['diff']:+.3f} (95% CI {x['ci95'][0]:+.3f} to {x['ci95'][1]:+.3f}; 90% lower {x['ci90'][0]:+.3f}; non-inferior {x['noninferior_5pp']})" for nm, x in v.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--convert", action="store_true")
    convert() if ap.parse_args().convert else evaluate()
