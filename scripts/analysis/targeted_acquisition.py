"""
Experiment E8 (docs/experiments/EXPERIMENT_LOG.md): targeted acquisition. How much tissue Glioma-SPARSE needs at 20x, and
whether stage B still works from the top-ranked patch-region alone, as from a single targeted capture.
  (1) per test slide (internal: fold of the test slide; TCGA: each of the five fold models): tissue tiles, tiles within the
      patch-regions and within the top-ranked patch-region (256-px tiles at 20x, 0.5 um/px: 0.0164 mm2 per tile)
  (2) stage B with the frozen 64-tile heads (results/resuni/rn50_f1/lr_n64_fold{k}.npz), grade decisions unchanged, on
        main        64 tiles from the patch-regions (reproduction check)
        top_region  64 tiles from the top-ranked patch-region only (all of them if fewer)
        one_field   all tiles of the 6 x 6-tile window (0.77 x 0.77 mm, about one field of a 20x objective) with the most
                    tissue tiles inside the top-ranked patch-region
      tile draws rng(1000 * draw + fold) as in resuni_nsweep (draws 0-2); one_field is deterministic.
Writes results/revision_oct22/e8_targeted_area.csv, e8_targeted_predictions.csv, e8_targeted.json.
    python -m scripts.analysis.targeted_acquisition      (mahmoodlab venv, h5py)
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from scripts.analysis import master_comparison as MC
from scripts.analysis import external_comparison_tcga as XT
from scripts.analysis import revision_oct22_eval as RV

OUT = REPO / "results" / "revision_oct22"
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]; N = 64; W = 6; TILE_MM2 = 0.128 ** 2; DRAWS = (0, 1, 2)
heads = {k: np.load(REPO / "results" / "resuni" / "rn50_f1" / f"lr_n64_fold{k}.npz") for k in range(1, 6)}
feat = {"internal": {Path(p).stem: p for p in glob.glob(str(REPO.parent / "mahmoodlab" / "trident_output" / "*" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))},
        "tcga": {Path(p).stem: p for p in glob.glob(str(REPO.parent / "mahmoodlab" / "trident_tcga_val" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}}


def head_probs(fold, F):
    z = heads[fold]; l = ((F.mean(0) - z["mean"]) / z["scale"]) @ z["coef"].T + z["intercept"]
    p = np.exp(l - l.max()); return p / p.sum()


def inside(cx, cy, boxes):
    m = np.zeros(len(cx), bool)
    for bx, by, bw, bh in boxes: m |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
    return m


def best_window(xy, ps):
    """Indices of the tiles in the W x W tile window with the most tiles."""
    ix, iy = np.round((xy[:, 0] - xy[:, 0].min()) / ps).astype(int), np.round((xy[:, 1] - xy[:, 1].min()) / ps).astype(int)
    occ = np.zeros((iy.max() + W, ix.max() + W), int); occ[iy, ix] = 1
    c = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = c[W:, W:] - c[:-W, W:] - c[W:, :-W] + c[:-W, :-W]
    y0, x0 = np.unravel_index(s.argmax(), s.shape)
    return np.where((ix >= x0) & (ix < x0 + W) & (iy >= y0) & (iy < y0 + W))[0]


def run(cohort, jobs):
    rows, area = [], []
    for sid, items in sorted(jobs.items()):
        with h5py.File(feat[cohort][sid]) as h:
            xy = h["coords"][:]; ps = int(h["coords"].attrs["patch_size_level0"]); F = h["features"][:]
        cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
        for fold, boxes in items:
            a = {"cohort": cohort, "fold": fold, "slide_id": sid, "n_regions": len(boxes), "tissue_tiles": len(F)}
            if boxes:
                mr, mt = inside(cx, cy, boxes), inside(cx, cy, boxes[:1])
                win = np.where(mt)[0][best_window(xy[mt], ps)] if mt.any() else np.array([], int)
                a.update(region_tiles=int(mr.sum()), top_tiles=int(mt.sum()), field_tiles=len(win))
                for v, idx in (("main", np.where(mr)[0]), ("top_region", np.where(mt)[0])):
                    if not len(idx): continue
                    for d in DRAWS:
                        sel = idx[np.random.default_rng(1000 * d + fold).permutation(len(idx))[:N]]
                        rows.append({"cohort": cohort, "variant": v, "draw": d, "fold": fold, "slide_id": sid,
                                     **{f"p_{k}": p for k, p in zip(SUB, head_probs(fold, F[sel]))}})
                if len(win):
                    for d in DRAWS:
                        rows.append({"cohort": cohort, "variant": "one_field", "draw": d, "fold": fold, "slide_id": sid,
                                     **{f"p_{k}": p for k, p in zip(SUB, head_probs(fold, F[win]))}})
            area.append(a)
    return rows, area


def main():
    reg = pd.read_csv(REPO / "results" / "e2e_regions_rn50_f1.csv")
    ji = {}
    for r in reg.itertuples(): ji.setdefault(r.slide_id, []).append((int(r.fold), json.loads(r.bboxes)))
    ext = XT.cascade_ext()["rn50_f1"]; ext = ext[ext.role == "external"].reset_index(drop=True)
    je = {}
    for r in ext.itertuples(): je.setdefault(r.slide_id, []).append((int(r.fold), json.loads(r.bboxes) if isinstance(r.bboxes, str) else []))
    ri, ai = run("internal", ji); print("internal done", flush=True)
    re_, ae = run("tcga", je); print("tcga done", flush=True)
    P = pd.DataFrame(ri + re_); A = pd.DataFrame(ai + ae)
    P.to_csv(OUT / "e8_targeted_predictions.csv", index=False); A.to_csv(OUT / "e8_targeted_area.csv", index=False)

    res = {"area": {}, "acc": {}}
    for coh, g in A.groupby("cohort"):
        g = g[g.n_regions > 0]
        res["area"][coh] = {"n": int(len(g)), "tissue_mm2_median": float(g.tissue_tiles.median() * TILE_MM2),
                            "regions_mm2_median": float(g.region_tiles.median() * TILE_MM2),
                            "top_mm2_median": float(g.top_tiles.median() * TILE_MM2),
                            "field_mm2_median": float(g.field_tiles.median() * TILE_MM2),
                            "regions_frac_median": float((g.region_tiles / g.tissue_tiles).median()),
                            "top_frac_median": float((g.top_tiles / g.tissue_tiles).median()),
                            "field_tiles_median": float(g.field_tiles.median()),
                            "top_tiles_lt64": float((g.top_tiles < N).mean())}
    bd, S = RV._base()
    okmain = MC.metrics(S["ResNet50 best-F1 Stage A + UNI2-h, 64 tiles"])[1]
    for v in ("main", "top_region", "one_field"):
        acc, ext_acc = [], []
        for d in DRAWS:
            g = P[(P.cohort == "internal") & (P.variant == v) & (P.draw == d)]
            r, ok = MC.metrics(MC.cascade_table(bd, RV._pb(bd, g)))
            acc.append(r["e2e_acc_pooled"])
            if d == 0: mc = MC.mcnemar(int((okmain & ~ok).sum()), int((~okmain & ok).sum())); type_acc = r["subtype_acc"]
            g = P[(P.cohort == "tcga") & (P.variant == v) & (P.draw == d)]
            t = MC.cascade_table(ext, RV._pb(ext, g)); t["true6"] = [MC.six_true(a, b) for a, b in zip(t.true_grade_class, t.true_mutation)]
            e = XT.ensemble(t); ext_acc.append(float((e.six_pred == e.true6).mean()))
        res["acc"][v] = {"internal": acc, "internal_mean": float(np.mean(acc)), "internal_type_acc_draw0": type_acc,
                         "mcnemar_vs_main_draw0": mc, "tcga": ext_acc, "tcga_mean": float(np.mean(ext_acc))}
    json.dump(res, open(OUT / "e8_targeted.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
