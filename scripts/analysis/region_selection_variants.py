"""
Experiment E1 (docs/experiments/EXPERIMENT_LOG.md), offline part: alternative Stage-A region-selection rules,
all evaluated with the SAME frozen Stage-B heads (results/resuni/rn50_f1/lr_n64_fold{k}.npz, fitted on the main
regions of the training slides) and the same 64-tile draw rule (rng(1000*draw + fold)) as the main analysis.

Variants (internal test predictions, ResNet-50 best-F1 Stage A of each fold; grade decisions unchanged):
  main            cells >= p95 of the rise in the predicted-grade logit, >= 25% tissue, ranked, max 4   (reproduction check)
  control_drop    same rule on the DROP of the control (normal) logit
  random_cells    the same NUMBER of cells as `main`, drawn uniformly from cells with >= 25% tissue  (draws 0-2)
  random_slide    64 tiles drawn uniformly from all tissue tiles of the slide (no regions)          (draws 0-2)
Needs results/region_selection_variants/injection_logits.jsonl (region_selection_variants_gpu.py).
Writes results/region_selection_variants/predictions.csv (+ regions.csv); with an argument N (e.g. 100) N draws instead of 3,
written to predictions_{N}draws.csv / regions_{N}draws.csv. Run with the mahmoodlab venv (h5py).
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, load_mapping, thumbnail_bbox_to_wsi

OUT = REPO / "results" / "region_selection_variants"
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]; G = 8; P = 95.0; CAP = 4; MIN_T = 0.25; N = 64
ND = int(sys.argv[1]) if len(sys.argv) > 1 else 3          # number of random draws (E1: 3; E1b: 100)
SUF = "" if ND == 3 else f"_{ND}draws"

def tissue_fraction(img, r, c):   # identical to inference_end_to_end._tissue_fraction
    w, h = img.size; tw, th = w // G, h // G
    a = np.asarray(img.crop((c * tw, r * th, c * tw + tw, r * th + th))).astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1e-8)
    return float(((mx < 0.9) & (sat > 0.05)).mean())

def select(score, tf):            # identical to inference_end_to_end.select_tiles
    thr = float(np.percentile(score.flatten(), P))
    kept = [(score[r, c], r, c) for r in range(G) for c in range(G) if score[r, c] >= thr and tf[r, c] >= MIN_T]
    return [(r, c) for _, r, c in sorted(kept, key=lambda t: -t[0])][:CAP]

feat_idx = {Path(p).stem: p for p in glob.glob(str(REPO.parent / "mahmoodlab" / "trident_output" / "*" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}
thumbs = {Path(p).stem: p for p in glob.glob(str(REPO / "data" / "ebrains_thumbnails" / "*" / "included" / "*.jpg"))}
heads = {k: np.load(REPO / "results" / "resuni" / "rn50_f1" / f"lr_n64_fold{k}.npz") for k in range(1, 6)}
truth = pd.read_csv(REPO / "results" / "e2e_regions_rn50_f1.csv")
main_boxes = {(int(r.fold), r.slide_id): json.loads(r.bboxes) for r in truth.itertuples()}

def head_probs(fold, F):
    z = heads[fold]; l = ((F.mean(0) - z["mean"]) / z["scale"]) @ z["coef"].T + z["intercept"]
    p = np.exp(l - l.max()); return p / p.sum()

rows, regs, checks = [], [], []
for line in open(OUT / "injection_logits.jsonl"):
    j = json.loads(line)
    if "shift" not in j: continue
    fold, sid = j["fold"], j["slide_id"]
    S = np.array(j["shift"]); k = ["control_grade", "low_grade", "high_grade"].index(j["pred_grade"])
    img = Image.open(thumbs[sid]).convert("RGB"); mp = load_mapping(Path(thumbs[sid]))
    tf = np.array([[tissue_fraction(img, r, c) for c in range(G)] for r in range(G)])
    def boxes(cells):
        out = []
        for r, c in cells:
            b = thumbnail_bbox_to_wsi(mp, grid_cell_bbox(r, c, G, G, mp.thumbnail_size))
            if b is not None: out.append([int(v) for v in b])
        return out
    main_cells = select(S[..., k], tf)
    mb = boxes(main_cells)
    checks.append(sorted(map(tuple, mb)) == sorted(map(tuple, main_boxes.get((fold, sid), []))))
    variants = {("main", 0): mb, ("control_drop", 0): boxes(select(-S[..., 0], tf))}
    eligible = [(r, c) for r in range(G) for c in range(G) if tf[r, c] >= MIN_T]
    for d in range(ND):
        rng = np.random.default_rng(10_000 + 100 * d + fold)
        pick = [eligible[i] for i in rng.choice(len(eligible), size=min(len(main_cells), len(eligible)), replace=False)] if main_cells else []
        variants[("random_cells", d)] = boxes(pick)
    with h5py.File(feat_idx[sid]) as h:
        xy = h["coords"][:]; ps = int(h["coords"].attrs["patch_size_level0"]); Fe = h["features"][:]
    cx, cy = xy[:, 0] + ps // 2, xy[:, 1] + ps // 2
    for d in range(ND):
        sel = np.random.default_rng(1000 * d + fold).permutation(len(Fe))[:N]
        p = head_probs(fold, Fe[sel])
        rows.append({"variant": "random_slide", "draw": d, "fold": fold, "slide_id": sid, **{f"p_{c}": p[i] for i, c in enumerate(SUB)}})
    for (v, d), bxs in variants.items():
        regs.append({"variant": v, "draw": d, "fold": fold, "slide_id": sid, "n_regions": len(bxs), "bboxes": json.dumps(bxs)})
        if not bxs: continue
        inside = np.zeros(len(xy), bool)
        for bx, by, bw, bh in bxs: inside |= (cx >= bx) & (cx < bx + bw) & (cy >= by) & (cy < by + bh)
        if not inside.any(): continue
        Fi = Fe[inside]
        # main / control_drop: three tile draws over the same regions; random_cells: region draw d with tile draw d
        for td in (range(ND) if v in ("main", "control_drop") else [d]):
            perm = np.random.default_rng(1000 * td + fold).permutation(len(Fi))[:N]
            p = head_probs(fold, Fi[perm])
            rows.append({"variant": v, "draw": td, "fold": fold, "slide_id": sid, **{f"p_{c}": p[i] for i, c in enumerate(SUB)}})
pd.DataFrame(rows).to_csv(OUT / f"predictions{SUF}.csv", index=False)
pd.DataFrame(regs).to_csv(OUT / f"regions{SUF}.csv", index=False)
print("slides", len(checks), "main regions reproduce archived regions:", sum(checks), "/", len(checks))
