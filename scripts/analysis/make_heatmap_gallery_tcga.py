"""
TCGA heatmap gallery: 10 example slides per WHO-2021 subtype/grade class (astro_low, astro_high,
oligo_low, oligo_high, GBM -- 50 slides total), each rendered as the same 4-panel figure used
throughout the paper: H&E | Stage-A patch injection (selected regions) | CLAM attention (5-fold
average) | UNI2-h-at-Stage-B evidence for the predicted subtype.

Because all three model families (ResNet cascade / CLAM benchmark / UNI2-h-at-Stage-B) are rendered
on the SAME slide in the SAME figure, the comparison is inherently apples-to-apples across models --
there is no separate "per-model" gallery, by design (this mirrors Figure 1 / make_heatmap_figures.py,
scaled from 1 example to 10 per class and without the correct/incorrect stratification,
since the goal here is a representative gallery, not curated agreement/disagreement cases).

Stage-A regions and the ResUNI head shown are the fold-1 model's; CLAM attention is averaged over all five fold models.

Usage: python -m scripts.analysis.make_heatmap_gallery_tcga [--n-per-class 10] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import heatmaps as H
from scripts.analysis.master_comparison import SIX

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "results" / "external_TCGA_val"
TRI_EXT = REPO.parent / "mahmoodlab" / "trident_tcga_val" / "20x_256px_0px_overlap" / "features_uni_v2"
OUT = REPO / "results" / "heatmaps" / "tcga_gallery"
SUBS = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
SUB_OF_SIX = {"astro_low": "IDH_mt", "astro_high": "IDH_mt", "oligo_low": "IDH_mt_1p19q",
              "oligo_high": "IDH_mt_1p19q", "GBM": "IDH_wt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from scripts.analysis.external_comparison_tcga import build_external_systems, ensemble
    lab = pd.read_csv(EXT / "labels.csv").set_index("slide_id")
    systems = {n: t for g, n, t, d in build_external_systems()}
    u = ensemble(systems["ResNet50 best-F1 Stage A + UNI2-h, 64 tiles"]).set_index("slide_id")
    c = ensemble(systems["TRIDENT + UNI2-h + CLAM-SB"]).set_index("slide_id")

    e2e = pd.read_csv(REPO / "results" / "e2e_external_rn50_f1_tcga" / "per_slide.csv")
    e2e_f1 = e2e[e2e.fold == 1].set_index("slide_id")

    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for cls in [c_ for c_ in SIX if c_ != "control"]:
        cand = lab[lab.true6 == cls].index.to_numpy()
        if len(cand) == 0:
            continue
        n = min(a.n_per_class, len(cand))
        chosen = rng.choice(cand, size=n, replace=False)
        for sid in chosen:
            r = lab.loc[sid]
            h5 = TRI_EXT / f"{sid}.h5"
            if not h5.exists() or sid not in e2e_f1.index:
                print("skip (missing features or Stage-A row):", sid)
                continue
            row = e2e_f1.loc[sid]
            risk = json.loads(row.risk_map) if isinstance(row.get("risk_map"), str) else None
            boxes = json.loads(row.bboxes) if isinstance(row.get("bboxes"), str) else []
            u_pred = u.loc[sid, "six_pred"] if sid in u.index else "NA"
            c_pred = c.loc[sid, "six_pred"] if sid in c.index else "NA"
            pred_idx = SUBS.index(SUB_OF_SIX.get(u_pred, "IDH_wt"))
            title = (f"TCGA slide {sid[:12]} | true: {cls} | UNI2-h at Stage B ens.: {u_pred} | "
                     f"CLAM ens.: {c_pred}   [gallery]")
            out = OUT / f"{cls}_{sid[:12]}.png"
            H.render_slide(out, title, r.path,
                           EXT / "thumbs_none" / f"{sid}.jpg", h5, risk, boxes,
                           [0, 1, 2, 3, 4], REPO / "results" / "resuni" / "rn50_f1" / "lr_n64_fold1.npz", pred_idx)
            made.append(out)
            print("wrote", out.name, flush=True)
    print(f"\ndone: {len(made)} figures -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
