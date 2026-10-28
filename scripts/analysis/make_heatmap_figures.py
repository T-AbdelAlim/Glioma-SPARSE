"""
Heatmap figures for the paper: EBRAINS example slides (test fold of each slide).

    python -m scripts.analysis.make_heatmap_figures ebrains

For every example: H&E | Stage A patch injection (RN50 best-F1) | CLAM attention | UNI2-h at Stage B tile evidence.
Panels use one coordinate frame; scores are normalised within the slide (see heatmaps.py).

The fold model in which the slide is a test slide is used for CLAM, Stage A and the UNI2-h head.
A per-class gallery (one correct example per class) is written next to the single-slide figures.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import heatmaps as H

REPO = Path(__file__).resolve().parents[2]
BRAIN = REPO.parent
OUT = REPO / "results" / "heatmaps"
TRI_EB = BRAIN / "mahmoodlab" / "trident_output"
WSI_EB = Path(r"C:\Users\TAbde\Documents\EMC_postdoc\Virtual_Biopsy\data\WSI_ebrains\WHO2021_data")
SUBS = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
SUB_OF_SIX = {"astro_low": "IDH_mt", "astro_high": "IDH_mt", "oligo_low": "IDH_mt_1p19q", "oligo_high": "IDH_mt_1p19q", "GBM": "IDH_wt"}


def ebrains_paths(sid):
    h5 = glob.glob(str(TRI_EB / "*" / "20x_256px_0px_overlap" / "features_uni_v2" / f"{sid}.h5"))[0]
    wsi = glob.glob(str(WSI_EB / "*" / "*" / f"{sid}.ndpi")) or glob.glob(str(WSI_EB / "*" / "*" / f"{sid}.*"))
    thumb = glob.glob(str(REPO / "data" / "ebrains_thumbnails" / "*" / "included" / f"{sid}.jpg"))
    return h5, wsi[0], (thumb[0] if thumb else None)


def run_ebrains():
    sel = pd.read_csv(OUT / "ebrains_selection.csv")
    made = []
    for r in sel.itertuples():
        h5, wsi, thumb = ebrains_paths(r.slide_id)
        risk, boxes, lr, pred_idx = None, [], None, None
        if r.true6 != "control":
            e = pd.read_csv(REPO / "results" / f"e2e_heatmap_ebrains_f{r.fold}" / "per_slide.csv")
            row = e[e.slide_id == r.slide_id].iloc[0]
            risk = json.loads(row.risk_map) if isinstance(row.get("risk_map"), str) else None
            boxes = json.loads(row.bboxes) if isinstance(row.get("bboxes"), str) else []
            lr = REPO / "results" / "resuni" / "rn50_f1" / f"lr_n64_fold{r.fold}.npz"
            pred_idx = SUBS.index(SUB_OF_SIX.get(r.u_pred, "IDH_wt")) if r.u_pred in SUB_OF_SIX else 2
        title = (f"EBRAINS test slide {r.slide_id[:8]} | true: {r.true6} | UNI2-h at Stage B: {r.u_pred} | CLAM: {r.c_pred} | "
                 f"fold {r.fold}   [{r.role}]")
        out = OUT / "ebrains" / f"{r.true6}_{r.slide_id[:8]}_{r.role.replace(', ', '_').replace(' ', '_')}.png"
        H.render_slide(out, title, wsi, thumb, h5, risk, boxes, [int(r.fold) - 1], lr, pred_idx)
        made.append(out)
        print("wrote", out.name, flush=True)
    return made


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_ebrains()
