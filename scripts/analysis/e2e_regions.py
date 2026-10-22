"""
Regions selected at end-to-end test time (no Stage B, no slide reading).

For every test slide of every fold, run the pipeline's Stage A + patch-injection + p95 tile
selection with the end-to-end settings (percentile 95, cap 4, tissue filter 0.25) and write the
level-0 bounding boxes of the selected regions. These are the regions the archived end-to-end
evaluation gave to Stage B, so a different Stage B (e.g. a foundation model on native-resolution
tiles) can be scored on exactly the same regions.

Output CSV: fold, slide_id, pred_grade, n_regions, bboxes (json list of [x, y, w, h], level 0).

    python -m scripts.analysis.e2e_regions --stage-a-glob "ResNet50_output/training_output/*_split_0*/best_f1.pth" \
        --model resnet50 --out results/e2e_regions_rn50_f1.csv
"""

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, load_mapping, thumbnail_bbox_to_wsi
from glioma_sparse.models.factory import build_model
from scripts.inference.e2e_multi import reroot
from scripts.inference.inference_end_to_end import (
    GRADE_CLASSES, GRID, SEED, _rel, _sidecar_for, select_tiles,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-glob", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--sidecar-root", default=r"data\ebrains_thumbnails")
    ap.add_argument("--control-image", default="data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--min-tissue", type=float, default=0.25)
    a = ap.parse_args()

    ckpts = sorted(glob.glob(str(_rel(a.stage_a_glob))))
    tf = build_eval_transform()
    ctrl = Image.open(_rel(a.control_image)).convert("RGB")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in sorted(_rel(a.splits_dir).glob("split_0*.csv")):
        fold = int(re.search(r"split_(\d+)", sc.stem).group(1))
        ck = [c for c in ckpts if sc.stem in str(c)][0]
        model = build_model(a.model, num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        test = [reroot(r["path"]) for r in csv.DictReader(open(sc)) if r["split"] == "test"]
        print(f"[fold {fold}] {len(test)} test slides | {ck}", flush=True)
        for p in test:
            if p.parent.name == "control":
                continue
            thumb = _sidecar_for(p.stem, _rel(a.sidecar_root))
            img = Image.open(thumb).convert("RGB")
            with torch.no_grad():
                probs = torch.softmax(model(tf(img).unsqueeze(0).to(DEVICE)), 1).cpu().numpy()[0]
            k = int(probs.argmax())
            boxes = []
            if GRADE_CLASSES[k] != "control":
                rm, _ = compute_risk_map(target_img=img, control_img=ctrl, model=model, transform=tf,
                                         grid=GRID, target_class=k, device=DEVICE,
                                         control_shuffle_seed=SEED, score="logit")
                sel, _ = select_tiles(rm, img, GRID, a.percentile, a.cap, a.min_tissue)
                mp = load_mapping(thumb)
                for s in sel:
                    b = thumbnail_bbox_to_wsi(mp, grid_cell_bbox(s["row"], s["col"], GRID[0], GRID[1],
                                                                mp.thumbnail_size))
                    if b is not None:
                        boxes.append([int(v) for v in b])
            rows.append({"fold": fold, "slide_id": p.stem, "pred_grade": GRADE_CLASSES[k],
                         "n_regions": len(boxes), "bboxes": json.dumps(boxes)})
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
