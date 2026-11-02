"""
Patch-regions of every slide, per fold, for training Stage B.

For each fold, that fold's Stage A model grades every tumour slide of the fold (training, validation and
test) from its thumbnail, patch injection computes the signal map for the predicted class, and the
patch-regions are selected: cells at or above the 95th percentile of the map with enough tissue, at most
four, mapped to level-0 coordinates of the whole-slide image with the thumbnail's JSON sidecar.

  * training and validation slides: at least 50% tissue (--min-tissue-train); regions are selected for the
    predicted class also when Stage A predicts control, so that every tumour slide contributes;
  * test slides: as at inference and in the dashboard, at least 25% tissue (--min-tissue-test) and no
    regions when Stage A predicts control.

Slides in the control folder of the split CSVs are skipped. The molecular type of a slide is read from its
folder in the thumbnail tree (<sidecar-root>/<subtype folder>/included/<slide_id>.jpg).

Output: one CSV with fold, split, slide_id, subtype, pred_grade, n_regions and bboxes (JSON list of
[x, y, w, h] in level-0 pixels), the input of encode_tiles.py and fit_heads.py.

    python -m scripts.stage_b.select_regions --model resnet50 ^
        --checkpoints-glob "ResNet50_output/training_output/*_split_0*/best_f1.pth" ^
        --out data/regions_rn50_f1.csv
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from glioma_sparse.data_utils.subtype_labels import subtype_from_folder
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import compute_risk_map
from glioma_sparse.interpret.wsi_mapping import grid_cell_bbox, load_mapping, thumbnail_bbox_to_wsi
from glioma_sparse.models.factory import build_model

REPO = Path(__file__).resolve().parents[2]
GRADE_CLASSES = ["control", "low_grade", "high_grade"]
GRID = (8, 8)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tissue_fraction(thumbnail, row, col, grid=GRID):
    """Fraction of a grid cell that is tissue (not white, not grey)."""
    w, h = thumbnail.size
    tw, th = w // grid[1], h // grid[0]
    arr = np.asarray(thumbnail.crop((col * tw, row * th, col * tw + tw, row * th + th))).astype(np.float32) / 255.0
    maxc, minc = arr.max(axis=2), arr.min(axis=2)
    mask = (maxc < 0.9) & ((maxc - minc) / (maxc + 1e-8) > 0.05)
    return float(mask.mean())


def select_cells(risk_map, thumbnail, percentile=95.0, cap=4, min_tissue=0.25):
    """Cells at or above the percentile of the signal map with enough tissue, strongest first, at most `cap`."""
    thresh = float(np.percentile(risk_map.flatten(), percentile))
    kept = [(float(risk_map[r, c]), r, c) for r in range(GRID[0]) for c in range(GRID[1])
            if risk_map[r, c] >= thresh and tissue_fraction(thumbnail, r, c) >= min_tissue]
    kept.sort(key=lambda t: -t[0])
    return [(r, c) for _, r, c in kept[:cap]]


def reroot(path):
    """Split CSVs store absolute paths of the machine that wrote them; keep the part from data/ onward."""
    s = str(path).replace("\\", "/")
    i = s.rfind("/data/")
    return REPO / s[i + 1:] if i >= 0 else Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--checkpoints-glob", required=True, help="one Stage A checkpoint per split, e.g. .../*_split_0*/best_f1.pth")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--sidecar-root", default="data/ebrains_thumbnails", help="thumbnails with their JSON sidecars")
    ap.add_argument("--control-image", default="data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--min-tissue-train", type=float, default=0.5)
    ap.add_argument("--min-tissue-test", type=float, default=0.25)
    ap.add_argument("--method", default="full", choices=["full", "crop"],
                    help="patch injection: full (as for the trained models) or crop (same map, about 4x faster)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sidecars = {p.stem: p for p in (REPO / a.sidecar_root).rglob("*.jpg") if p.with_suffix(".json").exists()}
    ckpts = glob.glob(str(REPO / a.checkpoints_glob))
    tf = build_eval_transform()
    ctrl = Image.open(REPO / a.control_image).convert("RGB")
    rows = []
    for sc in sorted((REPO / a.splits_dir).glob("split_0*.csv")):
        fold = int(re.search(r"split_(\d+)", sc.stem).group(1))
        ck = sorted(c for c in ckpts if sc.stem in c)
        if not ck:
            raise SystemExit(f"no checkpoint for {sc.stem} in {a.checkpoints_glob}")
        model = build_model(a.model, num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(ck[0], map_location=DEVICE, weights_only=True))
        model.to(DEVICE).eval()
        slides = [(reroot(r["path"]), r["split"]) for r in csv.DictReader(open(sc))]
        print(f"[fold {fold}] {len(slides)} slides | {ck[0]}", flush=True)
        for path, split in slides:
            if path.parent.name == "control":
                continue
            thumb = sidecars.get(path.stem)
            if thumb is None:
                raise SystemExit(f"no thumbnail with a JSON sidecar for {path.stem} under {a.sidecar_root}")
            img = Image.open(thumb).convert("RGB")
            with torch.no_grad():
                k = int(torch.softmax(model(tf(img).unsqueeze(0).to(DEVICE)), 1).argmax())
            boxes = []
            if split != "test" or GRADE_CLASSES[k] != "control":
                rm, _ = compute_risk_map(target_img=img, control_img=ctrl, model=model, transform=tf, grid=GRID,
                                         target_class=k, device=DEVICE, control_shuffle_seed=SEED, score="logit",
                                         method=a.method)
                mapping = load_mapping(thumb)
                min_tissue = a.min_tissue_test if split == "test" else a.min_tissue_train
                for r, c in select_cells(rm, img, a.percentile, a.cap, min_tissue):
                    b = thumbnail_bbox_to_wsi(mapping, grid_cell_bbox(r, c, GRID[0], GRID[1], mapping.thumbnail_size))
                    if b is not None:
                        boxes.append([int(v) for v in b])
            rows.append({"fold": fold, "split": split, "slide_id": path.stem,
                         "subtype": subtype_from_folder(thumb.parent.parent.name), "pred_grade": GRADE_CLASSES[k],
                         "n_regions": len(boxes), "bboxes": json.dumps(boxes)})
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} (fold, slide) rows, {sum(r['n_regions'] > 0 for r in rows)} with regions")


if __name__ == "__main__":
    main()
