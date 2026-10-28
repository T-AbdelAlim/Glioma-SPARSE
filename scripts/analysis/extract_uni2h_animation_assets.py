"""
Run the real Stage A + UNI2-h (ResUNI) pipeline on one example slide and save every visual
asset an animation of the pipeline needs: the thumbnail, the grade prediction, the
patch-injection signal map, the selected regions, a mosaic of the actual sampled UNI2-h
tiles, a real (PCA-projected, for display only) view of their embeddings, the mean-pooled
embedding, the per-fold and ensembled subtype probabilities, the per-tile evidence
(contribution to the predicted class), and the final integrated diagnosis.

    python -m scripts.analysis.extract_uni2h_animation_assets
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "dashboard"))

import backend as B  # noqa: E402
from glioma_sparse.interpret.patch_injection import tile_image, reconstruct_image, shuffle_control_once  # noqa: E402
from glioma_sparse.models import resuni  # noqa: E402
import openslide  # noqa: E402

DEMO_SIZE = 560  # display size for the injection-demo panels; 560/8 = 70px/tile, exact

SLIDE = Path(r"C:\Users\TAbde\Documents\EMC_postdoc\Virtual_Biopsy\Presentations\CPG\example_wsi\GBM\GL_S01_P021478_C0001_B101_T02_V01_L01_A01_CA_0081.mrxs")
OUT = REPO / "results" / "uni2h_animation_assets"
OUT.mkdir(parents=True, exist_ok=True)


def b64png(img, size=None, quality=82):
    """Despite the name (kept so call sites read the same), encodes as JPEG: these are all
    photographic H&E crops, where JPEG is far smaller than PNG at a visually identical size."""
    if size:
        img = img.copy()
        img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    print("importing + Stage A...")
    r1 = B.api_import(B.ImportReq(slide_path=str(SLIDE)))
    sid = r1["session_id"]
    a = B.api_stage_a(B.StageAReq(session_id=sid, arch="resnet50", metric="f1"))
    print("grade:", a["grade_pred"], a["probs"])

    print("risk map...")
    rm = B.api_risk_map(B.RiskReq(session_id=sid, percentile=95.0, cap=4, min_tissue=0.25))
    print("selected:", rm["selected"])

    s = B.SESSIONS[sid]
    mapping = s["mapping"]
    bboxes = B._selected_bboxes(mapping, s["selected"])
    print("wsi bboxes:", bboxes)

    print("sampling UNI2-h tiles + embedding...")
    slide = openslide.OpenSlide(mapping.wsi_path)
    try:
        tiles, n_tissue = resuni.sample_region_tiles(slide, mapping.base_mpp, bboxes)
    finally:
        slide.close()
    print(f"{len(tiles)} tiles sampled from {n_tissue} tissue tiles found")
    feats = resuni.embed_tiles(tiles, B.DEVICE)
    config = resuni.resuni_config_available("resnet50", "f1")
    out = resuni.predict_from_features(feats, config)
    print("config:", config, "pred_idx:", out["pred_idx"], "ens_probs:", out["ens_probs"])

    gc = B.grade_class_short(a["grade_pred"])
    subtype = B.SUBTYPE_CLASSES[out["pred_idx"]]
    dx = "Control tissue" if gc == "control" else B.INTEGRATED.get((gc, subtype), "Unresolved")
    print("integrated diagnosis:", dx)

    print("patch-injection demo (real per-position injection, same mechanism as the risk map)...")
    grid = (8, 8)
    rows, cols = grid
    stageA_model = B.get_model("stage_a", "resnet50", "f1")
    target_class = B.GRADE_CLASSES.index(a["grade_pred"])
    ctrl = B.control_img()
    control_shuffled = shuffle_control_once(ctrl, grid, B.SEED)
    order = np.random.default_rng(B.SEED).permutation(rows * cols).tolist()  # same call shuffle_control_once makes

    target_tiles = tile_image(s["thumbnail"], grid)
    control_tiles = tile_image(control_shuffled, grid)

    def fwd(img):
        probs, logits = B.forward_probs(stageA_model, img)
        return float(logits[target_class]), float(probs[target_class])

    base_logit, base_prob = fwd(control_shuffled)

    # sample cells: the real top-4 selected regions (high signal) + the 2 lowest-signal cells
    # in the whole grid (contrast) -- the same raw map the real pipeline selects regions from
    raw_map = s["risk_map"]
    selected_idx = {(sel["row"], sel["col"]) for sel in rm["selected"]}
    all_cells = sorted(((float(raw_map[r, c]), r, c) for r in range(rows) for c in range(cols)
                        if (r, c) not in selected_idx))
    low_cells = [(r, c) for _, r, c in all_cells[:2]]
    demo_cells = [(sel["row"], sel["col"]) for sel in rm["selected"]] + low_cells

    injection_frames = []
    for (r, c) in demo_cells:
        idx = r * cols + c
        injected_tiles = list(control_tiles)
        injected_tiles[idx] = target_tiles[idx]
        injected_img = reconstruct_image(injected_tiles, grid, control_shuffled.size)
        inj_logit, inj_prob = fwd(injected_img)
        injection_frames.append({
            "row": r, "col": c, "selected": (r, c) in selected_idx,
            "baseline_logit": base_logit, "injected_logit": inj_logit, "delta_logit": inj_logit - base_logit,
            "baseline_prob": base_prob, "injected_prob": inj_prob, "delta_prob": inj_prob - base_prob,
        })
        print(f"  r{r}c{c} selected={(r,c) in selected_idx}: logit {base_logit:.3f} -> {inj_logit:.3f} "
              f"(prob {base_prob:.3f} -> {inj_prob:.3f})")

    # ---- save assets ----
    assets = {}
    assets["injection_target_class"] = B.GRADE_CLASSES[target_class]
    assets["injection_shuffle_order"] = order
    assets["injection_base_logit"] = base_logit
    assets["injection_base_prob"] = base_prob
    assets["injection_frames"] = injection_frames
    assets["injection_control_shuffled"] = b64png(control_shuffled, (DEMO_SIZE, DEMO_SIZE))
    assets["injection_control_orig"] = b64png(ctrl.resize((2048, 2048)), (DEMO_SIZE, DEMO_SIZE))
    assets["injection_wsi_thumb"] = b64png(s["thumbnail"], (DEMO_SIZE, DEMO_SIZE))
    assets["injection_demo_size"] = DEMO_SIZE
    assets["injection_grid"] = list(grid)

    assets["thumbnail"] = b64png(s["thumbnail"], (900, 900))
    assets["grade_pred"] = a["grade_pred"]
    assets["grade_probs"] = a["probs"]
    assets["risk_grid"] = rm["risk_grid"]
    assets["grid_rows"] = rm["grid_rows"]
    assets["grid_cols"] = rm["grid_cols"]
    assets["selected"] = rm["selected"]
    assets["n_tissue_tiles_found"] = n_tissue
    assets["n_tiles_sampled"] = len(tiles)
    assets["subtype_pred"] = subtype
    assets["ens_probs"] = {resuni.SUB[i]: float(p) for i, p in enumerate(out["ens_probs"])}
    assets["per_fold_probs"] = [{resuni.SUB[i]: float(p) for i, p in enumerate(row)} for row in out["per_fold_probs"]]
    assets["integrated_diagnosis"] = dx
    assets["config"] = config

    # a handful of real, representative tile crops (evenly spaced through the sampled list,
    # not cherry-picked) for the "what goes in" mosaic -- 36 keeps the mosaic legible
    n_show = min(36, len(tiles))
    idx = np.linspace(0, len(tiles) - 1, n_show).astype(int)
    assets["tile_thumbs"] = [b64png(tiles[i][1], (110, 110)) for i in idx]

    # per-tile contribution to the predicted class (real numbers), for the evidence view,
    # normalised 0..1 for coloring
    contrib = out["contributions"]
    lo, hi = contrib.min(), contrib.max()
    contrib_n = ((contrib - lo) / (hi - lo + 1e-9)).tolist()
    assets["tile_contrib_norm"] = [contrib_n[i] for i in idx]

    # a real, low-dimensional (PCA) look at the tile embeddings -- for display only, to make
    # the "1536-d embedding" step concrete rather than a black box. 2 components on the
    # actual 64 tile embeddings sampled from this slide.
    Xc = feats - feats.mean(0)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = (Xc @ Vt[:2].T)
    pr_lo, pr_hi = proj.min(0), proj.max(0)
    proj_n = ((proj - pr_lo) / (pr_hi - pr_lo + 1e-9)).tolist()
    assets["tile_embed_2d"] = proj_n
    mean_feat = feats.mean(0)
    mean_proj = (np.zeros(2) - pr_lo) / (pr_hi - pr_lo + 1e-9)  # mean-centred embedding projects to the origin
    assets["mean_embed_2d"] = mean_proj.tolist()
    # first 24 raw dims of the mean embedding (z-scored for display), a concrete "vector" visual
    head = resuni.get_resuni_heads(config)[0]
    z = (mean_feat - head["mean"]) / head["scale"]
    assets["mean_embed_bars"] = z[:24].tolist()

    # region crops (the actual 2048x2048 re-extracted regions, downsized for display), for the
    # "regions -> tiles" step
    for sel in rm["selected"]:
        if (sel["row"], sel["col"]) not in s["patches"]:
            B.api_extract(B.ExtractReq(session_id=sid, row=sel["row"], col=sel["col"]))
    assets["region_crops"] = [b64png(s["patches"][(sel["row"], sel["col"])], (260, 260))
                              for sel in rm["selected"]]

    (OUT / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    print("wrote", OUT / "assets.json", f"({(OUT / 'assets.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
