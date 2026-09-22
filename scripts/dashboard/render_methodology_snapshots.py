#!/usr/bin/env python3
r"""
Render 300-DPI manuscript figures of the EXACT scenes shown in the dashboard's
"Methodology explained" walkthrough, using the REAL cached model output.

This reads the methodology cache the backend writes on first open
(scripts/dashboard/methodology_cache/methodology_cache.json) so the figures use
the genuine pipeline result for the preset slide — identical to what the
animation shows, no recomputation.

Run the dashboard, open "Methodology explained" once (this builds the cache),
then:

    python render_methodology_snapshots.py \
        --cache scripts/dashboard/methodology_cache/methodology_cache.json \
        --out figures/methodology_real

OUTPUT (300 DPI, white background + transparent PNG variants)
    snap1_target_grade.jpg
    snap2_permuted_control.jpg
    snap3_injection.jpg
    snap4_signal_map.jpg
    snap5_p95_extraction.jpg
    snap6_stageB_probs.jpg
    snap7_occlusion_diagnosis.jpg
    methodology_real_overview.jpg
"""

import argparse, base64, io, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

RISK = "RdYlBu_r"
INK = "#1d2733"; MUT = "#6b7686"
COLB = {"IDH_mt": "#a8dadc", "IDH_mt_1p19q": "#f1faee", "IDH_wt": "#457b9d"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 300})


def b64img(s):
    if not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def _clean(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


def _title(ax, step, text):
    ax.text(0.02, 0.97, f"Step {step}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#457b9d", va="top")
    ax.text(0.02, 0.915, text, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="top")


def _imshow(ax, im, x, y, w, h):
    # NOTE: deliberately no aspect= kwarg here. Passing aspect="auto" to
    # imshow doesn't just style this one image - it resets the AXES-level
    # aspect setting, silently overriding the ax.set_aspect("equal") from
    # _clean(). That stretched the whole 10x10 data square to fill the wider
    # (8.2x5.0in) figure canvas, squishing every image/patch/box vertically.
    # Leaving aspect unset lets the persistent "equal" setting hold, so square
    # extents (all boxes here are square) render undistorted.
    ax.imshow(np.asarray(im), extent=[x, x + w, y, y + h], zorder=1)


def _heat_overlay(ax, grid, x, y, w, h, alpha=0.45, boxes=None):
    g = np.array(grid, dtype=float)
    ax.imshow(g, extent=[x, x + w, y, y + h], cmap=RISK,   # no aspect= (see _imshow note)
              alpha=alpha, vmin=0, vmax=1, zorder=2)
    if boxes:
        gr, gc = g.shape; cw, ch = w / gc, h / gr
        for (r, c) in boxes:
            ax.add_patch(Rectangle((x + c * cw, y + h - (r + 1) * ch), cw, ch,
                                    fill=False, edgecolor="#e11d1d", lw=2, zorder=3))


# ---- individual snapshots ----

def snap_target(ax, d):
    _clean(ax); _title(ax, 1, "Target slide + Stage A grade")
    im = b64img(d["target_thumb"])
    _imshow(ax, im, 3.3, 2.6, 3.4, 3.4)
    sa = d.get("stage_a") or {}
    g = sa.get("grade_pred", "")
    col = {"control": "#c9cba3", "low_grade": "#ffe1a8", "high_grade": "#e26d5c"}.get(g, "#64748b")
    ax.add_patch(FancyBboxPatch((3.6, 1.5), 2.8, 0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=col))
    ax.text(5.0, 1.85, "Stage A: " + g.replace("_", " "), ha="center", fontsize=12,
            fontweight="bold", color="#fff" if g == "high_grade" else "#1d2733")
    ax.text(5.0, 0.9, f"confidence {sa.get('confidence', 0):.2f}", ha="center",
            fontsize=9.5, color=MUT)


def snap_permuted(ax, d):
    _clean(ax); _title(ax, 2, "Patch-permutation: the neutral control")
    _imshow(ax, b64img(d["control_orig"]), 0.7, 3.0, 3.2, 3.2)
    ax.text(2.3, 6.4, "control slide", ha="center", fontsize=10.5, color=INK)
    _imshow(ax, b64img(d["control_permuted"]), 6.1, 3.0, 3.2, 3.2)
    ax.text(7.7, 6.4, "permuted control", ha="center", fontsize=10.5, color=INK)
    ax.add_patch(FancyArrowPatch((4.1, 4.6), (5.9, 4.6), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, lw=2))
    ax.text(5.0, 4.9, "shuffle", ha="center", fontsize=9.5, color=MUT, style="italic")
    ax.text(5.0, 1.9, "Layout destroyed so each patch is judged on content alone.",
            ha="center", fontsize=9.3, color=INK)


def snap_injection(ax, d):
    _clean(ax); _title(ax, 3, "Patch injection: one tile flips, the logit moves")
    inj = d.get("injection") or {}
    frames = inj.get("frames", [])
    # show the final injected control on the left
    last = frames[-1] if frames else None
    if last:
        _imshow(ax, b64img(last["injected_img"]), 0.6, 2.7, 3.2, 3.2)
    grid = d["grid"]; gr, gc = grid
    if last:
        cw, ch = 3.2 / gc, 3.2 / gr
        sr, sc = inj["slot_row"], inj["slot_col"]
        ax.add_patch(Rectangle((0.6 + sc * cw, 2.7 + 3.2 - (sr + 1) * ch), cw, ch,
                               fill=False, edgecolor="#e11d1d", lw=2.5))
    ax.text(2.2, 6.1, "permuted control (one slot injected)", ha="center", fontsize=9.5, color=INK)
    # logit progression bar chart
    base = inj.get("base_logit", 0)
    logits = [base] + [f["logit"] for f in frames]
    labels = ["base"] + [f"+tile\n({f['row']},{f['col']})" for f in frames]
    xs = np.arange(len(logits))
    ax2 = ax.inset_axes([0.52, 0.28, 0.44, 0.5])
    colors = ["#6b7686"] + ["#e26d5c"] * len(frames)
    ax2.bar(xs, logits, color=colors)
    ax2.axhline(base, color="#6b7686", ls="--", lw=1)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Stage A logit\n(target class)", fontsize=8)
    ax2.tick_params(labelsize=7)
    for i, v in enumerate(logits):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_title("injecting high-signal tiles raises the logit", fontsize=8.5)
    ax.text(5.0, 1.0, "The logit shift from injecting a tile is that tile's signal score.",
            ha="center", fontsize=9.3, color=INK)


def snap_signal(ax, d):
    _clean(ax); _title(ax, 4, "Stage A signal map (all tiles scored)")
    _imshow(ax, b64img(d["target_thumb"]), 3.0, 2.3, 4.0, 4.0)
    _heat_overlay(ax, d["risk_grid"], 3.0, 2.3, 4.0, 4.0, alpha=0.5)
    ax.text(5.0, 6.55, "signal map", ha="center", fontsize=11, color=INK)
    cax = ax.inset_axes([0.80, 0.30, 0.03, 0.42])
    grad = np.linspace(0, 1, 100).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap=RISK, origin="lower")
    cax.set_xticks([]); cax.set_yticks([0, 99]); cax.set_yticklabels(["low", "high"], fontsize=8)
    ax.text(5.0, 1.5, "Warm cells drive the grade — localised with no pixel annotation.",
            ha="center", fontsize=9.3, color=INK)


def snap_extraction(ax, d):
    _clean(ax); _title(ax, 5, "p95 selection → level-0 extraction")
    boxes = [(p["row"], p["col"]) for p in d["patches"][:4]]
    _imshow(ax, b64img(d["target_thumb"]), 0.5, 2.6, 3.6, 3.6)
    _heat_overlay(ax, d["risk_grid"], 0.5, 2.6, 3.6, 3.6, alpha=0.5, boxes=boxes)
    ax.text(2.3, 6.4, "signal map (p95 boxed)", ha="center", fontsize=10, color=INK)
    for i, p in enumerate(d["patches"][:4]):
        py = 5.6 - i * 1.25
        im = b64img(p["patch"])
        _imshow(ax, im, 6.2, py, 1.05, 1.05)
        ax.add_patch(Rectangle((6.2, py), 1.05, 1.05, fill=False, edgecolor="#3b82f6", lw=1.4))
        ax.add_patch(FancyArrowPatch((4.2, 4.4 - i * 0.05), (6.05, py + 0.5),
                                     arrowstyle="-|>", mutation_scale=9, color="#3b82f6",
                                     lw=1.2, connectionstyle="arc3,rad=0.1"))
    ax.text(6.7, 6.85, "level-0 patches", ha="center", fontsize=9.5, color=INK)


def snap_stageB(ax, d):
    _clean(ax); _title(ax, 6, "Stage B: molecular probabilities")
    for i, p in enumerate(d["patches"][:4]):
        py = 5.4 - i * 1.15
        _imshow(ax, b64img(p["patch"]), 0.6, py, 0.95, 0.95)
        ax.add_patch(Rectangle((0.6, py), 0.95, 0.95, fill=False, edgecolor="#3b82f6", lw=1.2))
    ax.add_patch(FancyBboxPatch((2.7, 3.6), 1.7, 1.6, boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#1c232d", edgecolor="#6366f1", lw=1.6))
    ax.text(3.55, 4.55, "Stage B", ha="center", fontsize=11, color="#fff", fontweight="bold")
    ax.text(3.55, 4.15, "ResNet50", ha="center", fontsize=9.5, color="#a9b2c0")
    sb = d.get("stage_b") or {}
    subs = [("IDH-mut (astro)", sb.get("prob_IDH_mt", 0), COLB["IDH_mt"]),
            ("IDH-mut 1p/19q", sb.get("prob_IDH_mt_1p19q", 0), COLB["IDH_mt_1p19q"]),
            ("IDH-wildtype", sb.get("prob_IDH_wt", 0), COLB["IDH_wt"])]
    bx, bw = 5.6, 3.6
    by = 5.55          # top row's baseline
    bar_h = 0.42        # bar height
    spacing = 1.05       # distance between consecutive bars' baselines.
    # spacing (1.05) comfortably exceeds bar_h (0.42), and each label sits just
    # above its OWN bar with 0.55 clear of the bar above it, so labels never
    # run through the bar above (the earlier 0.72 spacing left only a 0.22 gap,
    # which the label offset then intruded into).
    for i, (name, p, col) in enumerate(subs):
        yy = by - i * spacing
        ax.add_patch(Rectangle((bx, yy), bw, bar_h, facecolor="#eef1f4", edgecolor="none"))
        ax.add_patch(Rectangle((bx, yy), bw * p, bar_h, facecolor=col, edgecolor="#c9ced6", lw=0.5))
        ax.text(bx, yy + bar_h + 0.12, name, fontsize=8.8, color=INK)
        ax.text(bx + bw + 0.12, yy + bar_h / 2, f"{p:.2f}", fontsize=9, color=INK, va="center")
    ax.text(bx + bw / 2, 6.45, "molecular probabilities (soft vote)", ha="center", fontsize=10, color=INK)
    ax.text(5.0, 1.4, f"slide-level call: {sb.get('subtype_pred','')}  "
                      f"(confidence {sb.get('confidence',0):.2f})",
            ha="center", fontsize=9.3, color=INK)


def snap_occlusion(ax, d):
    _clean(ax); _title(ax, 7, "Occlusion map + integrated diagnosis")
    p = next((x for x in d["patches"] if x.get("occl_grid")), d["patches"][0])
    _imshow(ax, b64img(p["patch"]), 0.8, 3.0, 3.4, 3.4)
    if p.get("occl_grid"):
        _heat_overlay(ax, p["occl_grid"], 0.8, 3.0, 3.4, 3.4, alpha=0.45)
        og = np.array(p["occl_grid"]); gr, gc = og.shape
        if p.get("occl_top"):
            tr, tc = p["occl_top"]["cell"]
            cw, ch = 3.4 / gc, 3.4 / gr
            ax.add_patch(Rectangle((0.8 + tc * cw, 3.0 + 3.4 - (tr + 1) * ch), cw, ch,
                                   fill=False, edgecolor="#fde047", lw=2.2))
    ax.text(2.5, 6.6, "occlusion signal map", ha="center", fontsize=10, color=INK)
    integ = d.get("integrated") or {}
    dx = integ.get("diagnosis") or integ.get("integrated_diagnosis") or "—"
    ax.add_patch(FancyArrowPatch((4.4, 4.7), (5.5, 4.7), arrowstyle="-|>",
                                 mutation_scale=14, color=INK, lw=2))
    ax.add_patch(FancyBboxPatch((5.7, 3.7), 3.7, 1.9, boxstyle="round,pad=0.05,rounding_size=0.14",
                                facecolor="#132029", edgecolor="#457b9d", lw=1.6))
    ax.text(7.55, 5.15, "Integrated diagnosis", ha="center", fontsize=9, color="#8fb4c9")
    # wrap the diagnosis
    words = dx.split(", ")
    line = ""
    yy = 4.75
    for w in words:
        if len(line + w) > 26:
            ax.text(7.55, yy, line.rstrip(", "), ha="center", fontsize=10, color="#fff", fontweight="bold")
            yy -= 0.38; line = ""
        line += w + ", "
    ax.text(7.55, yy, line.rstrip(", "), ha="center", fontsize=10, color="#fff", fontweight="bold")


def snap_overview(ax, d):
    _clean(ax); _title(ax, 8, "Overview: whole pipeline in one view")
    # Layout budget (all x in 0..10): thumbnail 0.2-2.0, Stage A 2.3-3.7,
    # p95 patches 4.0-7.1, Stage B bars 7.4-9.8. Verified to fit with no
    # clipping or overlap (checked programmatically, not just by eye).
    _imshow(ax, b64img(d["target_thumb"]), 0.2, 6.6, 1.8, 1.8)
    ax.text(1.1, 8.75, "thumbnail", ha="center", fontsize=8.5, color=MUT)
    ax.add_patch(FancyArrowPatch((2.05, 7.5), (2.25, 7.5), arrowstyle="-|>",
                                 mutation_scale=9, color="#3b82f6", lw=1.3))
    sa = d.get("stage_a") or {}
    g = sa.get("grade_pred", "")
    gcol = {"control": "#c9cba3", "low_grade": "#ffe1a8", "high_grade": "#e26d5c"}.get(g, "#64748b")
    ax.add_patch(FancyBboxPatch((2.3, 7.2), 1.4, 0.6, boxstyle="round,pad=0.02,rounding_size=0.3",
                                facecolor=gcol))
    ax.text(3.0, 7.5, g.replace("_", " "), ha="center", va="center", fontsize=7.8,
            fontweight="bold", color="#fff" if g == "high_grade" else "#1d2733")
    ax.text(3.0, 8.75, "Stage A", ha="center", fontsize=8.5, color=MUT)
    ax.add_patch(FancyArrowPatch((3.75, 7.5), (3.95, 7.5), arrowstyle="-|>",
                                 mutation_scale=9, color="#3b82f6", lw=1.3))
    n = min(len(d["patches"]), 4)
    pw = 0.7; gap = 0.12
    for i, p in enumerate(d["patches"][:n]):
        px = 4.0 + i * (pw + gap)
        _imshow(ax, b64img(p["patch"]), px, 7.15, pw, pw)
        ax.add_patch(Rectangle((px, 7.15), pw, pw, fill=False, edgecolor="#3b82f6", lw=1.0))
    patches_end = 4.0 + n * (pw + gap) - gap
    ax.text((4.0 + patches_end) / 2, 8.75, f"p95 patches ({len(d['selected'])})",
            ha="center", fontsize=8.5, color=MUT)
    ax.add_patch(FancyArrowPatch((patches_end + 0.1, 7.5), (patches_end + 0.3, 7.5),
                                 arrowstyle="-|>", mutation_scale=9, color="#6366f1", lw=1.3))
    # Stage B: compact stacked bars with values to the right, no left-side
    # labels (kept the block's total width safely inside the canvas)
    sb = d.get("stage_b") or {}
    subs = [("IDH-mt", sb.get("prob_IDH_mt", 0), "#a8dadc"),
            ("1p19q", sb.get("prob_IDH_mt_1p19q", 0), "#f1faee"),
            ("IDH-wt", sb.get("prob_IDH_wt", 0), "#457b9d")]
    bx0 = min(patches_end + 0.7, 7.6); bw2 = min(9.85 - bx0, 1.9)
    bar_h = 0.22; spacing = 0.55   # same clearance ratio as the main Stage B step
    for i, (nm, p, col) in enumerate(subs):
        yy = 8.3 - i * spacing
        ax.add_patch(Rectangle((bx0, yy), bw2, bar_h, facecolor="#eef1f4", edgecolor="none"))
        ax.add_patch(Rectangle((bx0, yy), bw2 * p, bar_h, facecolor=col, edgecolor="none"))
        ax.text(bx0, yy + bar_h + 0.1, f"{nm} {p:.2f}", fontsize=6.6, color=INK)
    ax.text(bx0 + bw2 / 2, 8.75, "Stage B", ha="center", fontsize=8.5, color=MUT)
    # integrated diagnosis banner, full width, near bottom
    ax.add_patch(FancyBboxPatch((0.3, 0.6), 9.4, 2.4, boxstyle="round,pad=0.05,rounding_size=0.14",
                                facecolor="#132029", edgecolor="#457b9d", lw=1.6))
    ax.text(5.0, 2.55, "Integrated diagnosis", ha="center", fontsize=10, color="#8fb4c9")
    integ = d.get("integrated") or {}
    dx = integ.get("diagnosis") or integ.get("integrated_diagnosis") or "unknown"
    # wrap long diagnosis strings across lines so they stay within the banner
    # width instead of overflowing off the canvas edges
    words = dx.replace(", ", " ,SPLIT ").split(" ")
    lines, line = [], ""
    for w in words:
        cand = (line + " " + w).strip()
        if len(cand.replace(",SPLIT", ",")) > 44:
            lines.append(line.replace(",SPLIT", ",")); line = w
        else:
            line = cand
    if line:
        lines.append(line.replace(",SPLIT", ","))
    y0 = 1.9 + (len(lines) - 1) * 0.24
    for i, ln in enumerate(lines):
        ax.text(5.0, y0 - i * 0.48, ln.strip(), ha="center", fontsize=12.5,
                fontweight="bold", color="#fff")
    conf = integ.get("confidence") or integ.get("integrated_confidence") or 0
    ax.text(5.0, 1.15 - max(0, len(lines) - 1) * 0.15, f"RN50 / RN50, p95, confidence {conf:.2f}",
            ha="center", fontsize=9.5, color="#e6edf3")
    ax.text(5.0, 5.7, "Nothing here is recomputed.", ha="center", fontsize=9.3, color=INK)
    ax.text(5.0, 5.4, "This is the same real result shown step by step above.",
            ha="center", fontsize=9.3, color=INK)


SNAPS = [
    ("snap1_target_grade", snap_target),
    ("snap2_permuted_control", snap_permuted),
    ("snap3_injection", snap_injection),
    ("snap4_signal_map", snap_signal),
    ("snap5_p95_extraction", snap_extraction),
    ("snap6_stageB_probs", snap_stageB),
    ("snap7_occlusion_diagnosis", snap_occlusion),
    ("snap8_overview", snap_overview),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="path to methodology_cache.json")
    ap.add_argument("--out", default="methodology_real")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(f"cache not found: {cache}\nOpen 'Methodology explained' "
                         f"in the dashboard once to build it.")
    d = json.loads(cache.read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for name, fn in SNAPS:
        for transparent in (False, True):
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            fn(ax, d)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            if transparent:
                fig.savefig(out / f"{name}_transparent.png", dpi=300, transparent=True)
            else:
                fig.savefig(out / f"{name}.jpg", dpi=300, facecolor="white",
                            pil_kwargs={"quality": 95})
            plt.close(fig)
        print(f"  {name}.jpg (+ _transparent.png)")

    # overview sheet
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    for (name, fn), ax in zip(SNAPS, axes.ravel()):
        fn(ax, d)
    # 8 snaps now exactly fill the 4x2 grid (no blank panel needed)
    fig.suptitle(f"Glioma-SPARSE methodology — {d.get('slide_id','example')}",
                 fontsize=15, fontweight="bold", color=INK, y=0.997)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01, hspace=0.1, wspace=0.04)
    fig.savefig(out / "methodology_real_overview.jpg", dpi=300, facecolor="white",
                pil_kwargs={"quality": 95})
    plt.close(fig)
    print("  methodology_real_overview.jpg")
    print(f"\nAll figures (300 DPI) in: {out}")


if __name__ == "__main__":
    main()
