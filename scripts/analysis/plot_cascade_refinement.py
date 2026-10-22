"""
Figures for the cascade-refinement study, from the tables written by
scripts/analysis/compare_systems.py (results/cascade_refinement_rn18/).

  fig_e2e_accuracy.(png|pdf)      pooled end-to-end accuracy per system with a 95%
                                  cluster-bootstrap CI (resampling unique slides) and
                                  the five fold accuracies as dots
  fig_error_decomposition.(png|pdf)  where the wrong slides come from, per system
  fig_confusion_S0_vs_S12.(png|pdf)  6-class confusion matrices, reference vs combined

Colour = the change a system carries: reference grey, Option 1 blue, Option 2
orange, both violet (fixed slots of the reference categorical palette). Systems
are grouped and labelled directly; colour is never the only cue.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"ref": "#8a8985", "opt1": "#2a78d6", "opt2": "#eb6834", "both": "#4a3aa7"}
GROUP = {"S0": "ref", "S0+TTA": "opt1", "S0-c": "ref", "S1": "opt1", "S1+TTA": "opt1",
         "A-ctrl": "ref", "A-mt": "opt2", "S2": "opt2", "S2+fuse": "opt2",
         "S12": "both", "S12+TTA": "both", "S12+TTA+fuse": "both"}
SIX = ["control", "astro_low", "astro_high", "oligo_low", "oligo_high", "GBM"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def cluster_ci(dec, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(dec.slide_id.to_numpy(), return_inverse=True)
    s = np.bincount(inv, weights=dec.e2e_correct.to_numpy(float), minlength=len(ids))
    c = np.bincount(inv, minlength=len(ids)).astype(float)
    d = rng.integers(0, len(ids), size=(n, len(ids)))
    b = s[d].sum(1) / c[d].sum(1)
    return np.percentile(b, [2.5, 97.5])


def six_label(row, which):
    if which == "true":
        g, m = row.true_grade_class, row.true_mutation
        if g == "control":
            return "control"
        return {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high",
                ("low", "IDH_mt_1p19q"): "oligo_low", ("high", "IDH_mt_1p19q"): "oligo_high",
                ("high", "IDH_wt"): "GBM"}.get((g, m))
    if row.grade_class == "control":
        return "control"
    if not row.has_stageB:
        return "no Stage B"
    return {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high",
            ("low", "IDH_mt_1p19q"): "oligo_low", ("high", "IDH_mt_1p19q"): "oligo_high",
            ("high", "IDH_wt"): "GBM"}.get((row.grade_class, row.pred_sub), "wt, low grade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/cascade_refinement_rn18")
    a = ap.parse_args()
    d = Path(a.dir)
    summ = pd.read_csv(d / "systems_summary.csv")
    dec = pd.read_csv(d / "per_slide_decisions.csv")
    order = list(summ.system)

    # ---- Fig 1: accuracy with CI and fold dots
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.2))
    for i, s in enumerate(order[::-1]):
        row = summ[summ.system == s].iloc[0]
        c = COL[GROUP.get(s, "ref")]
        lo, hi = cluster_ci(dec[dec.system == s])
        ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.plot(row.e2e_acc_pooled, i, "o", ms=7, color=c, mec="white", mew=1.5, zorder=3)
        folds = [float(x) for x in str(row.e2e_acc_per_fold).split()]
        ax.plot(folds, [i] * len(folds), "|", color=c, alpha=.55, ms=7, zorder=1)
        ax.text(0.80, i, f"{row.e2e_acc_pooled:.3f}", va="center", ha="right", color=INK, fontsize=8)
    ax.set_xlim(0.36, 0.82)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ref = summ[summ.system == "S0"].e2e_acc_pooled.iloc[0]
    ax.axvline(ref, color=MUTED, lw=.8, ls=(0, (3, 3)), zorder=0)
    ax.set_xlabel("End-to-end (6-way integrated diagnosis) accuracy, 5 held-out test folds")
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title("Dot = pooled accuracy, bar = 95% CI (bootstrap over slides), ticks = folds;"
                 "\ndashed line = reference S0", loc="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"fig_e2e_accuracy.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig 2: error decomposition
    e = pd.read_csv(d / "error_decomposition.csv").set_index("system").loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.2))
    parts = [("grade_wrong", "Grade wrong", "#e34948"),
             ("grade_ok_no_stageB", "Grade right, no region", "#eda100"),
             ("grade_ok_subtype_wrong", "Grade right, subtype wrong", "#4a3aa7")]
    left = np.zeros(len(order))
    y = np.arange(len(order))[::-1]
    for key, lab, c in parts:
        v = e[key].to_numpy(float)
        ax.barh(y, v, left=left, color=c, height=.62, edgecolor="white", linewidth=1.5,
                label=lab)
        left += v
    for yy, tot in zip(y, left):
        ax.text(tot + 1, yy, f"{int(tot)}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Wrongly diagnosed test slides (of 220)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"fig_error_decomposition.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig 3: confusion matrices, S0 vs S12
    pair = [s for s in ("S0", "S12") if s in order]
    if len(pair) == 2:
        cols = SIX + ["no Stage B", "wt, low grade"]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
        for ax, s in zip(axes, pair):
            x = dec[dec.system == s].copy()
            x["t"] = x.apply(lambda r: six_label(r, "true"), axis=1)
            x["p"] = x.apply(lambda r: six_label(r, "pred"), axis=1)
            m = pd.crosstab(x.t, x.p).reindex(index=SIX, columns=cols, fill_value=0)
            norm = m.div(m.sum(1).replace(0, 1), axis=0)
            ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
            for i in range(len(SIX)):
                for j in range(len(cols)):
                    if m.iloc[i, j]:
                        ax.text(j, i, int(m.iloc[i, j]), ha="center", va="center", fontsize=8,
                                color="white" if norm.iloc[i, j] > .55 else INK)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=40, ha="right")
            ax.set_yticks(range(len(SIX)))
            ax.set_yticklabels(SIX)
            acc = summ[summ.system == s].e2e_acc_pooled.iloc[0]
            ax.set_title(f"{s}  (accuracy {acc:.3f})", loc="left", fontsize=9)
            ax.set_xlabel("Predicted")
        axes[0].set_ylabel("True (WHO 2021)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(d / f"fig_confusion_S0_vs_S12.{ext}", dpi=300)
        plt.close(fig)
    print("figures written to", d)


if __name__ == "__main__":
    main()
