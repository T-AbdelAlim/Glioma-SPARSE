"""
Figures for every experiment and combined overlays.

results/figures/<experiment>/          independent figures per experiment (PNG + PDF)
    01_confusion_6class     confusion matrix of the 6-way integrated diagnosis (counts, row-normalised colour)
    02_roc_6class           one-vs-rest ROC per class + macro average (pooled over the 5 test folds)
    03_roc_stages           ROC of the grade step, the subtype step and IDH mutant-vs-wildtype
    04_pr_6class            precision-recall curves per class (average precision)
    05_per_class_metrics    precision / recall / F1 per class
    06_per_fold             accuracy, balanced accuracy, macro-F1 and macro-AUC per test fold
    07_calibration          reliability diagram + confidence histogram (correct vs incorrect)
    08_error_breakdown      outcome per true class: correct / grade wrong / no region / subtype wrong
results/figures/combined/              overlays of systems on the same axes (see README.md there)

All curves use the same per-slide probabilities and metric code as MASTER_COMPARISON.md
(scripts/analysis/master_comparison.py). ROC/PR curves pool the five test folds (220 rows);
AUCs in legends are pooled AUCs, so they differ slightly from the fold-mean AUCs in the tables.
Six-class scores: P(grade) x P(subtype) for cascades (control: P(control)), the six-class softmax for CLAM.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from scripts.analysis.master_comparison import (
    GRADES, SIX, SUB, archived, build_systems, cascade_table, metrics, sub_probs_archived,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
MASTER = REPO / "results" / "master_comparison" / "master_table.csv"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 8})
CLS_COL = {"control": "#8a8985", "astro_low": "#2a78d6", "astro_high": "#4a3aa7", "oligo_low": "#1baf7a",
           "oligo_high": "#eda100", "GBM": "#e34948"}
NICE = {"control": "control", "astro_low": "astro low", "astro_high": "astro high", "oligo_low": "oligo low",
        "oligo_high": "oligo high", "GBM": "GBM"}
FAMILY = {"ResNet cascades (archived)": "#2a78d6", "ResNet-18 refinement study": "#2a78d6", "UNI2-h at Stage B (sparse)": "#eb6834",
          "Benchmark (full slide)": "#4a3aa7", "TCGA-extended (different test set)": "#1baf7a"}


# ------------------------------------------------------------------ data
def arrays(t):
    t = t.copy()
    if "true6" not in t:
        from scripts.analysis.master_comparison import six_true
        t["true6"] = [six_true(g, m) for g, m in zip(t.true_grade_class, t.true_mutation)]
    y6 = t.true6.map({c: i for i, c in enumerate(SIX)}).to_numpy()
    S = np.stack(t.six_score.to_numpy())
    gy = t.true_grade_class.map({c: i for i, c in enumerate(GRADES)}).to_numpy()
    G = np.stack(t.grade_prob.to_numpy())
    tum = (t.true_grade_class != "control").to_numpy()
    hb = t.has_b.to_numpy(bool)
    m = tum & hb
    sy = t.true_mutation[m].map({k: i for i, k in enumerate(SUB)}).to_numpy()
    P = np.stack(t.sub_prob.to_numpy())[m]
    return dict(t=t, y6=y6, S=S, gy=gy, G=G, sy=sy, P=P, m=m)


def roc_set(y, S, names):
    """per-class OvR ROC (pooled) + macro-average; returns dict name -> (fpr, tpr, auc)."""
    grid = np.linspace(0, 1, 201)
    out, tprs = {}, []
    for k, n in enumerate(names):
        pos = (y == k)
        if pos.sum() == 0 or pos.sum() == len(y):
            continue
        f, tp, _ = roc_curve(pos.astype(int), S[:, k])
        out[n] = (f, tp, auc(f, tp))
        tprs.append(np.interp(grid, f, tp))
    macro = np.mean(tprs, axis=0)
    out["macro"] = (grid, macro, float(np.mean([v[2] for v in out.values()])))
    return out


def pr_set(y, S, names):
    grid = np.linspace(0, 1, 201)
    out, precs = {}, []
    for k, n in enumerate(names):
        pos = (y == k)
        if pos.sum() == 0 or pos.sum() == len(y):
            continue
        p, r, _ = precision_recall_curve(pos.astype(int), S[:, k])
        ap = average_precision_score(pos.astype(int), S[:, k])
        out[n] = (r, p, ap)
        env = np.maximum.accumulate(p[::-1])[::-1]                # precision envelope
        precs.append(np.interp(grid, r[::-1], env[::-1], left=env.max()))
    out["macro"] = (grid, np.mean(precs, axis=0), float(np.mean([v[2] for v in out.values()])))
    return out


def acc_ci(ok, ids, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=ok.astype(float), minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u)))
    return np.percentile(s[dr].sum(1) / c[dr].sum(1), [2.5, 97.5])


def outcomes(t):
    """per row: correct / grade wrong / no region / subtype wrong."""
    ok = (t.true6 == t.six_pred).to_numpy()
    gok = (t.grade_pred == t.true_grade_class).to_numpy()
    tum = (t.true_grade_class != "control").to_numpy()
    hb = t.has_b.to_numpy(bool)
    out = np.where(ok, "correct", np.where(~gok, "grade wrong", np.where(tum & ~hb, "no region", "subtype wrong")))
    return out


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------ per-experiment figs
def fig_confusion(a, title, folder):
    t = a["t"]
    extra = [c for c in ("no_stageB", "wt_low_grade") if (t.six_pred == c).any()]
    cols = SIX + extra
    m = pd.crosstab(t.true6, t.six_pred).reindex(index=[c for c in SIX if (t.true6 == c).any()], columns=cols, fill_value=0)
    norm = m.div(m.sum(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m.iloc[i, j]:
                ax.text(j, i, int(m.iloc[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if norm.iloc[i, j] > .55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NICE.get(c, c.replace("_", " ")) for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([NICE[c] for c in m.index])
    acc = (t.true6 == t.six_pred).mean()
    ax.set_title(f"{title}\n6-way accuracy {acc:.3f}  (counts; colour = share of the true class)", loc="left", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (WHO 2021)")
    save(fig, folder, "01_confusion_6class")


def draw_roc(ax, r, cols=None, macro_style=True):
    for n, (f, tp, a_) in r.items():
        if n == "macro":
            continue
        ax.plot(f, tp, color=(cols or CLS_COL).get(n, "#555"), lw=1.6, label=f"{NICE.get(n, n)}  AUC {a_:.3f}")
    f, tp, a_ = r["macro"]
    ax.plot(f, tp, color=INK, lw=2.4, ls=(0, (4, 2)), label=f"macro-average  AUC {a_:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")


def fig_roc6(a, title, folder):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    draw_roc(ax, roc_set(a["y6"], a["S"], SIX))
    ax.set_title(f"{title}\nOne-vs-rest ROC, 6-way integrated diagnosis (pooled 5 test folds)", loc="left", fontsize=9)
    save(fig, folder, "02_roc_6class")


def fig_roc_stages(a, title, folder):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    gcol = {"control": CLS_COL["control"], "low": "#2a78d6", "high": "#e34948"}
    draw_roc(axs[0], roc_set(a["gy"], a["G"], GRADES), gcol)
    axs[0].set_title("Grade (control / low / high), all rows", loc="left", fontsize=9)
    scol = {"IDH_mt": "#2a78d6", "IDH_mt_1p19q": "#1baf7a", "IDH_wt": "#e34948"}
    n_sub = len(a["sy"])
    if n_sub > 0 and len(np.unique(a["sy"])) == 3:
        draw_roc(axs[1], roc_set(a["sy"], a["P"], SUB), scol)
        f, tp, _ = roc_curve((a["sy"] == 2).astype(int), a["P"][:, 2])
        axs[2].plot(f, tp, color="#e34948", lw=2, label=f"wildtype vs mutant  AUC {auc(f, tp):.3f}")
        axs[2].plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
        axs[2].set_xlabel("False positive rate")
        axs[2].set_ylabel("True positive rate")
        axs[2].legend(loc="lower right")
    axs[1].set_title(f"Molecular subtype (3-way), n = {n_sub} tumour slides", loc="left", fontsize=9)
    axs[2].set_title("IDH wildtype vs mutant", loc="left", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    save(fig, folder, "03_roc_stages")


def fig_pr(a, title, folder):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    r = pr_set(a["y6"], a["S"], SIX)
    for n, (rec, pr, ap) in r.items():
        if n == "macro":
            continue
        ax.plot(rec, pr, color=CLS_COL[n], lw=1.6, label=f"{NICE[n]}  AP {ap:.3f}")
    g, p, ap = r["macro"]
    ax.plot(g, p, color=INK, lw=2.4, ls=(0, (4, 2)), label=f"macro-average  AP {ap:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.set_title(f"{title}\nPrecision-recall, 6-way (pooled 5 test folds)", loc="left", fontsize=9)
    save(fig, folder, "04_pr_6class")


def prf(t):
    rows = []
    for c in SIX:
        tp = ((t.true6 == c) & (t.six_pred == c)).sum()
        fn = ((t.true6 == c) & (t.six_pred != c)).sum()
        fp = ((t.true6 != c) & (t.six_pred == c)).sum()
        if (t.true6 == c).sum() == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((c, p, r, f, int((t.true6 == c).sum())))
    return pd.DataFrame(rows, columns=["cls", "precision", "recall", "f1", "n"])


def fig_prf(a, title, folder):
    d = prf(a["t"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(d))
    w = .26
    for i, (col, lab, c) in enumerate((("precision", "precision", "#2a78d6"), ("recall", "recall", "#eb6834"), ("f1", "F1", "#4a3aa7"))):
        ax.bar(x + (i - 1) * w, d[col], w * .92, color=c, label=lab)
        for xi, v in zip(x + (i - 1) * w, d[col]):
            ax.text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{NICE[c]}\n(n={n})" for c, n in zip(d.cls, d.n)])
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=3, loc="upper right")
    ax.set_title(f"{title}\nPer-class precision / recall / F1 of the 6-way decision (pooled)", loc="left", fontsize=9)
    save(fig, folder, "05_per_class_metrics")


def per_fold(t):
    rows = []
    for f in sorted(t.fold.unique()):
        r, _ = metrics(t[t.fold == f].reset_index(drop=True))
        rows.append((f, r["e2e_acc_pooled"], r["e2e_balacc_mean"], r["e2e_macroF1_pooled"], r["e2e_macroAUC_pooled"]))
    return pd.DataFrame(rows, columns=["fold", "accuracy", "balanced accuracy", "macro-F1", "macro-AUC"])


def fig_perfold(a, title, folder):
    d = per_fold(a["t"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for col, c, mk in (("accuracy", "#2a78d6", "o"), ("balanced accuracy", "#eb6834", "s"),
                       ("macro-F1", "#4a3aa7", "^"), ("macro-AUC", "#1baf7a", "D")):
        ax.plot(d.fold, d[col], color=c, marker=mk, ms=6, lw=1.6, mec="white", label=f"{col}  (mean {d[col].mean():.3f})")
    ax.set_xticks(d.fold)
    ax.set_xlabel("Test fold (split_0k)")
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title(f"{title}\nPerformance per test fold (44 rows each)", loc="left", fontsize=9)
    save(fig, folder, "06_per_fold")


def calib(a):
    S = a["S"]
    conf = S.max(1)
    ok = (S.argmax(1) == a["y6"])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    rows, ece = [], 0.0
    for b in range(10):
        m = idx == b
        if m.sum():
            rows.append((conf[m].mean(), ok[m].mean(), int(m.sum())))
            ece += m.sum() / len(conf) * abs(conf[m].mean() - ok[m].mean())
    return np.array(rows), ece, conf, ok


def fig_calibration(a, title, folder):
    rows, ece, conf, ok = calib(a)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    axs[0].plot(rows[:, 0], rows[:, 1], marker="o", color="#2a78d6", lw=1.8, mec="white")
    for x_, y_, n in rows:
        axs[0].text(x_, y_ + .03, str(int(n)), fontsize=6.5, ha="center", color=MUTED)
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_xlabel("Confidence (highest six-class probability)")
    axs[0].set_ylabel("Fraction correct")
    axs[0].set_title(f"Reliability diagram, ECE {ece:.3f} (numbers = rows per bin)", loc="left", fontsize=9)
    bins = np.linspace(0, 1, 11)
    axs[1].hist([conf[ok], conf[~ok]], bins=bins, stacked=True, color=["#1baf7a", "#e34948"], label=["correct", "incorrect"],
                edgecolor="white")
    axs[1].set_xlabel("Confidence")
    axs[1].set_ylabel("Rows")
    axs[1].legend()
    axs[1].set_title("Confidence of correct and incorrect predictions", loc="left", fontsize=9)
    fig.suptitle(f"{title}   (argmax of the six-class score)", x=0.01, ha="left", fontsize=10)
    save(fig, folder, "07_calibration")


OUTCOME_COL = [("correct", "#1baf7a"), ("grade wrong", "#e34948"), ("no region", "#eda100"), ("subtype wrong", "#4a3aa7")]


def fig_errors(a, title, folder):
    t = a["t"]
    o = outcomes(t)
    cls = [c for c in SIX if (t.true6 == c).any()]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    left = np.zeros(len(cls))
    y = np.arange(len(cls))[::-1]
    for name, c in OUTCOME_COL:
        v = np.array([((t.true6 == k) & (o == name)).sum() for k in cls], float)
        ax.barh(y, v, left=left, color=c, height=.62, edgecolor="white", linewidth=1.5, label=name)
        for yy, l, vv in zip(y, left, v):
            if vv >= 3:
                ax.text(l + vv / 2, yy, int(vv), ha="center", va="center", fontsize=7.5, color="white")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NICE[c]} (n={int((t.true6 == c).sum())})" for c in cls])
    ax.set_xlabel("Test rows")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"{title}\nOutcome per true class", loc="left", fontsize=9, pad=24)
    save(fig, folder, "08_error_breakdown")


def per_experiment(a, title, folder):
    for fn in (fig_confusion, fig_roc6, fig_roc_stages, fig_pr, fig_prf, fig_perfold, fig_calibration, fig_errors):
        try:
            fn(a, title, folder)
        except Exception as e:                           # keep going, report at the end
            print(f"   !! {folder.name}/{fn.__name__}: {e}")
            plt.close("all")


# ----------------------------------------------------------- combined figs
def style_for(name):
    """(colour, linestyle, linewidth) per system in the headline overlays."""
    tbl = {
        "ResNet18 best-F1 (S0)": ("#2a78d6", "-", 1.8), "ResNet18 best-AUC": ("#2a78d6", ":", 1.8),
        "ResNet50 best-F1": ("#1baf7a", "-", 1.8), "ResNet50 best-AUC": ("#1baf7a", ":", 1.8),
        "ResNet18 best-F1 Stage A + UNI2-h, 64 tiles": ("#eb6834", "-", 2.0),
        "ResNet50 best-F1 Stage A + UNI2-h, 64 tiles": ("#e34948", "-", 2.0),
        "ResNet50 best-AUC Stage A + UNI2-h, 64 tiles": ("#c98500", "--", 2.0),
        "TRIDENT + UNI2-h + CLAM-SB": ("#4a3aa7", "-", 2.4),
    }
    return tbl[name]


LABEL = {"ResNet18 best-F1 (S0)": "ResNet18-F1 cascade", "ResNet18 best-AUC": "ResNet18-AUC cascade",
         "ResNet50 best-F1": "ResNet50-F1 cascade", "ResNet50 best-AUC": "ResNet50-AUC cascade",
         "ResNet18 best-F1 Stage A + UNI2-h, 64 tiles": "UNI2-h at Stage B RN18-F1 (64 tiles)",
         "ResNet50 best-F1 Stage A + UNI2-h, 64 tiles": "UNI2-h at Stage B RN50-F1 (64 tiles)",
         "ResNet50 best-AUC Stage A + UNI2-h, 64 tiles": "UNI2-h at Stage B RN50-AUC (64 tiles)",
         "TRIDENT + UNI2-h + CLAM-SB": "UNI2-h + CLAM (full slide)"}
HEAD = list(LABEL)


def overlay_roc(ax, systems, kind, which):
    """which: 'macro' or a class name; kind: 6 | grade | subtype | idh"""
    for name in systems:
        a = A[name]
        c, ls, lw = style_for(name) if name in LABEL else SPECIAL[name]
        if kind == 6:
            r = roc_set(a["y6"], a["S"], SIX)
        elif kind == "grade":
            r = roc_set(a["gy"], a["G"], GRADES)
        elif kind == "subtype":
            r = roc_set(a["sy"], a["P"], SUB)
        else:
            f, tp, _ = roc_curve((a["sy"] == 2).astype(int), a["P"][:, 2])
            r = {"idh": (f, tp, auc(f, tp))}
            which = "idh"
        if which not in r:
            continue  # class absent from this system's data (e.g. no "control" slides externally)
        f, tp, au = r[which]
        ax.plot(f, tp, color=c, ls=ls, lw=lw, label=f"{LABEL.get(name, name)}  {au:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")


A, SPECIAL = {}, {}


def zoomed_roc_inset(ax, systems, kind, which):
    ins = ax.inset_axes([0.42, 0.10, 0.5, 0.42])
    overlay_roc(ins, systems, kind, which)
    ins.set_xlim(0, 0.35)
    ins.set_ylim(0.55, 1.0)
    ins.get_legend() and ins.get_legend().remove()
    ins.set_xlabel("")
    ins.set_ylabel("")
    ins.tick_params(labelsize=6)
    ins.set_title("zoom: low FPR", fontsize=6.5)


def combined(R, tables):
    C = OUT / "combined"
    head = [n for n in HEAD if n in A]
    # 01 macro ROC 6-class
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    overlay_roc(ax, head, 6, "macro")
    ax.legend(loc="lower right", title="system  (pooled macro-AUC)", fontsize=7.5, bbox_to_anchor=(1.0, 0.0))
    ax.set_title("6-way integrated diagnosis: macro-average one-vs-rest ROC", loc="left", fontsize=9)
    save(fig, C, "01_headline_roc_macro_6class")
    # 02 per-class ROC grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axs.ravel(), SIX):
        overlay_roc(ax, head, 6, c)
        ax.set_title(f"{NICE[c]}", loc="left", fontsize=9)
        if ax.get_legend_handles_labels()[0]:  # ax.plot always draws the unlabelled diagonal reference line
            ax.legend(loc="lower right", fontsize=6.5)
        else:  # e.g. "control": absent from every system in an external cohort with no controls
            ax.text(0.5, 0.5, "no slides of this class\nin this cohort", ha="center", va="center",
                    fontsize=9, color=MUTED, transform=ax.transAxes)
    fig.suptitle("Per-class one-vs-rest ROC of the 6-way diagnosis (legend: pooled AUC)", x=0.01, ha="left", fontsize=10)
    save(fig, C, "02_headline_roc_per_class")
    # 03-05 stage ROCs
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    overlay_roc(axs[0], head, "grade", "macro")
    axs[0].set_title("Grade (control / low / high), macro one-vs-rest", loc="left", fontsize=9)
    overlay_roc(axs[1], head, "subtype", "macro")
    axs[1].set_title("Molecular subtype (3-way), macro one-vs-rest", loc="left", fontsize=9)
    overlay_roc(axs[2], head, "idh", "idh")
    axs[2].set_title("IDH wildtype vs mutant", loc="left", fontsize=9)
    for ax in axs:
        ax.legend(loc="lower right", fontsize=7)
    save(fig, C, "03_headline_roc_stages")
    # 04 macro PR
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for n in head:
        c, ls, lw = style_for(n)
        g, p, ap = pr_set(A[n]["y6"], A[n]["S"], SIX)["macro"]
        ax.plot(g, p, color=c, ls=ls, lw=lw, label=f"{LABEL[n]}  AP {ap:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Macro-average precision-recall, 6-way diagnosis", loc="left", fontsize=9)
    save(fig, C, "04_headline_pr_macro_6class")
    # 05 metric bars (accuracy w/ CI; others fold mean +- SD)
    Rh = R.set_index("system").loc[head]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    y = np.arange(len(head))[::-1]
    spec = [("e2e_acc_pooled", None, "End-to-end accuracy\n(pooled; bar = 95% CI, bootstrap over slides)"),
            ("e2e_balacc_mean", "e2e_balacc_sd", "Balanced accuracy\n(fold mean ± SD)"),
            ("e2e_macroF1_mean", "e2e_macroF1_sd", "Macro-F1\n(fold mean ± SD)"),
            ("e2e_macroAUC_mean", "e2e_macroAUC_sd", "Macro-AUC (6-class)\n(fold mean ± SD)")]
    for ax, (col, sd, ttl) in zip(axs, spec):
        for yy, n in zip(y, head):
            c, ls, lw = style_for(n)
            v = Rh.loc[n, col]
            if sd is None:
                ok = (A[n]["t"].true6 == A[n]["t"].six_pred).to_numpy()
                lo, hi = acc_ci(ok, A[n]["t"].slide_id.to_numpy())
                ax.plot([lo, hi], [yy, yy], color=c, lw=2.5, solid_capstyle="round")
            else:
                ax.plot([v - Rh.loc[n, sd], v + Rh.loc[n, sd]], [yy, yy], color=c, lw=2.5, solid_capstyle="round")
            ax.plot(v, yy, "o", color=c, ms=8, mec="white", mew=1.5)
            ax.text(1.03, yy, f"{v:.3f}", va="center", ha="left", fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_xlim(*((0.78, 0.98) if col == "e2e_macroAUC_mean" else (0.3, 0.9)))
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([LABEL[n] for n in head])
    save(fig, C, "05_headline_metrics")
    # 06 per-class recall / F1 grouped
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), sharey=True)
    for ax, col in zip(axs, ("recall", "f1")):
        w = .8 / len(head)
        for i, n in enumerate(head):
            d = prf(A[n]["t"]).set_index("cls").reindex(SIX)
            ax.bar(np.arange(6) + (i - (len(head) - 1) / 2) * w, d[col], w * .92, color=style_for(n)[0],
                   hatch={"-": None, ":": "..", "--": "//"}[style_for(n)[1]], edgecolor="white", label=LABEL[n])
        ax.set_xticks(range(6))
        ax.set_xticklabels([NICE[c] for c in SIX])
        ax.set_title({"recall": "Per-class recall (sensitivity)", "f1": "Per-class F1"}[col], loc="left", fontsize=9)
        ax.set_ylim(0, 1.05)
    axs[0].legend(ncol=2, fontsize=7, loc="lower left")
    save(fig, C, "06_headline_per_class_recall_f1")
    # 07 per-fold
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True)
    pf = {n: per_fold(A[n]["t"]) for n in head}
    for ax, col in zip(axs, ("accuracy", "balanced accuracy", "macro-F1", "macro-AUC")):
        for n in head:
            c, ls, lw = style_for(n)
            ax.plot(pf[n].fold, pf[n][col], color=c, ls=ls, lw=lw, marker="o", ms=4, label=LABEL[n])
        ax.set_title(col, loc="left", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Test fold")
    axs[0].legend(fontsize=6.5, loc="lower left")
    save(fig, C, "07_headline_per_fold")
    # 08 stage metrics bars
    fig, ax = plt.subplots(figsize=(14, 4.6))
    cols = [("grade_acc", "grade acc"), ("grade_macroAUC", "grade AUC"), ("subtype_acc", "subtype acc"),
            ("subtype_macroF1", "subtype F1"), ("subtype_macroAUC", "subtype AUC"), ("IDH_mut_vs_wt_AUC", "IDH mut-vs-wt AUC")]
    w = .8 / len(head)
    for i, n in enumerate(head):
        v = [Rh.loc[n, c] for c, _ in cols]
        ax.bar(np.arange(len(cols)) + (i - (len(head) - 1) / 2) * w, v, w * .92, color=style_for(n)[0],
               hatch={"-": None, ":": "..", "--": "//"}[style_for(n)[1]], edgecolor="white", label=LABEL[n])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([l for _, l in cols])
    ax.set_ylim(0.4, 1.02)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Stage-level performance (grade: all 220 rows; subtype: tumour slides reaching the subtype step)",
                 loc="left", fontsize=9, pad=30)
    save(fig, C, "08_headline_stage_metrics")
    # 09 error breakdown all systems
    fig, ax = plt.subplots(figsize=(9, 5.6))
    names = list(R.system)
    ordr = [n for n in names if n in A]
    y = np.arange(len(ordr))[::-1]
    left = np.zeros(len(ordr))
    for name, c in OUTCOME_COL[1:]:
        v = np.array([(outcomes(A[n]["t"]) == name).sum() for n in ordr], float)
        ax.barh(y, v, left=left, color=c, height=.66, edgecolor="white", linewidth=1.2, label=name)
        left += v
    for yy, tot in zip(y, left):
        ax.text(tot + .8, yy, int(tot), va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(n, n) for n in ordr], fontsize=7.5)
    ax.set_xlabel("Wrongly diagnosed rows (slides x fold models)")
    ax.legend(loc="lower right")
    ax.set_title("Where the errors come from, all systems", loc="left", fontsize=9)
    save(fig, C, "09_error_breakdown_all_systems")
    # 10 calibration overlay
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":")
    for n in head:
        rows, ece, _, _ = calib(A[n])
        c, ls, lw = style_for(n)
        ax.plot(rows[:, 0], rows[:, 1], color=c, ls=ls, lw=lw, marker="o", ms=3.5, label=f"{LABEL[n]}  ECE {ece:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Fraction correct")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Reliability of the six-class score", loc="left", fontsize=9)
    save(fig, C, "10_headline_calibration")
    # 11 accuracy vs compute
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, xcol, xl in ((axs[0], "gflops_per_slide", "Inference FLOPs per slide (log scale, TFLOPs)"),
                         (axs[1], "params_trained", "Trained parameters (log scale, millions)")):
        for r in R.itertuples():
            if r.system not in A:
                continue
            fam = FAMILY.get(r.group, "#555")
            x = getattr(r, xcol) / (1e3 if xcol == "gflops_per_slide" else 1e6)
            ax.scatter(x, r.e2e_acc_pooled, s=46 if r.system in LABEL else 22, color=fam, edgecolor="white", linewidth=1,
                       alpha=1 if r.system in LABEL else .55, zorder=3)
            if r.system in LABEL and r.system != "ResNet50 best-AUC":      # RN50-F1/AUC cascades coincide
                lab = "ResNet50-F1 / AUC cascades" if r.system == "ResNet50 best-F1" else LABEL[r.system]
                right = r.system == "TRIDENT + UNI2-h + CLAM-SB"
                ax.annotate(lab, (x, r.e2e_acc_pooled), xytext=(-6 if right else 5, 5), textcoords="offset points",
                            fontsize=7, ha="right" if right else "left")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("End-to-end accuracy (pooled)")
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
                           (("ResNet cascades / refinement", "#2a78d6"), ("UNI2-h at Stage B (Stage A regions, n UNI2-h tiles)", "#eb6834"),
                            ("full-slide UNI2-h + CLAM", "#4a3aa7"))], loc="lower right")
    fig.suptitle("Performance vs compute (UNI2-h weights are frozen and excluded from 'trained parameters' on the right)",
                 x=0.01, ha="left", fontsize=10)
    save(fig, C, "11_accuracy_vs_compute")
    # 12 accuracy vs tiles
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    cfgs = (("ResNet18 best-F1", "#eb6834", "RN18-F1 Stage A"), ("ResNet50 best-F1", "#e34948", "RN50-F1 Stage A"),
            ("ResNet50 best-AUC", "#c98500", "RN50-AUC Stage A"))
    xs = {"8": 8, "32": 32, "64": 64, "all": 2900}
    for pre, c, lab in cfgs:
        pts = []
        for n in ("8", "32", "64", "all"):
            m = R[R.system == f"{pre} Stage A + UNI2-h, {n if n != 'all' else 'all'} tiles"]
            if len(m):
                r = m.iloc[0]
                pts.append((xs[n], r.e2e_acc_pooled_over_tile_draws_mean, r.e2e_acc_pooled_over_tile_draws_sd))
        if pts:
            p = np.array(pts)
            ax.errorbar(p[:, 0], p[:, 1], yerr=p[:, 2], color=c, marker="o", lw=1.8, capsize=3, label=f"UNI2-h at Stage B, {lab}")
        base = R[R.system == (pre + (" (S0)" if pre == "ResNet18 best-F1" else ""))].iloc[0].e2e_acc_pooled
        ax.axhline(base, color=c, ls=":", lw=1.2)
    ax.axhline(R[R.system == "TRIDENT + UNI2-h + CLAM-SB"].iloc[0].e2e_acc_pooled, color="#4a3aa7", lw=1.6, label="UNI2-h + CLAM, all ~11,900 tiles")
    ax.set_xscale("log")
    ax.set_xticks([8, 32, 64, 256, 2900])
    ax.set_xticklabels(["8", "32", "64", "256", "all region\ntiles (~2,900)"])
    ax.set_xlabel("UNI2-h tiles per slide")
    ax.set_ylabel("End-to-end accuracy (mean ± SD over 3 tile draws)")
    ax.set_ylim(0.62, 0.78)
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.text(0.01, 0.03, "dotted lines: the same Stage A with its own ResNet Stage B (RN18-F1 0.636; RN50-F1 and RN50-AUC both 0.645)",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    ax.set_title("Accuracy vs number of UNI2-h tiles", loc="left", fontsize=9)
    save(fig, C, "12_resuni_accuracy_vs_tiles")
    # 13 refinement overlay ROC + metrics
    ref = [n for n in R.system if n in A and R.set_index("system").loc[n, "group"] in
           ("ResNet-18 refinement study",) or n == "ResNet18 best-F1 (S0)"]
    cols = plt.cm.tab10(np.arange(len(ref)))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for n, c in zip(ref, cols):
        r = roc_set(A[n]["y6"], A[n]["S"], SIX)
        f, tp, au = r["macro"]
        axs[0].plot(f, tp, color=c, lw=1.8, label=f"{n}  {au:.3f}")
        s = roc_set(A[n]["sy"], A[n]["P"], SUB)["macro"]
        axs[1].plot(s[0], s[1], color=c, lw=1.8, label=f"{n}  {s[2]:.3f}")
    for ax, ttl in zip(axs, ("6-way diagnosis, macro ROC", "Molecular subtype (3-way), macro ROC")):
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("ResNet-18 refinement study (Options 1 and 2), legend: pooled AUC", x=0.01, ha="left", fontsize=10)
    save(fig, C, "13_refinement_roc")
    # 14 confusion differences
    pairs = (("ResNet50 best-F1", "ResNet50 best-F1 Stage A + UNI2-h, 64 tiles", "UNI2-h at Stage B RN50-F1 minus ResNet50-F1 cascade"),
             ("ResNet50 best-F1 Stage A + UNI2-h, 64 tiles", "TRIDENT + UNI2-h + CLAM-SB", "CLAM minus UNI2-h at Stage B RN50-F1"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (a_, b_, ttl) in zip(axs, pairs):
        def norm(n):
            t = A[n]["t"]
            m = pd.crosstab(t.true6, t.six_pred).reindex(index=SIX, columns=SIX, fill_value=0)
            return m.div(m.sum(1).replace(0, 1), axis=0)
        d = norm(b_) - norm(a_)
        im = ax.imshow(d.to_numpy(), cmap="RdBu", vmin=-.5, vmax=.5)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                if abs(d.iloc[i, j]) >= .04:
                    ax.text(j, i, f"{d.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels([NICE[c] for c in SIX], rotation=35, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels([NICE[c] for c in SIX])
        ax.set_title(ttl + "\n(change in the share of each true class; diagonal + = better)", loc="left", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    save(fig, C, "14_confusion_differences")
    (C / "README.md").write_text(COMBINED_README, encoding="utf8")


COMBINED_README = """# Combined figures

Overlays of several systems on the same axes. Headline systems: the four archived ResNet cascades
(blue = ResNet-18, green = ResNet-50; solid = best-F1 checkpoints, dotted = best-AUC), the three UNI2-h at Stage B
systems with 64 UNI2-h tiles per slide (orange / red / dark yellow) and the full-slide UNI2-h + CLAM benchmark (violet).
Curves pool the five held-out test folds (220 rows, 190 unique slides); legend AUCs are pooled AUCs
(the fold-mean AUCs are in docs/experiments/MASTER_COMPARISON.md). Source data: results/master_comparison/.

| File | Content |
|---|---|
| 01_headline_roc_macro_6class | macro-average one-vs-rest ROC of the 6-way diagnosis |
| 02_headline_roc_per_class | per-class ROC (control, astro low/high, oligo low/high, GBM) |
| 03_headline_roc_stages | ROC of grade, molecular subtype and IDH wildtype-vs-mutant |
| 04_headline_pr_macro_6class | macro precision-recall |
| 05_headline_metrics | accuracy (95% CI), balanced accuracy, macro-F1, macro-AUC |
| 06_headline_per_class_recall_f1 | per-class recall and F1 |
| 07_headline_per_fold | the four metrics per test fold |
| 08_headline_stage_metrics | grade / subtype / IDH metrics |
| 09_error_breakdown_all_systems | wrong slides: grade / no region / subtype, all 25 systems |
| 10_headline_calibration | reliability curves and ECE |
| 11_accuracy_vs_compute | accuracy against FLOPs per slide and against trained parameters |
| 12_resuni_accuracy_vs_tiles | UNI2-h at Stage B accuracy vs number of tiles |
| 13_refinement_roc | ROC of the ResNet-18 refinement study (Options 1 and 2) |
| 14_confusion_differences | change in confusion matrices: UNI2-h at Stage B vs ResNet50 cascade, CLAM vs UNI2-h at Stage B |
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    args = ap.parse_args()
    global OUT
    R = pd.read_csv(MASTER)
    extra = []
    systems, *_ = build_systems()
    # TCGA-extended archived runs (different test set, no control class)
    for name, path in (("RN50 TCGA-extended best-AUC", "results/e2e_testset_tcga_ext/e2e_testset_report.xlsx"),
                       ("RN50 TCGA-extended best-F1", "results/e2e_testset_tcga_ext_f1/e2e_testset_report.xlsx")):
        d = archived(path)
        if "prob_control" not in d:
            d["prob_control"] = 0.0
        extra.append(("TCGA-extended (different test set)", name, cascade_table(d, sub_probs_archived(d)), None, d))
    keep = []
    for g, n, t, c, d in systems + extra:
        if g == "UNI2-h at Stage B (sparse)" and not n.endswith(", 64 tiles"):
            continue
        keep.append((g, n, t))
    only = set(args.only)
    for g, n, t in keep:
        a = arrays(t)
        A[n] = a
    for i, (g, n, t) in enumerate(keep, 1):
        folder = OUT / re.sub(r"[^A-Za-z0-9]+", "_", f"{i:02d}_{n}").strip("_")
        if only and not any(o in folder.name for o in only):
            continue
        title = f"{n}   [{g}]"
        print("experiment", folder.name, flush=True)
        per_experiment(A[n], title, folder)
    A_head = {k: v for k, v in A.items()}
    R2 = R.copy()
    combined(R2, keep)
    # top-level index
    lines = ["# Figures\n", "Independent figures per experiment (one folder each) and `combined/` overlays. "
             "See the docstring of `scripts/analysis/plot_all_experiments.py` for what each numbered figure shows.\n",
             "| Experiment folder | Group |", "|---|---|"]
    for i, (g, n, t) in enumerate(keep, 1):
        lines.append(f"| `{re.sub(r'[^A-Za-z0-9]+', '_', f'{i:02d}_{n}').strip('_')}/` | {g} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
