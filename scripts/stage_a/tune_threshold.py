"""Per-class decision-threshold tuning for Stage A.

Motivation: low_grade AUC is high (~0.92) while low_grade recall at the default
argmax is low (~0.44). The ranking is good; the operating point is the problem.
This script recovers low_grade recall by re-weighting the softmax scores before
the argmax, WITHOUT retraining and WITHOUT touching AUC.

Method: replace argmax(p) by argmax(w * p), where w is a per-class weight vector.
w is fitted on each fold's VALIDATION predictions, frozen, then applied ONCE to
that fold's TEST predictions. This keeps the test estimate honest, exactly as
early stopping and checkpoint selection already use validation only.

The weight search is 1D per non-anchor class over a grid; control is anchored to
1.0 (it is already perfect and needs no help). Default objective is macro-F1.
An alternative objective targets a minimum low_grade recall subject to a
low_grade precision floor.

Run (from repo root):
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --objective macro_f1
    python scripts/tune_threshold.py --objective low_grade_recall --min-precision 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lib.eval_metrics import compute_metrics, confusion, HEADLINE_METRICS

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# LOADING
# ------------------------------------------------------------

def find_run_dirs(base_dir):
    base = Path(base_dir)
    return sorted(
        d for d in base.iterdir()
        if d.is_dir()
        and (d / "val_predictions.npz").exists()
        and (d / "test_predictions.npz").exists()
    )


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["y_true"].astype(int), np.asarray(d["y_prob"], dtype=float), \
        [str(c) for c in d["class_order"]]


# ------------------------------------------------------------
# WEIGHTED DECISION
# ------------------------------------------------------------

def predict_weighted(y_prob, weights):
    """argmax over weighted probabilities."""
    return (y_prob * np.asarray(weights)[None, :]).argmax(axis=1)


def tuned_metrics(y_true, y_prob, weights, class_order):
    """Metrics after applying the weight vector.

    The weighted argmax changes the hard-decision metrics (accuracy, recall,
    precision, F1, kappa). AUC is rank-based and per-class scaling preserves the
    within-class ranking, so AUC is taken from the UNSCALED probabilities and
    stays identical to the default. This keeps the AUC columns populated and
    correct instead of leaving them blank.
    """
    weights = np.asarray(weights, dtype=float)
    scaled = y_prob * weights[None, :]
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    m = compute_metrics(y_true, scaled, class_order)

    # overwrite AUC entries with the unscaled (unchanged) values
    m_auc = compute_metrics(y_true, y_prob, class_order)
    for k in list(m.keys()):
        if k == "macro_auc" or k.startswith("auc_"):
            m[k] = m_auc[k]
    return m


def score_objective(y_true, y_pred, class_order, objective, target_idx,
                    min_precision):
    from sklearn.metrics import f1_score, balanced_accuracy_score, \
        recall_score, precision_score
    labels = list(range(len(class_order)))
    if objective == "macro_f1":
        return f1_score(y_true, y_pred, labels=labels, average="macro",
                        zero_division=0)
    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if objective == "low_grade_recall":
        rec = recall_score(y_true, y_pred, labels=[target_idx], average=None,
                           zero_division=0)[0]
        prec = precision_score(y_true, y_pred, labels=[target_idx], average=None,
                               zero_division=0)[0]
        # maximise recall, but reject solutions below the precision floor
        return rec if prec >= min_precision else -1.0 + rec * 0.001
    raise ValueError(objective)


def fit_weights(y_true, y_prob, class_order, objective, target_class,
                min_precision, grid):
    """Search a per-class weight vector on validation data.

    Control is anchored at 1.0. low_grade and high_grade weights are searched
    over `grid`. Returns the best weight vector.
    """
    target_idx = class_order.index(target_class)
    idx_control = class_order.index("control")

    best_w = np.ones(len(class_order))
    best_score = -np.inf

    # 2D search over low_grade and high_grade weights (control fixed at 1.0)
    search_idx = [i for i in range(len(class_order)) if i != idx_control]
    for w1 in grid:
        for w2 in grid:
            w = np.ones(len(class_order))
            w[search_idx[0]] = w1
            w[search_idx[1]] = w2
            y_pred = predict_weighted(y_prob, w)
            s = score_objective(y_true, y_pred, class_order, objective,
                                target_idx, min_precision)
            if s > best_score:
                best_score = s
                best_w = w.copy()
    return best_w, best_score


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str,
                   default=str(REPO_ROOT / "training_output"))
    p.add_argument("--runs", type=str, nargs="*", default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--objective", type=str, default="macro_f1",
                   choices=["macro_f1", "balanced_accuracy", "low_grade_recall"])
    p.add_argument("--target-class", type=str, default="low_grade")
    p.add_argument("--min-precision", type=float, default=0.5,
                   help="Precision floor for the low_grade_recall objective.")
    p.add_argument("--grid-max", type=float, default=3.0)
    p.add_argument("--grid-steps", type=int, default=61)
    p.add_argument("--pooled", action="store_true",
                   help="Fit ONE shared weight vector on all folds' validation "
                        "predictions pooled together, then apply that single "
                        "rule to every fold's test set. More stable than "
                        "per-fold fitting when validation sets are small.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = find_run_dirs(args.base_dir)
    if not run_dirs:
        raise SystemExit(f"No runs with val+test predictions under {args.base_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.base_dir) / "aggregate_tuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(1.0, args.grid_max, args.grid_steps)
    # also allow down-weighting, not only up-weighting
    grid = np.unique(np.concatenate([np.linspace(0.5, 1.0, 11), grid]))

    mode = "pooled" if args.pooled else "per-fold"
    print(f"Tuning on {len(run_dirs)} folds | objective = {args.objective} "
          f"| mode = {mode}")

    # load every fold once
    folds = []
    class_order_ref = None
    for d in run_dirs:
        yv, pv, class_order = load_npz(d / "val_predictions.npz")
        yt, pt, _ = load_npz(d / "test_predictions.npz")
        if class_order_ref is None:
            class_order_ref = class_order
        folds.append((d.name, yv, pv, yt, pt, class_order))

    # in pooled mode, fit a single weight vector on all validation data
    pooled_w = None
    if args.pooled:
        yv_all = np.concatenate([f[1] for f in folds])
        pv_all = np.concatenate([f[2] for f in folds])
        pooled_w, pooled_score = fit_weights(
            yv_all, pv_all, class_order_ref, args.objective,
            args.target_class, args.min_precision, grid)
        print(f"  pooled weight (fit on {len(yv_all)} val slides): "
              f"w={np.round(pooled_w, 2).tolist()} | "
              f"val score {pooled_score:.4f}")

    default_rows, tuned_rows, weight_rows = [], [], []

    for name, yv, pv, yt, pt, class_order in folds:
        if args.pooled:
            w, val_score = pooled_w, float("nan")
        else:
            # fit on this fold's validation, freeze, apply once to its test
            w, val_score = fit_weights(yv, pv, class_order, args.objective,
                                       args.target_class, args.min_precision, grid)

        m_default = compute_metrics(yt, pt, class_order)
        m_default["run"] = name
        default_rows.append(m_default)

        m_tuned = tuned_metrics(yt, pt, w, class_order)
        m_tuned["run"] = name
        tuned_rows.append(m_tuned)

        weight_rows.append({"run": name,
                            **{f"w_{c}": round(float(w[i]), 3)
                               for i, c in enumerate(class_order)},
                            "val_objective_score": round(float(val_score), 4)})

        print(f"  {name}: w={np.round(w, 2).tolist()} | "
              f"low_grade recall {m_default['recall_low_grade']:.2f} -> "
              f"{m_tuned['recall_low_grade']:.2f}")

    # ---- assemble tables ----
    ddf = pd.DataFrame(default_rows).set_index("run")
    tdf = pd.DataFrame(tuned_rows).set_index("run")
    wdf = pd.DataFrame(weight_rows).set_index("run")

    num = [c for c in ddf.columns if ddf[c].dtype != object]

    def summ(df):
        rows = {}
        for c in num:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            v = v[~np.isnan(v)]
            if len(v):
                m = v.mean(); s = v.std(ddof=1) if len(v) > 1 else 0.0
                rows[c] = (m, s)
        return rows

    sd, st = summ(ddf), summ(tdf)

    comp_rows = []
    for c in num:
        if c in sd and c in st:
            comp_rows.append({
                "metric": c,
                "default": f"{sd[c][0]:.3f} \u00b1 {sd[c][1]:.3f}",
                "tuned": f"{st[c][0]:.3f} \u00b1 {st[c][1]:.3f}",
                "delta": round(st[c][0] - sd[c][0], 3),
            })
    comp = pd.DataFrame(comp_rows).set_index("metric")

    tag = "pooled" if args.pooled else "perfold"
    ddf.to_csv(out_dir / f"tuned_default_per_split_{tag}.csv")
    tdf.to_csv(out_dir / f"tuned_tuned_per_split_{tag}.csv")
    wdf.to_csv(out_dir / f"tuned_weights_per_split_{tag}.csv")
    comp.to_csv(out_dir / f"tuned_vs_default_summary_{tag}.csv")

    # ---- console summary ----
    print("\n=== DEFAULT vs TUNED (mean \u00b1 std across folds) ===")
    show = HEADLINE_METRICS + ["recall_low_grade", "precision_low_grade",
                               "recall_high_grade", "macro_recall"]
    for m in show:
        if m in comp.index:
            print(f"  {m:26s} {comp.loc[m,'default']:>18s}  ->  "
                  f"{comp.loc[m,'tuned']:>18s}   (d {comp.loc[m,'delta']:+.3f})")

    print("\nNote: macro_auc and per-class AUC are unchanged by design "
          "(ranking is preserved), and are carried over from the default.")
    print(f"\nMode: {mode}. Wrote outputs (tag '{tag}') to {out_dir}")


if __name__ == "__main__":
    main()
