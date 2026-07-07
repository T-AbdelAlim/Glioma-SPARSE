"""Shared classification metrics for Stage A.

Used by both train.py (per run) and aggregate_splits.py (across runs) so the
definitions never drift. Everything is computed from y_true + y_prob, which is
exactly what gets saved in test_predictions.npz.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_prob, class_order):
    """Return a flat dict of metrics.

    y_true: (N,) int labels
    y_prob: (N, C) softmax probabilities
    class_order: list of class names, index == label
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)

    n_classes = len(class_order)
    labels = list(range(n_classes))

    out = {}
    out["n_samples"] = int(len(y_true))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels,
                                     average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, labels=labels,
                                        average="weighted", zero_division=0))

    # Quadratic weighted kappa: the right metric for ordinal grading, since it
    # penalises a control->high_grade error more than a control->low_grade one.
    out["quadratic_weighted_kappa"] = float(
        cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    )

    # Per-class recall (sensitivity) and precision (PPV).
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(class_order):
        out[f"recall_{name}"] = float(rec[i])
        out[f"precision_{name}"] = float(prec[i])
    out["macro_recall"] = float(np.mean(rec))
    out["macro_precision"] = float(np.mean(prec))

    # AUC (one-vs-rest). Guard against a class missing from y_true in a split.
    present = np.unique(y_true)
    try:
        if len(present) == n_classes:
            out["macro_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                              labels=labels)
            )
            per_class_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=None, labels=labels
            )
            for i, name in enumerate(class_order):
                out[f"auc_{name}"] = float(per_class_auc[i])
        else:
            out["macro_auc"] = float("nan")
            for name in class_order:
                out[f"auc_{name}"] = float("nan")
    except ValueError:
        out["macro_auc"] = float("nan")
        for name in class_order:
            out[f"auc_{name}"] = float("nan")

    return out


def confusion(y_true, y_prob, n_classes):
    """Raw confusion matrix (rows = true, cols = predicted)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_prob, dtype=float).argmax(axis=1)
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


# Metrics shown in the summary table / figure, in display order.
HEADLINE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auc",
    "quadratic_weighted_kappa",
]
