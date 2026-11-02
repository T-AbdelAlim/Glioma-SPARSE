import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(probs, labels):

    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            # multi_class='ovr' expects >2 classes; binary needs a 1D score
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }