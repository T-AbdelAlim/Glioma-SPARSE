import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_curves(history, output_dir):

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # -------------------------
    # LOSS
    # -------------------------
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    # -------------------------
    # AUC
    # -------------------------
    plt.figure()
    plt.plot(epochs, history["auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(output_dir / "auc_curve.png")
    plt.close()

    # -------------------------
    # F1
    # -------------------------
    plt.figure()
    plt.plot(epochs, history["f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig(output_dir / "f1_curve.png")
    plt.close()

    # -------------------------
    # ACCURACY
    # -------------------------
    plt.figure()
    plt.plot(epochs, history["acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, preds, class_names, out_path, normalize=False):

    labels = np.array(labels)
    preds = np.array(preds)

    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(len(class_names)))
    )

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()

    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    plt.title(title)

    plt.savefig(out_path)
    plt.close()

# ============================================================
# ROC CURVES
# ============================================================

def plot_roc_curve(labels, probs, class_names, out_path):

    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    plt.figure()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(out_path)
    plt.close()