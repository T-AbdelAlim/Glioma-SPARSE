import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(labels, preds, class_names, out_path):

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(labels, probs, class_names, out_path):

    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    plt.figure()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label="{} (AUC={:.2f})".format(class_names[i], roc_auc))

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(out_path)
    plt.close()