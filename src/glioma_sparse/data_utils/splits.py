import random
import csv
from pathlib import Path
from collections import defaultdict


def stratified_split(samples, labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    samples: list of file paths
    labels:  list of integer labels
    """

    random.seed(seed)

    class_to_items = defaultdict(list)

    for path, label in zip(samples, labels):
        class_to_items[label].append(path)

    train, val, test = [], [], []

    for label, items in class_to_items.items():
        random.shuffle(items)

        n = len(items)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


# ============================================================
# CSV SAVE / LOAD
# ============================================================

def save_split_csv(path, train, val, test):
    path = Path(path)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split"])

        for p in train:
            writer.writerow([str(p), "train"])
        for p in val:
            writer.writerow([str(p), "val"])
        for p in test:
            writer.writerow([str(p), "test"])


def load_split_csv(path):
    path = Path(path)

    train, val, test = [], [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            p = Path(row["path"])
            split = row["split"]

            if split == "train":
                train.append(p)
            elif split == "val":
                val.append(p)
            elif split == "test":
                test.append(p)

    return train, val, test