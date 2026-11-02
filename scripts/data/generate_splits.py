"""
Generate N reproducible, stratified train/val/test split CSVs.

Each split keeps the same per-class proportions (default 80/10/10) and each
split is distinct (different seed). Paths are taken from SlideDataset, so they
match exactly what train.py will look up at load time.

Output: splits/split_01.csv ... split_0N.csv
Each CSV has columns: path,split   (split in {train, val, test})

Run:
    python -m scripts.data.generate_splits
    python -m scripts.data.generate_splits --n-splits 5 --base-seed 42
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from glioma_sparse.data_utils.slide_dataset import SlideDataset


# ============================================================
# CONFIG (kept in sync with train.py)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "included"
SPLITS_DIR = REPO_ROOT / "splits"

CLASS_ORDER = ["control", "low_grade", "high_grade"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test ratio is the remainder (1 - TRAIN_RATIO - VAL_RATIO)

N_SPLITS = 5
BASE_SEED = 42


# ============================================================
# SPLIT LOGIC
# ============================================================

def stratified_split(paths, labels, train_ratio, val_ratio, seed):
    """Per-class shuffle then proportional cut. Robust to small classes."""
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[y].append(p)

    train, val, test = [], [], []
    for y in sorted(by_class):
        items = by_class[y][:]
        rng.shuffle(items)
        n = len(items)

        n_val = round(n * val_ratio)
        n_test = round(n * (1.0 - train_ratio - val_ratio))
        # keep val/test non-empty once a class has enough samples
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        n_train = n - n_val - n_test
        if n_train < 0:  # safety for tiny classes
            n_train, n_val, n_test = n, 0, 0

        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]

    return train, val, test


def save_split_csv(csv_path, train, val, test):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split"])
        for p in train:
            writer.writerow([str(p), "train"])
        for p in val:
            writer.writerow([str(p), "val"])
        for p in test:
            writer.writerow([str(p), "test"])


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SlideDataset(
        root_dir=DATA_DIR,
        transform=None,
        patch_transform=None,
        class_names=CLASS_ORDER,
    )
    paths = dataset.paths
    labels = dataset.labels
    assert len(paths) > 0, f"No data found in {DATA_DIR}"

    label_of = dict(zip(paths, labels))

    def name(y):
        return CLASS_ORDER[y]

    print(f"Data dir : {DATA_DIR}")
    print(f"Samples  : {len(paths)}")
    print(f"Per class: "
          + ", ".join(f"{name(y)}={c}" for y, c in sorted(Counter(labels).items())))
    print(f"Ratios   : train={args.train_ratio} val={args.val_ratio} "
          f"test={round(1 - args.train_ratio - args.val_ratio, 4)}")
    print()

    test_sets = []
    for k in range(args.n_splits):
        seed = args.base_seed + k
        train, val, test = stratified_split(
            paths, labels, args.train_ratio, args.val_ratio, seed
        )

        # integrity checks
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
        assert len(train) + len(val) + len(test) == len(paths)

        csv_path = SPLITS_DIR / f"split_{k + 1:02d}.csv"
        save_split_csv(csv_path, train, val, test)

        def counts(subset):
            c = Counter(label_of[p] for p in subset)
            return " ".join(f"{name(y)}={c.get(y, 0)}" for y in range(len(CLASS_ORDER)))

        print(f"[split {k + 1:02d}] seed={seed}  -> {csv_path.name}")
        print(f"           train {len(train):>4}  ({counts(train)})")
        print(f"           val   {len(val):>4}  ({counts(val)})")
        print(f"           test  {len(test):>4}  ({counts(test)})")

        test_sets.append(frozenset(test))

    # confirm the splits are distinct
    identical = sum(
        test_sets[i] == test_sets[j]
        for i in range(len(test_sets))
        for j in range(i + 1, len(test_sets))
    )
    print()
    print(f"Wrote {args.n_splits} splits to {SPLITS_DIR}")
    print(f"Identical test-set pairs: {identical} (should be 0)")


if __name__ == "__main__":
    main()
