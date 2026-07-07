"""
Stage B split resolution from the extraction manifest.

The manifest produced by build_stageB_cohort.py holds every region with a
`fold` and `split` column. For a given fold, this returns the train, val, and
test patch lists, guaranteeing that all patches from one slide share a split
(the split is defined at slide level, per fold). Reused for end-to-end
inference, where you take a fold's test patches and soft-vote per slide.
"""

import csv
from collections import defaultdict
from pathlib import Path


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_fold_split(manifest_path, fold):
    """Return dict with 'train'/'val'/'test' -> list of manifest rows for a fold."""
    rows = load_manifest(manifest_path)
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        if int(r["fold"]) == fold and r["split"] in out:
            out[r["split"]].append(r)
    return out


def patches_by_slide(rows):
    """Group patch rows by slide_id (for soft-voting at slide level)."""
    g = defaultdict(list)
    for r in rows:
        g[r["slide_id"]].append(r)
    return dict(g)


def check_integrity(manifest_path):
    """Assert no slide straddles two splits within a fold; report coverage."""
    rows = load_manifest(manifest_path)
    folds = sorted({int(r["fold"]) for r in rows})
    problems = []
    for fold in folds:
        slide_split = {}
        for r in rows:
            if int(r["fold"]) != fold:
                continue
            sid, sp = r["slide_id"], r["split"]
            if sid in slide_split and slide_split[sid] != sp:
                problems.append((fold, sid, slide_split[sid], sp))
            slide_split[sid] = sp
        n = {"train": 0, "val": 0, "test": 0}
        for sp in slide_split.values():
            n[sp] += 1
        print(f"fold {fold}: slides train={n['train']} val={n['val']} "
              f"test={n['test']} | regions={sum(1 for r in rows if int(r['fold'])==fold)}")
    if problems:
        print("\nINTEGRITY PROBLEMS (slide in >1 split within a fold):")
        for f, s, a, b in problems[:20]:
            print(f"  fold {f} slide {s}: {a} and {b}")
    else:
        print("\nOK: within each fold, every slide is in exactly one split.")
    return not problems


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--fold", type=int, default=None)
    args = p.parse_args()
    check_integrity(args.manifest)
    if args.fold is not None:
        sp = get_fold_split(args.manifest, args.fold)
        print(f"\nfold {args.fold}: "
              f"{len(sp['train'])} train / {len(sp['val'])} val / "
              f"{len(sp['test'])} test regions")
        for split in ("train", "val", "test"):
            print(f"  {split}: {len(patches_by_slide(sp[split]))} slides")
