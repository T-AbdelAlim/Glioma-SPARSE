r"""
Tune the Stage B selection and aggregation knobs, honestly.

The question: how many tiles should reach Stage B (percentile / cap), and how
should their predictions be combined into a slide call (aggregation rule, class
weights)?

WHY THIS IS CHEAP
-----------------
inference_report.py caches every candidate tile's Stage B softmax together with
its Stage A risk. A (percentile, cap) setting is then just a SUBSET of that
cache, and an aggregation rule is a function of the subset. So the entire grid
replays with no GPU and no WSI reads. Build the cache once at a low percentile
and a high cap; sweep for free thereafter.

WHY THE DESIGN MATTERS MORE THAN THE SWEEP
------------------------------------------
There are ~38 tumour slides per fold. Sweeping ~100 configurations on the test
set and reporting the best one will find several accuracy points of pure noise:
the expected maximum over K configurations grows with K even when every
configuration is identical in truth. That is the same order as the entire
RN18-vs-RN50 difference, so an unguarded sweep would manufacture a result.

This harness therefore does VALIDATION-SELECTED, TEST-REPORTED tuning:

    for each fold:
        pick the configuration that maximises the objective on that fold's VAL
        apply it, once, to that fold's TEST
    report the mean +/- SD of those five test numbers

The test set is touched once per fold, with a configuration chosen without
seeing it. The full val grid is written out so you can inspect the surface, and
a test grid is written ONLY as a diagnostic, clearly marked as not reportable.

It also reports the honest baseline: the current setting (p95, cap 4, mean),
evaluated the same way. If validation-selected tuning does not beat that, the
answer is that the knobs do not matter, and that is a finding worth reporting.

CAVEAT ON DISTRIBUTION SHIFT
----------------------------
Stage B was TRAINED on p95 / cap 4 / min_tissue 0.5 regions (build_stageB_cohort.py).
Evaluating it on p50 tiles asks it to classify regions unlike its training
distribution. Gains or losses at low percentiles therefore confound "more
evidence" with "out-of-distribution input". The script flags this; the clean
version of the experiment retrains Stage B at the chosen percentile.

USAGE
-----
1. Build caches for the test+val slides of every fold:

       python -m scripts.inference.inference_report \
           --input <fold k slides> --run-name cache_fold1 --model resnet50 \
           --stage-a-ckpt .../best_f1.pth --stage-b-ckpt .../best_f1.pth \
           --cache-percentile 50 --cache-cap 32

2. Sweep:

       python -m scripts.analysis.tune_selection \
           --cache-root results/ --splits-dir splits \
           --out-dir results/tuning_rn50 --objective balanced_accuracy

OUTPUTS
-------
    tuning_report.xlsx
        val_grid           mean val score per configuration, all folds
        selected           the configuration chosen per fold, and its test score
        headline           validation-selected test vs the p95/cap4/mean baseline
        test_grid          DIAGNOSTIC ONLY, marked; do not report
        class_weights      the weight sweep, if enabled
        sensitivity        how stable the winner is across folds
    tuning_surface.png     val objective vs cap, one line per aggregation rule
"""

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
MUTANT_SET = {"IDH_mt", "IDH_mt_1p19q"}


# ============================================================
# AGGREGATION  (kept in sync with inference_report.py)
# ============================================================

def aggregate(tile_probs, risks, rule="mean", **kw):
    P = np.asarray(tile_probs, dtype=float)
    if len(P) == 0:
        return None
    R = np.asarray(risks, dtype=float)

    if rule == "mean":
        out = P.mean(axis=0)
    elif rule == "max":
        out = P.max(axis=0); out = out / out.sum()
    elif rule == "topq_mean":
        q = kw.get("q", 0.25)
        k = max(1, int(np.ceil(q * len(P))))
        out = np.array([np.sort(P[:, c])[::-1][:k].mean() for c in range(P.shape[1])])
        out = out / out.sum()
    elif rule == "risk_weighted":
        w = R - R.min()
        w = w / (w.sum() + 1e-12) if w.sum() > 0 else np.ones(len(P)) / len(P)
        out = (P * w[:, None]).sum(axis=0)
    elif rule == "logodds_mean":
        eps = 1e-6
        L = np.log(np.clip(P, eps, 1 - eps) / np.clip(1 - P, eps, 1 - eps))
        out = 1.0 / (1.0 + np.exp(-L.mean(axis=0))); out = out / out.sum()
    elif rule == "noisy_or":
        out = 1.0 - np.prod(1.0 - P, axis=0); out = out / out.sum()
    elif rule == "trimmed_mean":
        if len(P) <= 2:
            out = P.mean(axis=0)
        else:
            out = np.array([np.sort(P[:, c])[1:-1].mean() for c in range(P.shape[1])])
            out = out / out.sum()
    else:
        raise ValueError(f"unknown rule {rule}")
    return out


def apply_class_weights(probs, weights):
    p = np.asarray(probs, float) * np.asarray(weights, float)
    return p / p.sum()


# ============================================================
# CACHE LOADING
# ============================================================

def load_caches(cache_root):
    """Every tiles.npz under cache_root, keyed by slide id."""
    out = {}
    for npz in Path(cache_root).rglob("cache/tiles.npz"):
        d = np.load(npz, allow_pickle=True)
        sid = str(d["slide_id"])
        out[sid] = {
            "risk": d["risk"], "tile_probs": d["tile_probs"],
            "row": d["row"], "col": d["col"], "tissue": d["tissue"],
            "risk_map": d["risk_map"], "stage_a_probs": d["stage_a_probs"],
        }
    return out


def subset_for(cache, percentile, cap):
    R = cache["risk"]
    if len(R) == 0:
        return np.array([], dtype=int)
    thresh = np.percentile(cache["risk_map"].flatten(), percentile)
    idx = np.where(R >= thresh)[0]
    return idx[np.argsort(-R[idx])][:cap]


def load_labels(cache_root, splits_dir, sidecar_root=None):
    """slide_id -> {fold, split, mutation}.

    Preferred source: a labels.csv written by inference_report.py (one
    authoritative file, derived the same way as build_stageB_cohort.py).
    Fallback: resolve each cached slide against the ebrains sidecar tree,
    whose <subtype>/included/<slide>.jpg layout carries the molecular class
    two levels up. The split CSV only holds the train/val/test role, never
    the mutation, so it cannot be the label source.
    """
    labels = {}

    # 1) labels.csv, if present anywhere under the cache root
    for lab_csv in Path(cache_root).rglob("labels.csv"):
        with open(lab_csv, newline="") as f:
            for row in csv.DictReader(f):
                mut = row.get("true_mutation", "")
                if mut in ("", "unknown"):
                    continue
                labels[(int(row["fold"]), row["slide_id"])] = {
                    "fold": int(row["fold"]), "split": row["split"],
                    "mutation": mut}
        if labels:
            return labels

    # 2) fallback: sidecar tree + split CSVs for the role
    if not sidecar_root:
        return labels
    role = {}          # slide_id -> {fold -> split}
    for split_csv in sorted(Path(splits_dir).glob("split_0*.csv")):
        fold = int(split_csv.stem.split("_")[-1])
        with open(split_csv, newline="") as f:
            for row in csv.DictReader(f):
                sid = Path(str(row["path"]).replace("\\", "/")).stem
                role.setdefault(sid, {})[fold] = row["split"]

    index = {}         # slide stem -> subtype folder (two levels up)
    for jpg in Path(sidecar_root).rglob("*.jpg"):
        if jpg.with_suffix(".json").exists():
            index[jpg.stem] = jpg.parent.parent.name

    def mut_of(folder):
        n = folder.lower()
        if "gbm" in n or "idhwt" in n:
            return "IDH_wt"
        if "oligo" in n and "1p19q" in n:
            return "IDH_mt_1p19q"
        if "astro" in n and "idhmt" in n:
            return "IDH_mt"
        return None

    for sid, folds in role.items():
        folder = index.get(sid)
        if folder is None:
            continue
        mut = mut_of(folder)
        if mut is None:
            continue
        for fold, split in folds.items():
            labels[(fold, sid)] = {"fold": fold, "split": split, "mutation": mut}
    return labels


# ============================================================
# SCORING
# ============================================================

def score(y_true, y_pred, objective="balanced_accuracy"):
    """y_true / y_pred are lists of class names."""
    if not y_true:
        return float("nan")
    yt = np.array([SUBTYPE_CLASSES.index(v) for v in y_true])
    yp = np.array([SUBTYPE_CLASSES.index(v) for v in y_pred])
    if objective == "accuracy":
        return float(np.mean(yt == yp))
    if objective == "balanced_accuracy":
        recs = [np.mean(yp[yt == c] == c) for c in range(3) if (yt == c).sum()]
        return float(np.mean(recs)) if recs else float("nan")
    if objective == "macro_f1":
        f1s = []
        for c in range(3):
            tp = np.sum((yp == c) & (yt == c))
            fp = np.sum((yp == c) & (yt != c))
            fn = np.sum((yp != c) & (yt == c))
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        return float(np.mean(f1s))
    if objective == "mutant_accuracy":
        tm = np.array([v in MUTANT_SET for v in y_true])
        pm = np.array([v in MUTANT_SET for v in y_pred])
        return float(np.mean(tm == pm))
    raise ValueError(objective)


def evaluate(caches, labels, fold, split, percentile, cap, rule,
             class_weights=None, objective="balanced_accuracy", q=0.25):
    """Score one configuration on one fold/split. Returns (score, n)."""
    y_true, y_pred = [], []
    for (f, sid), meta in labels.items():
        if f != fold or meta["split"] != split:
            continue
        cache = caches.get(sid)
        if cache is None or len(cache["tile_probs"]) == 0:
            continue
        idx = subset_for(cache, percentile, cap)
        if len(idx) == 0:
            continue
        probs = aggregate(cache["tile_probs"][idx], cache["risk"][idx],
                          rule=rule, q=q)
        if probs is None:
            continue
        if class_weights is not None:
            probs = apply_class_weights(probs, class_weights)
        y_true.append(meta["mutation"])
        y_pred.append(SUBTYPE_CLASSES[int(np.argmax(probs))])
    return score(y_true, y_pred, objective), len(y_true)


# ============================================================
# SWEEP
# ============================================================

def build_grid(percentiles, caps, rules, qs):
    grid = []
    for p, c, r in itertools.product(percentiles, caps, rules):
        if r == "topq_mean":
            for q in qs:
                grid.append({"percentile": p, "cap": c, "rule": r, "q": q})
        else:
            grid.append({"percentile": p, "cap": c, "rule": r, "q": 0.25})
    return grid


def config_label(cfg):
    base = f"p{cfg['percentile']:g}/cap{cfg['cap']}/{cfg['rule']}"
    return base + (f"(q={cfg['q']:g})" if cfg["rule"] == "topq_mean" else "")


def sweep(caches, labels, folds, grid, objective):
    """val and test score for every configuration on every fold."""
    val = defaultdict(dict)
    test = defaultdict(dict)
    for cfg in grid:
        lab = config_label(cfg)
        for fold in folds:
            sv, nv = evaluate(caches, labels, fold, "val", cfg["percentile"],
                              cfg["cap"], cfg["rule"], objective=objective,
                              q=cfg["q"])
            st, nt = evaluate(caches, labels, fold, "test", cfg["percentile"],
                              cfg["cap"], cfg["rule"], objective=objective,
                              q=cfg["q"])
            val[lab][fold] = (sv, nv)
            test[lab][fold] = (st, nt)
    return val, test


def select_and_report(val, test, folds, grid):
    """Per fold: argmax on val, read off test. This is the reportable number."""
    labels = [config_label(c) for c in grid]
    chosen, test_scores = {}, []
    for fold in folds:
        best, best_s = None, -np.inf
        for lab in labels:
            s, n = val[lab].get(fold, (np.nan, 0))
            if not np.isnan(s) and s > best_s:
                best, best_s = lab, s
        chosen[fold] = (best, best_s)
        ts, tn = test[best][fold] if best else (np.nan, 0)
        test_scores.append(ts)
    return chosen, np.array(test_scores, dtype=float)


def class_weight_sweep(caches, labels, folds, base_cfg, objective, steps):
    """Sweep a per-class multiplier, validation-selected. Tests the 'focal
    IDH_wt evidence should count more' hypothesis explicitly."""
    out = []
    for w_wt in steps:
        for w_cod in steps:
            weights = (1.0, w_cod, w_wt)
            vals, tests = [], []
            for fold in folds:
                sv, _ = evaluate(caches, labels, fold, "val",
                                 base_cfg["percentile"], base_cfg["cap"],
                                 base_cfg["rule"], class_weights=weights,
                                 objective=objective, q=base_cfg["q"])
                st, _ = evaluate(caches, labels, fold, "test",
                                 base_cfg["percentile"], base_cfg["cap"],
                                 base_cfg["rule"], class_weights=weights,
                                 objective=objective, q=base_cfg["q"])
                vals.append(sv); tests.append(st)
            out.append({"w_IDH_mt": 1.0, "w_IDH_mt_1p19q": w_cod, "w_IDH_wt": w_wt,
                        "val_mean": float(np.nanmean(vals)),
                        "test_mean": float(np.nanmean(tests)),
                        "test_sd": float(np.nanstd(tests, ddof=1))})
    return out


# ============================================================
# REPORT
# ============================================================

HEAD = "305496"
GREEN, AMBER, RED = "C6EFCE", "FFEB9C", "FFC7CE"


def _fill(c):
    from openpyxl.styles import PatternFill
    return PatternFill("solid", fgColor=c)


def _head(ws, n):
    from openpyxl.styles import Font, Alignment
    for j in range(1, n + 1):
        c = ws.cell(row=1, column=j)
        c.font = Font(bold=True, color="FFFFFF"); c.fill = _fill(HEAD)
        c.alignment = Alignment(horizontal="center")


def _autosize(ws, mx=44):
    for col in ws.columns:
        w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(w + 2, mx)


def write_report(path, val, test, folds, grid, chosen, sel_scores,
                 baseline_scores, objective, cw_rows, notes):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()

    # ---- headline ----
    ws = wb.active; ws.title = "headline"
    ws.append(["Stage B selection and aggregation tuning"])
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append([])
    ws.append(["objective", objective])
    ws.append([])
    ws.append(["approach", "test score (mean ± SD over folds)", "note"])
    _head(ws, 3)  # styles row 1; re-style the real header below
    for j in range(1, 4):
        c = ws.cell(row=5, column=j)
        c.font = Font(bold=True, color="FFFFFF"); c.fill = _fill(HEAD)
    ws.append(["Current setting (p95 / cap 4 / mean)",
               f"{np.nanmean(baseline_scores):.4f} ± {np.nanstd(baseline_scores, ddof=1):.4f}",
               "the pipeline as it stands"])
    ws.append(["Validation-selected tuning",
               f"{np.nanmean(sel_scores):.4f} ± {np.nanstd(sel_scores, ddof=1):.4f}",
               "config chosen on val per fold, applied once to test"])
    diff = np.nanmean(sel_scores) - np.nanmean(baseline_scores)
    ws.append(["Difference", f"{diff:+.4f}",
               "if this is small relative to the fold SD, the knobs do not matter"])
    r = ws.max_row
    ws.cell(row=r, column=2).fill = _fill(GREEN if diff > 0.02 else
                                          AMBER if diff > -0.02 else RED)
    ws.append([])
    ws.append(["Per-fold test scores under validation-selected tuning"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.append(["fold", "chosen config", "val score", "test score"])
    for j in range(1, 5):
        c = ws.cell(row=ws.max_row, column=j)
        c.font = Font(bold=True, color="FFFFFF"); c.fill = _fill(HEAD)
    for i, fold in enumerate(folds):
        lab, vs = chosen[fold]
        ws.append([fold, lab, round(vs, 4) if vs == vs else "",
                   round(sel_scores[i], 4) if sel_scores[i] == sel_scores[i] else ""])
    ws.append([])
    for n in notes:
        ws.append([n])
        ws.cell(row=ws.max_row, column=1).font = Font(italic=True, color="8B4000")
    _autosize(ws, 70)

    # ---- val grid ----
    vs_ = wb.create_sheet("val_grid")
    hdr = ["config"] + [f"fold{f}" for f in folds] + ["mean", "sd"]
    vs_.append(hdr); _head(vs_, len(hdr))
    rows = []
    for cfg in grid:
        lab = config_label(cfg)
        vals = [val[lab].get(f, (np.nan, 0))[0] for f in folds]
        rows.append((float(np.nanmean(vals)), lab, vals))
    for m, lab, vals in sorted(rows, key=lambda t: -t[0]):
        vs_.append([lab] + [round(v, 4) if v == v else "" for v in vals] +
                   [round(m, 4), round(float(np.nanstd(vals, ddof=1)), 4)])
    for r in range(2, vs_.max_row + 1):
        v = vs_.cell(row=r, column=len(folds) + 2).value
        if isinstance(v, float):
            vs_.cell(row=r, column=len(folds) + 2).fill = _fill(
                GREEN if v >= 0.7 else AMBER if v >= 0.55 else RED)
    vs_.freeze_panes = "A2"; _autosize(vs_)

    # ---- test grid, diagnostic only ----
    ts = wb.create_sheet("test_grid")
    ts.append(["DIAGNOSTIC ONLY - selecting a row from this sheet and reporting "
               "it would be tuning on the test set. Use the headline sheet."])
    ts.cell(row=1, column=1).font = Font(bold=True, color="C00000")
    ts.append(hdr)
    for j in range(1, len(hdr) + 1):
        c = ts.cell(row=2, column=j)
        c.font = Font(bold=True, color="FFFFFF"); c.fill = _fill(HEAD)
    trows = []
    for cfg in grid:
        lab = config_label(cfg)
        vals = [test[lab].get(f, (np.nan, 0))[0] for f in folds]
        trows.append((float(np.nanmean(vals)), lab, vals))
    for m, lab, vals in sorted(trows, key=lambda t: -t[0]):
        ts.append([lab] + [round(v, 4) if v == v else "" for v in vals] +
                  [round(m, 4), round(float(np.nanstd(vals, ddof=1)), 4)])
    ts.freeze_panes = "A3"; _autosize(ts)

    # ---- sensitivity: how often does each config win on val? ----
    ss = wb.create_sheet("sensitivity")
    ss.append(["config", "folds won on val", "interpretation"])
    _head(ss, 3)
    wins = defaultdict(int)
    for fold in folds:
        lab, _ = chosen[fold]
        wins[lab] += 1
    for lab, n in sorted(wins.items(), key=lambda kv: -kv[1]):
        note = ("stable winner" if n >= 4 else
                "fold-dependent" if n >= 2 else "won once; likely noise")
        ss.append([lab, n, note])
    ss.append([])
    ss.append(["If no configuration wins on 4+ folds, the surface is flat and "
               "the tuning gain is not real."])
    _autosize(ss, 60)

    # ---- class weights ----
    if cw_rows:
        cw = wb.create_sheet("class_weights")
        hdr2 = ["w_IDH_mt", "w_IDH_mt_1p19q", "w_IDH_wt", "val_mean",
                "test_mean", "test_sd"]
        cw.append(hdr2); _head(cw, len(hdr2))
        for r in sorted(cw_rows, key=lambda d: -d["val_mean"]):
            cw.append([r["w_IDH_mt"], r["w_IDH_mt_1p19q"], r["w_IDH_wt"],
                       round(r["val_mean"], 4), round(r["test_mean"], 4),
                       round(r["test_sd"], 4)])
        cw.append([])
        cw.append(["Weights are selected on val_mean. Reading test_mean off the "
                   "best val row is legitimate; reading the best test_mean is not."])
        cw.freeze_panes = "A2"; _autosize(cw)

    wb.save(path)


def plot_surface(val, folds, grid, out_path, objective):
    by_rule = defaultdict(lambda: defaultdict(list))
    for cfg in grid:
        lab = config_label(cfg)
        m = float(np.nanmean([val[lab].get(f, (np.nan, 0))[0] for f in folds]))
        by_rule[cfg["rule"]][cfg["cap"]].append(m)
    fig, ax = plt.subplots(figsize=(8, 5))
    for rule, caps in sorted(by_rule.items()):
        xs = sorted(caps)
        ys = [float(np.nanmean(caps[c])) for c in xs]
        ax.plot(xs, ys, marker="o", label=rule)
    ax.set_xlabel("cap (max tiles to Stage B)")
    ax.set_ylabel(f"validation {objective}")
    ax.set_title("Validation surface: aggregation rule vs number of tiles")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True,
                   help="root containing */cache/tiles.npz from inference_report.py")
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--sidecar-root", default="",
                   help="ebrains thumbnail tree; used only if no labels.csv is "
                        "found under --cache-root")
    p.add_argument("--out-dir", default="results/tuning")
    p.add_argument("--objective", default="balanced_accuracy",
                   choices=["accuracy", "balanced_accuracy", "macro_f1",
                            "mutant_accuracy"])
    p.add_argument("--percentiles", type=float, nargs="*",
                   default=[50, 75, 90, 95, 98])
    p.add_argument("--caps", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--rules", nargs="*",
                   default=["mean", "max", "topq_mean", "risk_weighted",
                            "logodds_mean", "noisy_or", "trimmed_mean"])
    p.add_argument("--qs", type=float, nargs="*", default=[0.1, 0.25, 0.5])
    p.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    p.add_argument("--class-weights", action="store_true",
                   help="also sweep per-class multipliers on the best base config")
    p.add_argument("--weight-steps", type=float, nargs="*",
                   default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading caches...")
    caches = load_caches(args.cache_root)
    print(f"  {len(caches)} slide caches")
    labels = load_labels(args.cache_root, args.splits_dir,
                         sidecar_root=args.sidecar_root or None)
    print(f"  {len(labels)} labelled (fold, slide) entries")
    if not caches:
        raise SystemExit("no caches found; run inference_report.py first")
    if not labels:
        raise SystemExit(
            "no molecular labels resolved. Either re-run inference_report.py "
            "(it now writes labels.csv), or pass --sidecar-root pointing at the "
            "<subtype>/included/ ebrains tree so mutations can be derived.")

    n_tiles = [len(c["tile_probs"]) for c in caches.values() if len(c["tile_probs"])]
    print(f"  cached tiles per slide: median {np.median(n_tiles):.0f}, "
          f"range {min(n_tiles)}-{max(n_tiles)}")

    grid = build_grid(args.percentiles, args.caps, args.rules, args.qs)
    print(f"Sweeping {len(grid)} configurations x {len(args.folds)} folds "
          f"(no GPU; replaying caches)...")
    val, test = sweep(caches, labels, args.folds, grid, args.objective)

    chosen, sel_scores = select_and_report(val, test, args.folds, grid)

    # baseline: the current pipeline setting
    baseline = []
    for fold in args.folds:
        s, _ = evaluate(caches, labels, fold, "test", 95.0, 4, "mean",
                        objective=args.objective)
        baseline.append(s)
    baseline = np.array(baseline, dtype=float)

    cw_rows = []
    if args.class_weights:
        best_lab = max(
            (config_label(c) for c in grid),
            key=lambda l: float(np.nanmean([val[l].get(f, (np.nan, 0))[0]
                                            for f in args.folds])))
        base_cfg = next(c for c in grid if config_label(c) == best_lab)
        print(f"Class-weight sweep on {best_lab}...")
        cw_rows = class_weight_sweep(caches, labels, args.folds, base_cfg,
                                     args.objective, args.weight_steps)

    notes = [
        "Note: Stage B was trained on p95 / cap 4 / min_tissue 0.5 regions. "
        "Low-percentile settings feed it out-of-distribution tiles, so a gain "
        "or loss there confounds 'more evidence' with 'unfamiliar input'. The "
        "clean experiment retrains Stage B at the chosen percentile.",
        "Note: with ~38 tumour slides per fold, sweeping this grid on the test "
        "set and reporting the winner inflates the score by roughly +0.18 even "
        "when every configuration is identical in truth (simulated, 126 configs, "
        "n=38). That is larger than the whole Glioma-SPARSE/CLAM gap. Differences "
        "below roughly 0.05 are within sampling noise; check the sensitivity "
        "sheet before believing any winner.",
        "Note: the five splits are independent stratifications, not a partition, "
        "so test sets overlap between folds. Fold scores are not independent.",
    ]

    report = out_dir / "tuning_report.xlsx"
    write_report(report, val, test, args.folds, grid, chosen, sel_scores,
                 baseline, args.objective, cw_rows, notes)
    plot_surface(val, args.folds, grid, out_dir / "tuning_surface.png",
                 args.objective)

    print("\n=== HEADLINE ===")
    print(f"  current (p95/cap4/mean):  {np.nanmean(baseline):.4f} "
          f"± {np.nanstd(baseline, ddof=1):.4f}")
    print(f"  validation-selected:      {np.nanmean(sel_scores):.4f} "
          f"± {np.nanstd(sel_scores, ddof=1):.4f}")
    print(f"  difference:               {np.nanmean(sel_scores) - np.nanmean(baseline):+.4f}")
    print("\n  chosen per fold:")
    for fold in args.folds:
        print(f"    fold {fold}: {chosen[fold][0]}")
    print(f"\nReport: {report}")
    print(f"Surface: {out_dir / 'tuning_surface.png'}")


if __name__ == "__main__":
    main()
