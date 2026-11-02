"""Aggregate results across the 5 split runs into publication-ready outputs.

Reads each run's <split>_predictions.npz (default: test), recomputes metrics
with eval_metrics.compute_metrics so definitions match train.py exactly, pulls
efficiency.json for the compute-cost numbers, then writes:

    aggregate_metrics_per_split.csv   one row per split, all metrics
    aggregate_summary.csv             mean, std, and "mean ± std" per metric
    aggregate_efficiency.csv          compute-cost numbers, mean ± std
    aggregate_figure.png / .pdf       4-panel overview figure

Run (from repo root):
    python -m scripts.data.aggregate_splits
    python -m scripts.data.aggregate_splits --base-dir training_output --split test
    python -m scripts.data.aggregate_splits --runs training_output/run1 training_output/run2 ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lib.eval_metrics import compute_metrics, confusion, HEADLINE_METRICS


# ============================================================
# DISCOVERY / LOADING
# ============================================================

def find_run_dirs(base_dir, split):
    """Any subfolder that has a <split>_predictions.npz, sorted by name."""
    base = Path(base_dir)
    runs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / f"{split}_predictions.npz").exists()
    )
    return runs


def load_run(run_dir, split):
    npz = np.load(run_dir / f"{split}_predictions.npz", allow_pickle=True)
    y_true = npz["y_true"]
    y_prob = npz["y_prob"]
    class_order = [str(c) for c in npz["class_order"]]

    eff = {}
    eff_path = run_dir / "efficiency.json"
    if eff_path.exists():
        with open(eff_path) as f:
            eff = json.load(f)

    return y_true, y_prob, class_order, eff


def extract_efficiency(eff, split):
    """Flatten the parts of efficiency.json used in the summary."""
    inf = eff.get(f"inference_{split}", {}) or {}
    train = eff.get("train", {}) or {}
    params = eff.get("model_params", {}) or {}
    flops = eff.get("model_flops", {}) or {}
    return {
        "params_millions": params.get("params_millions"),
        "gflops_per_image": flops.get("gflops_per_image"),
        "latency_ms_per_image": inf.get("latency_ms_per_image"),
        "throughput_img_per_sec": inf.get("throughput_img_per_sec"),
        "energy_mj_per_image": inf.get("energy_mj_per_image"),
        "peak_gpu_mem_mb": inf.get("peak_gpu_mem_mb"),
        "train_time_sec": train.get("elapsed_sec"),
        "train_energy_wh": train.get("energy_wh"),
    }


# ============================================================
# AGGREGATION
# ============================================================

def fmt(mean, std, prec=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    return f"{mean:.{prec}f} \u00b1 {std:.{prec}f}"


def summarize(df):
    """Mean/std for every numeric column, ignoring NaNs."""
    rows = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            rows.append({"metric": col, "mean": np.nan, "std": np.nan,
                         "n": 0, "formatted": "n/a"})
        else:
            m, s = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            prec = 4 if abs(m) < 1 else 2
            rows.append({"metric": col, "mean": m, "std": s, "n": int(len(vals)),
                         "formatted": fmt(m, s, prec)})
    return pd.DataFrame(rows).set_index("metric")


# ============================================================
# FIGURE
# ============================================================

def make_figure(metrics_df, per_class_recall, mean_cm_norm, class_order,
                eff_summary, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Stage A: cross-split performance and compute cost "
                 f"(n = {len(metrics_df)} splits)", fontsize=14, y=0.98)

    # --- Panel A: headline metrics, mean +/- std ---
    ax = axes[0, 0]
    names = [m for m in HEADLINE_METRICS if m in metrics_df.columns]
    means = [metrics_df[m].mean() for m in names]
    stds = [metrics_df[m].std(ddof=1) if len(metrics_df) > 1 else 0 for m in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.9)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + (s if not np.isnan(s) else 0) + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Overall metrics", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: per-class recall, mean +/- std ---
    ax = axes[0, 1]
    rmeans = [np.nanmean(per_class_recall[c]) for c in class_order]
    rstds = [np.nanstd(per_class_recall[c], ddof=1)
             if len(per_class_recall[c]) > 1 else 0 for c in class_order]
    x = np.arange(len(class_order))
    ax.bar(x, rmeans, yerr=rstds, capsize=4, color="#DD8452", alpha=0.9)
    for xi, m, s in zip(x, rmeans, rstds):
        ax.text(xi, m + (s if not np.isnan(s) else 0) + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("recall (sensitivity)")
    ax.set_title("Per-class recall", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel C: mean normalized confusion matrix ---
    ax = axes[1, 0]
    im = ax.imshow(mean_cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order, fontsize=9)
    ax.set_yticklabels(class_order, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Mean confusion matrix (row-normalized)", fontsize=11)
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            v = mean_cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel D: efficiency table ---
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Compute cost (mean \u00b1 std)", fontsize=11)
    rows = [
        ("Parameters (M)", "params_millions"),
        ("GFLOPs / slide", "gflops_per_image"),
        ("Latency (ms / slide)", "latency_ms_per_image"),
        ("Throughput (slides / s)", "throughput_img_per_sec"),
        ("Energy (mJ / slide)", "energy_mj_per_image"),
        ("Peak GPU mem (MB)", "peak_gpu_mem_mb"),
        ("Train time (s)", "train_time_sec"),
        ("Train energy (Wh)", "train_energy_wh"),
    ]
    table_rows = []
    for label, key in rows:
        if key in eff_summary.index and eff_summary.loc[key, "formatted"] != "n/a":
            table_rows.append([label, eff_summary.loc[key, "formatted"]])
    if table_rows:
        tbl = ax.table(cellText=table_rows, colLabels=["metric", "value"],
                       loc="center", cellLoc="left", colLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
    else:
        ax.text(0.5, 0.5, "No efficiency logs found", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default="training_output")
    p.add_argument("--runs", type=str, nargs="*", default=None,
                   help="Explicit run dirs. Overrides --base-dir discovery.")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write outputs (default: <base-dir>/aggregate).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = find_run_dirs(args.base_dir, args.split)

    if not run_dirs:
        raise SystemExit(
            f"No runs with {args.split}_predictions.npz found under "
            f"{args.base_dir}. Run training first, or pass --runs explicitly."
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.base_dir) / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(run_dirs)} runs (split = {args.split}):")
    for d in run_dirs:
        print("  ", d.name)

    per_split_metrics = []
    per_split_eff = []
    per_class_recall = None
    cms_norm = []
    class_order_ref = None

    for d in run_dirs:
        y_true, y_prob, class_order, eff = load_run(d, args.split)
        if class_order_ref is None:
            class_order_ref = class_order
            per_class_recall = {c: [] for c in class_order}
        assert class_order == class_order_ref, \
            f"class order mismatch in {d.name}"

        m = compute_metrics(y_true, y_prob, class_order)
        m["run"] = d.name
        per_split_metrics.append(m)

        for c in class_order:
            per_class_recall[c].append(m[f"recall_{c}"])

        # row-normalized confusion matrix (per-class recall structure)
        cm = confusion(y_true, y_prob, len(class_order)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cms_norm.append(cm_norm)

        per_split_eff.append(extract_efficiency(eff, args.split))

    # ---- per-split metrics table ----
    mdf = pd.DataFrame(per_split_metrics).set_index("run")
    numeric_cols = [c for c in mdf.columns if mdf[c].dtype != object]
    mdf_num = mdf[numeric_cols]
    mdf.to_csv(out_dir / "aggregate_metrics_per_split.csv")

    # ---- metric summary (mean +/- std) ----
    summary = summarize(mdf_num)
    summary.to_csv(out_dir / "aggregate_summary.csv")

    # ---- efficiency summary ----
    edf = pd.DataFrame(per_split_eff)
    edf.insert(0, "run", [d.name for d in run_dirs])
    edf_num = edf.drop(columns=["run"])
    eff_summary = summarize(edf_num)

    # derived performance-per-compute numbers, if available
    def mean_of(col, frame):
        if col in frame.columns:
            v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            v = v[~np.isnan(v)]
            return float(v.mean()) if len(v) else np.nan
        return np.nan

    auc_mean = mean_of("macro_auc", mdf_num)
    gflops_mean = mean_of("gflops_per_image", edf_num)
    energy_mean = mean_of("energy_mj_per_image", edf_num)
    derived = {}
    if not np.isnan(auc_mean) and gflops_mean and not np.isnan(gflops_mean):
        derived["macro_auc_per_gflop"] = round(auc_mean / gflops_mean, 4)
    if not np.isnan(auc_mean) and energy_mean and not np.isnan(energy_mean):
        derived["macro_auc_per_mj_per_slide"] = round(auc_mean / energy_mean, 4)
    if derived:
        with open(out_dir / "aggregate_efficiency_derived.json", "w") as f:
            json.dump(derived, f, indent=2)
    edf.to_csv(out_dir / "aggregate_efficiency_per_split.csv", index=False)
    eff_summary.to_csv(out_dir / "aggregate_efficiency.csv")

    # ---- mean normalized confusion matrix ----
    mean_cm_norm = np.nanmean(np.stack(cms_norm, axis=0), axis=0)

    # ---- figure ----
    make_figure(mdf_num, per_class_recall, mean_cm_norm, class_order_ref,
                eff_summary, out_dir / "aggregate_figure")

    # ---- console summary ----
    print("\n=== HEADLINE (mean \u00b1 std across splits) ===")
    for m in HEADLINE_METRICS:
        if m in summary.index:
            print(f"  {m:28s} {summary.loc[m, 'formatted']}")
    print("\n=== PER-CLASS RECALL ===")
    for c in class_order_ref:
        key = f"recall_{c}"
        if key in summary.index:
            print(f"  {c:28s} {summary.loc[key, 'formatted']}")
    print("\n=== COMPUTE COST ===")
    for key in ["params_millions", "gflops_per_image", "latency_ms_per_image",
                "throughput_img_per_sec", "energy_mj_per_image",
                "peak_gpu_mem_mb", "train_time_sec", "train_energy_wh"]:
        if key in eff_summary.index and eff_summary.loc[key, "formatted"] != "n/a":
            print(f"  {key:28s} {eff_summary.loc[key, 'formatted']}")
    for k, v in derived.items():
        print(f"  {k:28s} {v}")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
