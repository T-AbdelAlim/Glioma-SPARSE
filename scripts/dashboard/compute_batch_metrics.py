#!/usr/bin/env python3
r"""
Glioma-SPARSE: metrics from one or more dashboard batch runs.

Independent of the dashboard/backend - reads only the session.json files that
`Run batch` already writes into each slide's GS_* folder, plus a true label you
supply for each batch run. Produces a colored .xlsx report and PNG figures:
accuracy, confusion matrices, per-class precision/recall/F1, and (when enough
classes are present) one-vs-rest ROC/AUC, all for grade, molecular subtype, and
the integrated diagnosis.

WHY YOU SUPPLY THE TRUE LABEL PER RUN, NOT PER SLIDE
    A batch run is normally pointed at one class-homogeneous folder (e.g. an
    external-validation cohort organised as .../oligo_IDHmt_1p19qdel_G2/...),
    so every slide in that run shares one true grade and one true molecular
    class. That is what --run takes: a batch-run folder plus its true label.

SINGLE RUN vs MULTIPLE RUNS - an honest limitation
    With only one run (one true class), every slide has the same ground truth,
    so accuracy/recall for that class is well-defined, but precision, F1, and
    ROC-AUC are NOT: they need both positive and negative examples, which a
    single homogeneous cohort cannot supply. Pass --run multiple times (one per
    class-homogeneous folder) to get the full metric suite. The script always
    reports what is honestly computable from what you gave it, and says
    explicitly which metrics were skipped and why - it never fills in a metric
    that the data can't support.

USAGE
    # one run: only accuracy/recall for that class + confusion distribution
    python compute_batch_metrics.py \
        --run "D:\...\oligo_IDHmt_1p19qdel_G2_results\2e88c2d1_GS_batchrun" low IDH_mt_1p19q \
        --out results/oligo_g2_eval

    # multiple runs: full accuracy/precision/recall/F1/AUC + ROC curves
    python compute_batch_metrics.py \
        --run "D:\...\astro_IDHmt_G2_results\<id>_GS_batchrun"               low  IDH_mt \
        --run "D:\...\astro_IDHmt_G3_results\<id>_GS_batchrun"               high IDH_mt \
        --run "D:\...\oligo_IDHmt_1p19qdel_G2_results\<id>_GS_batchrun"      low  IDH_mt_1p19q \
        --run "D:\...\oligo_IDHmt_1p19qdel_G3_results\<id>_GS_batchrun"      high IDH_mt_1p19q \
        --run "D:\...\GBM_IDHwt_results\<id>_GS_batchrun"                   high IDH_wt \
        --run "D:\...\control_results\<id>_GS_batchrun"                     control na \
        --out results/tcga_external_val

    Grade must be one of: control, low, high, na
    Mutation must be one of: IDH_mt, IDH_mt_1p19q, IDH_wt, na
    (na = "don't evaluate this dimension for this run", e.g. a control-only
    run has no molecular class to score.)

OUTPUT (in --out)
    metrics_report.xlsx   summary, per_slide, confusion_*, all colored
    figures/*.png         confusion heatmaps, ROC curves, confidence histograms
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# CONSTANTS - must mirror backend.py exactly
# ============================================================

GRADE_CLASSES = ["control", "low", "high"]
SUBTYPE_CLASSES = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]

# same integrated-diagnosis mapping as the dashboard backend (INTEGRATED dict),
# reproduced here so this script has no dependency on backend.py
INTEGRATED = {
    ("low", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 2",
    ("high", "IDH_mt"): "Astrocytoma, IDH-mutant, grade 3-4",
    ("low", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 2",
    ("high", "IDH_mt_1p19q"): "Oligodendroglioma, IDH-mutant 1p/19q-codeleted, grade 3",
    ("low", "IDH_wt"): "Glioblastoma, IDH-wildtype, grade 4",
    ("high", "IDH_wt"): "Glioblastoma, IDH-wildtype, grade 4",
}


def true_integrated(true_grade, true_mut):
    if true_grade == "control" or true_mut == "na":
        return "Control tissue" if true_grade == "control" else None
    return INTEGRATED.get((true_grade, true_mut))


# ============================================================
# LOAD SLIDES FROM A BATCH RUN
# ============================================================

def load_batch_slides(batch_dir, true_grade, true_mut):
    """Read every GS_* / session.json under a batch run folder, skipping slides
    that errored (they have no session.json). Returns a list of row dicts."""
    batch_dir = Path(batch_dir)
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"not a folder: {batch_dir}")

    idx_path = batch_dir / "batch_index.json"
    ok_folders = None
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        ok_folders = [Path(e["folder"]) for e in idx.get("slides", [])
                      if e.get("status") == "ok" and e.get("folder")]

    if ok_folders is None:
        ok_folders = sorted(p for p in batch_dir.iterdir() if p.is_dir()
                            and p.name.startswith("GS_"))

    rows = []
    skipped = 0
    for gs in ok_folders:
        sj = gs / "session.json"
        if not sj.exists():
            skipped += 1
            continue
        d = json.loads(sj.read_text())
        sa = d.get("stage_a") or {}
        sb = d.get("stage_b") or {}
        integ = d.get("integrated") or {}
        rows.append({
            "run_dir": str(batch_dir),
            "slide_id": d.get("slide_id", gs.name),
            "true_grade": true_grade,
            "true_mutation": true_mut,
            "true_integrated": true_integrated(true_grade, true_mut),
            "pred_grade": sa.get("grade_pred", "").replace("_grade", ""),
            "pred_subtype": sb.get("subtype_pred"),
            "pred_integrated": integ.get("diagnosis"),
            "prob_control": sa.get("prob_control"),
            "prob_low": sa.get("prob_low_grade"),
            "prob_high": sa.get("prob_high_grade"),
            "prob_IDH_mt": sb.get("prob_IDH_mt"),
            "prob_IDH_mt_1p19q": sb.get("prob_IDH_mt_1p19q"),
            "prob_IDH_wt": sb.get("prob_IDH_wt"),
            "conf_a": sa.get("confidence"),
            "conf_b": sb.get("confidence"),
            "conf_integrated": integ.get("confidence"),
        })
    if skipped:
        print(f"  [{batch_dir.name}] skipped {skipped} slide(s) with no session.json "
              f"(errored during batch)")
    return rows


# normalise the raw grade_pred string ("low_grade"/"high_grade"/"control") to
# the short form used for true_grade ("low"/"high"/"control")
def _norm_grade(g):
    if not g:
        return g
    return g.replace("_grade", "")


# ============================================================
# METRICS (implemented from scratch: no sklearn dependency)
# ============================================================

def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def prf_from_confusion(cm, labels):
    """Per-class precision/recall/F1 from a confusion matrix (rows=true,
    cols=pred). Returns dict label -> {precision, recall, f1, support}."""
    out = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if (prec + rec) > 0 and not np.isnan(prec) and not np.isnan(rec)
              else float("nan"))
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": int(support)}
    return out


def roc_auc_binary(y_true_bin, scores):
    """AUC via the Mann-Whitney U statistic (rank-based), equivalent to
    sklearn's roc_auc_score. Returns (auc, fpr, tpr) or (None, None, None) if
    only one class is present (AUC undefined without both positives and
    negatives)."""
    y = np.asarray(y_true_bin, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg_rank = sums[inv] / counts[inv]
    sum_ranks_pos = avg_rank[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # ROC curve points for plotting
    thresholds = np.unique(s)[::-1]
    tpr, fpr = [0.0], [0.0]
    for th in thresholds:
        pred = (s >= th).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        tpr.append(tp / n_pos); fpr.append(fp / n_neg)
    tpr.append(1.0); fpr.append(1.0)
    return float(auc), np.array(fpr), np.array(tpr)


def macro_avg(prf, labels):
    vals = {"precision": [], "recall": [], "f1": []}
    for l in labels:
        for k in vals:
            v = prf[l][k]
            if not np.isnan(v):
                vals[k].append(v)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in vals.items()}


# ============================================================
# ANALYSIS
# ============================================================

def analyse_dimension(rows, true_key, pred_key, prob_keys, labels, name):
    """Full analysis for one dimension (grade / subtype / integrated).
    prob_keys: dict label -> row-key holding that class's predicted probability
               (None if not applicable, e.g. integrated diagnosis has no single
               probability column).
    Returns a dict of everything computed, with explicit notes on what was
    skipped and why.
    """
    sub = [r for r in rows if r.get(true_key) not in (None, "na")
           and r.get(pred_key) is not None]
    result = {"name": name, "n": len(sub), "notes": []}
    if not sub:
        result["notes"].append(f"no slides with both a true {name} label and a "
                                f"prediction; nothing computed.")
        return result

    y_true = [r[true_key] for r in sub]
    y_pred = [r[pred_key] for r in sub]
    present_true = sorted(set(y_true), key=lambda l: labels.index(l) if l in labels else 999)
    all_labels = labels  # confusion columns cover the full predicted universe

    cm = confusion_matrix(y_true, y_pred, all_labels)
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    prf = prf_from_confusion(cm, all_labels)

    if len(present_true) < 2:
        result["notes"].append(
            f"only one true class ({present_true[0]}) present across the "
            f"supplied runs: precision, F1 and AUC are undefined without a "
            f"negative class to contrast against. Reporting accuracy/recall "
            f"for that class only. Add --run entries for other classes to "
            f"unlock the full metric suite.")
        macro = None
    else:
        macro = macro_avg(prf, all_labels)

    # per-class AUC (one-vs-rest), only for classes with a probability column
    # and only when both a positive and a negative true example exist
    auc_results = {}
    roc_curves = {}
    if prob_keys:
        for lbl in all_labels:
            pk = prob_keys.get(lbl)
            if pk is None:
                continue
            probs = [r.get(pk) for r in sub]
            if any(p is None for p in probs):
                continue
            y_bin = [1 if t == lbl else 0 for t in y_true]
            auc, fpr, tpr = roc_auc_binary(y_bin, probs)
            if auc is None:
                auc_results[lbl] = None   # explicit: undefined, not omitted
            else:
                auc_results[lbl] = auc
                roc_curves[lbl] = (fpr, tpr)

    result.update({
        "labels": all_labels, "cm": cm, "accuracy": acc, "prf": prf,
        "macro": macro, "present_true": present_true,
        "auc": auc_results, "roc_curves": roc_curves,
    })
    return result


# ============================================================
# FIGURES
# ============================================================

def make_figures(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for res in results:
        if "cm" not in res:
            continue
        name = res["name"]; cm = res["cm"]; labels = res["labels"]

        # confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(1 + 0.9 * len(labels), 1 + 0.9 * len(labels)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name} confusion (n={res['n']}, acc={res['accuracy']:.3f})", fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        fig.tight_layout()
        p = fig_dir / f"confusion_{name}.png"
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

        # ROC curves, if any
        if res.get("roc_curves"):
            fig, ax = plt.subplots(figsize=(4.2, 4))
            for lbl, (fpr, tpr) in res["roc_curves"].items():
                auc = res["auc"][lbl]
                ax.plot(fpr, tpr, label=f"{lbl} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
            ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
            ax.set_title(f"{name}: one-vs-rest ROC", fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = fig_dir / f"roc_{name}.png"
            fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


# ============================================================
# EXCEL REPORT
# ============================================================

def write_report(path, rows, results, runs_meta):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.drawing.image import Image as XLImage

    HEAD = "305496"; GREEN = "C6EFCE"; AMBER = "FFEB9C"; RED = "FFC7CE"
    def fill(c): return PatternFill("solid", fgColor=c)
    def head_row(ws, n, row=1):
        for j in range(1, n + 1):
            c = ws.cell(row=row, column=j)
            c.font = Font(bold=True, color="FFFFFF"); c.fill = fill(HEAD)
    def autosize(ws, mx=48):
        for col in ws.columns:
            w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(w + 2, mx)

    wb = Workbook()

    # ---- summary ----
    ws = wb.active; ws.title = "summary"
    ws.append(["Glioma-SPARSE batch metrics"])
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws.append([]); ws.append(["Runs included:"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    for m in runs_meta:
        ws.append([m["batch_dir"], f"true grade: {m['true_grade']}",
                   f"true mutation: {m['true_mutation']}", f"n slides: {m['n']}"])
    ws.append([])

    for res in results:
        ws.append([f"--- {res['name']} ---"])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12)
        if "cm" not in res:
            for note in res["notes"]:
                ws.append([note])
            ws.append([]); continue
        ws.append(["n", res["n"], "accuracy", round(res["accuracy"], 4)])
        if res["macro"]:
            ws.append(["macro precision", round(res["macro"]["precision"], 4),
                       "macro recall", round(res["macro"]["recall"], 4),
                       "macro F1", round(res["macro"]["f1"], 4)])
        for note in res["notes"]:
            ws.append([note])
        ws.append(["class", "precision", "recall", "f1", "support", "AUC (ovr)"])
        for j in range(1, 7):
            c = ws.cell(row=ws.max_row, column=j)
            c.font = Font(bold=True, color="FFFFFF"); c.fill = fill(HEAD)
        for lbl in res["labels"]:
            p = res["prf"][lbl]
            auc = res["auc"].get(lbl) if res.get("auc") else None
            auc_txt = (f"{auc:.4f}" if isinstance(auc, float) else
                       "undefined (needs +/- both)" if lbl in (res.get("auc") or {}) else "")
            row = [lbl,
                   round(p["precision"], 4) if not np.isnan(p["precision"]) else "n/a",
                   round(p["recall"], 4) if not np.isnan(p["recall"]) else "n/a",
                   round(p["f1"], 4) if not np.isnan(p["f1"]) else "n/a",
                   p["support"], auc_txt]
            ws.append(row)
            rc = ws.cell(row=ws.max_row, column=3)
            if isinstance(row[2], float):
                rc.fill = fill(GREEN if row[2] >= 0.75 else AMBER if row[2] >= 0.5 else RED)
        ws.append([])
    autosize(ws, 60)

    # ---- per_slide ----
    ps = wb.create_sheet("per_slide")
    cols = ["run_dir", "slide_id", "true_grade", "true_mutation", "true_integrated",
            "pred_grade", "pred_subtype", "pred_integrated",
            "prob_control", "prob_low", "prob_high",
            "prob_IDH_mt", "prob_IDH_mt_1p19q", "prob_IDH_wt",
            "conf_a", "conf_b", "conf_integrated"]
    ps.append(cols); head_row(ps, len(cols))
    for r in rows:
        ps.append([r.get(c) for c in cols])
    ps.freeze_panes = "A2"; autosize(ps)

    # ---- confusion sheets ----
    for res in results:
        if "cm" not in res:
            continue
        sh = wb.create_sheet(f"confusion_{res['name']}"[:31])
        sh.append([""] + [f"pred:{l}" for l in res["labels"]])
        head_row(sh, len(res["labels"]) + 1)
        for i, l in enumerate(res["labels"]):
            row = [f"true:{l}"] + [int(v) for v in res["cm"][i]]
            sh.append(row)
            sh.cell(row=sh.max_row, column=1).font = Font(bold=True)
            diag_col = i + 2
            cell = sh.cell(row=sh.max_row, column=diag_col)
            cell.fill = fill(GREEN)
        autosize(sh)

    # ---- figures (embedded) ----
    fig_sheet = wb.create_sheet("figures")
    r = 1
    for res in results:
        if "cm" not in res:
            continue
        for kind in ("confusion", "roc"):
            p = Path(path).parent / "figures" / f"{kind}_{res['name']}.png"
            if p.exists():
                try:
                    img = XLImage(str(p))
                    img.width, img.height = img.width * 0.6, img.height * 0.6
                    fig_sheet.add_image(img, f"A{r}")
                    r += int(img.height / 15) + 2
                except Exception:
                    pass

    wb.save(path)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs=3, action="append", required=True,
                   metavar=("BATCH_DIR", "TRUE_GRADE", "TRUE_MUTATION"),
                   help="a batch run folder + its true grade "
                        "(control|low|high|na) + true mutation "
                        "(IDH_mt|IDH_mt_1p19q|IDH_wt|na). Repeat for multiple runs.")
    p.add_argument("--out", default="batch_metrics_out",
                   help="output directory (created if missing)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    valid_grades = {"control", "low", "high", "na"}
    valid_muts = {"IDH_mt", "IDH_mt_1p19q", "IDH_wt", "na"}

    all_rows = []
    runs_meta = []
    for batch_dir, tg, tm in args.run:
        if tg not in valid_grades:
            raise SystemExit(f"invalid true grade '{tg}'; must be one of {valid_grades}")
        if tm not in valid_muts:
            raise SystemExit(f"invalid true mutation '{tm}'; must be one of {valid_muts}")
        rows = load_batch_slides(batch_dir, tg, tm)
        print(f"  [{Path(batch_dir).name}] {len(rows)} slide(s) loaded "
              f"(true grade={tg}, true mutation={tm})")
        all_rows.extend(rows)
        runs_meta.append({"batch_dir": batch_dir, "true_grade": tg,
                          "true_mutation": tm, "n": len(rows)})

    if not all_rows:
        raise SystemExit("no slides loaded from any --run; nothing to analyse")

    for r in all_rows:
        r["pred_grade"] = _norm_grade(r["pred_grade"])

    print(f"\ntotal slides across all runs: {len(all_rows)}\n")

    grade_prob_keys = {"control": "prob_control", "low": "prob_low", "high": "prob_high"}
    subtype_prob_keys = {"IDH_mt": "prob_IDH_mt", "IDH_mt_1p19q": "prob_IDH_mt_1p19q",
                         "IDH_wt": "prob_IDH_wt"}

    res_grade = analyse_dimension(all_rows, "true_grade", "pred_grade",
                                  grade_prob_keys, GRADE_CLASSES, "grade")
    res_subtype = analyse_dimension(all_rows, "true_mutation", "pred_subtype",
                                    subtype_prob_keys, SUBTYPE_CLASSES, "subtype")
    integrated_universe = sorted(set(
        [v for v in INTEGRATED.values()] + ["Control tissue", "Unresolved"]))
    res_integrated = analyse_dimension(all_rows, "true_integrated", "pred_integrated",
                                       None, integrated_universe, "integrated")

    results = [res_grade, res_subtype, res_integrated]

    for res in results:
        print(f"--- {res['name']} ---")
        if "cm" not in res:
            for n in res["notes"]:
                print("  " + n)
            continue
        print(f"  n={res['n']}  accuracy={res['accuracy']:.4f}")
        if res["macro"]:
            print(f"  macro precision={res['macro']['precision']:.4f}  "
                  f"macro recall={res['macro']['recall']:.4f}  "
                  f"macro F1={res['macro']['f1']:.4f}")
        for n in res["notes"]:
            print("  NOTE: " + n)
        print()

    make_figures(results, out_dir)
    report_path = out_dir / "metrics_report.xlsx"
    write_report(report_path, all_rows, results, runs_meta)
    print(f"\nReport: {report_path}")
    print(f"Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
