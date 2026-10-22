"""
Compare end-to-end Glioma-SPARSE systems on identical held-out slides.

Reads the per-slide probability tables written by scripts/inference/e2e_multi.py
and turns each configured "system" into a per-slide decision with the SAME
integration rule as the original pipeline (inference_end_to_end.score_row):

  control slide : correct  <=> Stage A predicts control
  tumour slide  : correct  <=> Stage A grade class (low/high) is right
                                AND a Stage B region exists
                                AND the Stage B subtype (argmax of the soft vote)
                                    equals the true molecular subtype

A system is (run, Stage B prefix, options):
  tta   : use the 8-view test-time-augmented Stage B probabilities
  fuse  : mix Stage A's auxiliary subtype head into the Stage B soft vote,
          p = (1 - w) * p_B + w * p_A_sub. w is chosen on the VALIDATION slides
          (grid maximising log-likelihood of the true subtype) from the saved
          validation predictions of the two stages, never on test data.
          Config: "fuse": {"stage_b_val": <glob of val_predictions[_tta].npz>,
                           "stage_a_val": <glob of Stage A val_predictions.npz>}.

Outputs (in --out-dir): systems_summary.csv/.md, per_slide_decisions.csv,
pairwise_vs_reference.csv, error_decomposition.csv, fusion_weights.json.

Statistics: pooled 220 test rows (190 unique slides: the five random splits
overlap), paired. McNemar exact test on discordant rows and a paired cluster
bootstrap (10,000 resamples of unique slides) 95% CI for the difference in
end-to-end accuracy. Per-fold accuracy is reported as mean +/- SD
(n=5 folds), as in the manuscript.

Usage:
    python -m scripts.analysis.compare_systems --config configs/cascade_rn18.json
    python -m scripts.analysis.compare_systems --verify results/e2e_multi_..../per_slide.csv \
        B0 results/e2e_testset_rn18_f1/e2e_testset_report_RN18_f1.xlsx
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]
SIX = ["control", "astro_low", "astro_high", "oligo_low", "oligo_high", "GBM"]
GRID_W = [round(0.1 * i, 1) for i in range(11)]


# ---------------------------------------------------------------- decisions
def system_probs(df, b, tta=False, fuse_w=None):
    """(N,3) Stage B subtype probabilities of a system; NaN rows = no region."""
    pre = f"{b}__tta_p_" if tta else f"{b}__p_"
    pb = df[[pre + k for k in SUB]].to_numpy(float)
    if fuse_w:
        pa = df[[f"stageA_sub_p_{k}" for k in SUB]].to_numpy(float)
        pb = (1 - fuse_w) * pb + fuse_w * pa
    return pb


def decide(df, pb):
    """Apply the pipeline's integration rule. Returns a per-slide DataFrame."""
    out = df[["slide_id", "fold", "role", "true_grade_class", "true_mutation",
              "grade_class"]].copy()
    ctrl = out.true_grade_class == "control"
    has_b = ~np.isnan(pb).any(axis=1)
    pred_sub = np.array([SUB[i] if h else "NA"
                         for i, h in zip(np.nan_to_num(pb).argmax(1), has_b)])
    out["has_stageB"] = has_b
    out["pred_sub"] = pred_sub
    grade_ok = out.grade_class == out.true_grade_class
    out["grade_ok"] = grade_ok
    out["subtype_ok"] = np.where(ctrl, np.nan, (pred_sub == out.true_mutation).astype(float))
    out["e2e_correct"] = np.where(ctrl, grade_ok, grade_ok & (pred_sub == out.true_mutation)
                                  ).astype(int)
    return out, pb


def six_labels(true_grade, mut):
    if true_grade == "control":
        return "control"
    return {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high",
            ("low", "IDH_mt_1p19q"): "oligo_low", ("high", "IDH_mt_1p19q"): "oligo_high",
            ("high", "IDH_wt"): "GBM"}.get((true_grade, mut))


def joint_macro_auc(df, pb):
    """Diagnostic only: one-vs-rest macro-AUC over the six integrated classes with
    score P(grade) x P(subtype) (control: P(control)); no-Stage-B slides get a flat
    subtype. Not the pipeline decision; see the accuracy for that."""
    pb = np.where(np.isnan(pb), 1 / 3, pb)
    pl, ph, pc = df.prob_low_grade.to_numpy(float), df.prob_high_grade.to_numpy(float), \
        df.prob_control.to_numpy(float)
    S = np.stack([pc, pl * pb[:, 0], ph * pb[:, 0], pl * pb[:, 1], ph * pb[:, 1],
                  ph * pb[:, 2]], 1)
    S = S / S.sum(1, keepdims=True)
    y = np.array([six_labels(g, m) for g, m in zip(df.true_grade_class, df.true_mutation)])
    aucs = []
    for f, g in df.groupby("fold").groups.items():
        idx = np.asarray([df.index.get_loc(i) for i in g])
        a = [roc_auc_score((y[idx] == c).astype(int), S[idx, k])
             for k, c in enumerate(SIX) if 0 < (y[idx] == c).sum() < len(idx)]
        aucs.append(np.mean(a))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def stage_metrics(df, dec, pb):
    """Per-stage diagnostics on the same slides."""
    tum = dec.true_grade_class != "control"
    m = {"grade_acc": float(dec.grade_ok.mean())}
    sub = tum & dec.has_stageB
    m["stageB_reached_pct"] = float(dec.has_stageB[tum].mean() * 100)
    m["subtype_acc_reached"] = float((dec.pred_sub[sub] == dec.true_mutation[sub]).mean())
    y = dec.true_mutation[sub].map({k: i for i, k in enumerate(SUB)}).to_numpy()
    p = pb[sub.to_numpy()]
    try:
        m["subtype_macro_auc_reached"] = float(roc_auc_score(y, p, multi_class="ovr"))
    except ValueError:
        m["subtype_macro_auc_reached"] = float("nan")
    return m


# --------------------------------------------------------------- statistics
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def paired_bootstrap(a, b, slide_ids, n=10000, seed=0):
    """Paired cluster bootstrap over unique slides. The five random splits overlap,
    so some slides are test slides in more than one fold (220 rows = 190 unique
    slides); resampling whole slides keeps such repeats together."""
    rng = np.random.default_rng(seed)
    d = (b - a).astype(float)
    ids, inv = np.unique(np.asarray(slide_ids), return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ids))
    cnts = np.bincount(inv, minlength=len(ids)).astype(float)
    draw = rng.integers(0, len(ids), size=(n, len(ids)))
    boots = sums[draw].sum(1) / cnts[draw].sum(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def choose_fusion_weight(stage_b_val_glob, stage_a_val_glob):
    """Fusion weight maximising the validation log-likelihood of the true subtype.

    Uses only validation slides: the Stage B slide-level soft votes on each fold's
    validation regions (val_predictions[_tta].npz written by finetune_stageB.py) and
    the Stage A subtype-head probabilities of the same slides (val_predictions.npz
    written by finetune_stageA_multitask.py). Test data are never touched.
    """
    import glob
    yb, pbs, pas = [], [], []
    for fold in range(1, 6):
        fb = [f for f in glob.glob(stage_b_val_glob) if f"fold{fold}" in f]
        fa = [f for f in glob.glob(stage_a_val_glob) if f"split_0{fold}" in f]
        assert len(fb) == 1 and len(fa) == 1, (fold, fb, fa)
        b, a = np.load(fb[0]), np.load(fa[0])
        sub_true = {str(s): int(t) for s, t in zip(a["slide_ids"], a["sub_true"])}
        sub_prob = {str(s): p for s, p in zip(a["slide_ids"], a["sub_prob"])}
        for sid, y, p in zip(b["slide_ids"], b["y_true"], b["y_prob"]):
            sid = str(sid)
            if sid in sub_prob and sub_true[sid] == int(y):   # labels must agree
                yb.append(int(y)); pbs.append(p); pas.append(sub_prob[sid])
    y, pb, pa = np.array(yb), np.array(pbs), np.array(pas)
    best, curve = None, {}
    for w in GRID_W:
        p = (1 - w) * pb + w * pa
        ll = float(np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))
        curve[w] = ll
        if best is None or ll > best[1]:
            best = (w, ll)
    return best[0], curve, int(len(y))


# ------------------------------------------------------------------- driver
def run(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {k: pd.read_csv(v) for k, v in cfg["runs"].items()}
    decisions, rows, weights, tables = {}, [], {}, {}
    for sysd in cfg["systems"]:
        label = sysd["label"]
        df_all = runs[sysd["run"]]
        w = None
        if sysd.get("fuse"):
            w, curve, n = choose_fusion_weight(sysd["fuse"]["stage_b_val"],
                                               sysd["fuse"]["stage_a_val"])
            weights[label] = {"w": w, "val_slides": n, "val_loglik_by_w": curve}
        df = df_all[df_all.role == "test"].reset_index(drop=True)
        pb = system_probs(df, sysd["b"], sysd.get("tta", False), w)
        dec, pb = decide(df, pb)
        decisions[label] = dec
        tables[label] = (df, pb)
        per_fold = dec.groupby("fold").e2e_correct.mean()
        auc_m, auc_s = joint_macro_auc(df, pb)
        row = {"system": label, "run": sysd["run"], "stageB": sysd["b"],
               "tta": sysd.get("tta", False), "fusion_w": w,
               "n_slides": len(dec), "n_correct": int(dec.e2e_correct.sum()),
               "e2e_acc_pooled": dec.e2e_correct.mean(),
               "e2e_acc_fold_mean": per_fold.mean(), "e2e_acc_fold_sd": per_fold.std(ddof=1),
               "e2e_acc_per_fold": " ".join(f"{v:.3f}" for v in per_fold),
               "joint_macro_auc_fold_mean": auc_m, "joint_macro_auc_fold_sd": auc_s,
               **stage_metrics(df, dec, pb)}
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "systems_summary.csv", index=False)

    # per-slide decisions
    long = pd.concat([d.assign(system=k) for k, d in decisions.items()])
    long.to_csv(out_dir / "per_slide_decisions.csv", index=False)

    # error decomposition
    dec_rows = []
    for k, d in decisions.items():
        tum = d.true_grade_class != "control"
        wrong = d.e2e_correct == 0
        dec_rows.append({
            "system": k, "n": len(d), "correct": int((~wrong).sum()),
            "grade_wrong": int((wrong & ~d.grade_ok).sum()),
            "grade_ok_no_stageB": int((wrong & d.grade_ok & tum & ~d.has_stageB).sum()),
            "grade_ok_subtype_wrong": int((wrong & d.grade_ok & tum & d.has_stageB).sum())})
    pd.DataFrame(dec_rows).to_csv(out_dir / "error_decomposition.csv", index=False)

    # paired comparisons
    ref = cfg["reference"]
    pair_rows = []
    pairs = cfg.get("pairs") or [[ref, s["label"]] for s in cfg["systems"]
                                 if s["label"] != ref]
    for a_lab, b_lab in pairs:
        da, db = decisions[a_lab], decisions[b_lab]
        assert (da.slide_id.values == db.slide_id.values).all(), "slide mismatch"
        a, b = da.e2e_correct.to_numpy(), db.e2e_correct.to_numpy()
        n01 = int(((a == 1) & (b == 0)).sum())   # A right, B wrong
        n10 = int(((a == 0) & (b == 1)).sum())   # A wrong, B right
        dmean, lo, hi = paired_bootstrap(a, b, da.slide_id.to_numpy())
        fa = da.groupby("fold").e2e_correct.mean().to_numpy()
        fb = db.groupby("fold").e2e_correct.mean().to_numpy()
        pair_rows.append({
            "A (comparator)": a_lab, "B (candidate)": b_lab,
            "acc_A": a.mean(), "acc_B": b.mean(), "delta_B_minus_A": dmean,
            "boot95_lo": lo, "boot95_hi": hi,
            "B_right_A_wrong": n10, "A_right_B_wrong": n01,
            "mcnemar_exact_p": mcnemar_exact(n01, n10),
            "folds_B_better": int((fb > fa).sum()), "folds_tied": int((fb == fa).sum()),
            "folds_B_worse": int((fb < fa).sum())})
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairwise_comparisons.csv", index=False)
    (out_dir / "fusion_weights.json").write_text(json.dumps(weights, indent=2))

    # markdown
    def f3(x):
        return "" if pd.isna(x) else f"{x:.3f}"
    md = ["## End-to-end (6-way integrated diagnosis), held-out test folds\n",
          "| System | E2E acc, pooled (n/N) | E2E acc, fold mean ± SD | Grade acc | "
          "Stage B reached | Subtype acc (reached) | Subtype macro-AUC (reached) | "
          "Joint 6-class macro-AUC |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['system']} | {r['e2e_acc_pooled']:.3f} "
                  f"({r['n_correct']}/{r['n_slides']}) | "
                  f"{r['e2e_acc_fold_mean']:.3f} ± {r['e2e_acc_fold_sd']:.3f} | "
                  f"{f3(r['grade_acc'])} | {r['stageB_reached_pct']:.1f}% | "
                  f"{f3(r['subtype_acc_reached'])} | {f3(r['subtype_macro_auc_reached'])} | "
                  f"{r['joint_macro_auc_fold_mean']:.3f} ± {r['joint_macro_auc_fold_sd']:.3f} |")
    md += ["\n## Paired comparisons (pooled test slides)\n",
           "| Comparator (A) | Candidate (B) | Acc A | Acc B | Δ (B−A) [95% bootstrap CI] | "
           "B right / A right | McNemar p | Folds B better/tied/worse |",
           "|---|---|---|---|---|---|---|---|"]
    for r in pair_rows:
        md.append(f"| {r['A (comparator)']} | {r['B (candidate)']} | {r['acc_A']:.3f} | "
                  f"{r['acc_B']:.3f} | {r['delta_B_minus_A']:+.3f} "
                  f"[{r['boot95_lo']:+.3f}, {r['boot95_hi']:+.3f}] | "
                  f"{r['B_right_A_wrong']} / {r['A_right_B_wrong']} | "
                  f"{r['mcnemar_exact_p']:.3f} | {r['folds_B_better']}/{r['folds_tied']}/"
                  f"{r['folds_B_worse']} |")
    md += ["\n## Error decomposition (wrong slides)\n",
           "| System | Correct | Grade wrong | Grade right, no Stage B | "
           "Grade right, subtype wrong |", "|---|---|---|---|---|"]
    for r in dec_rows:
        md.append(f"| {r['system']} | {r['correct']}/{r['n']} | {r['grade_wrong']} | "
                  f"{r['grade_ok_no_stageB']} | {r['grade_ok_subtype_wrong']} |")
    if weights:
        md += ["\n## Fusion weights (chosen on validation slides)\n"]
        for k, v in weights.items():
            md.append(f"- {k}: w = {v['w']} (n = {v['val_slides']} validation tumour slides)")
    (out_dir / "systems_summary.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(md))


def verify(per_slide_csv, b, archived_xlsx):
    """Check that the plain Stage B system reproduces an archived e2e report."""
    df = pd.read_csv(per_slide_csv)
    df = df[df.role == "test"].reset_index(drop=True)
    dec, _ = decide(df, system_probs(df, b))
    arch = pd.read_excel(archived_xlsx, sheet_name="per_slide")
    m = dec.merge(arch[["slide_id", "fold", "end_to_end_correct", "grade_class", "subtype_pred"]],
                  on=["slide_id", "fold"], how="outer", suffixes=("", "_arch"), indicator=True)
    both = m[m._merge == "both"]
    print(f"slides: mine {len(dec)}, archived {len(arch)}, in both {len(both)}")
    print(f"e2e accuracy: mine {dec.e2e_correct.mean():.4f} | archived "
          f"{arch.end_to_end_correct.astype(float).mean():.4f}")
    diff = both[both.e2e_correct != both.end_to_end_correct.astype(int)]
    print(f"slides whose e2e outcome differs: {len(diff)}")
    print(f"grade_class differs: {(both.grade_class != both.grade_class_arch).sum()}")
    sub = both[both.has_stageB & both.subtype_pred.notna()]
    print(f"Stage B subtype differs (both reached Stage B): "
          f"{(sub.pred_sub != sub.subtype_pred).sum()} of {len(sub)}")
    if len(diff):
        print(diff[["slide_id", "fold", "true_grade_class", "true_mutation", "grade_class",
                    "pred_sub", "subtype_pred", "e2e_correct", "end_to_end_correct"]]
              .to_string())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # tables contain "±" / "Δ"; Windows default is cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out-dir")
    ap.add_argument("--verify", nargs=3, metavar=("PER_SLIDE_CSV", "B_PREFIX", "ARCHIVED_XLSX"))
    a = ap.parse_args()
    if a.verify:
        verify(*a.verify)
    else:
        cfg = json.loads(Path(a.config).read_text())
        run(cfg, Path(a.out_dir or cfg.get("out_dir", "results/compare_systems")))
