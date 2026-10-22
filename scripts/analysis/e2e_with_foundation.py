"""
End-to-end (6-way integrated diagnosis) accuracy when Stage B is replaced by the UNI2-h
region classifier from region_foundation_features.py.

Uses the archived RN18 Stage A decisions on the same 220 test rows as every other system
(results/e2e_multi_rn18_A0/per_slide.csv) and the pipeline's integration rule
(control: Stage A must say control; tumour: grade class right AND a region exists AND the
subtype is right). The subtype comes from the UNI2-h classifier trained on that fold's train
slides (regions = that fold's Stage A regions from the cohort manifest).

Rows whose region is missing in the feature-based table (the e2e run used a looser tissue
filter than the cohort builder) count as "no Stage B" - i.e. as errors - and are reported.
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SUB = ["IDH_mt", "IDH_mt_1p19q", "IDH_wt"]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(d, ids, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    dr = rng.integers(0, len(u), size=(n, len(u)))
    b = s[dr].sum(1) / c[dr].sum(1)
    return np.percentile(b, [2.5, 97.5])


def joint_auc(e, pb):
    """Six-class one-vs-rest macro-AUC with score P(grade) x P(subtype) (control: P(control));
    slides without a subtype prediction get a flat subtype. Fold mean and SD."""
    from sklearn.metrics import roc_auc_score
    six = ["control", "astro_low", "astro_high", "oligo_low", "oligo_high", "GBM"]
    t = {("low", "IDH_mt"): "astro_low", ("high", "IDH_mt"): "astro_high", ("low", "IDH_mt_1p19q"): "oligo_low",
         ("high", "IDH_mt_1p19q"): "oligo_high", ("high", "IDH_wt"): "GBM"}
    y = np.array(["control" if g == "control" else t[(g, m)] for g, m in zip(e.true_grade_class, e.true_mutation)])
    pb = np.where(np.isnan(pb), 1 / 3, pb)
    pl, ph, pc = (e.prob_low_grade.to_numpy(float), e.prob_high_grade.to_numpy(float),
                  e.prob_control.to_numpy(float))
    S = np.stack([pc, pl * pb[:, 0], ph * pb[:, 0], pl * pb[:, 1], ph * pb[:, 1], ph * pb[:, 2]], 1)
    S = S / S.sum(1, keepdims=True)
    per = []
    for f in sorted(e.fold.unique()):
        i = np.where(e.fold.to_numpy() == f)[0]
        per.append(np.mean([roc_auc_score((y[i] == c).astype(int), S[i, k])
                            for k, c in enumerate(six) if 0 < (y[i] == c).sum() < len(i)]))
    return float(np.mean(per)), float(np.std(per, ddof=1))


def main(pred_csv="results/region_foundation/test_predictions.csv", variant="regions",
         e2e_csv="results/e2e_multi_rn18_A0/per_slide.csv", label="UNI2-h on regions",
         ref_label="S0 archived (ResNet18 Stage B)", out_dir="results/region_foundation"):
    if str(e2e_csv).endswith(".xlsx"):     # archived report -> same columns as e2e_multi (system "B0")
        e = pd.read_excel(REPO / e2e_csv, sheet_name="per_slide")
        e = e.rename(columns={f"prob_{k}": f"B0__p_{k}" for k in SUB})
        e["role"] = "test"
    else:
        e = pd.read_csv(REPO / e2e_csv)
    e = e[e.role == "test"].reset_index(drop=True)
    P = pd.read_csv(REPO / pred_csv)
    P = P[P.variant == variant].set_index(["fold", "slide_id"])
    ctrl = e.true_grade_class == "control"
    grade_ok = e.grade_class == e.true_grade_class

    def decide(sub_pred, has_b):
        ok = np.where(ctrl, grade_ok, grade_ok & has_b & (sub_pred == e.true_mutation.to_numpy()))
        return ok.astype(int)

    # reference: archived Stage B
    b0 = e[[f"B0__p_{k}" for k in SUB]].to_numpy(float)
    has0 = ~np.isnan(b0).any(1)
    ref = decide(np.array([SUB[i] for i in np.nan_to_num(b0).argmax(1)]), has0)

    sub, has = [], []
    for f, s in zip(e.fold, e.slide_id):
        if (f, s) in P.index:
            r = P.loc[(f, s)]
            sub.append(SUB[int(np.argmax([r[f"p_{k}"] for k in SUB]))])
            has.append(True)
        else:
            sub.append("NA")
            has.append(False)
    sub, has = np.array(sub), np.array(has)
    new = decide(sub, has)
    tum = ~ctrl.to_numpy()
    print(f"tumour rows without a feature-based prediction: {int((tum & ~has & grade_ok.to_numpy()).sum())} "
          f"(of {int((tum & grade_ok.to_numpy()).sum())} with correct grade)")

    n01 = int(((ref == 1) & (new == 0)).sum())
    n10 = int(((ref == 0) & (new == 1)).sum())
    d = (new - ref).astype(float)
    lo, hi = boot(d, e.slide_id.to_numpy())
    pf = pd.Series(new).groupby(e.fold.values).mean()
    pr = pd.Series(ref).groupby(e.fold.values).mean()
    wrong = new == 0
    dec = {"grade wrong": int((wrong & ~grade_ok).sum()),
           "grade ok, no Stage B": int((wrong & grade_ok & tum & ~has).sum()),
           "grade ok, subtype wrong": int((wrong & grade_ok & tum & has).sum())}
    wr = ref == 0
    rdec = {"g": int((wr & ~grade_ok).sum()), "nb": int((wr & grade_ok & tum & ~has0).sum()),
            "sub": int((wr & grade_ok & tum & has0).sum())}
    auc_ref, auc_new = joint_auc(e, b0), joint_auc(e, np.array(
        [[P.loc[(f, s_)][f"p_{k}"] for k in SUB] if (f, s_) in P.index else [np.nan] * 3
         for f, s_ in zip(e.fold, e.slide_id)]))
    txt = [f"| System | E2E accuracy pooled (n/N) | fold mean ± SD | Wrong: grade / no region / subtype |",
           "|---|---|---|---|",
           f"| {ref_label} | {ref.mean():.3f} ({ref.sum()}/{len(ref)}) | "
           f"{pr.mean():.3f} ± {pr.std(ddof=1):.3f} | {rdec['g']} / {rdec['nb']} / {rdec['sub']} |",
           f"| {label} | {new.mean():.3f} ({new.sum()}/{len(new)}) | {pf.mean():.3f} ± {pf.std(ddof=1):.3f} | "
           f"{dec['grade wrong']} / {dec['grade ok, no Stage B']} / {dec['grade ok, subtype wrong']} |",
           "", f"Six-class macro-AUC (fold mean ± SD): reference {auc_ref[0]:.3f} ± {auc_ref[1]:.3f}, new {auc_new[0]:.3f} ± {auc_new[1]:.3f}",
           f"Δ vs S0 = {d.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] (bootstrap over unique slides), "
           f"McNemar p = {mcnemar(n01, n10):.4f} (new right/S0 right: {n10}/{n01}); per-fold new: "
           + " ".join(f"{v:.3f}" for v in pf)]
    print("\n".join(txt))
    Path(REPO / out_dir / "e2e_summary.md").write_text("\n".join(txt), encoding="utf8")
    e.assign(uni_sub=sub, uni_ok=new, s0_ok=ref).to_csv(REPO / out_dir / "e2e_per_slide.csv", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="results/region_foundation/test_predictions.csv")
    ap.add_argument("--e2e", default="results/e2e_multi_rn18_A0/per_slide.csv")
    ap.add_argument("--label", default="UNI2-h on regions")
    ap.add_argument("--ref-label", default="S0 archived (ResNet18 Stage B)")
    ap.add_argument("--out-dir", default="results/region_foundation")
    a = ap.parse_args()
    main(pred_csv=a.pred_csv, e2e_csv=a.e2e, label=a.label, ref_label=a.ref_label, out_dir=a.out_dir)
