"""
Experiment E6: is the two-stage design worth it? Single-stage six-class prediction
directly from n UNI2-h tiles drawn at random from the WHOLE slide (no Stage A), n = 64, 128, 256, 512, 1024 and all.

  A  benchmark CLAM-SB (the five trained fold models, trained on all tiles) applied to the n random tiles at test time
     ("the benchmark, run cheaper"); n = all reproduces the benchmark.
  B  single-stage counterpart of our Stage B: mean of n random UNI2-h tile features -> StandardScaler -> class-balanced
     multinomial logistic regression over the six classes, fitted per fold and per n on n-tile embeddings of the training
     slides (C chosen on the validation slides by log-loss), same recipe as resuni_predictions.py.

Internal: the 220 test predictions of the five folds (test sets = the CLAM benchmark's, split_{k}_results.pkl).
External: TCGA (180 slides), ensemble of the five fold models. Three random tile draws per n (all: one).
Tiles per slide come from TRIDENT (20x, 256 px, features_uni_v2), as in the benchmark.
    python -m scripts.analysis.single_stage_ntiles        (mahmoodlab venv: h5py + CLAM; GPU)
Output: results/revision_oct22/e6_single_stage_predictions.csv (+ e6_single_stage.json via --eval)
"""
import glob, json, pickle, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[2]; BRAIN = REPO.parent
sys.path[:0] = [str(REPO), str(REPO / "src"), str(BRAIN / "mahmoodlab" / "CLAM")]
from scripts.analysis import heatmaps as H

OUT = REPO / "results" / "revision_oct22"; OUT.mkdir(parents=True, exist_ok=True)
SIX = ["control", "astro_low", "astro_high", "oligo_low", "oligo_high", "GBM"]
NAME = {"control": "control", "low_grade_astro": "astro_low", "high_grade_astro": "astro_high",
        "low_grade_oligo": "oligo_low", "high_grade_oligo": "oligo_high", "GBM": "GBM"}
NS = [64, 128, 256, 512, 1024, "all"]; DRAWS = (0, 1, 2)
RES = BRAIN / "mahmoodlab" / "CLAM" / "results" / "glioma_subtype_clam_s1"

# ---- labels, folds and CLAM output order
ds = pd.read_csv(BRAIN / "clam_workspace" / "dataset_csv" / "dpwsi_6class.csv").set_index("slide_id").label
lab_int = {sid: NAME[l] for sid, l in ds.items()}
pk = [pickle.load(open(RES / f"split_{k}_results.pkl", "rb")) for k in range(5)]
idx_of = {}                                   # CLAM output index -> SIX name
for r in pk:
    for sid, v in r.items():
        idx_of[int(v["label"])] = lab_int[str(sid)]
ORDER = [next(i for i, c in idx_of.items() if c == s) for s in SIX]   # columns to take, in SIX order
TEST = {k + 1: [str(s) for s in pk[k]] for k in range(5)}
SPLIT = {}
for k in range(1, 6):
    sp = pd.read_csv(REPO / "splits" / f"split_0{k}.csv"); sp["slide_id"] = [Path(p).stem for p in sp.path]
    SPLIT[k] = sp.set_index("slide_id").split.to_dict()
    assert sorted(s for s, r in SPLIT[k].items() if r == "test") == sorted(TEST[k]), f"fold {k}: test sets differ"
tcga = pd.read_csv(REPO / "results" / "external_TCGA_val" / "labels.csv").set_index("slide_id").true6.to_dict()

feat_int = {Path(p).stem: p for p in glob.glob(str(BRAIN / "mahmoodlab" / "trident_output" / "*" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}
feat_ext = {Path(p).stem: p for p in glob.glob(str(BRAIN / "mahmoodlab" / "trident_tcga_val" / "20x_256px_0px_overlap" / "features_uni_v2" / "*.h5"))}


def subsets(sid, m):
    """{(n, draw): tile indices}; nested random subsets of the slide's m tiles (slide-specific seed)."""
    out = {("all", 0): np.arange(m)}
    for d in DRAWS:
        perm = np.random.default_rng([d, zlib.crc32(sid.encode())]).permutation(m)
        for n in NS[:-1]:
            out[(n, d)] = perm[:n]
    return out


@torch.no_grad()
def clam_probs(model, F):
    logits = model(F)[0]
    return torch.softmax(logits, 1)[0].cpu().numpy()[ORDER]


def extract():
    """One pass over the slides: n-tile mean embeddings (variant B) and CLAM probabilities (variant A)."""
    models = {k: H.clam_model(k - 1) for k in range(1, 6)}
    emb, rows = {}, []
    slides = [(s, "internal", feat_int[s]) for s in sorted(set().union(*[set(SPLIT[k]) for k in SPLIT])) if s in feat_int] + \
             [(s, "tcga", feat_ext[s]) for s in sorted(tcga) if s in feat_ext]
    print("slides:", len(slides), flush=True)
    for i, (sid, coh, path) in enumerate(slides):
        with h5py.File(path) as h:
            F = h["features"][:].astype(np.float32)
        sub = subsets(sid, len(F)); Ft = torch.from_numpy(F).to(H.DEV)
        for key, idx in sub.items():
            emb[(sid, *key)] = F[idx].mean(0)
        folds = [k for k in TEST if sid in TEST[k]] if coh == "internal" else [1, 2, 3, 4, 5]
        for k in folds:
            for (n, d), idx in sub.items():
                p = clam_probs(models[k], Ft[torch.from_numpy(idx).to(H.DEV)])
                rows.append({"variant": "A_clam", "cohort": coh, "n": n, "draw": d, "fold": k, "slide_id": sid,
                             **{f"p_{c}": float(v) for c, v in zip(SIX, p)}})
        if i % 50 == 0: print(f"  {i}/{len(slides)}", flush=True)
    return emb, rows


def fit(Xtr, ytr, Xva, yva, Cs=(1e-3, 1e-2, 1e-1, 1, 10)):
    sc = StandardScaler().fit(Xtr); best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        ll = log_loss(yva, clf.predict_proba(sc.transform(Xva)), labels=list(range(6)))
        if best is None or ll < best[0]:
            best = (ll, clf)
    return sc, best[1]


def single_stage(emb):
    rows = []
    for k in range(1, 6):
        tr = [s for s, r in SPLIT[k].items() if r == "train" and s in feat_int]
        va = [s for s, r in SPLIT[k].items() if r == "val" and s in feat_int]
        ytr = [SIX.index(lab_int[s]) for s in tr]; yva = [SIX.index(lab_int[s]) for s in va]
        for n in NS:
            for d in ((0,) if n == "all" else DRAWS):
                X = lambda ids: np.stack([emb[(s, n, d)] for s in ids])
                sc, clf = fit(X(tr), ytr, X(va), yva)
                for coh, ids in (("internal", TEST[k]), ("tcga", [s for s in tcga if s in feat_ext])):
                    P = clf.predict_proba(sc.transform(X(ids)))
                    P6 = np.zeros((len(ids), 6)); P6[:, clf.classes_] = P
                    for s, p in zip(ids, P6):
                        rows.append({"variant": "B_meanLR", "cohort": coh, "n": n, "draw": d, "fold": k, "slide_id": s,
                                     **{f"p_{c}": float(v) for c, v in zip(SIX, p)}})
            print(f"fold {k} n={n} done", flush=True)
    return rows


if __name__ == "__main__":
    emb, rows = extract()
    rows += single_stage(emb)
    D = pd.DataFrame(rows)
    D["true6"] = [lab_int[s] if c == "internal" else tcga[s] for s, c in zip(D.slide_id, D.cohort)]
    D.to_csv(OUT / "e6_single_stage_predictions.csv", index=False)
    print("wrote", len(D), "rows")
