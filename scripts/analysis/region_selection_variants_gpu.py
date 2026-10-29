"""
Experiment E1, GPU part: patch-injection logit shifts for ALL three Stage-A
classes on every internal test prediction, so alternative region-selection rules can be compared offline.

For each fold k (ResNet-50 best-F1 checkpoint of split_0k) and each test slide of that fold that Stage A does not
call control: the control slide is shuffled once (seed 42, as in the pipeline), every target cell (r, c) is injected
at the same position, and the three logits of the injected image and of the shuffled control are stored.
  shift[r, c, j] = logit_j(injected) - logit_j(shuffled control)
The main pipeline uses shift[..., predicted grade] (largest increase); variant "control drop" uses -shift[..., 0].
Results are appended per slide (resumable after an interruption):
  results/region_selection_variants/injection_logits.jsonl
Run from the repo root with the Glioma-SPARSE venv:  python -m scripts.analysis.region_selection_variants_gpu
"""
import csv, glob, json, re, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from glioma_sparse.data_utils.transforms import build_eval_transform
from glioma_sparse.interpret.patch_injection import shuffle_control_once, tile_image, reconstruct_image
from glioma_sparse.models.factory import build_model
from scripts.inference.e2e_multi import reroot
from scripts.inference.inference_end_to_end import GRADE_CLASSES, GRID, SEED, _rel, _sidecar_for

OUT = REPO / "results" / "region_selection_variants"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "injection_logits.jsonl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
done = set()
if LOG.exists():
    for line in open(LOG):
        try: j = json.loads(line); done.add((j["fold"], j["slide_id"]))
        except Exception: pass
tf = build_eval_transform()
ctrl = Image.open(_rel("data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")).convert("RGB")
ckpts = sorted(glob.glob(str(_rel("ResNet50_output/training_output/*_split_0*/best_f1.pth"))))

@torch.no_grad()
def logits(imgs, model, bs=8):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(torch.stack([tf(im) for im in imgs[i:i + bs]]).to(DEV)).float().cpu().numpy())
    return np.concatenate(out)

for sc in sorted(_rel("splits").glob("split_0*.csv")):
    fold = int(re.search(r"split_(\d+)", sc.stem).group(1))
    ck = [c for c in ckpts if sc.stem in str(c)][0]
    model = build_model("resnet50", num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True)); model.to(DEV).eval()
    test = [reroot(r["path"]) for r in csv.DictReader(open(sc)) if r["split"] == "test"]
    for p in test:
        if p.parent.name == "control" or (fold, p.stem) in done:
            continue
        thumb = _sidecar_for(p.stem, _rel(r"data\ebrains_thumbnails"))
        img = Image.open(thumb).convert("RGB")
        lt = logits([img], model)[0]; k = int(np.argmax(lt))
        rec = {"fold": fold, "slide_id": p.stem, "pred_grade": GRADE_CLASSES[k], "target_logits": lt.round(5).tolist()}
        if GRADE_CLASSES[k] != "control":
            cs = shuffle_control_once(ctrl, GRID, SEED)
            lb = logits([cs], model)[0]
            T, C = tile_image(img, GRID), tile_image(cs, GRID)
            variants = []
            for i in range(GRID[0] * GRID[1]):
                t = list(C); t[i] = T[i]; variants.append(reconstruct_image(t, GRID, cs.size))
            L = logits(variants, model)
            rec.update({"base_logits": lb.round(5).tolist(), "shift": (L - lb).reshape(GRID[0], GRID[1], 3).round(5).tolist()})
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(fold, p.stem[:8], GRADE_CLASSES[k], flush=True)
print("done")
