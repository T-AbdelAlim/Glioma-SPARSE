"""
Check that the environment works and that the trained models are in place. Run from the repo root:

    python -m scripts.check_setup
"""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
CONTROL_IMAGE = REPO_ROOT / "data" / "included" / "control" / "86242943-7775-11eb-827d-001a7dda7111.jpg"

ok = True


def line(status, what, note=""):
    print(f"  [{status:^7}] {what}" + (f"  ({note})" if note else ""))


print("Environment")
print(f"  python {sys.version.split()[0]}  ({sys.executable})")
for mod in ["numpy", "torch", "torchvision", "timm", "openslide", "PIL", "sklearn", "pandas", "h5py",
            "fastapi", "uvicorn", "webview", "glioma_sparse"]:
    try:
        m = importlib.import_module(mod)
        line("ok", mod, getattr(m, "__version__", ""))
    except Exception as e:
        ok = False
        line("FAIL", mod, f"{type(e).__name__}: {e}")

try:
    import torch
    if torch.cuda.is_available():
        line("ok", "CUDA", torch.cuda.get_device_name(0))
    else:
        line("cpu", "CUDA not available", "runs on CPU, slower")
except Exception:
    pass

print("\nTrained models (Glioma-SPARSE-models.zip from the GitHub release, unzipped in the repository root)")
for arch in ["ResNet18_output", "ResNet50_output"]:
    n = len(list((REPO_ROOT / arch / "training_output").glob("*/best_*.pth")))
    line("ok" if n else "MISSING", f"{arch}/", f"{n} Stage A checkpoints")
line("ok" if CONTROL_IMAGE.exists() else "MISSING", "control image", "data/included/control/; needed for patch injection")
try:
    from glioma_sparse.models import resuni
    for enc, v in resuni.ENCODERS.items():
        cfgs = [c + v["suffix"] for c in resuni.CONFIG_FOR_STAGE_A.values() if resuni.heads_available(c + v["suffix"], 64)]
        line("ok" if cfgs else "MISSING", f"Stage B heads, {v['label']}", ", ".join(cfgs) or "data/resuni_heads/")
        found = v["available"]()
        line("ok" if found else "missing", f"{v['label']} weights", "found or downloadable" if found else "see README")
except Exception as e:
    line("FAIL", "Stage B", f"{type(e).__name__}: {e}")

if (REPO_ROOT / "scripts" / "stage_a").exists():
    print("\nRetraining")
    n_splits = len(list((REPO_ROOT / "splits").glob("split_0*.csv")))
    line("ok" if n_splits == 5 else "missing", "splits/", f"{n_splits}/5 CSVs")
    n_thumbs = len(list((REPO_ROOT / "data" / "included").rglob("*.jpg")))
    line("ok" if n_thumbs else "missing", "data/included/", f"{n_thumbs} thumbnails (scripts/data/process_wsi_folder.py)")

print("\n" + ("Environment OK." if ok else "Environment has problems, see FAIL lines above."))
sys.exit(0 if ok else 1)
