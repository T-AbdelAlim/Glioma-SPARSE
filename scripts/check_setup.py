"""
Check that the environment works and which files that live outside git are
present (checkpoints, data, dashboard cache). Run from the repo root:

    python -m scripts.check_setup
"""

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_IMAGE = REPO_ROOT / "data" / "included" / "control" / "86242943-7775-11eb-827d-001a7dda7111.jpg"

ok = True


def line(status, what, note=""):
    print(f"  [{status:^7}] {what}" + (f"  ({note})" if note else ""))


print("Environment")
print(f"  python {sys.version.split()[0]}  ({sys.executable})")
for mod in ["numpy", "torch", "torchvision", "openslide", "PIL", "sklearn", "scipy", "pandas",
            "matplotlib", "openpyxl", "fastapi", "uvicorn", "webview", "glioma_sparse"]:
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

print("\nFiles outside git (copy from the old machine, see README 'Moving to another machine')")
for arch in ["ResNet18_output", "ResNet50_output"]:
    root = REPO_ROOT / arch
    a = list((root / "training_output").glob("*/best_*.pth"))
    b = [p for d in ["training_output_stageB", "training_output_stageB_F1"]
         for p in (root / d).glob("*/best_*.pth")]
    status = "ok" if a and b else "MISSING"
    line(status, f"{arch}/", f"{len(a)} Stage A + {len(b)} Stage B checkpoints; needed by the dashboard")

line("ok" if CONTROL_IMAGE.exists() else "MISSING", "control image",
     "data/included/control/...; needed for the risk map")
cache = REPO_ROOT / "scripts" / "dashboard" / "methodology_cache" / "methodology_cache.json"
line("ok" if cache.exists() else "missing", "methodology cache",
     "optional; otherwise rebuilt once from GLIOMA_SPARSE_METHOD_SLIDE")
n_splits = len(list((REPO_ROOT / "splits").glob("split_0*.csv")))
line("ok" if n_splits == 5 else "missing", "splits/", f"{n_splits}/5 CSVs; needed for training")
n_thumbs = len(list((REPO_ROOT / "data" / "included").rglob("*.jpg")))
line("ok" if n_thumbs else "missing", "data/included/", f"{n_thumbs} thumbnails; needed for training")
slide = os.environ.get("GLIOMA_SPARSE_METHOD_SLIDE")
if slide:
    line("ok" if Path(slide).exists() else "missing", "GLIOMA_SPARSE_METHOD_SLIDE", slide)

print("\n" + ("Environment OK." if ok else "Environment has problems, see FAIL lines above."))
sys.exit(0 if ok else 1)
