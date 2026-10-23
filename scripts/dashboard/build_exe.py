"""
Builds the standalone dashboard exe with PyInstaller (onedir: fast startup,
reliable with torch/CUDA -- a single-file build would re-extract several GB
on every launch).

The exe carries a model bundle (build/exe_bundle -> bundle/ inside the exe), so it
runs on a machine without this repo: for ResNet-18 and ResNet-50 the Stage A
checkpoint of the best fold per validation metric (F1, AUC, accuracy, the same
choice the dashboard makes) with its metrics_val.json, the control image used for
patch injection, the logo and the Stage B heads (data/resuni_heads). Started inside
a repo that has the trained models, the exe uses the repo instead. The UNI2-h
weights are never bundled (gated model): the app finds them via
GLIOMA_SPARSE_UNI2H_CKPT, models/UNI2-h/pytorch_model.bin next to the exe, the
Hugging Face cache, or the user locates the file once in the app.

Windowed (no console): the app opens as a native window (pywebview /
WebView2), not a browser tab. Since there's no console to show errors,
run_dashboard.py redirects stdout/stderr to dashboard.log next to the exe.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\dashboard\\build_exe.py

Output: dist\\GliomaSPARSE-Dashboard\\GliomaSPARSE-Dashboard.exe
"""

import json
import shutil
from pathlib import Path

import PyInstaller.__main__

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "build" / "exe_bundle"
METRIC_COLUMN = {"f1": "macro_f1", "auc": "macro_auc", "acc": "accuracy"}   # as in backend.py
CONTROL = Path("data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")


def stage_bundle():
    """Copy the files the app reads from the repo into build/exe_bundle (same relative layout)."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for arch_dir in ("ResNet18_output", "ResNet50_output"):
        base = REPO / arch_dir / "training_output"
        for metric, col in METRIC_COLUMN.items():
            best = None
            for fold in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("aggregate")):
                mf, ck = fold / "metrics_val.json", fold / f"best_{metric}.pth"
                if mf.exists() and ck.exists():
                    v = json.loads(mf.read_text()).get(col)
                    if v is not None and (best is None or v > best[0]):
                        best = (v, fold)
            if best is None:
                raise SystemExit(f"no {metric} checkpoint under {base}")
            dst = STAGE / arch_dir / "training_output" / best[1].name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best[1] / "metrics_val.json", dst / "metrics_val.json")
            shutil.copy2(best[1] / f"best_{metric}.pth", dst / f"best_{metric}.pth")
            print(f"bundle: {arch_dir} {metric} -> {best[1].name} (val {col} {best[0]:.4f})")
    (STAGE / CONTROL.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / CONTROL, STAGE / CONTROL)
    (STAGE / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "docs" / "logo.png", STAGE / "docs" / "logo.png")
    shutil.copytree(REPO / "data" / "resuni_heads", STAGE / "data" / "resuni_heads")
    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"bundle: {size / 1e6:.0f} MB in {STAGE}")


stage_bundle()
PyInstaller.__main__.run([
    "scripts/dashboard/run_dashboard.py",
    "--name", "GliomaSPARSE-Dashboard",
    "--onedir",
    "--windowed",
    "--icon", "docs/app_icon.ico",
    "--noconfirm",
    "--paths", "src",
    "--paths", "scripts/dashboard",
    "--add-data", "scripts/dashboard/index.html;.",
    # methodology explainer media (talk animations + figures, ~13 MB), served at /explainer/
    "--add-data", "scripts/dashboard/explainer;explainer",
    "--add-data", "src/glioma_sparse/preprocessing/stain_profiles;glioma_sparse/preprocessing/stain_profiles",
    # model bundle (see stage_bundle); NOT the UNI2-h encoder weights (gated download)
    "--add-data", f"{STAGE};bundle",
    "--collect-all", "openslide_bin",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "timm",
    "--collect-all", "webview",
    "--collect-all", "clr_loader",
])
