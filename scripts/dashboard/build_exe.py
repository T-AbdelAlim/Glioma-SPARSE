"""
Builds the standalone dashboard exe with PyInstaller (onedir: fast startup,
reliable with torch/CUDA -- a single-file build would re-extract several GB
on every launch).

The exe carries a model bundle (build/exe_bundle -> bundle/ inside the exe), so it
runs on a machine without this repo: for ResNet-18 and ResNet-50 the Stage A
checkpoint of the fold the app uses (resuni.APP_FOLD), for each validation metric with
Stage B heads in this version, with its metrics_val.json, the control image used for
patch injection, the logo and the Stage B heads (data/resuni_heads) of the encoders of
this version (DistillPath; also UNI2-h in the full version). Started inside a repo that
has the trained models, the exe uses the repo instead. Encoder weights are never
bundled. DistillPath (non-commercial licence, no access request): found via
GLIOMA_SPARSE_DISTILLPATH_CKPT, models/DistillPath-KS16-UNI2h/model.safetensors next to
the exe or the Hugging Face cache, or downloaded from the hub on first use (86 MB).
UNI2-h (full version, gated model): found via GLIOMA_SPARSE_UNI2H_CKPT,
models/UNI2-h/pytorch_model.bin next to the exe, the Hugging Face cache, or the user
locates the file once in the app.

The build follows the platform and the PyTorch of the environment it runs in: on Windows
with a CPU-only PyTorch GliomaSPARSE-Dashboard-Windows (runs on any Windows machine), with a
CUDA PyTorch GliomaSPARSE-Dashboard-Windows-GPU (needs an NVIDIA GPU, about 3.5 GB); on macOS
GliomaSPARSE-Dashboard-macOS (an .app bundle, runs on the CPU). PyInstaller builds for the
platform it runs on, so the macOS app is built on a Mac.

Windowed (no console): the app opens as a native window (pywebview /
WebView2), not a browser tab. Since there's no console to show errors,
run_dashboard.py redirects stdout/stderr to dashboard.log next to the exe.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\dashboard\\build_exe.py

Output: dist\\GliomaSPARSE-Dashboard-Windows[-GPU]\\GliomaSPARSE-Dashboard-Windows[-GPU].exe, or
dist/GliomaSPARSE-Dashboard-macOS.app on macOS
"""

import json
import os
import shutil
import sys
from pathlib import Path

import PyInstaller.__main__
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from glioma_sparse.models import resuni  # noqa: E402

NAME = "GliomaSPARSE-Dashboard-" + ("macOS" if sys.platform == "darwin" else "Windows-GPU" if torch.version.cuda else "Windows")
SEP = os.pathsep   # PyInstaller's SOURCE<sep>DEST separator: ";" on Windows, ":" elsewhere
STAGE = REPO / "build" / "exe_bundle"
METRIC_COLUMN = {"f1": "macro_f1", "auc": "macro_auc", "acc": "accuracy"}   # as in backend.py
CONTROL = Path("data/included/control/86242943-7775-11eb-827d-001a7dda7111.jpg")


ENCODERS_ALL = {"distillpath": "_dp_uni2h", "uni2h": ""}   # encoder -> head config suffix, in any version


def stage_bundle():
    """Copy the files the app reads from the repo into build/exe_bundle (same relative layout)."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for arch_dir, arch in (("ResNet18_output", "resnet18"), ("ResNet50_output", "resnet50")):
        base = REPO / arch_dir / "training_output"
        for metric in METRIC_COLUMN:
            # the app's fold, for the checkpoint choices with Stage B heads in this version
            if not any(resuni.resuni_config_available(arch, metric, e) for e in resuni.ENCODERS):
                continue
            for k in (resuni.APP_FOLD,):
                folds = sorted(base.glob(f"*_split_{k:02d}"))
                if not folds or not (folds[-1] / f"best_{metric}.pth").exists():
                    raise SystemExit(f"no {metric} checkpoint of fold {k} under {base}")
                dst = STAGE / arch_dir / "training_output" / folds[-1].name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folds[-1] / "metrics_val.json", dst / "metrics_val.json")
                shutil.copy2(folds[-1] / f"best_{metric}.pth", dst / f"best_{metric}.pth")
                print(f"bundle: {arch_dir} {metric} fold {k} -> {folds[-1].name}")
    (STAGE / CONTROL.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / CONTROL, STAGE / CONTROL)
    (STAGE / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "docs" / "logo.png", STAGE / "docs" / "logo.png")
    for cfg in sorted((REPO / "data" / "resuni_heads").iterdir()):
        # a head config belongs to the encoder with the longest matching suffix ("" = UNI2-h)
        enc = max((e for e, v in ENCODERS_ALL.items() if cfg.name.endswith(v)), key=lambda e: len(ENCODERS_ALL[e]))
        if cfg.is_dir() and enc in resuni.ENCODERS:
            shutil.copytree(cfg, STAGE / "data" / "resuni_heads" / cfg.name)
            print(f"bundle: Stage B heads {cfg.name} ({resuni.ENCODERS[enc]['label']})")
    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"bundle: {size / 1e6:.0f} MB in {STAGE}")


stage_bundle()
PyInstaller.__main__.run([
    "scripts/dashboard/run_dashboard.py",
    "--name", NAME,
    "--onedir",
    "--windowed",
    "--icon", "docs/app_icon.ico",
    "--noconfirm",
    "--paths", "src",
    "--paths", "scripts/dashboard",
    "--add-data", f"scripts/dashboard/index.html{SEP}.",
    # methodology explainer media (talk animations + figures, ~13 MB), served at /explainer/
    "--add-data", f"scripts/dashboard/explainer{SEP}explainer",
    "--add-data", f"src/glioma_sparse/preprocessing/stain_profiles{SEP}glioma_sparse/preprocessing/stain_profiles",
    # model bundle (see stage_bundle); never the encoder weights
    "--add-data", f"{STAGE}{SEP}bundle",
    "--collect-all", "openslide_bin",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "timm",
    # DistillPath: local weights (safetensors) or a download from the hub on first use
    "--collect-all", "safetensors",
    "--collect-all", "huggingface_hub",
    "--collect-all", "webview",
] + (["--collect-all", "clr_loader"] if sys.platform == "win32" else []))   # pythonnet bridge of WebView2
