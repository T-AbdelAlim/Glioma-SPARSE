"""
Builds the standalone dashboard exe with PyInstaller (onedir: fast startup,
reliable with torch/CUDA -- a single-file build would re-extract several GB
on every launch). Model checkpoints and data paths stay hardcoded absolute
paths on this machine, same as the dev server; the exe is not portable to
another machine's directory layout.

Windowed (no console): the app opens as a native window (pywebview /
WebView2), not a browser tab. Since there's no console to show errors,
run_dashboard.py redirects stdout/stderr to dashboard.log next to the exe.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\dashboard\\build_exe.py

Output: dist\\GliomaSPARSE-Dashboard\\GliomaSPARSE-Dashboard.exe
"""

import PyInstaller.__main__

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
    # ResUNI's trained per-fold heads (small, ~700 KB total; NOT the UNI2-h encoder weights
    # themselves -- those are a gated download, see glioma_sparse.models.resuni / README).
    "--add-data", "data/resuni_heads;data/resuni_heads",
    "--collect-all", "openslide_bin",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "timm",
    "--collect-all", "webview",
    "--collect-all", "clr_loader",
])
