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
    "--add-data", "src/glioma_sparse/preprocessing/stain_profiles;glioma_sparse/preprocessing/stain_profiles",
    "--collect-all", "openslide_bin",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "webview",
    "--collect-all", "clr_loader",
])
