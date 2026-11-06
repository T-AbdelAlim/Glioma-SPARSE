"""
Standalone entry point for the packaged dashboard app (see build_exe.py).
Runs the FastAPI server in a background thread and shows it in a native
OS window (pywebview: WebView2 on Windows, WebKit on macOS) instead of a browser tab.

    --check   start the server, report the device, encoders and trained models found, and exit
              (exit code 0 when the app can run the full pipeline)
"""

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

# --windowed builds have no console, so sys.stdout/stderr are None -- code
# that prints (uvicorn, our own prints) would crash without this redirect.
CHECK = "--check" in sys.argv   # packaged apps write its report to dashboard.log as well
if FROZEN:
    # next to the app, or (e.g. a macOS app in /Applications) in the user's GliomaSPARSE folder
    for log_dir in (Path(sys.executable).resolve().parent,
                    Path(os.environ.get("APPDATA") or Path.home()) / "GliomaSPARSE"):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(log_dir / "dashboard.log", "a", buffering=1, encoding="utf-8")
            break
        except OSError:
            continue
    sys.stdout = log_file
    sys.stderr = log_file
else:
    # `python -m scripts.dashboard.run_dashboard` puts the repo root on the path,
    # not this folder, so `from backend import app` needs it added
    sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(os.environ.get("GLIOMA_SPARSE_ROOT") or (
    Path(sys.executable).resolve().parents[2] if FROZEN
    else Path(__file__).resolve().parents[2]))

import socket
import threading
import time

import uvicorn
import webview

HOST = "127.0.0.1"
PORT = 8000
ICON_PATH = str(REPO_ROOT / "docs" / "app_icon.ico")


def _unblock_dlls():
    """Windows marks the files of a downloaded zip as coming from the internet, and .NET refuses to load such a library
    ("Failed to resolve Python.Runtime.Loader.Initialize"). Remove the mark from the .NET libraries of the app window."""
    if not (FROZEN and sys.platform == "win32"):
        return
    for folder in ("pythonnet", "clr_loader", "webview"):
        for dll in (Path(sys._MEIPASS) / folder).rglob("*.dll"):
            try:
                os.remove(f"{dll}:Zone.Identifier")
            except OSError:     # no mark, or a read-only folder
                pass


def _run_server():
    from backend import app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def _wait_until_up(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)


def check():
    import json
    import urllib.request
    threading.Thread(target=_run_server, daemon=True).start()
    _wait_until_up(120)
    c = json.load(urllib.request.urlopen(f"http://{HOST}:{PORT}/api/config", timeout=60))
    ok_a = [k for k, v in c["checkpoints_present"].items() if k.startswith("stage_a") and v]
    print("device:", c["device"])
    print("Stage A checkpoints:", ", ".join(ok_a) or "none")
    for e, v in c["encoders"].items():
        heads = [k for k, ok in v["configs"].items() if ok]
        print(f"Stage B {v['label']}: weights {'found or downloadable' if v['available'] else 'missing'}, "
              f"heads for {', '.join(heads) or 'none'}")
    good = bool(ok_a) and any(v["available"] and any(v["configs"].values()) for v in c["encoders"].values())
    if good:
        # a small computation on the bundled control thumbnail, to compare across machines
        import numpy as np
        import backend as B
        img = B.control_img()
        model = B.get_model("stage_a", "resnet50", "f1")
        probs = B.forward_probs(model, img)[0]
        rm, _ = B.compute_risk_map(target_img=img, control_img=img, model=model, transform=B._transform, grid=B.GRID,
                                   target_class=int(np.argmax(probs)), device=B.DEVICE,
                                   control_shuffle_seed=B.SEED, score="logit", method="crop")
        tiles = [((0, 0, 0, 0), img.crop((x, y, x + 256, y + 256))) for x in (512, 1024) for y in (512, 768, 1024, 1280)]
        feats = B.resuni.embed_tiles(tiles, B.DEVICE, encoder="distillpath")
        print("Stage A on the control thumbnail:", " ".join(f"{p:.4f}" for p in probs))
        print(f"patch injection: signal map mean {rm.mean():.4f}, max {rm.max():.4f}")
        print(f"DistillPath on 8 tiles: mean feature {feats.mean():.4f}, norm {np.linalg.norm(feats.mean(0)):.3f}")
    print("check OK" if good else "check FAILED")
    return 0 if good else 1


def main():
    threading.Thread(target=_run_server, daemon=True).start()
    _wait_until_up()
    _unblock_dlls()
    webview.create_window("Glioma-SPARSE", f"http://{HOST}:{PORT}",
                          width=1440, height=900, min_size=(1024, 700))
    webview.start(icon=ICON_PATH)


if __name__ == "__main__":
    # batch processing runs each slide in a spawned process; in the packaged app that process starts this
    # executable again, and freeze_support() makes it run the worker instead of a second dashboard
    import multiprocessing
    multiprocessing.freeze_support()
    if CHECK:
        sys.exit(check())
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
