"""
Standalone entry point for the packaged dashboard exe (see build_exe.py).
Runs the FastAPI server in a background thread and shows it in a native
OS window (pywebview + WebView2 on Windows) instead of a browser tab.
"""

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

# --windowed builds have no console, so sys.stdout/stderr are None -- code
# that prints (uvicorn, our own prints) would crash without this redirect.
if FROZEN:
    log_path = Path(sys.executable).resolve().parent / "dashboard.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
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


def main():
    threading.Thread(target=_run_server, daemon=True).start()
    _wait_until_up()
    webview.create_window("Glioma-SPARSE", f"http://{HOST}:{PORT}",
                          width=1440, height=900, min_size=(1024, 700))
    webview.start(icon=ICON_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
