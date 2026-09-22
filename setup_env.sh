#!/usr/bin/env bash
# One-time setup on a new Linux machine (Windows: setup_env.ps1). Creates .venv with
# the exact package versions from requirements-lock.txt, installs glioma_sparse, then
# checks which files that live outside git still need to be copied over.
#   bash setup_env.sh            (PYTHON=/path/to/python3.14 to pick the interpreter)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.14}
command -v "$PY" >/dev/null || { echo "Python 3.14 not found; install it or set PYTHON=/path/to/python3.14"; exit 1; }

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m scripts.check_setup
