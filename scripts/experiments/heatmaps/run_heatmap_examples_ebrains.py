"""
Stage A patch-injection maps and region boxes for the selected EBRAINS heatmap examples (RN50 best-F1),
one e2e_multi call per test fold. Reads results/heatmaps/ebrains_selection.csv (written when the examples were
chosen); writes results/e2e_heatmap_ebrains_f<fold>/per_slide.csv (columns risk_map, bboxes, ...).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")

sel = pd.read_csv(REPO / "results" / "heatmaps" / "ebrains_selection.csv")
sel = sel[sel.true6 != "control"]
for fold, g in sel.groupby("fold"):
    cmd = [PY, "-u", "-m", "scripts.inference.e2e_multi",
           "--run-name", f"e2e_heatmap_ebrains_f{fold}", "--model", "resnet50",
           "--stage-a-glob", "ResNet50_output/training_output/*_split_0*/best_f1.pth",
           "--system", "B0=ResNet50_output/training_output_stageB_F1/*fold*/best_f1.pth",
           "--roles", "test", "--folds", str(fold), "--slide-ids", ",".join(g.slide_id),
           "--risk-batch", "2", "--resume"]
    print("running fold", fold, len(g), "slides", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
print("done")
