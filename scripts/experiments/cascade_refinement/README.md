# Cascade refinement: run scripts

Bash scripts (Git Bash / WSL), run from the repository root with the project
venv on PATH (`source .venv/Scripts/activate`). Each phase writes logs to `logs/`
and skips work that already exists, so it can be re-run after an interruption.

| Phase | Script | What |
|---|---|---|
| 0 | `phase0_baseline_check.sh` | re-score the archived RN18/F1 models with the new evaluator and compare with the archived report |
| 1 | `phase1_option1_stageB.sh` | Option 1: Stage B "continue" and "refined" fine-tuning on the original cohort (5 folds) |
| 2 | `phase2_option2_stageA.sh` | Option 2: Stage A multitask fine-tuning and the grade-only control (5 folds each) |
| 3 | `phase3_cohorts_and_stageB.sh` | rebuild the Stage B cohort for each new Stage A and fine-tune Stage B on it |
| 4 | `phase4_e2e.sh` | held-out end-to-end scoring of all systems (`e2e_multi`) |
| 5 | `phase5_analysis.sh` | pairwise statistics and tables (`compare_systems`) |
