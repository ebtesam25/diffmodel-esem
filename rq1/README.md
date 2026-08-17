# RQ1 — Can static task features predict agent success?

**Question.** How well do patch, repository, and prompt features predict agent outcomes?

## Scripts

| File | Purpose |
|------|---------|
| `run_model_comparison.py` | Baseline models, VIF, CV, component ablation (fixed hyperparameters) |
| `run_tuned_evaluation.py` | **Tuned** models: VIF + `RandomizedSearchCV`, held-out metrics, ROC, calibration |

## Outputs

- `results/analysis/rq1_predictive_accuracy/model_comparison/` — comparison script
- `results/analysis/rq1_predictive_accuracy/tuned_models/` — tuned evaluation + saved XGBoost joblibs

## Run

```bash
export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"
python rq1/run_model_comparison.py
python rq1/run_tuned_evaluation.py --verbose                 # published XGB hyperparameters
python rq1/run_tuned_evaluation.py --search-xgboost --verbose  # optional: re-search XGBoost
```

Default tuned evaluation **refits** the `paper_run` XGBoost hyperparameters and writes a **new** pickle under `results/analysis/`. It does not load or overwrite `results/paper_run/`. Use `./run_paper.sh` to score the archived models. Use `./run_all.sh` for the full from-scratch path.
