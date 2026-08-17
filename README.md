# Replication package — *What makes software issue resolution tasks difficult?*

Two ways to run this package.


| Command                 | What it does                                                                   |
| ----------------------- | ------------------------------------------------------------------------------ |
| `./run_paper.sh`        | Loads the frozen `paper_run` XGBoost pickles. Does not train.                  |
| `./run_all.sh`          | Retrains into `results/analysis/` using the published XGBoost hyperparameters. |
| `./run_all.sh --search` | Retrains with `RandomizedSearchCV` for XGBoost.                                |


Local re-runs go under `results/analysis/` (gitignored).

## Quick start

```bash
cd diffmodel_esem_replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"

chmod +x run_paper.sh run_all.sh

# 1) Reproduce the paper run from the archived models
./run_paper.sh

# 2) Retrain the full pipeline from scratch (writes results/analysis/ only)
./run_all.sh
```

## Layout

```
diffmodel_esem_replication/
├── README.md
├── run_paper.sh               # load frozen paper_run pickles
├── run_all.sh                 # train from scratch → results/analysis/
├── requirements.txt
├── data/
├── preprocessing/             # archival upstream scripts (not used by run_*.sh)
│   └── requirements-extract.txt
├── lib/replication/
├── rq1/  rq2/  rq3/
└── results/
    ├── paper_run/             # paper run artifacts
    └── analysis/              # local re-runs only
```

## From-scratch step order (`./run_all.sh`)


| Step | RQ  | Script                                                   | Writes                                                       |
| ---- | --- | -------------------------------------------------------- | ------------------------------------------------------------ |
| 1    | RQ1 | `rq1/run_model_comparison.py`                            | `results/analysis/rq1_predictive_accuracy/model_comparison/` |
| 2    | RQ1 | `rq1/run_tuned_evaluation.py`                            | `results/analysis/rq1_predictive_accuracy/tuned_models/`     |
| 3    | RQ3 | `rq3/run_model_shap.py`                                  | `results/analysis/rq3_feature_importance/model_shap/`        |
| 4    | RQ3 | `rq3/run_stratified_shap_analysis.py --load-saved-model` | `results/analysis/rq3_feature_importance/stratified_shap/`   |
| 5    | RQ2 | `rq2/run_family_ablation.py`                             | `results/analysis/rq2_feature_groups/family_ablation/`       |


Default step 2 refits the published XGBoost hyperparameters (400 trees, lr=0.03, …). Pass `--search` to let `RandomizedSearchCV` choose XGBoost settings.

See [data/README.md](data/README.md) for column definitions.
