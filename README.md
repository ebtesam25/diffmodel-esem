# Replication package — *What makes software issue resolution tasks difficult?*


## Layout (by research question)

```
diffmodel_esem_replication/
├── README.md
├── run_all.sh                 # RQ1 → RQ3 → RQ2 (see order below)
├── requirements.txt
├── data/                      # shared dataset
│   ├── task_level_dataset.parquet
│   └── splits/train_test_split.csv
├── lib/replication/           # data loading, paths, tuned_evaluation, shap_analysis
├── rq1/                       # RQ1 — predictive accuracy
│   ├── README.md
│   ├── run_model_comparison.py
│   └── run_tuned_evaluation.py
├── rq2/                       # RQ2 — feature-group ablation
│   ├── README.md
│   └── run_family_ablation.py
├── rq3/                       # RQ3 — SHAP / individual features
│   ├── README.md
│   ├── run_model_shap.py
│   ├── plot_shap_waterfalls.py
│   └── plot_shap_waterfalls_by_difficulty.py
└── results/
    ├── paper_run/             # reference
    └── paper_tables/          # reference
    
```

## Quick start

```bash
cd diffmodel_esem_replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"

chmod +x run_all.sh
./run_all.sh
```

## Run order

| Step | RQ | Script | Depends on |
|------|-----|--------|------------|
| 1 | RQ1 | `rq1/run_model_comparison.py` | `data/task_level_dataset.parquet` |
| 2 | RQ1 | `rq1/run_tuned_evaluation.py` | same dataset → `tuned_models/` |
| 3 | RQ3 | `rq3/run_model_shap.py` | `tuned_models/` (saved XGBoost) |
| 4 | RQ2 | `rq2/run_family_ablation.py` | `best_params` from step 2 |


See [data/README.md](data/README.md) for column definitions.
