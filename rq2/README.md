# RQ2 — Which feature groups account for predictive performance?

**Question.** How much do patch, repository, and prompt features contribute alone and in combination?

**Script.** `run_family_ablation.py`

**Prerequisite.** Run `rq3/run_model_shap.py` first (reads tuned XGBoost hyperparameters and VIF feature set from that run).

**Outputs.** `results/analysis/rq2_feature_groups/family_ablation/`

- `family_ablation_metrics.csv` — patch / repo / prompt subsets (Table component ablation)

**Run.**

```bash
export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"
python rq2/run_family_ablation.py --verbose
```

Optional: `--shap-outdir results/analysis/rq3_feature_importance/model_shap`
