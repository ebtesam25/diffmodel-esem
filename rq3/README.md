# RQ3 — Which individual features are most strongly associated with agent success?

**Question.** Which features drive predictions, in what direction, and how do they interact?

## Scripts


| File                                    | Paper RQ       | Purpose                                                               |
| --------------------------------------- | -------------- | --------------------------------------------------------------------- |
| `run_model_shap.py`                     | **RQ3**        | SHAP + interactions on tuned XGBoost from RQ1 `tuned_models/`         |
| `run_stratified_shap_analysis.py`       | **RQ3 §4.3.2** | Easy / mid-band / hard SHAP composition, prompt top-5 rates, and R(x) |
| `plot_shap_waterfalls.py`               | RQ3            | Waterfalls for borderline / average-difficulty tasks                  |
| `plot_shap_waterfalls_by_difficulty.py` | RQ3            | Waterfalls for easy / hard / borderline buckets                       |


Tuning and calibration are **RQ1** (`rq1/run_tuned_evaluation.py` → `lib/replication/tuned_evaluation.py`).

## Outputs

`results/analysis/rq3_feature_importance/model_shap/`

## Run

```bash
export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"

# After RQ1 has written models under results/analysis/:
python rq3/run_model_shap.py --verbose
python rq3/run_stratified_shap_analysis.py --load-saved-model --verbose

# Paper run
python rq3/run_stratified_shap_analysis.py \
  --load-saved-model \
  --tuned-dir results/paper_run/task_level_run \
  --outdir results/analysis/rq3_feature_importance/stratified_shap_paper \
  --assert-paper-values --verbose
```

`./run_paper.sh` is the paper run path. `./run_all.sh` trains new models and then runs stratified analysis with `--load-saved-model` on those new models.

`--assert-paper-values` is only for the archived `any_success` pickle. A from-scratch refit is nearby, not bit-identical.

**Optional figures** (after `run_model_shap.py`):

```bash
OUT=results/analysis/rq3_feature_importance/model_shap

python rq3/plot_shap_waterfalls.py \
  --outdir "$OUT" --targets any_success --average-n 10 --plot-suffix _avg10

python rq3/plot_shap_waterfalls_by_difficulty.py \
  --outdir "$OUT" --targets any_success --n-each 10 --plot-suffix _top10
```

