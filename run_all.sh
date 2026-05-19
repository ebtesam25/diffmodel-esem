#!/usr/bin/env bash
# Full replication pipeline ordered by research question dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/lib:${PYTHONPATH:-}"
export CODERFORGE_REPLICATION_ROOT="${ROOT}"

echo "=== RQ1 — predictive accuracy (models + VIF) ==="
python rq1/run_model_comparison.py

echo "=== RQ1 (tuned) — RandomizedSearchCV + held-out metrics + calibration ==="
python rq1/run_tuned_evaluation.py --verbose

echo "=== RQ3 — SHAP on tuned XGBoost (requires RQ1 tuned_models) ==="
python rq3/run_model_shap.py --verbose

echo "=== RQ2 — feature-group ablation (requires tuned XGB params from RQ1) ==="
python rq2/run_family_ablation.py --verbose

echo "Done."
echo "  RQ1 → results/analysis/rq1_predictive_accuracy/"
echo "  RQ2 → results/analysis/rq2_feature_groups/"
echo "  RQ3 → results/analysis/rq3_feature_importance/"
