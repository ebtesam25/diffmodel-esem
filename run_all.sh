#!/usr/bin/env bash
# From-scratch pipeline. Trains new models into results/analysis/.
# Never writes into results/paper_run/.
#
#   ./run_all.sh              # refit published XGBoost hyperparameters (nearby Finding 3)
#   ./run_all.sh --search     # RandomizedSearchCV for XGBoost (may pick a different model)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/lib:${PYTHONPATH:-}"
export CODERFORGE_REPLICATION_ROOT="${ROOT}"

SEARCH=0
for arg in "$@"; do
  case "${arg}" in
    --search|--search-xgboost) SEARCH=1 ;;
    -h|--help)
      echo "Usage: $0 [--search]"
      echo "  default   refit paper_run XGBoost hyperparameters; new trees in results/analysis/"
      echo "  --search  RandomizedSearchCV for XGBoost (not guaranteed to match the PDF)"
      echo "Camera-ready pickle path: ./run_paper.sh"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [--search]" >&2
      exit 1
      ;;
  esac
done

echo "=== From-scratch path: write only to results/analysis/ ==="
if [[ "${SEARCH}" -eq 1 ]]; then
  echo "XGBoost: RandomizedSearchCV (may not match ESEM-CR.pdf)"
else
  echo "XGBoost: refit published paper_run hyperparameters (new trees, not the pickle)"
fi

echo "=== RQ1 — predictive accuracy (models + VIF) ==="
python rq1/run_model_comparison.py

echo "=== RQ1 (tuned) — held-out metrics + calibration ==="
if [[ "${SEARCH}" -eq 1 ]]; then
  python rq1/run_tuned_evaluation.py --search-xgboost --verbose
else
  python rq1/run_tuned_evaluation.py --verbose
fi

echo "=== RQ3 — SHAP on the models just trained ==="
python rq3/run_model_shap.py --verbose

echo "=== RQ3 — prediction-stratified SHAP on the models just trained ==="
python rq3/run_stratified_shap_analysis.py --load-saved-model --verbose

echo "=== RQ2 — feature-group ablation ==="
python rq2/run_family_ablation.py --verbose

echo "Done."
echo "  RQ1 → results/analysis/rq1_predictive_accuracy/"
echo "  RQ2 → results/analysis/rq2_feature_groups/"
echo "  RQ3 → results/analysis/rq3_feature_importance/"
echo "  Frozen camera-ready artifacts were not modified (results/paper_run/)."
