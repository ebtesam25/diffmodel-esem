#!/usr/bin/env bash
# Reproduce camera-ready numbers from the frozen paper_run models.
# Does not train. Does not write into results/paper_run/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/lib:${PYTHONPATH:-}"
export CODERFORGE_REPLICATION_ROOT="${ROOT}"

PAPER="${ROOT}/results/paper_run/task_level_run"
PICKLE="${PAPER}/any_success_xgb/any_success_xgb_final_xgb_model.pkl"
if [[ ! -f "${PICKLE}" ]]; then
  echo "Missing frozen any_success model: ${PICKLE}" >&2
  echo "The replication package must ship results/paper_run/ model pickles." >&2
  exit 1
fi

echo "=== Camera-ready path: load frozen paper_run XGBoost (no training) ==="
echo "  model: ${PICKLE}"
echo "  outputs: results/analysis/ (paper_run is read-only)"

python rq3/run_stratified_shap_analysis.py \
  --load-saved-model \
  --tuned-dir "${PAPER}" \
  --outdir "${ROOT}/results/analysis/rq3_feature_importance/stratified_shap_paper" \
  --assert-paper-values \
  --verbose

echo "Done. Exact §4.3.2 numbers should match ESEM-CR.pdf."
echo "  Frozen models remain in results/paper_run/"
echo "  This run wrote only under results/analysis/rq3_feature_importance/stratified_shap_paper/"
