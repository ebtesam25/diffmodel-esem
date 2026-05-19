"""Path helpers for the replication bundle."""

from __future__ import annotations

import os
from pathlib import Path

# Paper research questions → results subdirectories
RQ_DIRS = {
    "rq1": "rq1_predictive_accuracy",
    "rq2": "rq2_feature_groups",
    "rq3": "rq3_feature_importance",
}


def get_replication_root() -> Path:
    """Root of ``diffmodel_esem_replication/``."""
    env = os.environ.get("CODERFORGE_REPLICATION_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # lib/replication/paths.py → diffmodel_esem_replication/
    return Path(__file__).resolve().parents[2]


def default_dataset_path() -> Path:
    return get_replication_root() / "data" / "task_level_dataset.parquet"


def default_split_path() -> Path:
    return get_replication_root() / "data" / "splits" / "train_test_split.csv"


def results_dir_for_rq(rq: str) -> Path:
    """``results/analysis/<rq_dir>/`` for one research question."""
    if rq not in RQ_DIRS:
        raise ValueError(f"Unknown rq {rq!r}; expected one of {sorted(RQ_DIRS)}")
    return get_replication_root() / "results" / "analysis" / RQ_DIRS[rq]


def default_results_dir() -> Path:
    """Base analysis output directory (all RQs)."""
    return get_replication_root() / "results" / "analysis"

def default_tuned_models_dir() -> Path:
    """Default directory for RQ1 tuned model artifacts."""
    return results_dir_for_rq("rq1") / "tuned_models"


def default_paper_run_dir() -> Path:
    """Frozen authors' task-level run (models, SHAP, tables)."""
    return get_replication_root() / "results" / "paper_run" / "task_level_run"

