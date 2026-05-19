"""
Load ``data/task_level_dataset.parquet`` for all RQ scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .paths import default_dataset_path, get_replication_root
from .split import attach_split_column, train_test_row_indices


def load_task_level_data(
    dataset_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one row per task: features, outcomes, and identifiers.

    Returns
    -------
    X : DataFrame
        Numeric ``prompt_*``, ``repo_*``, ``patch_*`` columns (model inputs).
        Index is ``task_id`` when that column exists.
    targets : DataFrame
        Outcomes, task/repo/patch identifiers, and ``split`` when present in the
        parquet or ``data/splits/train_test_split.csv``.
    """
    path = Path(dataset_path).expanduser().resolve() if dataset_path else default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Task-level dataset not found: {path}\n"
            "Expected: diffmodel_esem_replication/data/task_level_dataset.parquet"
        )

    df = pd.read_parquet(path)

    outcome_cols = ["pass_rate", "any_success", "maj_success", "n_runs"]
    meta_cols = [
        c
        for c in [
            "task_id",
            "trajectory_id",
            "cf_split",
            "cf_row_index",
            "base_repo",
            "base_patch",
            "base_base_commit",
            "base_instance_id",
            "split",
        ]
        if c in df.columns
    ]

    feature_cols = [
        c
        for c in df.columns
        if (c.startswith("prompt_") or c.startswith("repo_") or c.startswith("patch_"))
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=False) > 1
    ]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    targets = df[outcome_cols + meta_cols].copy()

    if "task_id" in df.columns:
        X.index = df["task_id"].astype(str).values
        targets.index = df["task_id"].astype(str).values

    had_split_in_parquet = "split" in targets.columns
    targets = attach_split_column(targets)
    split_from_csv = not had_split_in_parquet and "split" in targets.columns

    print(f"Task-level dataset: {len(X)} tasks, {X.shape[1]} features")
    print(f"  pass_rate mean={targets['pass_rate'].mean():.3f}")
    print(f"  any_success rate={targets['any_success'].mean():.3f}")
    if "split" in targets.columns:
        n_train = int((targets["split"] == "train").sum())
        n_test = int((targets["split"] == "test").sum())
        source = "parquet" if had_split_in_parquet else "train_test_split.csv"
        print(f"  train/test: {n_train} / {n_test} ({source})")
    else:
        print("  train/test: computed (80/20 stratified on any_success, seed 42)")

    return X, targets


def load_vif_selected_features() -> list[str]:
    """54 features kept after VIF filtering (threshold 10) on the training fold."""
    manifest = get_replication_root() / "data" / "feature_manifest" / "features_vif_selected_54.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing VIF manifest: {manifest}")
    return json.loads(manifest.read_text())


def train_test_indices_from_targets(targets: pd.DataFrame, y_any):
    """
    Positional row indices for train and test folds.

    Uses column ``split`` when present; otherwise applies the paper protocol
    (see ``replication.split``).
    """
    return train_test_row_indices(targets)
