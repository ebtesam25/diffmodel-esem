"""Train/test split for task-level analysis (paper protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import default_split_path

# Paper / replication protocol (task_level_model_shap experiment_manifest)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_STRATIFY_COL = "any_success"


def assign_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify_col: str = DEFAULT_STRATIFY_COL,
    split_column: str = "split",
) -> pd.DataFrame:
    """
    Add a ``split`` column (``train`` / ``test``) with one row per task.

    Task-level stratified holdout: ``test_size`` fraction held out, stratified on
    ``stratify_col`` (binary ``any_success`` in the paper).
    """
    if len(df) == 0:
        raise ValueError("Cannot split an empty dataframe")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratify column {stratify_col!r} not in dataframe")

    out = df.copy()
    y = out[stratify_col].values
    if pd.api.types.is_float_dtype(out[stratify_col]):
        y = (y >= 0.5).astype(int)
    else:
        y = y.astype(int)

    strat = y if len(np.unique(y)) >= 2 else None
    idx = np.arange(len(out))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    labels = np.empty(len(out), dtype=object)
    labels[:] = "train"
    labels[test_idx] = "test"
    out[split_column] = labels
    return out


def attach_split_column(
    targets: pd.DataFrame,
    *,
    split_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Add ``split`` from parquet, ``train_test_split.csv``, or leave unchanged.

    Lookup order: existing column → CSV at ``split_path`` (default under ``data/splits/``).
    """
    if "split" in targets.columns:
        return targets

    path = Path(split_path).expanduser().resolve() if split_path else default_split_path()
    if not path.is_file():
        return targets

    table = pd.read_csv(path)
    if "task_id" not in table.columns or "split" not in table.columns:
        raise ValueError(f"Split table must have task_id and split columns: {path}")

    by_task = table.set_index(table["task_id"].astype(str))["split"]
    out = targets.copy()
    if out.index.name == "task_id" or (
        out.index.dtype == object and out.index.astype(str).isin(by_task.index).any()
    ):
        out["split"] = out.index.astype(str).map(by_task)
    elif "task_id" in out.columns:
        out["split"] = out["task_id"].astype(str).map(by_task)
    else:
        raise ValueError("Cannot align split table: targets need task_id index or column")

    if out["split"].isna().any():
        n = int(out["split"].isna().sum())
        raise ValueError(f"Split table missing labels for {n} tasks")
    return out


def train_test_row_indices(
    targets: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify_col: str = DEFAULT_STRATIFY_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train and test row indices (positional, aligned with ``targets`` rows)."""
    if "split" in targets.columns:
        train_idx = np.where(targets["split"].values == "train")[0]
        test_idx = np.where(targets["split"].values == "test")[0]
        return train_idx, test_idx

    labeled = assign_train_test_split(
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify_col=stratify_col,
    )
    train_idx = np.where(labeled["split"].values == "train")[0]
    test_idx = np.where(labeled["split"].values == "test")[0]
    return train_idx, test_idx


def write_split_table(
    targets: pd.DataFrame,
    path: Path,
    *,
    task_id_col: str = "task_id",
    extra_cols: Optional[list[str]] = None,
) -> Path:
    """Write ``task_id``, ``split``, and optional columns to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in [task_id_col, "split"] if c in targets.columns]
    if extra_cols:
        cols.extend(c for c in extra_cols if c in targets.columns and c not in cols)
    targets[cols].to_csv(path, index=False)
    return path
