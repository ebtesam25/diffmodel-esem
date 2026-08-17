"""Merge upstream prompt/repo/patch feature parquets to one row per task."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

OUTCOME_COLUMNS = ["pass_rate", "n_runs", "n_success", "any_success", "maj_success"]
TASK_META_COLUMNS = [
    "task_id",
    "trajectory_id",
    "cf_split",
    "cf_row_index",
    "base_repo",
    "base_patch",
    "base_base_commit",
    "base_instance_id",
]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if (c.startswith("prompt_") or c.startswith("repo_") or c.startswith("patch_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_task_level_dataset(
    prompt_path: Path | str,
    repo_path: Path | str,
    patch_path: Path | str,
    mapping_path: Optional[Path | str] = None,
    exclude_splits: Iterable[str] = ("R2E_Gym",),
) -> pd.DataFrame:
    prompt_path = Path(prompt_path).expanduser()
    repo_path = Path(repo_path).expanduser()
    patch_path = Path(patch_path).expanduser()

    prompt_df = pd.read_parquet(prompt_path)
    repo_df = pd.read_parquet(repo_path)
    patch_df = pd.read_parquet(patch_path)

    merged = prompt_df.merge(repo_df, on="trajectory_id", suffixes=("", "_repo"))
    merged = merged.merge(patch_df, on="trajectory_id", suffixes=("", "_patch"))

    if mapping_path is not None:
        mapping_path = Path(mapping_path).expanduser()
        if mapping_path.is_file():
            mdf = pd.read_parquet(mapping_path)
            map_cols = ["trajectory_id"] + [
                c for c in TASK_META_COLUMNS if c in mdf.columns and c != "trajectory_id"
            ]
            mdf = mdf[map_cols].drop_duplicates(subset=["trajectory_id"])
            merged = merged.merge(mdf, on="trajectory_id", how="left", suffixes=("", "_map"))

    if "patch_present" in merged.columns:
        merged = merged[pd.to_numeric(merged["patch_present"], errors="coerce").fillna(0) != 0].copy()

    if "cf_split" in merged.columns and exclude_splits:
        merged = merged[~merged["cf_split"].isin(list(exclude_splits))].copy()

    if "reward" not in merged.columns:
        raise ValueError("prompt parquet must include trajectory-level reward for outcomes")

    merged["is_success"] = (merged["reward"] >= 1.0).astype(int)

    outcome_agg = (
        merged.groupby("task_id")["is_success"]
        .agg(pass_rate="mean", n_runs="count", n_success="sum")
        .reset_index()
    )
    outcome_agg["any_success"] = (outcome_agg["n_success"] >= 1).astype(int)
    outcome_agg["maj_success"] = (outcome_agg["pass_rate"] >= 0.5).astype(int)

    feature_cols = _feature_columns(merged)
    task_features = (
        merged[["task_id"] + feature_cols].drop_duplicates(subset=["task_id"]).set_index("task_id")
    )
    task_features = task_features.loc[:, task_features.nunique(dropna=False) > 1]
    task_features = task_features.fillna(task_features.median(numeric_only=True))

    meta_cols = [c for c in TASK_META_COLUMNS if c in merged.columns and c != "task_id"]
    task_meta = merged.groupby("task_id")[meta_cols].first() if meta_cols else pd.DataFrame()

    out = task_features.join(outcome_agg.set_index("task_id"), how="inner")
    if not task_meta.empty:
        out = out.join(task_meta, how="left")

    out = out.reset_index()
    front = [c for c in TASK_META_COLUMNS if c in out.columns]
    front += [c for c in OUTCOME_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]
