"""Extract repo-only feature tables from mapping and tree parquets."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

import pandas as pd

from repo_trees import read_existing_commit_tree
from task_row_features import _tree_features, save_features_parquet

REPO_META_COLUMNS = [
    "trajectory_id",
    "task_id",
    "cf_split",
    "cf_row_index",
    "row_uid",
    "tree_file",
]

REPO_FEATURE_COLUMNS = [
    "tree_present",
    "tree_truncated",
    "repo_file_count",
    "repo_dir_count",
    "repo_top_level_dir_count",
    "repo_has_docs_dir",
    "repo_has_tests_dir",
    "repo_has_examples_dir",
    "repo_max_depth",
    "repo_mean_file_depth",
    "repo_basename_collision_count",
    "repo_python_file_count",
    "repo_test_file_count",
    "repo_test_ratio",
    "repo_total_known_size",
    "repo_mean_known_file_size",
]


def _empty_repo_features() -> dict[str, Any]:
    return {
        "tree_present": 0,
        "tree_truncated": 0,
        "repo_file_count": 0,
        "repo_dir_count": 0,
        "repo_top_level_dir_count": 0,
        "repo_has_docs_dir": 0,
        "repo_has_tests_dir": 0,
        "repo_has_examples_dir": 0,
        "repo_max_depth": 0,
        "repo_mean_file_depth": 0.0,
        "repo_basename_collision_count": 0,
        "repo_python_file_count": 0,
        "repo_test_file_count": 0,
        "repo_test_ratio": 0.0,
        "repo_total_known_size": 0,
        "repo_mean_known_file_size": 0.0,
    }


@functools.lru_cache(maxsize=32)
def _tree_payload(path_raw: Any) -> dict[str, Any] | None:
    if path_raw is None:
        return None
    s = str(path_raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_file():
        return None
    return read_existing_commit_tree(p)


def extract_repo_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*REPO_META_COLUMNS, *REPO_FEATURE_COLUMNS])

    unique_tree_files = [t for t in df["tree_file"].unique() if pd.notna(t) and str(t).strip()]
    features_list: list[dict[str, Any]] = []
    for tree_file in unique_tree_files:
        payload = _tree_payload(tree_file)
        if payload is None:
            features = _empty_repo_features()
        else:
            features = _tree_features(payload, [])
            features = {k: features.get(k, 0) for k in REPO_FEATURE_COLUMNS}
        row = {"tree_file": tree_file}
        row.update(features)
        features_list.append(row)

    if features_list:
        feat_df = pd.DataFrame(features_list)
        res_df = df.merge(feat_df, on="tree_file", how="left")
    else:
        res_df = df.copy()

    empty = _empty_repo_features()
    for col in REPO_FEATURE_COLUMNS:
        if col not in res_df.columns:
            res_df[col] = empty[col]
        else:
            res_df[col] = res_df[col].fillna(empty[col])

    ordered_cols = [c for c in [*REPO_META_COLUMNS, *REPO_FEATURE_COLUMNS] if c in res_df.columns]
    return res_df[ordered_cols]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping-parquet", type=Path, required=True)
    p.add_argument("--trees-parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def _subset_mapping_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards - 1}]")
    if args.num_shards > 1:
        n = len(df)
        shard_rows = (n + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_rows
        end = min(start + shard_rows, n)
        df = df.iloc[start:end].copy()
    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()
    return df


def main() -> int:
    args = parse_args()
    mapping_path = args.mapping_parquet.expanduser().resolve()
    trees_path = args.trees_parquet.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    if not mapping_path.is_file():
        raise SystemExit(f"Mapping parquet not found: {mapping_path}")
    if not trees_path.is_file():
        raise SystemExit(f"Trees parquet not found: {trees_path}")

    df = pd.read_parquet(mapping_path)
    if "trajectory_id" not in df.columns:
        raise SystemExit("Mapping parquet missing trajectory_id.")

    df = _subset_mapping_rows(df, args)
    if len(df) > 0:
        trees_df = pd.read_parquet(trees_path)
        key_cols = ["cf_split", "cf_row_index"]
        for c in key_cols:
            if c not in df.columns or c not in trees_df.columns:
                raise SystemExit(f"Missing key column {c!r} in mapping or trees parquet.")
        need_tree_cols = [c for c in ["tree_file"] if c in trees_df.columns]
        df = df.merge(
            trees_df[key_cols + need_tree_cols].drop_duplicates(key_cols),
            on=key_cols,
            how="left",
        )
        feat_df = extract_repo_features_for_df(df)
    else:
        feat_df = pd.DataFrame(columns=[*REPO_META_COLUMNS, *REPO_FEATURE_COLUMNS])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_features_parquet(feat_df, out_path, index=False)
    print(f"Wrote {len(feat_df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
