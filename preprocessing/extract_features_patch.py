"""Extract patch-only feature tables from a mapping parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task_row_features import extract_features_many, save_features_parquet

PATCH_META_COLUMNS = ["trajectory_id", "task_id", "cf_split", "cf_row_index", "row_uid"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping-parquet", type=Path, required=True)
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


def _select_patch_columns(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("patch_", "ast_", "cfg_", "func_")
    special = {"patch_present", "patch_lines_changed_total", "patch_is_single_file"}
    cols = [
        c
        for c in df.columns
        if c in PATCH_META_COLUMNS or c in special or any(c.startswith(p) for p in prefixes)
    ]
    ordered = [c for c in PATCH_META_COLUMNS if c in cols] + [c for c in cols if c not in PATCH_META_COLUMNS]
    return df[ordered]


def main() -> int:
    args = parse_args()
    mapping_path = args.mapping_parquet.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    if not mapping_path.is_file():
        raise SystemExit(f"Mapping parquet not found: {mapping_path}")

    df = pd.read_parquet(mapping_path)
    if "trajectory_id" not in df.columns:
        raise SystemExit("Mapping parquet missing trajectory_id.")

    df = _subset_mapping_rows(df, args)
    if len(df) == 0:
        feat_df = pd.DataFrame(columns=PATCH_META_COLUMNS)
    else:
        feat_df = extract_features_many(df.to_dict("records"))
        feat_df = _select_patch_columns(feat_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_features_parquet(feat_df, out_path, index=False)
    print(f"Wrote {len(feat_df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
