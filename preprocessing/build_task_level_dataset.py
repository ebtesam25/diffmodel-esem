#!/usr/bin/env python3
"""Build task_level_dataset.parquet from upstream feature parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

from load_upstream import build_task_level_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-parquet", type=Path, required=True)
    p.add_argument("--repo-parquet", type=Path, required=True)
    p.add_argument("--patch-parquet", type=Path, required=True)
    p.add_argument("--mapping-parquet", type=Path, default=None)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "task_level_dataset.parquet",
    )
    p.add_argument("--include-r2e", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exclude = () if args.include_r2e else ("R2E_Gym",)
    df = build_task_level_dataset(
        prompt_path=args.prompt_parquet,
        repo_path=args.repo_parquet,
        patch_path=args.patch_parquet,
        mapping_path=args.mapping_parquet,
        exclude_splits=exclude,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output} ({len(df):,} tasks, {df.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
