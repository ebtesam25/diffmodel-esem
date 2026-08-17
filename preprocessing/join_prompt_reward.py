#!/usr/bin/env python3
"""Attach trajectory ``reward`` to a prompt-features parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-parquet", type=Path, required=True)
    p.add_argument(
        "--trajectory-parquet",
        type=Path,
        action="append",
        required=True,
        help="CoderForge-Preview trajectory split(s) with trajectory_id and reward.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--join",
        choices=("left", "inner"),
        default="left",
        help="How to join rewards onto prompt rows (default: left).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prompt_df = pd.read_parquet(args.prompt_parquet.expanduser().resolve())
    if "trajectory_id" not in prompt_df.columns:
        raise SystemExit("prompt parquet missing trajectory_id")

    reward_parts: list[pd.DataFrame] = []
    for path in args.trajectory_parquet:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"trajectory parquet not found: {path}")
        part = pd.read_parquet(path, columns=["trajectory_id", "reward"])
        reward_parts.append(part)

    rewards = pd.concat(reward_parts, ignore_index=True).drop_duplicates("trajectory_id", keep="first")
    out_df = prompt_df.merge(rewards, on="trajectory_id", how=args.join)

    missing = int(out_df["reward"].isna().sum()) if "reward" in out_df.columns else len(out_df)
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df):,} rows; missing reward on {missing:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
