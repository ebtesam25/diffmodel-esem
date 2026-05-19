#!/usr/bin/env python3
"""
Regenerate ``data/splits/train_test_split.csv`` (paper train/test protocol).

Example::

    export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
    export CODERFORGE_REPLICATION_ROOT="$(pwd)"
    python data/make_train_test_split.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPL_ROOT / "lib"))

from replication.data import load_task_level_data
from replication.split import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    assign_train_test_split,
    write_split_table,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=str,
        default=str(_REPL_ROOT / "data" / "splits" / "train_test_split.csv"),
        help="Output CSV path (default: data/splits/train_test_split.csv).",
    )
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument(
        "--with-parquet-split",
        action="store_true",
        help="Also add a ``split`` column to task_level_dataset.parquet.",
    )
    args = p.parse_args()

    _, targets = load_task_level_data()
    targets = targets.drop(columns=["split"], errors="ignore")
    labeled = assign_train_test_split(
        targets,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    out_path = write_split_table(
        labeled,
        Path(args.output),
        extra_cols=["any_success"],
    )
    n_train = int((labeled["split"] == "train").sum())
    n_test = int((labeled["split"] == "test").sum())
    print(f"Wrote {out_path}")
    print(f"  train={n_train} test={n_test} (stratified on any_success, seed={args.random_state})")

    if args.with_parquet_split:
        from replication.paths import default_dataset_path

        full = pd.read_parquet(default_dataset_path())
        if "split" in full.columns:
            full = full.drop(columns=["split"])
        if "task_id" in labeled.columns:
            split_df = labeled.set_index("task_id")[["split"]]
            full = full.set_index("task_id")
            full["split"] = split_df["split"]
            full = full.reset_index()
        else:
            full["split"] = labeled["split"].values
        path = default_dataset_path()
        full.to_parquet(path, index=False)
        print(f"Updated {path} with ``split`` column (--with-parquet-split)")


if __name__ == "__main__":
    main()
