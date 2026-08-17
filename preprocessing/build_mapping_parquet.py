#!/usr/bin/env python3
"""Build mapping parquet linking CoderForge trajectories to upstream benchmark rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from mapping_lib import SplitLoc, load_split
from mapping_resolver import BaseMatch, MappingResolver
from task_row_features import task_id_from_trajectory_id

HEX8_RE = re.compile(r"([0-9a-fA-F]{8,40})")


def _first_str(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _smith_base_repo(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    org, rest = instance_id.split("__", 1)
    if "." not in rest:
        return f"swesmith/{org}__{rest}"
    repo_part, after_dot = rest.split(".", 1)
    m = HEX8_RE.search(after_dot)
    sha = m.group(1)[:8] if m else ""
    if sha:
        return f"swesmith/{org}__{repo_part}.{sha}"
    return f"swesmith/{org}__{repo_part}"


def _smith_base_commit(instance_id: str) -> str:
    m = HEX8_RE.search(instance_id)
    return m.group(1).lower() if m else ""


def _base_repo(source: str, cf_row: dict[str, Any], base_row: dict[str, Any]) -> str:
    if source == "SWE-rebench":
        return _first_str(base_row, "repo")
    if source == "SWE-smith":
        inst = _first_str(base_row, "instance_id")
        return _smith_base_repo(inst) if inst else ""
    if source == "R2E-Gym":
        repo = _first_str(base_row, "repo_name")
        if repo:
            return repo
        tid = cf_row.get("trajectory_id")
        return str(tid).split("_")[0] if isinstance(tid, str) else ""
    return ""


def _base_commit(source: str, cf_row: dict[str, Any], base_row: dict[str, Any]) -> str:
    if source == "SWE-rebench":
        return _first_str(base_row, "base_commit", "commit", "base_commit_sha")
    if source == "SWE-smith":
        inst = _first_str(base_row, "instance_id")
        return _smith_base_commit(inst)
    if source == "R2E-Gym":
        return _first_str(base_row, "commit_hash")
    return ""


def _mapping_record(
    *,
    cf_split: str,
    cf_row_index: int,
    cf_row: dict[str, Any],
    source: str,
    match: BaseMatch,
) -> dict[str, Any]:
    tid = str(cf_row.get("trajectory_id") or "")
    base_row = match.row
    inst = _first_str(base_row, "instance_id")
    patch = _first_str(base_row, "patch", "model_patch", "gold_patch")
    return {
        "trajectory_id": tid,
        "task_id": task_id_from_trajectory_id(tid),
        "cf_split": cf_split,
        "cf_row_index": cf_row_index,
        "row_uid": f"{cf_split}__{cf_row_index}",
        "base_dataset": match.loc.hf_id,
        "base_split": match.loc.split,
        "base_row_index": match.idx,
        "base_instance_id": inst,
        "base_repo": _base_repo(source, cf_row, base_row),
        "base_base_commit": _base_commit(source, cf_row, base_row),
        "base_patch": patch,
        "patch": patch,
        "base_problem_statement": _first_str(base_row, "problem_statement", "issue", "text"),
        "messages": cf_row.get("messages"),
        "image": cf_row.get("image"),
    }


def build_mapping_dataframe(processed_dir: Path, *, cf_splits: tuple[str, ...] = ("R2E_Gym", "SWE_Smith", "SWE_Rebench")) -> pd.DataFrame:
    resolver = MappingResolver(processed_dir)
    source_by_split = {
        "R2E_Gym": "R2E-Gym",
        "SWE_Smith": "SWE-smith",
        "SWE_Rebench": "SWE-rebench",
    }
    rows: list[dict[str, Any]] = []

    for cf_split in cf_splits:
        resolve = resolver.resolver_for_split(cf_split)
        source = source_by_split[cf_split]
        cf_loc = SplitLoc("togethercomputer/CoderForge-Preview", "trajectories", cf_split)
        ds = load_split(processed_dir, cf_loc.hf_id, cf_loc.config, cf_loc.split)
        n_mapped = 0
        for cf_idx in range(len(ds)):
            cf_row = dict(ds[cf_idx])
            match = resolve(cf_row)
            if match is None:
                continue
            rows.append(
                _mapping_record(
                    cf_split=cf_split,
                    cf_row_index=cf_idx,
                    cf_row=cf_row,
                    source=source,
                    match=match,
                )
            )
            n_mapped += 1
        print(f"{cf_split}: mapped {n_mapped:,} / {len(ds):,} rows")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing hf_arrow/ or hf_parquet/ exports from download_datasets.py.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--split",
        action="append",
        default=None,
        help="CoderForge split to map (repeatable). Default: R2E_Gym, SWE_Smith, SWE_Rebench.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    splits = tuple(args.split) if args.split else ("R2E_Gym", "SWE_Smith", "SWE_Rebench")
    df = build_mapping_dataframe(args.processed_dir.expanduser().resolve(), cf_splits=splits)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, {df.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
