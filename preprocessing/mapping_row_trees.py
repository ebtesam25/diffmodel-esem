"""Fetch commit trees for mapped rows in a parquet and emit row-level tree refs."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from repo_trees import (
    REPO_PATTERN,
    commit_tree_filename,
    fetch_tree_for_commit,
    find_cached_commit_tree,
    should_refresh_commit_tree,
    write_gzip_json,
)

SMITH_REPO_META_RE = re.compile(
    r"^(?:swesmith/)?(?P<org>[A-Za-z0-9_.-]+)__(?P<repo>[A-Za-z0-9_.-]+)\.(?P<sha>[0-9a-fA-F]{7,40})$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Mapping parquet with base_repo/base_base_commit.")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for mapping_row_trees.parquet output.")
    p.add_argument(
        "--commit-cache-dir",
        type=Path,
        required=True,
        help="Directory for cached GitHub commit tree JSON (gzip).",
    )
    p.add_argument("--github-token", type=str, default=None, help="Default: GITHUB_TOKEN env var.")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def _derive_repo_commit(base_repo: str, base_commit: str) -> tuple[str, str]:
    repo = base_repo.strip()
    commit = base_commit.strip().lower()
    if repo and commit and REPO_PATTERN.fullmatch(repo):
        return repo, commit
    m = SMITH_REPO_META_RE.match(repo)
    if m:
        derived_repo = f"{m.group('org')}/{m.group('repo')}"
        derived_commit = m.group("sha").lower()
        return derived_repo, (commit or derived_commit)
    return repo, commit


def main() -> int:
    args = parse_args()
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN or pass --github-token")

    commit_dir = args.commit_cache_dir.expanduser().resolve()
    commit_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    required = {"cf_split", "cf_row_index", "base_repo", "base_base_commit"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Input parquet missing required columns: {missing}")

    df = df[df["base_repo"].notna()].copy()
    df["base_repo"] = df["base_repo"].astype(str).str.strip()
    df["base_base_commit"] = df.get("base_base_commit", "").fillna("").astype(str).str.strip().str.lower()
    df = df[df["base_repo"] != ""]

    derived = [_derive_repo_commit(r.base_repo, r.base_base_commit) for r in df.itertuples(index=False)]
    df["tree_repo"] = [x[0] for x in derived]
    df["tree_commit"] = [x[1] for x in derived]
    df = df[df["tree_repo"].str.match(REPO_PATTERN, na=False)]
    df = df[df["tree_commit"].astype(str).str.len() >= 7]

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    df["row_uid"] = df["cf_split"].astype(str) + "__" + df["cf_row_index"].astype(str)
    unique_pairs = (
        df[["tree_repo", "tree_commit"]]
        .drop_duplicates()
        .rename(columns={"tree_repo": "repo", "tree_commit": "commit"})
    )
    pair_list = list(unique_pairs.itertuples(index=False, name=None))

    print(f"Rows considered: {len(df)}")
    print(f"Unique (repo, commit) pairs: {len(pair_list)}")

    def fetch_pair(pair: tuple[str, str]) -> tuple[str, str, Path]:
        repo, commit = pair
        refresh, _reason = should_refresh_commit_tree(repo, commit, commit_dir, args.force_refresh)
        if refresh:
            payload = fetch_tree_for_commit(repo, commit, token)
            full_sha = str(payload.get("commit_sha", commit))
            out_path = commit_dir / commit_tree_filename(repo, full_sha)
            write_gzip_json(out_path, payload)
            return repo, commit, out_path

        cached = find_cached_commit_tree(commit_dir, repo, commit)
        if cached is None:
            payload = fetch_tree_for_commit(repo, commit, token)
            full_sha = str(payload.get("commit_sha", commit))
            out_path = commit_dir / commit_tree_filename(repo, full_sha)
            write_gzip_json(out_path, payload)
            return repo, commit, out_path
        return repo, commit, cached

    pair_to_file: dict[tuple[str, str], str] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_pair, p): p for p in pair_list}
        for fut in as_completed(futs):
            repo, commit = futs[fut]
            try:
                _repo, _commit, path = fut.result()
                pair_to_file[(repo, commit)] = str(path)
            except Exception as e:  # noqa: BLE001
                failures.append({"repo": repo, "commit": commit, "error": str(e)})

    if failures:
        fail_path = out_dir / "mapping_row_trees_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        print(f"Failures: {len(failures)} (details in {fail_path})")

    df["tree_file"] = [pair_to_file.get((r.tree_repo, r.tree_commit), "") for r in df.itertuples(index=False)]
    df["tree_cached_or_fetched"] = df["tree_file"] != ""

    out_cols = [
        "row_uid",
        "cf_split",
        "cf_row_index",
        "trajectory_id",
        "base_dataset",
        "base_split",
        "base_row_index",
        "base_repo",
        "base_base_commit",
        "tree_repo",
        "tree_commit",
        "tree_file",
        "tree_cached_or_fetched",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].copy()

    out_parquet = out_dir / "mapping_row_trees.parquet"
    out_jsonl = out_dir / "mapping_row_trees.jsonl"
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)

    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {out_jsonl}")
    print(f"Rows with tree file: {int(out_df['tree_cached_or_fetched'].sum())}/{len(out_df)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
