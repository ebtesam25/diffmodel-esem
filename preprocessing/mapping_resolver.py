"""Resolve CoderForge-Preview trajectories to upstream benchmark rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset

from mapping_lib import (
    ParquetRowDataset,
    SplitLoc,
    _best_fuzzy_idx,
    _build_first_index_by_key,
    _normalize_rebench_instance_id,
    _normalize_smith_instance_like,
    _r2e_commit_from_cf_row,
    _rebench_instance_from_image,
    _rebench_owner_repo_key,
    _repo_from_image,
    _repo_from_r2e_trajectory_id,
    _smith_pr_instance_from_cf_trajectory_id,
    _smith_repo_sha_from_base_instance_id,
    _smith_repo_sha_from_cf_trajectory_id,
    _strip_run_suffix,
    load_split,
)


@dataclass(frozen=True)
class BaseMatch:
    loc: SplitLoc
    idx: int
    row: dict[str, Any]


class MappingResolver:
    """Index upstream benchmark tables and resolve CF rows (see mapping_lib.py)."""

    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir.expanduser().resolve()

        self.base_r2e_loc = SplitLoc("R2E-Gym/R2E-Gym-V1", "default", "train")
        self.base_smith_locs = (SplitLoc("SWE-bench/SWE-smith", "default", "train"),)
        self.base_rebench_loc = SplitLoc("nebius/SWE-rebench", "default", "test")
        self.base_rebench_v2_loc = SplitLoc("nebius/SWE-rebench-V2", "default", "train")

        self.base_r2e_ds = load_split(
            self.processed_dir,
            self.base_r2e_loc.hf_id,
            self.base_r2e_loc.config,
            self.base_r2e_loc.split,
        )
        self.base_rebench_ds = load_split(
            self.processed_dir,
            self.base_rebench_loc.hf_id,
            self.base_rebench_loc.config,
            self.base_rebench_loc.split,
        )
        try:
            self.base_rebench_v2_ds: Dataset | ParquetRowDataset | None = load_split(
                self.processed_dir,
                self.base_rebench_v2_loc.hf_id,
                self.base_rebench_v2_loc.config,
                self.base_rebench_v2_loc.split,
            )
        except FileNotFoundError:
            self.base_rebench_v2_ds = None

        self.smith_indexes: list[
            tuple[
                SplitLoc,
                Dataset | ParquetRowDataset,
                dict[str, int],
                dict[str, int],
                dict[str, int],
                dict[str, int],
                dict[str, list[tuple[int, str]]],
            ]
        ] = []
        for loc in self.base_smith_locs:
            ds = load_split(self.processed_dir, loc.hf_id, loc.config, loc.split)
            exact_idx = _build_first_index_by_key(ds, "instance_id")
            exact_idx_ci = {k.lower(): v for k, v in exact_idx.items()}
            normalized_idx = {_normalize_smith_instance_like(k): v for k, v in exact_idx.items()}
            repo_sha_idx: dict[str, int] = {}
            repo_sha_candidates: dict[str, list[tuple[int, str]]] = {}
            for inst, idx in exact_idx.items():
                key = _smith_repo_sha_from_base_instance_id(inst)
                if key and key not in repo_sha_idx:
                    repo_sha_idx[key] = idx
                if key:
                    repo_sha_candidates.setdefault(key, []).append((idx, _normalize_smith_instance_like(inst)))
            self.smith_indexes.append(
                (loc, ds, exact_idx, repo_sha_idx, exact_idx_ci, normalized_idx, repo_sha_candidates)
            )

        self.rebench_index = _build_first_index_by_key(self.base_rebench_ds, "instance_id")
        self.rebench_index_ci = {k.lower(): v for k, v in self.rebench_index.items()}
        self.rebench_owner_repo_candidates: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for inst, idx in self.rebench_index.items():
            key = _rebench_owner_repo_key(inst)
            if key:
                self.rebench_owner_repo_candidates.setdefault(key, []).append((idx, inst.lower()))

        self.rebench_v2_index: dict[str, int] = {}
        self.rebench_v2_index_ci: dict[str, int] = {}
        self.rebench_v2_owner_repo_candidates: dict[tuple[str, str], list[tuple[int, str]]] = {}
        if self.base_rebench_v2_ds is not None:
            self.rebench_v2_index = _build_first_index_by_key(self.base_rebench_v2_ds, "instance_id")
            self.rebench_v2_index_ci = {k.lower(): v for k, v in self.rebench_v2_index.items()}
            for inst, idx in self.rebench_v2_index.items():
                key = _rebench_owner_repo_key(inst)
                if key:
                    self.rebench_v2_owner_repo_candidates.setdefault(key, []).append((idx, inst.lower()))

        self.r2e_index: dict[tuple[str, str], int] = {}
        if isinstance(self.base_r2e_ds, Dataset):
            commits = self.base_r2e_ds["commit_hash"]
            repos = self.base_r2e_ds["repo_name"]
            for i, (commit, repo) in enumerate(zip(commits, repos)):
                if not isinstance(commit, str) or not isinstance(repo, str):
                    continue
                key = (commit.lower(), repo.lower())
                if key not in self.r2e_index:
                    self.r2e_index[key] = i
        else:
            for i in range(len(self.base_r2e_ds)):
                row = self.base_r2e_ds[i]
                commit = row.get("commit_hash")
                repo = row.get("repo_name")
                if not isinstance(commit, str) or not isinstance(repo, str):
                    continue
                key = (commit.lower(), repo.lower())
                if key not in self.r2e_index:
                    self.r2e_index[key] = i

    def resolve_r2e(self, cf_row: dict[str, Any]) -> BaseMatch | None:
        tid = cf_row.get("trajectory_id")
        image = cf_row.get("image")
        if not isinstance(tid, str) or not isinstance(image, str):
            return None
        commit = _r2e_commit_from_cf_row(tid, image)
        if not commit:
            return None
        repo = _repo_from_image(image) or _repo_from_r2e_trajectory_id(tid)
        idx = self.r2e_index.get((commit, repo)) if repo else None
        if idx is None:
            repo = _repo_from_r2e_trajectory_id(tid)
            idx = self.r2e_index.get((commit, repo)) if repo else None
        if not repo or idx is None:
            return None
        return BaseMatch(self.base_r2e_loc, idx, dict(self.base_r2e_ds[idx]))

    def resolve_smith(self, cf_row: dict[str, Any]) -> BaseMatch | None:
        tid = cf_row.get("trajectory_id")
        if not isinstance(tid, str):
            return None
        inst = _strip_run_suffix(tid)
        for base_loc, base_ds, idx_map, _a, _b, _c, _d in self.smith_indexes:
            idx = idx_map.get(inst)
            if idx is not None:
                return BaseMatch(base_loc, idx, dict(base_ds[idx]))
        pr_inst = _smith_pr_instance_from_cf_trajectory_id(tid)
        if pr_inst:
            pr_inst_l = pr_inst.lower()
            for base_loc, base_ds, _a, _b, idx_map_ci, _c, _d in self.smith_indexes:
                idx = idx_map_ci.get(pr_inst_l)
                if idx is not None:
                    return BaseMatch(base_loc, idx, dict(base_ds[idx]))
        normalized_inst = _normalize_smith_instance_like(inst)
        for base_loc, base_ds, _a, _b, _c, normalized_idx, _d in self.smith_indexes:
            idx = normalized_idx.get(normalized_inst)
            if idx is not None:
                return BaseMatch(base_loc, idx, dict(base_ds[idx]))
        repo_sha = _smith_repo_sha_from_cf_trajectory_id(tid)
        if repo_sha:
            for base_loc, base_ds, _a, repo_sha_map, _b, _c, repo_sha_candidates in self.smith_indexes:
                idx = repo_sha_map.get(repo_sha)
                if idx is not None:
                    return BaseMatch(base_loc, idx, dict(base_ds[idx]))
                fuzzy_idx = _best_fuzzy_idx(normalized_inst, repo_sha_candidates.get(repo_sha, []))
                if fuzzy_idx is not None:
                    return BaseMatch(base_loc, idx=fuzzy_idx, row=dict(base_ds[fuzzy_idx]))
        return None

    def resolve_rebench(self, cf_row: dict[str, Any]) -> BaseMatch | None:
        tid = cf_row.get("trajectory_id")
        image = cf_row.get("image")
        if not isinstance(tid, str):
            return None
        inst = _normalize_rebench_instance_id(tid)
        idx = self.rebench_index.get(inst)
        if idx is not None:
            return BaseMatch(self.base_rebench_loc, idx, dict(self.base_rebench_ds[idx]))

        img_inst = _rebench_instance_from_image(image) if isinstance(image, str) else None
        if img_inst:
            idx = self.rebench_index_ci.get(img_inst.lower())
            if idx is not None:
                return BaseMatch(self.base_rebench_loc, idx, dict(self.base_rebench_ds[idx]))

        key = _rebench_owner_repo_key(inst)
        if key:
            idx = _best_fuzzy_idx(inst.lower(), self.rebench_owner_repo_candidates.get(key, []))
            if idx is not None:
                return BaseMatch(self.base_rebench_loc, idx, dict(self.base_rebench_ds[idx]))

        if self.base_rebench_v2_ds is None:
            return None
        idx = self.rebench_v2_index.get(inst)
        if idx is None and img_inst:
            idx = self.rebench_v2_index_ci.get(img_inst.lower())
        if idx is None:
            key = _rebench_owner_repo_key(inst)
            if key:
                idx = _best_fuzzy_idx(inst.lower(), self.rebench_v2_owner_repo_candidates.get(key, []))
        if idx is None:
            return None
        return BaseMatch(self.base_rebench_v2_loc, idx, dict(self.base_rebench_v2_ds[idx]))

    def resolver_for_split(self, cf_split: str) -> Callable[[dict[str, Any]], BaseMatch | None]:
        if cf_split == "R2E_Gym":
            return self.resolve_r2e
        if cf_split == "SWE_Smith":
            return self.resolve_smith
        if cf_split == "SWE_Rebench":
            return self.resolve_rebench
        raise ValueError(f"Unknown cf_split: {cf_split}")
