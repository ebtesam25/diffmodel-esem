"""Dump a handful of full CoderForge rows as JSON (no derived commentary)."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, load_from_disk


SAFE_NAME_TRANS = str.maketrans({"/": "__", ":": "__"})


def _require_env(name: str) -> str:
    import os

    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def safe_name(hf_id: str) -> str:
    return hf_id.translate(SAFE_NAME_TRANS)


class ParquetRowDataset:
    """Single-split `*.parquet` export with row-group-scoped reads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pf = pq.ParquetFile(path)
        meta = self._pf.metadata
        if meta is None:
            raise RuntimeError(f"Parquet file has no metadata: {path}")
        self._num_rows = meta.num_rows
        self._bounds: list[tuple[int, int, int]] = []
        offset = 0
        for rg in range(self._pf.num_row_groups):
            n = meta.row_group(rg).num_rows
            self._bounds.append((offset, offset + n, rg))
            offset += n

    def __len__(self) -> int:
        return int(self._num_rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(idx)
        lo, hi = 0, len(self._bounds)
        while lo < hi:
            mid = (lo + hi) // 2
            start, end, rg = self._bounds[mid]
            if idx < start:
                hi = mid
            elif idx >= end:
                lo = mid + 1
            else:
                table = self._pf.read_row_group(rg)
                local = idx - start
                return {name: table.column(name)[local].as_py() for name in table.column_names}
        raise RuntimeError("Parquet row group lookup failed (internal error)")

    def first_index_by_column(self, key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        offset = 0
        for rg in range(self._pf.num_row_groups):
            col = self._pf.read_row_group(rg, columns=[key]).column(key)
            vals = col.to_pylist()
            for i, val in enumerate(vals):
                if isinstance(val, str) and val and val not in out:
                    out[val] = offset + i
            offset += len(vals)
        return out


def load_split(processed_dir: Path, hf_id: str, config: str, split: str) -> Dataset | ParquetRowDataset:
    arrow_dir = processed_dir / "hf_arrow" / safe_name(hf_id) / safe_name(config) / safe_name(split)
    pq_path = processed_dir / "hf_parquet" / safe_name(hf_id) / safe_name(config) / f"{safe_name(split)}.parquet"

    if arrow_dir.exists():
        try:
            ds = load_from_disk(str(arrow_dir))
            if isinstance(ds, Dataset):
                return ds
        except TypeError as e:
            if "dataclass" not in str(e).lower():
                raise
        if pq_path.exists():
            return ParquetRowDataset(pq_path)
        raise RuntimeError(
            f"Could not load Arrow split at {arrow_dir} (datasets metadata issue) "
            f"and no Parquet fallback exists at {pq_path}"
        )

    if pq_path.exists():
        return ParquetRowDataset(pq_path)

    raise FileNotFoundError(f"Missing split export (no {arrow_dir} or {pq_path})")


def _jsonable(x: Any) -> Any:
    """Best-effort structure for `json.dumps` without truncating cell values."""

    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist") and callable(x.tolist):
        try:
            return x.tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(x)


def format_full_row(row: dict[str, Any]) -> str:
    return json.dumps(_jsonable(dict(row)), ensure_ascii=False, indent=2)


def _rough_json_size(value: Any) -> int:
    # Lightweight estimate used to avoid selecting pathological multi-MB rows.
    return len(str(value))


@dataclass(frozen=True)
class SplitLoc:
    hf_id: str
    config: str
    split: str


RUN_SUFFIX_RE = re.compile(r"_run\d+$")
SMITH_BASE_KEY_RE = re.compile(r"^[^_]+__([^.]+)\.([0-9a-fA-F]+)\.")
SMITH_CF_PR_RE = re.compile(r"^(?P<org>[^_]+)__(?P<repo>[A-Za-z0-9_]+)_(?P<sha>[0-9a-fA-F]+)_pr_(?P<pr>\d+)$")
HEX40_ANY_RE = re.compile(r"([0-9a-fA-F]{40})")
INCOMPLETE_SUFFIX_RE = re.compile(r"_incomplete_\d{8}_\d{6}$")
REBENCH_IMAGE_RE = re.compile(r"sweb\.eval\.x86_64\.(?P<owner>[^_]+)_\d+_(?P<repo>.+)-(?P<num>\d+)$", re.IGNORECASE)


def _strip_run_suffix(trajectory_id: str) -> str:
    return RUN_SUFFIX_RE.sub("", trajectory_id)


def _build_first_index_by_key(ds: Dataset | ParquetRowDataset, key: str) -> dict[str, int]:
    if isinstance(ds, ParquetRowDataset):
        return ds.first_index_by_column(key)
    if isinstance(ds, Dataset):
        out: dict[str, int] = {}
        col = ds[key]
        for i, val in enumerate(col):
            if isinstance(val, str) and val and val not in out:
                out[val] = i
        return out
    out: dict[str, int] = {}
    for i in range(len(ds)):
        val = ds[i].get(key)
        if isinstance(val, str) and val and val not in out:
            out[val] = i
    return out


def _smith_repo_sha_from_cf_trajectory_id(trajectory_id: str) -> str | None:
    core = _strip_run_suffix(trajectory_id)
    if "__" not in core:
        return None
    _, rest = core.split("__", 1)
    if "__" not in rest:
        return None
    head, _tail = rest.split("__", 1)
    bits = head.split("_")
    if len(bits) < 2:
        return None
    repo, sha = bits[0].lower(), bits[1].lower()
    if not repo or not sha:
        return None
    return f"{repo}.{sha}"


def _smith_pr_instance_from_cf_trajectory_id(trajectory_id: str) -> str | None:
    """
    Map CF token like `HIPS__autograd_ac044f0d_pr_672_run4`
    to canonical base instance id `HIPS__autograd.ac044f0d.pr_672`.
    """
    core = _strip_run_suffix(trajectory_id)
    core = INCOMPLETE_SUFFIX_RE.sub("", core)
    m = SMITH_CF_PR_RE.match(core)
    if not m:
        return None
    return f"{m.group('org')}__{m.group('repo')}.{m.group('sha')}.pr_{m.group('pr')}"


def _smith_repo_sha_from_base_instance_id(instance_id: str) -> str | None:
    m = SMITH_BASE_KEY_RE.match(instance_id)
    if not m:
        return None
    return f"{m.group(1).lower()}.{m.group(2).lower()}"


def _repo_from_image(image: str) -> str | None:
    # qingyangwu/aiohttp_final:<sha> -> aiohttp
    name = image.split(":", 1)[0]
    repo = name.split("/")[-1].lower()
    if repo.endswith("_final"):
        repo = repo[: -len("_final")]
    return repo or None


def _repo_from_r2e_trajectory_id(trajectory_id: str) -> str | None:
    """
    Recover repo from malformed R2E identifiers such as:
    `qingyangwu_coveragepy_final_<sha>_incomplete_..._run2` -> `coveragepy`.
    """
    core = _strip_run_suffix(trajectory_id)
    m = HEX40_ANY_RE.search(core)
    if not m:
        return None
    prefix = core[: m.start()].rstrip("_")
    if not prefix:
        return None
    bits = prefix.split("_")
    if len(bits) >= 2 and bits[0].lower() in {"qingyangwu", "namanjain12"}:
        bits = bits[1:]
    if bits and bits[-1].lower() == "final":
        bits = bits[:-1]
    repo = "_".join(bits).strip("_").lower()
    return repo or None


def _r2e_commit_from_cf_row(trajectory_id: str, image: str) -> str | None:
    core = _strip_run_suffix(trajectory_id)
    m = HEX40_ANY_RE.search(core)
    if not m:
        m = HEX40_ANY_RE.search(image)
    if not m:
        return None
    return m.group(1).lower()


def _normalize_rebench_instance_id(trajectory_id: str) -> str:
    core = _strip_run_suffix(trajectory_id)
    return INCOMPLETE_SUFFIX_RE.sub("", core)


def _normalize_smith_instance_like(value: str) -> str:
    s = value.casefold()
    s = INCOMPLETE_SUFFIX_RE.sub("", s)
    s = RUN_SUFFIX_RE.sub("", s)
    # Canonicalize separators so cf/base format differences do not block matching.
    s = s.replace("__", "_")
    s = s.replace(".", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _rebench_owner_repo_key(instance_id: str) -> tuple[str, str] | None:
    if "__" not in instance_id or "-" not in instance_id:
        return None
    owner, rest = instance_id.split("__", 1)
    repo, _num = rest.rsplit("-", 1)
    if not owner or not repo:
        return None
    return (owner.lower(), repo.lower())


def _rebench_instance_from_image(image: str) -> str | None:
    m = REBENCH_IMAGE_RE.search(image.strip())
    if not m:
        return None
    owner = m.group("owner")
    repo = m.group("repo")
    num = m.group("num")
    return f"{owner}__{repo}-{num}"


def _best_fuzzy_idx(query: str, candidates: list[tuple[int, str]], *, min_ratio: float = 0.97) -> int | None:
    query_norm = query.casefold()
    best_idx: int | None = None
    best_ratio = min_ratio
    for idx, candidate in candidates:
        ratio = difflib.SequenceMatcher(a=query_norm, b=candidate.casefold()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx


