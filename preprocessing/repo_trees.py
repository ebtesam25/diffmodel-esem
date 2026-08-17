"""Discover GitHub repos in local Arrow datasets and fetch recursive trees."""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import load_from_disk


REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
HTTP_REPO_PATTERN = re.compile(
    r"github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?$",
    flags=re.IGNORECASE,
)
GITHUB_REPO_IN_TEXT_PATTERN = re.compile(
    r"(?:https?://|git@)github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:\.git)?(?:[/?#:]|$)",
    flags=re.IGNORECASE,
)
SMITH_INSTANCE_REPO_PATTERN = re.compile(
    r"^(?:swesmith/)?(?P<owner>[A-Za-z0-9_.-]+)__(?P<repo>[A-Za-z0-9_.-]+)\.(?P<sha>[0-9a-fA-F]{7,40})$"
)
SWEB_ENCODED_REPO_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+/(?:(?:swesmith|sweb\.eval|swerebench)\.x86_64\.)"
    r"(?P<owner>[A-Za-z0-9_.-]+)_\d+_(?P<repo>[A-Za-z0-9_.-]+)\.[0-9a-fA-F]{7,40}$",
    flags=re.IGNORECASE,
)
REPO_KEYS = {
    "repo",
    "repository",
    "repo_name",
    "repo_full_name",
    "github_repo",
    "full_name",
}

EVAL_TRAJECTORIES_DATASET_ID = (
    "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
)


def _safe_hf_dir_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__").replace(":", "__")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _normalize_repo(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None

    if REPO_PATTERN.fullmatch(value):
        return value

    if "github.com" in value.lower():
        m = HTTP_REPO_PATTERN.search(value)
        if m:
            return f"{m.group('owner')}/{m.group('repo')}"
    return None


def _repos_from_text(value: str, *, key_hint: str | None = None) -> set[str]:
    out: set[str] = set()

    stripped = value.strip()

    # Decode synthetic benchmark IDs into canonical owner/repo when present.
    m_encoded = SWEB_ENCODED_REPO_PATTERN.fullmatch(stripped)
    if m_encoded:
        out.add(f"{m_encoded.group('owner')}/{m_encoded.group('repo')}")

    normalized = _normalize_repo(value)
    if normalized:
        owner, repo = normalized.split("/", 1)
        # Avoid treating synthetic benchmark IDs as literal GitHub repos.
        if not (
            repo.startswith("swesmith.x86_64.")
            or repo.startswith("sweb.eval.x86_64.")
            or repo.startswith("swerebench.x86_64.")
        ):
            out.add(f"{owner}/{repo}")

    if key_hint in {"messages", "message", "text", "content", "prompt", "problem_statement", "description", "body"}:
        for m in GITHUB_REPO_IN_TEXT_PATTERN.finditer(value):
            out.add(f"{m.group('owner')}/{m.group('repo')}")

    # SWE-Smith style IDs (owner__repo.<sha>) appear in structured IDs and metadata fields.
    if key_hint in {"instance_id", "id", "task_id", "trajectory_id", "base_repo"}:
        m = SMITH_INSTANCE_REPO_PATTERN.fullmatch(stripped)
        if m:
            out.add(f"{m.group('owner')}/{m.group('repo')}")

    return out


def _collect_repos_from_obj(obj: Any, out: set[str], depth: int = 0) -> None:
    if depth > 4:
        return

    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, str):
                key_l = key.lower()
                for repo in _repos_from_text(val, key_hint=key_l):
                    out.add(repo)

                # Many datasets store structured fields as JSON strings (`messages`, `ds`, `tools`).
                stripped = val.strip()
                if stripped and stripped[0] in "[{" and stripped[-1] in "]}":
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        parsed = None
                    if parsed is not None:
                        _collect_repos_from_obj(parsed, out, depth + 1)
            elif isinstance(val, (dict, list)):
                _collect_repos_from_obj(val, out, depth + 1)
        return

    if isinstance(obj, list):
        for item in obj:
            _collect_repos_from_obj(item, out, depth + 1)


def discover_split_dirs(arrow_root: Path) -> list[Path]:
    split_dirs: list[Path] = []
    for dataset_dir in arrow_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
            for split_dir in config_dir.iterdir():
                if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
                    split_dirs.append(split_dir)
    return split_dirs


def extract_repos_from_local_arrow(arrow_root: Path, row_limit_per_split: int | None) -> set[str]:
    repos: set[str] = set()
    split_dirs = discover_split_dirs(arrow_root)
    print(f"Scanning {len(split_dirs)} local split directories for repository identifiers.")

    for split_dir in split_dirs:
        ds = load_from_disk(str(split_dir))
        n_rows = len(ds) if row_limit_per_split is None else min(len(ds), row_limit_per_split)
        if n_rows == 0:
            continue
        for idx in range(n_rows):
            _collect_repos_from_obj(ds[idx], repos)
        print(f"  - scanned {split_dir.name} ({n_rows} rows), repos_found_so_far={len(repos)}")
    return repos


def github_json(url: str, token: str, timeout_s: int = 120, *, _max_retries: int = 5) -> dict[str, Any]:
    req = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    for attempt in range(_max_retries + 1):
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as err:
            # Retry on rate-limit responses (GitHub uses 403 for secondary limits, 429 for primary)
            if err.code in (403, 429) and attempt < _max_retries:
                retry_after = err.headers.get("Retry-After")
                if retry_after:
                    # GitHub explicitly told us how long to wait
                    wait = float(retry_after) + 1.0
                else:
                    reset_ts = err.headers.get("X-RateLimit-Reset")
                    if reset_ts:
                        # Primary rate limit: wait until the reset timestamp
                        wait = max(float(reset_ts) - time.time() + 1.0, 1.0)
                    else:
                        # Secondary rate limit or unknown: exponential backoff
                        wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 s
                print(f"[rate-limit] {err.code} on {url} (attempt {attempt + 1}/{_max_retries}), waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("github_json: exceeded max retries")


def get_remote_default_branch_and_tree_sha(repo: str, token: str) -> tuple[str, str]:
    repo_meta = github_json(f"https://api.github.com/repos/{repo}", token)
    default_branch = repo_meta.get("default_branch", "main")
    branch_meta = github_json(f"https://api.github.com/repos/{repo}/branches/{default_branch}", token)
    commit_sha = branch_meta["commit"]["sha"]
    commit_meta = github_json(f"https://api.github.com/repos/{repo}/git/commits/{commit_sha}", token)
    return default_branch, commit_meta["tree"]["sha"]


def fetch_tree_for_repo(repo: str, token: str) -> dict[str, Any]:
    default_branch, _tip_tree_sha = get_remote_default_branch_and_tree_sha(repo, token)
    tree = github_json(f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1", token)
    return {
        "repo": repo,
        "kind": "branch_tip",
        "default_branch": default_branch,
        "tree_sha": tree.get("sha"),
        "truncated": tree.get("truncated"),
        "tree": tree.get("tree", []),
        "fetched_at_unix": int(time.time()),
    }


def fetch_tree_for_commit(repo: str, commit_sha: str, token: str) -> dict[str, Any]:
    """Recursive tree at a **specific commit** (SWE-bench `base_commit`, Docker tag SHA, …)."""

    commit_sha = commit_sha.strip()
    if len(commit_sha) < 7:
        raise ValueError(f"commit SHA too short for {repo!r}")

    commit_meta = github_json(f"https://api.github.com/repos/{repo}/commits/{commit_sha}", token)
    full_sha = commit_meta.get("sha") or commit_sha
    tree_sha = commit_meta["commit"]["tree"]["sha"]
    tree = github_json(f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}?recursive=1", token)
    return {
        "repo": repo,
        "kind": "commit",
        "commit_sha": full_sha,
        "root_tree_sha": tree_sha,
        "tree_sha": tree.get("sha"),
        "truncated": tree.get("truncated"),
        "tree": tree.get("tree", []),
        "fetched_at_unix": int(time.time()),
    }


def repo_to_filename(repo: str) -> str:
    return repo.replace("/", "__") + ".json.gz"


def write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    tmp_path.replace(path)


def read_existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None

    required = {"repo", "default_branch", "tree_sha", "tree"}
    if not required.issubset(payload.keys()):
        return None
    if not isinstance(payload.get("tree"), list):
        return None
    if not payload.get("tree_sha"):
        return None
    return payload


def commit_tree_filename(repo: str, commit_sha: str) -> str:
    c = commit_sha.strip().lower()
    return f"{repo.replace('/', '__')}__{c}.json.gz"


def _sha_compatible(stored: str, requested: str) -> bool:
    a, b = (stored or "").strip().lower(), (requested or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a))


def find_cached_commit_tree(commit_dir: Path, repo: str, commit_sha: str) -> Path | None:
    """Resolve a cache path when `base_commit` in data is abbreviated or matches canonical filename."""

    direct = commit_dir / commit_tree_filename(repo, commit_sha)
    if read_existing_commit_tree(direct):
        return direct
    prefix = f"{repo.replace('/', '__')}__"
    for candidate in sorted(commit_dir.glob(f"{prefix}*.json.gz")):
        payload = read_existing_commit_tree(candidate)
        if (
            payload
            and payload.get("repo") == repo
            and _sha_compatible(str(payload.get("commit_sha", "")), commit_sha)
        ):
            return candidate
    return None


def read_existing_commit_tree(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    required = {"repo", "commit_sha", "root_tree_sha", "tree"}
    if not required.issubset(payload.keys()):
        return None
    if payload.get("kind") != "commit":
        return None
    if not isinstance(payload.get("tree"), list):
        return None
    return payload


def discover_eval_repo_commit_pairs(
    processed_dir: Path,
    *,
    row_limit: int | None,
) -> set[tuple[str, str]]:
    """`(owner/repo, base_commit)` from eval trajectories `ds` JSON."""

    pairs: set[tuple[str, str]] = set()
    arrow_dir = (
        processed_dir / "hf_arrow" / _safe_hf_dir_name(EVAL_TRAJECTORIES_DATASET_ID) / "trajectory" / "train"
    )
    if not arrow_dir.is_dir() or not (arrow_dir / "dataset_info.json").exists():
        return pairs

    ds = load_from_disk(str(arrow_dir))
    n = len(ds) if row_limit is None else min(len(ds), row_limit)
    for i in range(n):
        raw = ds[i].get("ds", "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        repo = obj.get("repo")
        commit = obj.get("base_commit")
        if isinstance(repo, str) and isinstance(commit, str) and repo.strip() and commit.strip():
            pairs.add((repo.strip(), commit.strip().lower()))
    return pairs


def should_refresh_commit_tree(
    repo: str,
    commit_sha: str,
    commit_dir: Path,
    force_refresh: bool,
) -> tuple[bool, str]:
    if force_refresh:
        return True, "force-refresh"
    cached = find_cached_commit_tree(commit_dir, repo, commit_sha)
    if cached is None:
        return True, "no-valid-local-file"
    existing = read_existing_commit_tree(cached)
    if existing is None:
        return True, "no-valid-local-file"
    if existing.get("repo") != repo:
        return True, "repo-mismatch"
    if not _sha_compatible(str(existing.get("commit_sha", "")), commit_sha):
        return True, "commit-mismatch"
    return False, "local-valid-skip"


def should_refresh_repo(
    repo: str,
    output_path: Path,
    token: str,
    validate_remote_sha: bool,
    force_refresh: bool,
) -> tuple[bool, str]:
    if force_refresh:
        return True, "force-refresh"

    existing = read_existing_payload(output_path)
    if existing is None:
        return True, "no-valid-local-file"

    if not validate_remote_sha:
        return False, "local-valid-skip"

    default_branch, remote_tree_sha = get_remote_default_branch_and_tree_sha(repo, token)
    if existing.get("default_branch") != default_branch:
        return True, "default-branch-changed"
    if existing.get("tree_sha") != remote_tree_sha:
        return True, "tree-sha-changed"
    return False, "tree-sha-match-skip"


def run_pull_trees(
    *,
    workers: int,
    row_limit_per_split: int | None,
    validate_remote_sha: bool,
    force_refresh: bool,
    fetch_eval_commit_trees: bool = False,
    max_commit_trees: int = 10_000,
) -> int:
    processed_dir = Path(_require_env("CODERFORGE_DATA_PROCESSED_DIR")).expanduser().resolve()
    external_dir = Path(_require_env("CODERFORGE_DATA_EXTERNAL_DIR")).expanduser().resolve()
    token = _require_env("GITHUB_TOKEN")
    arrow_root = processed_dir / "hf_arrow"
    output_root = external_dir / "github_trees"
    output_root.mkdir(parents=True, exist_ok=True)

    if not arrow_root.exists():
        raise RuntimeError(f"Local Arrow root does not exist: {arrow_root}")

    repos = sorted(extract_repos_from_local_arrow(arrow_root, row_limit_per_split))
    if not repos:
        print("No repositories found in local datasets.")
        return 1

    print(
        f"Discovered {len(repos)} unique repositories. "
        f"workers={workers}, validate_remote_sha={validate_remote_sha}, force_refresh={force_refresh}"
    )

    failures: list[dict[str, str]] = []
    successes = 0
    skipped = 0
    to_fetch: list[str] = []
    for repo in repos:
        output_path = output_root / repo_to_filename(repo)
        try:
            refresh, reason = should_refresh_repo(
                repo=repo,
                output_path=output_path,
                token=token,
                validate_remote_sha=validate_remote_sha,
                force_refresh=force_refresh,
            )
            if refresh:
                to_fetch.append(repo)
            else:
                skipped += 1
            print(f"[PLAN] {repo}: {reason}")
        except Exception as err:  # noqa: BLE001
            to_fetch.append(repo)
            print(f"[PLAN] {repo}: validation-error ({err}), scheduling refresh")

    print(f"Fetch plan: {len(to_fetch)} to fetch, {skipped} skipped.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetch_tree_for_repo, repo, token): repo for repo in to_fetch}
        for future in as_completed(future_map):
            repo = future_map[future]
            try:
                payload = future.result()
                output_path = output_root / repo_to_filename(repo)
                write_gzip_json(output_path, payload)
                successes += 1
                print(f"[OK] {repo} -> {output_path.name} ({len(payload.get('tree', []))} entries)")
            except HTTPError as err:
                failures.append({"repo": repo, "error": f"HTTP {err.code}"})
                print(f"[FAIL] {repo}: HTTP {err.code}")
            except URLError as err:
                failures.append({"repo": repo, "error": f"URL error: {err.reason}"})
                print(f"[FAIL] {repo}: URL error: {err.reason}")
            except Exception as err:  # noqa: BLE001
                failures.append({"repo": repo, "error": str(err)})
                print(f"[FAIL] {repo}: {err}")

    manifest = {
        "repo_count": len(repos),
        "fetch_planned": len(to_fetch),
        "skipped_count": skipped,
        "success_count": successes,
        "failure_count": len(failures),
        "output_root": str(output_root),
        "validate_remote_sha": validate_remote_sha,
        "force_refresh": force_refresh,
        "failures": failures,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if not fetch_eval_commit_trees:
        return 0 if not failures else 1

    # --- Per-commit recursive trees (eval `ds.repo` + `ds.base_commit`) ---
    commit_dir = output_root / "commits"
    commit_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_eval_repo_commit_pairs(processed_dir, row_limit=row_limit_per_split)
    pair_list = sorted(pairs)
    if len(pair_list) > max_commit_trees:
        print(f"Capping eval (repo, commit) pairs from {len(pair_list)} to {max_commit_trees}.")
        pair_list = pair_list[:max_commit_trees]

    print(f"Eval commit trees: {len(pair_list)} unique (repo, base_commit) pairs after cap.")

    commit_failures: list[dict[str, str]] = []
    commit_success = 0
    commit_skipped = 0
    to_fetch_commits: list[tuple[str, str]] = []

    for repo, commit in pair_list:
        if not REPO_PATTERN.fullmatch(repo):
            continue
        try:
            refresh, reason = should_refresh_commit_tree(repo, commit, commit_dir, force_refresh)
            if refresh:
                to_fetch_commits.append((repo, commit))
            else:
                commit_skipped += 1
            short = commit[:12] if len(commit) > 12 else commit
            print(f"[COMMIT PLAN] {repo}@{short}…: {reason}")
        except Exception as err:  # noqa: BLE001
            to_fetch_commits.append((repo, commit))
            print(f"[COMMIT PLAN] {repo}@{commit[:12]}…: validation-error ({err})")

    print(
        f"Commit-tree fetch plan: {len(to_fetch_commits)} to fetch, {commit_skipped} skipped "
        f"(cache hits)."
    )

    def _fetch_pair(pair: tuple[str, str]) -> tuple[str, str, dict[str, Any]]:
        r, c = pair
        payload = fetch_tree_for_commit(r, c, token)
        return r, c, payload

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_fetch_pair, p): p for p in to_fetch_commits}
        for future in as_completed(future_map):
            repo, commit = future_map[future]
            try:
                _, _, payload = future.result()
                full_sha = str(payload.get("commit_sha", commit))
                out_path = commit_dir / commit_tree_filename(repo, full_sha)
                write_gzip_json(out_path, payload)
                commit_success += 1
                print(
                    f"[COMMIT OK] {repo}@{full_sha[:12]}… "
                    f"-> {out_path.name} ({len(payload.get('tree', []))} entries, "
                    f"truncated={payload.get('truncated')})"
                )
            except HTTPError as err:
                commit_failures.append({"repo": repo, "commit": commit, "error": f"HTTP {err.code}"})
                print(f"[COMMIT FAIL] {repo}@{commit[:12]}…: HTTP {err.code}")
            except URLError as err:
                commit_failures.append({"repo": repo, "commit": commit, "error": f"URL: {err.reason}"})
                print(f"[COMMIT FAIL] {repo}@{commit[:12]}…: {err.reason}")
            except Exception as err:  # noqa: BLE001
                commit_failures.append({"repo": repo, "commit": commit, "error": str(err)})
                print(f"[COMMIT FAIL] {repo}@{commit[:12]}…: {err}")

    commit_manifest = {
        "pairs_discovered_total": len(pairs),
        "pairs_in_run": len(pair_list),
        "fetch_planned": len(to_fetch_commits),
        "skipped_count": commit_skipped,
        "success_count": commit_success,
        "failure_count": len(commit_failures),
        "commit_dir": str(commit_dir),
        "failures": commit_failures,
    }
    (output_root / "manifest_commits.json").write_text(
        json.dumps(commit_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote commit manifest: {output_root / 'manifest_commits.json'}")

    return 0 if not failures and not commit_failures else 1
