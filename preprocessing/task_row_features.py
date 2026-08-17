"""Row-level task/problem features (prompt, patch, tree, linking).

Rows may use either nested keys (`trajectory_row`, `base_row`, `github_tree`) or
flat aliases seen in mapping parquet (`trajectory_id`, `messages`, `base_patch`).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import tiktoken

# --- Tokenizer (cl100k_base: deterministic, widely used for NLP features) ---
_ENC = tiktoken.get_encoding("cl100k_base")

_ISSUE_BLOCK_RE = re.compile(
    r"<issue_description>\s*(.*?)\s*</issue_description>",
    re.IGNORECASE | re.DOTALL,
)

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")

_ERROR_KEYWORD_RE = re.compile(
    r"(Error|Exception|Traceback|failed|raises)",
    re.IGNORECASE,
)

# Paths ending in common code/doc extensions (repo-relative or absolute-style).
_FILE_PATH_MENTION_RE = re.compile(
    r"(?:(?:/[\w.\-]+)+|[\w.\-]+(?:/[\w.\-]+)+)\."
    r"(?:py|js|ts|java|go|rb|php|c|cpp|h|hpp|md|txt|json|yaml|yml|toml|ini|cfg)\b",
    re.IGNORECASE,
)

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

_FUNC_CALL_LIKE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(")

_PHASE_RE = re.compile(r"\bphase\s+\d+\b", re.IGNORECASE)

_FENCE_LINE_RE = re.compile(r"^[ \t]*```")

_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

_DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_CONTROL_FLOW_RE = re.compile(r"^\s*(?:if|elif|for|while|match|case|try|except|with)\b")
_SYNTAX_NODE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|if|elif|for|while|match|case|try|except|with|return|raise|import|from|lambda)\b"
)

_TASK_ID_RUN_SUFFIX_RE = re.compile(r"_run\d+$")

_PROMPT_PARSE_NLP: Any | None = None
_PROMPT_PARSE_LOAD_ATTEMPTED = False
_PROMPT_PARSE_LAST_ERROR: str | None = None


def constituency_nlp_last_error() -> str | None:
    """Last failure reason from loading the prompt-parse spaCy model (alias for compatibility)."""
    return _PROMPT_PARSE_LAST_ERROR


def _truncate_text_for_prompt_nlp(text: str) -> str:
    """Return the prompt unchanged.

    The parser now always receives the full prompt text. The helper is kept for
    compatibility with older call sites and to avoid forcing broader refactors.
    """
    return text


def _truncate_text_for_benepar_encoder(text: str) -> str:
    """Backward-compatible no-op for :func:`_truncate_text_for_prompt_nlp`."""
    return _truncate_text_for_prompt_nlp(text)


def _spacy_model_name_candidates() -> list[str]:
    raw = os.getenv("CODERFORGE_SPACY_MODELS", "").strip()
    if raw:
        names = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        primary = os.getenv("CODERFORGE_SPACY_MODEL", "en_core_web_sm").strip()
        names = [primary, "en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

__all__ = [
    "constituency_nlp_last_error",
    "extract_issue_text",
    "parse_patch_changed_files",
    "extract_features",
    "extract_features_many",
    "prompt_parse_ambiguity_features_many",
    "save_features_parquet",
    "task_id_from_trajectory_id",
]


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _as_json_messages(messages_field: Any) -> list[Any]:
    if _is_missing(messages_field):
        return []
    if isinstance(messages_field, list):
        return messages_field
    if isinstance(messages_field, str):
        try:
            data = json.loads(messages_field)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []
    return []


def extract_issue_text(messages_json_str: Any) -> str:
    """Return main issue text from chat messages JSON (or list)."""
    messages = _as_json_messages(messages_json_str)
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        m = _ISSUE_BLOCK_RE.search(content)
        if m:
            return m.group(1).strip()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _issue_block_found(messages_field: Any) -> int:
    messages = _as_json_messages(messages_field)
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and _ISSUE_BLOCK_RE.search(content):
            return 1
    return 0


def task_id_from_trajectory_id(trajectory_id: Any) -> str:
    """Strip trailing ``_run<digits>`` from a trajectory id."""
    if _is_missing(trajectory_id):
        return ""
    tid = str(trajectory_id).strip()
    return _TASK_ID_RUN_SUFFIX_RE.sub("", tid)


def _strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", text)


def _count_inline_backtick_spans(text: str) -> int:
    body = _strip_fenced_code_blocks(text)
    return len(re.findall(r"`([^`\n]+)`", body))


def _count_fence_lines(text: str) -> int:
    return len(_FENCE_LINE_RE.findall(text))


def _prompt_token_count_simple(text: str) -> int:
    """Length of cl100k token ids for ``text`` (``tiktoken`` ``cl100k_base``).
    """
    if not text:
        return 0
    return len(_ENC.encode(text))


def _parse_diff_git_paths(line: str) -> str | None:
    if not line.startswith("diff --git "):
        return None
    rest = line[len("diff --git ") :].strip()
    if rest.startswith('"'):
        return None  # quoted paths: skip without full git quote parser
    if " b/" not in rest:
        return None
    left, right = rest.split(" b/", 1)
    if not left.startswith("a/"):
        return None
    path = left[2:]
    if right != path:
        return None
    return path


def parse_patch_changed_files(patch_str: Any) -> list[str]:
    """List file paths from ``diff --git a/<path> b/<path>`` lines (unquoted)."""
    if _is_missing(patch_str):
        return []
    s = str(patch_str)
    if s.lower() == "nan":
        return []
    out: list[str] = []
    for line in s.splitlines():
        p = _parse_diff_git_paths(line)
        if p is not None:
            out.append(p)
    return out


def _patch_line_stats(patch_str: str) -> dict[str, Any]:
    files_order: list[str] = []
    num_hunks = 0
    lines_added = 0
    lines_deleted = 0
    current: str | None = None
    per_file: dict[str, int] = defaultdict(int)
    function_keys_touched: set[tuple[str, str]] = set()
    class_keys_touched: set[tuple[str, str]] = set()
    function_changed_line_counts: dict[tuple[str, str], int] = defaultdict(int)
    current_function_key: tuple[str, str] | None = None
    syntax_node_changes = 0
    control_flow_added = 0
    control_flow_deleted = 0
    hunk_modified_spans: list[int] = []
    hunk_new_starts_by_file: dict[str, list[int]] = defaultdict(list)

    for line in patch_str.splitlines():
        if line.startswith("diff --git "):
            p = _parse_diff_git_paths(line)
            if p is not None:
                current = p
                files_order.append(p)
            else:
                current = None
            current_function_key = None
        elif line.startswith("@@"):
            num_hunks += 1
            m_h = _HUNK_HEADER_RE.match(line)
            if m_h:
                old_span = int(m_h.group(2) or "1")
                new_start = int(m_h.group(3))
                new_span = int(m_h.group(4) or "1")
                hunk_modified_spans.append(max(old_span, new_span))
                if current is not None:
                    hunk_new_starts_by_file[current].append(new_start)
            current_function_key = None
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
            if current is not None:
                per_file[current] += 1
            body = line[1:]
            m_def = _DEF_RE.match(body)
            if m_def:
                name = m_def.group(1)
                if current is not None:
                    key = (current, name)
                    function_keys_touched.add(key)
                    current_function_key = key
            m_class = _CLASS_RE.match(body)
            if m_class:
                name = m_class.group(1)
                if current is not None:
                    class_keys_touched.add((current, name))
            if _SYNTAX_NODE_RE.match(body):
                syntax_node_changes += 1
            if _CONTROL_FLOW_RE.match(body):
                control_flow_added += 1
            if current_function_key is not None and body.strip():
                function_changed_line_counts[current_function_key] += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_deleted += 1
            if current is not None:
                per_file[current] += 1
            body = line[1:]
            m_def = _DEF_RE.match(body)
            if m_def:
                name = m_def.group(1)
                if current is not None:
                    key = (current, name)
                    function_keys_touched.add(key)
                    current_function_key = key
            m_class = _CLASS_RE.match(body)
            if m_class:
                name = m_class.group(1)
                if current is not None:
                    class_keys_touched.add((current, name))
            if _SYNTAX_NODE_RE.match(body):
                syntax_node_changes += 1
            if _CONTROL_FLOW_RE.match(body):
                control_flow_deleted += 1
            if current_function_key is not None and body.strip():
                function_changed_line_counts[current_function_key] += 1

    files = files_order
    n_files = len(files)
    unique_files = list(dict.fromkeys(files))

    def parent_dir(p: str) -> str:
        p = p.replace("\\", "/").strip("/")
        if "/" not in p:
            return "."
        return str(Path(p).parent).replace("\\", "/") or "."

    unique_dirs = {parent_dir(f) for f in unique_files}
    depths = [p.count("/") for p in unique_files]
    max_depth = max(depths) if depths else 0
    mean_depth = sum(depths) / len(depths) if depths else 0.0

    exts = set()
    for f in unique_files:
        suf = Path(f).suffix.lower().lstrip(".")
        exts.add(suf if suf else "")

    def is_test_path(p: str) -> bool:
        norm = p.replace("\\", "/")
        base = Path(norm).name
        return base.startswith("test_") or "/tests/" in norm

    py_count = sum(1 for f in unique_files if f.lower().endswith(".py"))
    test_count = sum(1 for f in unique_files if is_test_path(f))
    src_count = sum(1 for f in unique_files if not is_test_path(f))

    entropy = 0.0
    if len(unique_files) <= 1:
        entropy = 0.0
    else:
        total_changed = sum(per_file.values())
        if total_changed > 0:
            h = 0.0
            for c in per_file.values():
                if c <= 0:
                    continue
                p = c / total_changed
                h -= p * math.log2(p)
            entropy = h

    primary = ""
    primary_depth = 0
    if per_file:
        best_c = max(per_file.values())
        candidates = sorted([f for f, c in per_file.items() if c == best_c])
        primary = candidates[0]
        primary_depth = primary.count("/")

    avg_function_size_modified = 0.0
    if function_changed_line_counts:
        avg_function_size_modified = sum(function_changed_line_counts.values()) / len(function_changed_line_counts)

    control_flow_delta = control_flow_added - control_flow_deleted

    # Mean gap between consecutive hunk starts within the same file.
    hunk_gaps: list[float] = []
    for starts in hunk_new_starts_by_file.values():
        if len(starts) <= 1:
            continue
        for i in range(1, len(starts)):
            gap = starts[i] - starts[i - 1]
            hunk_gaps.append(float(gap if gap > 0 else 0))
    patch_hunk_gap_mean = (sum(hunk_gaps) / len(hunk_gaps)) if hunk_gaps else 0.0

    patch_edit_fragmentation = (num_hunks / n_files) if n_files > 0 else 0.0

    patch_churn_gini_over_files = 0.0
    if unique_files:
        churn_values = [float(per_file.get(f, 0)) for f in unique_files]
        churn_total = sum(churn_values)
        if churn_total > 0 and len(churn_values) > 1:
            x_sorted = sorted(churn_values)
            n = len(x_sorted)
            weighted_sum = sum((i + 1) * x for i, x in enumerate(x_sorted))
            gini = (2.0 * weighted_sum) / (n * churn_total) - (n + 1) / n
            patch_churn_gini_over_files = max(0.0, min(1.0, gini))

    return {
        "patch_num_files_changed": n_files,
        "patch_num_hunks": num_hunks,
        "patch_lines_added": lines_added,
        "patch_lines_deleted": lines_deleted,
        "patch_num_unique_dirs_changed": len(unique_dirs),
        "patch_max_file_depth": max_depth,
        "patch_mean_file_depth": mean_depth,
        "patch_extensions_changed_count": len(exts),
        "patch_python_files_changed": py_count,
        "patch_test_files_changed": test_count,
        "patch_src_files_changed": src_count,
        "patch_entropy_over_files": entropy,
        "patch_churn_gini_over_files": patch_churn_gini_over_files,
        "patch_edit_fragmentation": patch_edit_fragmentation,
        "patch_primary_file_path": primary,
        "patch_primary_file_depth": primary_depth,
        "ast_num_functions_modified": len(function_keys_touched),
        "ast_num_classes_modified": len(class_keys_touched),
        "ast_syntax_node_change_count": syntax_node_changes,
        "ast_edit_distance_proxy": float(
            len(function_keys_touched) + len(class_keys_touched) + syntax_node_changes
        ),
        "cfg_conditionals_added": control_flow_added,
        "cfg_conditionals_deleted": control_flow_deleted,
        "cfg_conditionals_delta": control_flow_delta,
        "cfg_conditionals_total_changed": control_flow_added + control_flow_deleted,
        "func_num_functions_touched": len(function_keys_touched),
        "func_avg_changed_lines_per_touched_function": avg_function_size_modified,
        "func_avg_hunk_span_modified": (
            (sum(hunk_modified_spans) / len(hunk_modified_spans)) if hunk_modified_spans else 0.0
        ),
        "patch_hunk_gap_mean": patch_hunk_gap_mean,
        "_per_file_changed": dict(per_file),
        "_unique_files": unique_files,
    }


def _coerce_github_tree(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def _tree_entries(github_tree: dict[str, Any] | None) -> tuple[list[dict[str, Any]], bool]:
    if not github_tree or "tree" not in github_tree:
        return [], False
    t = github_tree["tree"]
    if not isinstance(t, list):
        return [], False
    entries: list[dict[str, Any]] = []
    for e in t:
        if isinstance(e, dict):
            entries.append(e)
    return entries, True


def _first_path_segment(path: str) -> str:
    path = path.replace("\\", "/").strip("/")
    if "/" in path:
        return path.split("/", 1)[0]
    return path


def _tree_features(
    github_tree: dict[str, Any] | None,
    changed_files: list[str],
) -> dict[str, Any]:
    entries, present = _tree_entries(github_tree)
    truncated_raw = bool(github_tree.get("truncated")) if github_tree else False

    blobs = [e for e in entries if e.get("type") == "blob"]
    trees = [e for e in entries if e.get("type") == "tree"]

    blob_paths: list[str] = []
    blob_path_set: set[str] = set()
    for e in blobs:
        p = e.get("path")
        if isinstance(p, str) and p:
            blob_paths.append(p)
            blob_path_set.add(p)

    def depth_of(path: str) -> int:
        return path.replace("\\", "/").count("/")

    all_paths = [e.get("path") for e in entries if isinstance(e.get("path"), str)]
    max_d = max((depth_of(p) for p in all_paths), default=0)
    blob_depths = [depth_of(p) for p in blob_paths]
    mean_blob_depth = sum(blob_depths) / len(blob_depths) if blob_depths else 0.0

    py_blobs = [p for p in blob_paths if p.lower().endswith(".py")]

    def is_test_blob(p: str) -> bool:
        norm = p.replace("\\", "/")
        return Path(norm).name.startswith("test_") or "/tests/" in norm

    test_blobs = [p for p in blob_paths if is_test_blob(p)]

    sizes: list[int] = []
    for e in blobs:
        sz = e.get("size")
        if isinstance(sz, (int, float)) and not (isinstance(sz, float) and math.isnan(sz)):
            sizes.append(int(sz))

    total_size = sum(sizes)
    mean_size = total_size / len(sizes) if sizes else 0.0

    top_segments = {_first_path_segment(p) for p in all_paths if p}

    def dir_top_present(name: str) -> bool:
        base = name.rstrip("/")
        pre = base + "/"
        return any(
            (pn.replace("\\", "/") == base) or pn.replace("\\", "/").startswith(pre)
            for pn in all_paths
        )

    basenames: dict[str, list[str]] = defaultdict(list)
    for p in blob_paths:
        basenames[Path(p.replace("\\", "/")).name].append(p)
    collision_groups = sum(1 for _bn, ps in basenames.items() if len(ps) > 1)

    # siblings: blobs per parent directory
    parent_to_blobs: dict[str, int] = defaultdict(int)
    for p in blob_paths:
        norm = p.replace("\\", "/")
        parent = str(Path(norm).parent).replace("\\", "/") if "/" in norm else "."
        if parent == "":
            parent = "."
        parent_to_blobs[parent] += 1

    mean_siblings: float | None = None
    if changed_files and blob_paths:
        counts: list[int] = []
        for cf in changed_files:
            norm = cf.replace("\\", "/")
            parent = str(Path(norm).parent).replace("\\", "/") if "/" in norm else "."
            if parent == "":
                parent = "."
            counts.append(parent_to_blobs.get(parent, 0))
        mean_siblings = sum(counts) / len(counts) if counts else None

    exist_ratio: float | None = None
    if changed_files:
        hit = sum(1 for f in changed_files if f in blob_path_set)
        exist_ratio = hit / len(changed_files)

    return {
        "tree_present": int(present),
        "tree_truncated": int(truncated_raw),
        "repo_file_count": len(blobs),
        "repo_dir_count": len(trees),
        "repo_max_depth": max_d,
        "repo_mean_file_depth": mean_blob_depth,
        "repo_python_file_count": len(py_blobs),
        "repo_test_file_count": len(test_blobs),
        "repo_test_ratio": (len(test_blobs) / len(blobs)) if blobs else 0.0,
        "repo_total_known_size": total_size,
        "repo_mean_known_file_size": mean_size,
        "repo_top_level_dir_count": len(top_segments),
        "repo_has_docs_dir": int(dir_top_present("docs")),
        "repo_has_tests_dir": int(dir_top_present("tests")),
        "repo_has_examples_dir": int(dir_top_present("examples")),
        "repo_basename_collision_count": collision_groups,
        "repo_mean_siblings_per_changed_file": mean_siblings,
        "repo_changed_files_exist_in_tree_ratio": exist_ratio,
        "_blob_paths": blob_paths,
        "_blob_path_set": blob_path_set,
        "_parent_to_blobs": dict(parent_to_blobs),
    }


def _prompt_word_tokens(issue_text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", issue_text.lower()))


def _filename_overlap_tokens_from_paths(paths: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for raw in paths:
        p = raw.replace("\\", "/").strip("/")
        if "/" in p:
            parent, base = p.rsplit("/", 1)
            parts = parent.split("/") + [base]
        else:
            parts = [p]
        for part in parts:
            for t in re.findall(r"[a-z0-9]+", part.lower()):
                out.add(t)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def _path_identifier_tokens(path: str) -> set[str]:
    return set(_IDENTIFIER_RE.findall(path))


def _linking_features(
    issue_text: str,
    changed_files: list[str],
    blob_paths: list[str],
) -> dict[str, Any]:
    prompt_words = _prompt_word_tokens(issue_text)
    patch_toks = _filename_overlap_tokens_from_paths(changed_files)
    tree_toks = _filename_overlap_tokens_from_paths(blob_paths)

    prompt_idents = set(_IDENTIFIER_RE.findall(issue_text))
    path_idents: set[str] = set()
    for p in changed_files:
        path_idents |= _path_identifier_tokens(p)

    prompt_patch_filename_j = _jaccard(prompt_words, patch_toks)
    prompt_tree_filename_j = _jaccard(prompt_words, tree_toks)
    ident_overlap = len(prompt_idents & path_idents)

    candidate_files = 0
    candidate_dirs: set[str] = set()
    for p in blob_paths:
        ptoks = set(re.findall(r"[a-z0-9]+", p.lower()))
        if ptoks & prompt_words:
            candidate_files += 1
            norm = p.replace("\\", "/")
            parent = str(Path(norm).parent).replace("\\", "/") if "/" in norm else "."
            if parent == "":
                parent = "."
            candidate_dirs.add(parent)

    rank_min: int | None = None
    if changed_files and blob_paths:
        scored: list[tuple[int, str]] = []
        for p in blob_paths:
            ptoks = set(re.findall(r"[a-z0-9]+", p.lower()))
            scored.append((len(ptoks & prompt_words), p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        rank_of: dict[str, int] = {}
        for i, (_sc, path) in enumerate(scored, start=1):
            rank_of[path] = i
        ranks = [rank_of[f] for f in changed_files if f in rank_of]
        rank_min = min(ranks) if ranks else None

    return {
        "prompt_patch_filename_token_overlap": prompt_patch_filename_j,
        "prompt_tree_filename_token_overlap": prompt_tree_filename_j,
        "prompt_patch_identifier_overlap_count": ident_overlap,
        "prompt_tree_candidate_file_count": candidate_files,
        "prompt_tree_candidate_dir_count": len(candidate_dirs),
        "changed_file_candidate_rank_min": rank_min,
    }


def _prompt_issue_from_trajectory_row(trajectory_row: Mapping[str, Any]) -> tuple[str, int]:
    """Prompt features use tabular ``problem_statement`` (first user message) when set; else full ``messages``."""
    ps = trajectory_row.get("problem_statement")
    if not _is_missing(ps):
        s = str(ps).strip()
        if s:
            single_user: list[Any] = [{"role": "user", "content": s}]
            return extract_issue_text(single_user), _issue_block_found(single_user)
    messages_field = trajectory_row.get("messages")
    return extract_issue_text(messages_field), _issue_block_found(messages_field)


def _normalize_input_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any]:
    tr = row.get("trajectory_row")
    if not isinstance(tr, dict):
        tr = {}
        if "trajectory_id" in row:
            tr["trajectory_id"] = row["trajectory_id"]
        if "messages" in row:
            tr["messages"] = row["messages"]
        if "problem_statement" in row:
            tr["problem_statement"] = row["problem_statement"]
    br = row.get("base_row")
    if not isinstance(br, dict):
        br = {}
        if "patch" in row:
            br["patch"] = row["patch"]
        elif "base_patch" in row:
            br["patch"] = row["base_patch"]
    gh = row.get("github_tree")
    return tr, br, gh


def _message_text(msg: Mapping[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                txt = chunk.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return ""


def _trajectory_behavior_features(messages_field: Any, primary_patch_file: str) -> dict[str, Any]:
    messages = _as_json_messages(messages_field)
    if not messages:
        return {
            "traj_first_test_run_success": None,
            "traj_files_opened_before_first_edit": None,
            "traj_repeated_edits_same_file_count": None,
            "traj_process_control_frequency": 0,
            "traj_time_to_first_correct_file_index": None,
            "traj_touches_before_first_correct_file": None,
        }

    file_touch_counts: Counter[str] = Counter()
    opened_before_edit: set[str] = set()
    seen_edit = False
    process_control_count = 0
    first_correct_idx: int | None = None
    first_test_result: int | None = None
    touches_before_correct = 0

    for idx, raw_msg in enumerate(messages):
        if not isinstance(raw_msg, dict):
            continue
        txt = _message_text(raw_msg)
        txt_l = txt.lower()
        tool_name = ""
        if isinstance(raw_msg.get("name"), str):
            tool_name = raw_msg["name"].lower()
        elif isinstance(raw_msg.get("tool_name"), str):
            tool_name = raw_msg["tool_name"].lower()

        if "process_control" in tool_name or "process_control" in txt_l:
            process_control_count += 1

        path_hits = _FILE_PATH_MENTION_RE.findall(txt)
        touched_files = {p.strip() for p in path_hits if p.strip()}
        is_open = ("open" in tool_name) or ("read" in tool_name) or ("view" in tool_name)
        is_edit = (
            ("edit" in tool_name)
            or ("write" in tool_name)
            or ("apply_patch" in tool_name)
            or ("replace" in tool_name)
        )

        if "pytest" in txt_l or "unittest" in txt_l or "test" in tool_name:
            if first_test_result is None:
                if any(k in txt_l for k in ("passed", "ok", "all tests passed", "0 failed")):
                    first_test_result = 1
                elif any(k in txt_l for k in ("failed", "traceback", "error", "assertionerror")):
                    first_test_result = 0

        if touched_files and first_correct_idx is None and primary_patch_file:
            if primary_patch_file in touched_files:
                first_correct_idx = idx
                touches_before_correct = sum(file_touch_counts.values())

        if is_open and not seen_edit:
            opened_before_edit |= touched_files
        if is_edit:
            seen_edit = True

        for p in touched_files:
            file_touch_counts[p] += 1

    repeated_edits_same_file = sum(max(0, c - 1) for c in file_touch_counts.values())
    return {
        "traj_first_test_run_success": first_test_result,
        "traj_files_opened_before_first_edit": len(opened_before_edit) if seen_edit else len(opened_before_edit),
        "traj_repeated_edits_same_file_count": repeated_edits_same_file,
        "traj_process_control_frequency": process_control_count,
        "traj_time_to_first_correct_file_index": first_correct_idx,
        "traj_touches_before_first_correct_file": (
            touches_before_correct if first_correct_idx is not None else None
        ),
    }


def _get_prompt_parse_nlp() -> Any | None:
    """Load spaCy once for dependency-based ``prompt_parse_*`` (no benepar / transformers)."""
    global _PROMPT_PARSE_NLP
    global _PROMPT_PARSE_LOAD_ATTEMPTED
    global _PROMPT_PARSE_LAST_ERROR
    if _PROMPT_PARSE_LOAD_ATTEMPTED:
        return _PROMPT_PARSE_NLP
    # ProcessPoolExecutor workers: loading spaCy per process can OOM the node. Skip unless allowed.
    if os.environ.get("CODERFORGE_FEATURE_SUBPROCESS") == "1" and os.environ.get(
        "CODERFORGE_ALLOW_NLP_IN_WORKERS"
    ) not in {"1", "true", "yes"}:
        _PROMPT_PARSE_LOAD_ATTEMPTED = True
        _PROMPT_PARSE_NLP = None
        _PROMPT_PARSE_LAST_ERROR = "skipped in worker subprocess (OOM guard)"
        return None
    _PROMPT_PARSE_LOAD_ATTEMPTED = True
    _PROMPT_PARSE_LAST_ERROR = None
    try:
        import spacy
    except ImportError as exc:
        _PROMPT_PARSE_LAST_ERROR = f"import spacy failed: {exc!r}"
        _PROMPT_PARSE_NLP = None
        return None

    nlp = None
    load_errors: list[str] = []
    for model_name in _spacy_model_name_candidates():
        try:
            # Parser + tagger: dependency edges and sentence boundaries; drop NER for speed/memory.
            nlp = spacy.load(model_name, disable=["ner"])
            break
        except Exception as exc:
            load_errors.append(f"{model_name}: {exc!r}")

    if nlp is None:
        _PROMPT_PARSE_LAST_ERROR = "spacy.load failed for all candidates: " + "; ".join(load_errors)
        _PROMPT_PARSE_NLP = None
        return None

    _PROMPT_PARSE_NLP = nlp
    return _PROMPT_PARSE_NLP


def _dep_ud_depth_to_root(token: Any) -> int:
    depth = 0
    seen: set[int] = set()
    x = token
    while x.head is not x:
        if id(x) in seen:
            break
        seen.add(id(x))
        depth += 1
        x = x.head
        if depth > 512:
            break
    return depth


def _dep_sentence_metrics(sent: Any) -> dict[str, float]:
    """Scalar dependency-tree stats for one sentence (UD-style), aligned to old constituency keys."""
    tokens = [t for t in sent if not t.is_space]
    if not tokens:
        return {
            "node_count": 0.0,
            "tree_depth": 0.0,
            "branching_factor": 0.0,
            "pp_count": 0.0,
            "conjunction_count": 0.0,
        }

    tok_set = set(tokens)
    max_depth = max(_dep_ud_depth_to_root(t) for t in tokens)
    branch_counts: list[int] = []
    for t in tokens:
        n_kids = sum(1 for c in t.children if c in tok_set)
        if n_kids > 0:
            branch_counts.append(n_kids)
    avg_branch = sum(branch_counts) / len(branch_counts) if branch_counts else 0.0
    pp_count = float(sum(1 for t in tokens if t.dep_.lower() == "prep"))
    conj_count = float(sum(1 for t in tokens if t.dep_.lower() == "conj"))
    return {
        "node_count": float(len(tokens)),
        "tree_depth": float(max_depth),
        "branching_factor": float(avg_branch),
        "pp_count": pp_count,
        "conjunction_count": conj_count,
    }


_CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl"}
_CONJ_DEPS = {"cc", "conj"}
_NOMINAL_MODIFIER_DEPS = {"amod", "compound", "poss", "nmod"}
_COMPETITION_DEPS = _NOMINAL_MODIFIER_DEPS | _CONJ_DEPS | {"prep", "mark"} | _CLAUSE_DEPS


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def _prompt_parse_feature_keys() -> list[str]:
    return [
        "prompt_parse_sent_count",
        "prompt_parse_avg_node_count",
        "prompt_parse_avg_tree_depth",
        "prompt_parse_avg_branching_factor",
        "prompt_parse_avg_pp_count",
        "prompt_parse_avg_conjunction_count",
        "prompt_parse_ambiguity",
        "prompt_parse_available",
        "prompt_parse_num_sentences",
        "prompt_parse_num_tokens",
        "prompt_parse_num_content_tokens",
        "prompt_parse_num_nouns",
        "prompt_parse_num_verbs",
        "prompt_parse_num_clauses",
        "prompt_parse_num_preps",
        "prompt_parse_num_conjunctions",
        "prompt_parse_num_pronouns",
        "prompt_parse_num_nominal_modifiers",
        "prompt_parse_num_subordinate_markers",
        "prompt_parse_coordination_density",
        "prompt_parse_verb_coordination_ratio",
        "prompt_parse_noun_coordination_ratio",
        "prompt_parse_max_conj_chain_len",
        "prompt_parse_mean_conj_chain_len",
        "prompt_parse_pp_density_per_noun",
        "prompt_parse_pp_density_per_clause",
        "prompt_parse_max_pp_chain_depth",
        "prompt_parse_mean_pp_chain_depth",
        "prompt_parse_multi_pp_head_ratio",
        "prompt_parse_modifiers_per_noun",
        "prompt_parse_max_modifiers_on_single_noun",
        "prompt_parse_mean_modifiers_on_noun",
        "prompt_parse_compound_noun_ratio",
        "prompt_parse_pronoun_ratio",
        "prompt_parse_pronouns_per_sentence",
        "prompt_parse_pronoun_candidate_antecedent_density",
        "prompt_parse_multi_antecedent_pronoun_ratio",
        "prompt_parse_clauses_per_sentence",
        "prompt_parse_subordination_density",
        "prompt_parse_max_clause_nesting_depth",
        "prompt_parse_mean_clause_nesting_depth",
        "prompt_parse_clausal_branching_factor",
        "prompt_parse_coordination_pp_interaction",
        "prompt_parse_coordination_clause_interaction",
        "prompt_parse_modifier_clause_interaction",
        "prompt_parse_pp_clause_interaction",
        "prompt_parse_attachment_competition_ratio",
        "prompt_parse_mean_competing_dependents_per_head",
        "prompt_parse_long_dependency_ratio",
        "prompt_parse_mean_dependency_distance",
        "prompt_parse_max_dependency_distance",
        "prompt_parse_mean_dep_entropy_per_sentence",
        "prompt_parse_max_dep_entropy_per_sentence",
    ]


def _prompt_parse_unavailable_dict() -> dict[str, Any]:
    out: dict[str, Any] = {k: None for k in _prompt_parse_feature_keys()}
    out["prompt_parse_available"] = 0
    out["prompt_parse_sent_count"] = 0
    out["prompt_parse_num_sentences"] = 0
    out["prompt_parse_num_tokens"] = 0
    out["prompt_parse_num_content_tokens"] = 0
    return out


def _prompt_parse_empty_dict() -> dict[str, Any]:
    out: dict[str, Any] = {k: 0.0 for k in _prompt_parse_feature_keys()}
    out["prompt_parse_available"] = 0
    out["prompt_parse_sent_count"] = 0
    out["prompt_parse_num_sentences"] = 0
    out["prompt_parse_num_tokens"] = 0
    out["prompt_parse_num_content_tokens"] = 0
    return out


def _prompt_parse_metrics_from_spacy_doc(doc: Any) -> dict[str, Any]:
    """Aggregate dependency-tree metrics from one spaCy Doc (English parser pipeline)."""
    if not doc.text.strip():
        return _prompt_parse_empty_dict()

    num_sentences = 0
    num_tokens = 0
    num_content_tokens = 0
    num_nouns = 0
    num_verbs = 0
    num_clauses = 0
    num_preps = 0
    num_conjunctions = 0
    num_pronouns = 0
    num_nominal_modifiers = 0
    num_subordinate_markers = 0

    rows: list[dict[str, float]] = []
    conj_head_types_total = 0
    conj_head_types_verb = 0
    conj_head_types_noun = 0
    conj_chains: dict[int, set[int]] = {}
    pp_chain_depths: list[int] = []
    prep_children_by_head: dict[int, int] = defaultdict(int)
    noun_modifier_counts: list[int] = []
    compound_noun_count = 0
    pronoun_antecedent_counts: list[int] = []
    multi_antecedent_pron_count = 0
    clause_depths: list[int] = []
    clause_child_counts: list[int] = []
    competing_head_ratios: list[float] = []
    competing_dependents_per_competing_head: list[float] = []
    dep_distances: list[float] = []
    dep_entropies: list[float] = []

    for sent in doc.sents:
        if sent.text.strip():
            num_sentences += 1
            sent_tokens = [t for t in sent if not t.is_space]
            tok_set = set(sent_tokens)
            num_tokens += len(sent_tokens)
            sent_content = [t for t in sent_tokens if not t.is_punct]
            num_content_tokens += len(sent_content)
            sent_nouns = [t for t in sent_tokens if t.pos_ in {"NOUN", "PROPN"}]
            num_nouns += len(sent_nouns)
            num_verbs += sum(1 for t in sent_tokens if t.pos_ in {"VERB", "AUX"})
            sent_clauses = [t for t in sent_tokens if t.dep_ in _CLAUSE_DEPS]
            num_clauses += len(sent_clauses)
            sent_preps = [t for t in sent_tokens if t.dep_ == "prep"]
            num_preps += len(sent_preps)
            num_conjunctions += sum(1 for t in sent_tokens if t.dep_ in _CONJ_DEPS)
            num_pronouns += sum(1 for t in sent_tokens if t.pos_ == "PRON")
            sent_nom_mods = [
                t for t in sent_tokens if t.dep_ in _NOMINAL_MODIFIER_DEPS and t.head.pos_ in {"NOUN", "PROPN"}
            ]
            num_nominal_modifiers += len(sent_nom_mods)
            num_subordinate_markers += sum(1 for t in sent_tokens if t.dep_ == "mark")

            for t in sent_tokens:
                if t.dep_ == "conj":
                    conj_head_types_total += 1
                    if t.head.pos_ in {"VERB", "AUX"}:
                        conj_head_types_verb += 1
                    if t.head.pos_ in {"NOUN", "PROPN"}:
                        conj_head_types_noun += 1
                    root = t
                    while root.dep_ == "conj" and root.head is not root:
                        root = root.head
                    rid = root.i
                    if rid not in conj_chains:
                        conj_chains[rid] = {root.i}
                    conj_chains[rid].add(t.i)

                if t.dep_ == "prep":
                    prep_children_by_head[t.head.i] += 1
                    depth = 1
                    x = t.head
                    while x is not x.head and x.dep_ == "prep":
                        depth += 1
                        x = x.head
                        if depth > 512:
                            break
                    pp_chain_depths.append(float(depth))

                if t.head is not t:
                    dep_distances.append(float(abs(t.i - t.head.i)))

                if t.pos_ == "PRON":
                    antecedent_count = sum(1 for n in sent_nouns if n.i < t.i)
                    pronoun_antecedent_counts.append(antecedent_count)
                    if antecedent_count >= 2:
                        multi_antecedent_pron_count += 1

            for noun in sent_nouns:
                mod_count = sum(1 for c in noun.children if c.dep_ in _NOMINAL_MODIFIER_DEPS)
                noun_modifier_counts.append(float(mod_count))
                if noun.dep_ == "compound" or any(c.dep_ == "compound" for c in noun.children):
                    compound_noun_count += 1

            clause_set = {t.i for t in sent_clauses}
            for ct in sent_clauses:
                d = 1
                x = ct.head
                while x is not x.head:
                    if x.i in clause_set:
                        d += 1
                    x = x.head
                    if d > 512:
                        break
                clause_depths.append(float(d))
                clause_child_counts.append(float(sum(1 for c in ct.children if c.dep_ in _CLAUSE_DEPS)))

            head_counts: list[int] = []
            for head in sent_tokens:
                c = sum(1 for ch in head.children if ch in tok_set and ch.dep_ in _COMPETITION_DEPS)
                if c > 0:
                    head_counts.append(c)
            if head_counts:
                n_comp = sum(1 for c in head_counts if c >= 2)
                competing_head_ratios.append(_safe_div(float(n_comp), float(len(head_counts))))
                comp_counts = [float(c) for c in head_counts if c >= 2]
                if comp_counts:
                    competing_dependents_per_competing_head.append(sum(comp_counts) / len(comp_counts))
                else:
                    competing_dependents_per_competing_head.append(0.0)
            else:
                competing_head_ratios.append(0.0)
                competing_dependents_per_competing_head.append(0.0)

            dep_cat = Counter()
            for t in sent_tokens:
                if t.dep_ in _CLAUSE_DEPS:
                    dep_cat["clause"] += 1
                elif t.dep_ == "prep":
                    dep_cat["prep"] += 1
                elif t.dep_ in _CONJ_DEPS:
                    dep_cat["coord"] += 1
                elif t.dep_ in _NOMINAL_MODIFIER_DEPS:
                    dep_cat["modifier"] += 1
                elif t.dep_ == "mark":
                    dep_cat["subordinate"] += 1
                else:
                    dep_cat["other"] += 1
            total = float(sum(dep_cat.values()))
            if total > 0:
                probs = [v / total for v in dep_cat.values() if v > 0]
                dep_entropies.append(float(-sum(p * math.log(p) for p in probs)))
            else:
                dep_entropies.append(0.0)

            rows.append(_dep_sentence_metrics(sent))
    if not rows:
        return _prompt_parse_unavailable_dict()

    sent_count = len(rows)
    avg_node_count = sum(r["node_count"] for r in rows) / sent_count
    avg_tree_depth = sum(r["tree_depth"] for r in rows) / sent_count
    avg_branching_factor = sum(r["branching_factor"] for r in rows) / sent_count
    avg_pp_count = sum(r["pp_count"] for r in rows) / sent_count
    avg_conj_count = sum(r["conjunction_count"] for r in rows) / sent_count
    ambiguity_parse = avg_tree_depth + avg_branching_factor + avg_pp_count + avg_conj_count
    conj_chain_lens = [float(len(v)) for v in conj_chains.values() if len(v) > 0]
    heads_with_prep = [c for c in prep_children_by_head.values() if c > 0]
    noun_count_f = float(num_nouns)
    clause_count_f = float(num_clauses)
    sent_count_f = float(num_sentences)
    content_count_f = float(num_content_tokens)
    pron_count_f = float(num_pronouns)

    coordination_density = _safe_div(float(num_conjunctions), max(clause_count_f, 1.0))
    pp_density_per_noun = _safe_div(float(num_preps), max(noun_count_f, 1.0))
    pp_density_per_clause = _safe_div(float(num_preps), max(clause_count_f, 1.0))
    modifiers_per_noun = _safe_div(float(num_nominal_modifiers), max(noun_count_f, 1.0))
    clauses_per_sentence = _safe_div(float(num_clauses), max(sent_count_f, 1.0))
    pronoun_ratio = _safe_div(pron_count_f, max(content_count_f, 1.0))
    pronouns_per_sentence = _safe_div(pron_count_f, max(sent_count_f, 1.0))

    return {
        "prompt_parse_sent_count": sent_count,
        "prompt_parse_avg_node_count": avg_node_count,
        "prompt_parse_avg_tree_depth": avg_tree_depth,
        "prompt_parse_avg_branching_factor": avg_branching_factor,
        "prompt_parse_avg_pp_count": avg_pp_count,
        "prompt_parse_avg_conjunction_count": avg_conj_count,
        "prompt_parse_ambiguity": ambiguity_parse,
        "prompt_parse_num_sentences": float(num_sentences),
        "prompt_parse_num_tokens": float(num_tokens),
        "prompt_parse_num_content_tokens": float(num_content_tokens),
        "prompt_parse_num_nouns": float(num_nouns),
        "prompt_parse_num_verbs": float(num_verbs),
        "prompt_parse_num_clauses": float(num_clauses),
        "prompt_parse_num_preps": float(num_preps),
        "prompt_parse_num_conjunctions": float(num_conjunctions),
        "prompt_parse_num_pronouns": float(num_pronouns),
        "prompt_parse_num_nominal_modifiers": float(num_nominal_modifiers),
        "prompt_parse_num_subordinate_markers": float(num_subordinate_markers),
        "prompt_parse_coordination_density": coordination_density,
        "prompt_parse_verb_coordination_ratio": _safe_div(
            float(conj_head_types_verb), max(float(conj_head_types_total), 1.0)
        ),
        "prompt_parse_noun_coordination_ratio": _safe_div(
            float(conj_head_types_noun), max(float(conj_head_types_total), 1.0)
        ),
        "prompt_parse_max_conj_chain_len": max(conj_chain_lens) if conj_chain_lens else 0.0,
        "prompt_parse_mean_conj_chain_len": _safe_div(sum(conj_chain_lens), float(len(conj_chain_lens))),
        "prompt_parse_pp_density_per_noun": pp_density_per_noun,
        "prompt_parse_pp_density_per_clause": pp_density_per_clause,
        "prompt_parse_max_pp_chain_depth": max(pp_chain_depths) if pp_chain_depths else 0.0,
        "prompt_parse_mean_pp_chain_depth": _safe_div(sum(pp_chain_depths), float(len(pp_chain_depths))),
        "prompt_parse_multi_pp_head_ratio": _safe_div(
            float(sum(1 for c in heads_with_prep if c >= 2)), float(len(heads_with_prep))
        ),
        "prompt_parse_modifiers_per_noun": modifiers_per_noun,
        "prompt_parse_max_modifiers_on_single_noun": max(noun_modifier_counts) if noun_modifier_counts else 0.0,
        "prompt_parse_mean_modifiers_on_noun": _safe_div(
            sum(noun_modifier_counts), float(len(noun_modifier_counts))
        ),
        "prompt_parse_compound_noun_ratio": _safe_div(float(compound_noun_count), max(noun_count_f, 1.0)),
        "prompt_parse_pronoun_ratio": pronoun_ratio,
        "prompt_parse_pronouns_per_sentence": pronouns_per_sentence,
        "prompt_parse_pronoun_candidate_antecedent_density": _safe_div(
            float(sum(pronoun_antecedent_counts)), float(len(pronoun_antecedent_counts))
        ),
        "prompt_parse_multi_antecedent_pronoun_ratio": _safe_div(
            float(multi_antecedent_pron_count), max(pron_count_f, 1.0)
        ),
        "prompt_parse_clauses_per_sentence": clauses_per_sentence,
        "prompt_parse_subordination_density": _safe_div(float(num_subordinate_markers), max(clause_count_f, 1.0)),
        "prompt_parse_max_clause_nesting_depth": max(clause_depths) if clause_depths else 0.0,
        "prompt_parse_mean_clause_nesting_depth": _safe_div(sum(clause_depths), float(len(clause_depths))),
        "prompt_parse_clausal_branching_factor": _safe_div(
            sum(clause_child_counts), float(len(clause_child_counts))
        ),
        "prompt_parse_coordination_pp_interaction": coordination_density * pp_density_per_noun,
        "prompt_parse_coordination_clause_interaction": coordination_density * clauses_per_sentence,
        "prompt_parse_modifier_clause_interaction": modifiers_per_noun * clauses_per_sentence,
        "prompt_parse_pp_clause_interaction": pp_density_per_clause * clauses_per_sentence,
        "prompt_parse_attachment_competition_ratio": _safe_div(
            sum(competing_head_ratios), float(len(competing_head_ratios))
        ),
        "prompt_parse_mean_competing_dependents_per_head": _safe_div(
            sum(competing_dependents_per_competing_head), float(len(competing_dependents_per_competing_head))
        ),
        "prompt_parse_long_dependency_ratio": _safe_div(
            float(sum(1 for d in dep_distances if d > 5.0)), float(len(dep_distances))
        ),
        "prompt_parse_mean_dependency_distance": _safe_div(sum(dep_distances), float(len(dep_distances))),
        "prompt_parse_max_dependency_distance": max(dep_distances) if dep_distances else 0.0,
        "prompt_parse_mean_dep_entropy_per_sentence": _safe_div(
            sum(dep_entropies), float(len(dep_entropies))
        ),
        "prompt_parse_max_dep_entropy_per_sentence": max(dep_entropies) if dep_entropies else 0.0,
        "prompt_parse_available": 1,
    }


_CONST_NLP_MISSING_MSG = """\
Prompt dependency parser (spaCy English) is not available or failed to load.

Install and download models in this environment, then retry:
  pip install -e ".[nlp]"
  python -m spacy download en_core_web_sm

If spaCy English is installed under another name, set e.g. CODERFORGE_SPACY_MODELS=en_core_web_md,en_core_web_sm

To skip parse columns (other features only): use extract_task_features --no-parse-ambiguity
or enrich_parse_ambiguity --allow-empty-parse

Details:
"""


def prompt_parse_ambiguity_features_many(
    issue_texts: list[str],
    *,
    pipe_batch_size: int | None = None,
    progress_every: int = 5000,
    require_nlp: bool = False,
) -> list[dict[str, Any]]:
    """Fill prompt_parse_* for many rows using one spaCy load and ``nlp.pipe`` (main process only).

    Metrics are **dependency-tree** statistics (UD), not constituency / benepar.

    Use this after parallel feature extraction so workers never duplicate heavy NLP models.

    If ``require_nlp`` is True and models are missing, raises ``RuntimeError`` instead of
    silently writing null parse columns for every row.
    """
    if not issue_texts:
        return []

    nlp = _get_prompt_parse_nlp()
    if nlp is None:
        if require_nlp:
            detail = constituency_nlp_last_error() or "unknown (no detail stored)"
            raise RuntimeError(_CONST_NLP_MISSING_MSG + detail)
        return [_prompt_parse_unavailable_dict() for _ in issue_texts]

    bs_env = os.getenv("CODERFORGE_PARSE_PIPE_BATCH")
    bs = pipe_batch_size if pipe_batch_size is not None else (int(bs_env) if bs_env and bs_env.isdigit() else 32)

    out: list[dict[str, Any]] = []
    n = len(issue_texts)
    for start in range(0, n, bs):
        chunk_raw = issue_texts[start : start + bs]
        try:
            for doc in nlp.pipe(chunk_raw, batch_size=min(bs, len(chunk_raw))):
                out.append(_prompt_parse_metrics_from_spacy_doc(doc))
        except Exception:
            for t in chunk_raw:
                try:
                    out.append(_prompt_parse_metrics_from_spacy_doc(nlp(t)))
                except Exception:
                    out.append(_prompt_parse_unavailable_dict())
        end = min(start + bs, n)
        if progress_every > 0 and (end % progress_every == 0 or end == n):
            print(f"  prompt_parse progress {end}/{n}", flush=True)
    return out


def _prompt_parse_ambiguity_features(issue_text: str) -> dict[str, Any]:
    if not issue_text.strip():
        return _prompt_parse_empty_dict()

    nlp = _get_prompt_parse_nlp()
    if nlp is None:
        return _prompt_parse_unavailable_dict()

    try:
        doc = nlp(issue_text)
    except Exception:
        return _prompt_parse_unavailable_dict()

    return _prompt_parse_metrics_from_spacy_doc(doc)


def extract_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compute all scalar features for one dataset row (dict-like)."""
    trajectory_row, base_row, gh_raw = _normalize_input_row(row)
    github_tree = _coerce_github_tree(gh_raw)

    trajectory_id = trajectory_row.get("trajectory_id")
    if _is_missing(trajectory_id):
        trajectory_id_str = ""
    else:
        trajectory_id_str = str(trajectory_id)

    task_id = task_id_from_trajectory_id(trajectory_id_str)
    issue_text, issue_block = _prompt_issue_from_trajectory_row(trajectory_row)

    lines = issue_text.splitlines()
    line_count = len(lines)
    char_count = len(issue_text)
    tok_simple = _prompt_token_count_simple(issue_text)
    fence_count = _count_fence_lines(issue_text)
    bullet_count = sum(1 for ln in lines if _BULLET_LINE_RE.match(ln))

    has_err_kw = int(bool(_ERROR_KEYWORD_RE.search(issue_text)))
    err_line_count = sum(1 for ln in lines if _ERROR_KEYWORD_RE.search(ln))

    file_mentions = len(_FILE_PATH_MENTION_RE.findall(issue_text))
    inline_code = _count_inline_backtick_spans(issue_text)
    ident_count = len(_IDENTIFIER_RE.findall(issue_text))
    func_like = len(_FUNC_CALL_LIKE_RE.findall(issue_text))
    phase_count = len(_PHASE_RE.findall(issue_text))

    patch_raw = base_row.get("patch")
    if _is_missing(patch_raw):
        patch_str = ""
    else:
        patch_str = str(patch_raw)
        if patch_str.lower() == "nan":
            patch_str = ""

    patch_present = int(len(patch_str) > 0)
    pstats = _patch_line_stats(patch_str)
    changed_files = parse_patch_changed_files(patch_str)

    tree_feats = _tree_features(github_tree, changed_files)
    blob_paths = tree_feats.pop("_blob_paths")
    tree_feats.pop("_blob_path_set", None)
    tree_feats.pop("_parent_to_blobs", None)

    link_feats = _linking_features(issue_text, changed_files, blob_paths)

    repo_name = ""
    commit_sha = ""
    if github_tree:
        r = github_tree.get("repo")
        if r is not None and not _is_missing(r):
            repo_name = str(r)
        c = github_tree.get("commit_sha")
        if c is not None and not _is_missing(c):
            commit_sha = str(c)

    out: dict[str, Any] = {
        "trajectory_id": trajectory_id_str,
        "task_id": task_id,
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "prompt_char_count": char_count,
        "prompt_line_count": line_count,
        "prompt_token_count_simple": tok_simple,
        "prompt_code_block_count": fence_count,
        "prompt_bullet_count": bullet_count,
        "prompt_has_error_keyword": has_err_kw,
        "prompt_error_line_count": err_line_count,
        "prompt_file_path_mention_count": file_mentions,
        "prompt_inline_code_count": inline_code,
        "prompt_code_identifier_count": ident_count,
        "prompt_function_call_like_count": func_like,
        "prompt_phase_count": phase_count,
        "prompt_issue_block_found": issue_block,
        "patch_present": patch_present,
        "patch_lines_changed_total": pstats["patch_lines_added"] + pstats["patch_lines_deleted"],
        "patch_is_single_file": int(pstats["patch_num_files_changed"] == 1),
        **{k: v for k, v in pstats.items() if not k.startswith("_")},
        **tree_feats,
        **link_feats,
        **_prompt_parse_ambiguity_features(issue_text),
        **_trajectory_behavior_features(trajectory_row.get("messages"), pstats["patch_primary_file_path"]),
    }
    return out


def extract_features_many(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Vector-friendly batch API: one feature dict per input row."""
    records = [extract_features(r) for r in rows]
    return pd.DataFrame.from_records(records)


def save_features_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write features to Parquet (efficient columnar storage for analysis)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, **kwargs)
    return p
