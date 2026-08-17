"""CLI: extract prompt-side feature tables from mapping/trajectory Parquets.

The extractor follows the same row organization as the task feature pipeline:
it prefers ``problem_statement`` when present, otherwise falls back to the first
user message in ``messages``. Output defaults to the workspace processed-data
directory, which is scratch-enforced by the workspace env loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import tiktoken

from task_row_features import (
    _get_prompt_parse_nlp,
    _prompt_issue_from_trajectory_row,
    constituency_nlp_last_error,
    save_features_parquet,
    task_id_from_trajectory_id,
)

_ENC = tiktoken.get_encoding("cl100k_base")

_CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl"}
_CONJ_DEPS = {"cc", "conj"}
_NOMINAL_MODIFIER_DEPS = {"amod", "compound", "poss", "nmod"}
_COMPETITION_DEPS = _NOMINAL_MODIFIER_DEPS | _CONJ_DEPS | {"prep", "mark"} | _CLAUSE_DEPS

PROMPT_META_COLUMNS = ["trajectory_id", "task_id", "cf_split", "cf_row_index", "row_uid"]

PROMPT_FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Component Load",
        [
            "prompt_len_bpe",
            "prompt_sent_count",
            "prompt_clauses_per_sentence",
            "prompt_content_tokens_per_sentence",
            "prompt_clause_length_mean",
        ],
    ),
    (
        "Coordinative Complexity",
        [
            "prompt_dep_depth_mean",
            "prompt_dep_distance_mean",
            "prompt_dep_distance_max",
            "prompt_branching_factor_mean",
            "prompt_clause_nesting_depth_mean",
            "prompt_clausal_branching_factor",
        ],
    ),
    (
        "Interpretive Uncertainty - Coordination",
        [
            "prompt_coordination_density",
            "prompt_mean_conj_chain_len",
            "prompt_max_conj_chain_len",
            "prompt_verb_coordination_ratio",
            "prompt_noun_coordination_ratio",
        ],
    ),
    (
        "Interpretive Uncertainty - Attachment",
        [
            "prompt_pp_density_per_noun",
            "prompt_pp_density_per_clause",
            "prompt_mean_pp_chain_depth",
            "prompt_max_pp_chain_depth",
            "prompt_multi_pp_head_ratio",
            "prompt_attachment_competition_ratio",
            "prompt_mean_competing_dependents_per_head",
        ],
    ),
    (
        "Interpretive Uncertainty - Referential",
        [
            "prompt_pronoun_ratio",
            "prompt_pronouns_per_sentence",
            "prompt_pronoun_candidate_antecedent_density",
            "prompt_multi_antecedent_pronoun_ratio",
        ],
    ),
    (
        "Cohesion",
        [
            "prompt_adjacent_lemma_overlap_mean",
            "prompt_adjacent_lemma_overlap_min",
            "prompt_sentence_embedding_sim_mean",
            "prompt_sentence_embedding_sim_min",
            "prompt_entity_reuse_ratio",
            "prompt_coref_chain_len_mean",
        ],
    ),
]

PROMPT_FEATURE_COLUMNS = [feature for _, features in PROMPT_FEATURE_GROUPS for feature in features]

PROMPT_FEATURE_FORMULAS: dict[str, str] = {
    "prompt_len_bpe": "|BPE_cl100k(x)|",
    "prompt_sent_count": "|S(x)|",
    "prompt_clauses_per_sentence": "|H_clause(x)| / max(|S(x)|, 1)",
    "prompt_content_tokens_per_sentence": "|C(x)| / max(|S(x)|, 1)",
    "prompt_clause_length_mean": "mean subtree size over clause heads",
    "prompt_dep_depth_mean": "mean(max dependency depth per sentence)",
    "prompt_dep_distance_mean": "mean(|idx(t) - idx(head(t))|)",
    "prompt_dep_distance_max": "max(|idx(t) - idx(head(t))|)",
    "prompt_branching_factor_mean": "mean(|children(t)| for non-leaf nodes)",
    "prompt_clause_nesting_depth_mean": "mean(number of clause ancestors per clause)",
    "prompt_clausal_branching_factor": "mean(number of clause children per clause)",
    "prompt_coordination_density": "|L_coord(x)| / max(|H_clause(x)|, 1)",
    "prompt_mean_conj_chain_len": "mean size of coordination chains",
    "prompt_max_conj_chain_len": "max chain size",
    "prompt_verb_coordination_ratio": "fraction of conj with verb heads",
    "prompt_noun_coordination_ratio": "fraction of conj with noun heads",
    "prompt_pp_density_per_noun": "|L_pp(x)| / max(|N(x)|, 1)",
    "prompt_pp_density_per_clause": "|L_pp(x)| / max(|H_clause(x)|, 1)",
    "prompt_mean_pp_chain_depth": "mean nested PP depth",
    "prompt_max_pp_chain_depth": "max nested PP depth",
    "prompt_multi_pp_head_ratio": "fraction of heads with >=2 PP children",
    "prompt_attachment_competition_ratio": "fraction of heads with >=2 competing dependents",
    "prompt_mean_competing_dependents_per_head": "mean competing dependents",
    "prompt_pronoun_ratio": "|P(x)| / max(|C(x)|, 1)",
    "prompt_pronouns_per_sentence": "|P(x)| / max(|S(x)|, 1)",
    "prompt_pronoun_candidate_antecedent_density": "mean antecedent candidates per pronoun",
    "prompt_multi_antecedent_pronoun_ratio": "fraction with >=2 antecedents",
    "prompt_adjacent_lemma_overlap_mean": "mean Jaccard overlap between adjacent sentences",
    "prompt_adjacent_lemma_overlap_min": "min overlap",
    "prompt_sentence_embedding_sim_mean": "mean cosine similarity",
    "prompt_sentence_embedding_sim_min": "min cosine similarity",
    "prompt_entity_reuse_ratio": "reused mentions / total mentions",
    "prompt_coref_chain_len_mean": "mean coref chain length",
}


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom > 0 else 0.0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _default_trajectory_paths(_data_processed: Path) -> list[Path]:
    return []


def _coalesce_strings(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    mask = primary.notna() & primary.astype(str).str.strip().ne("")
    return primary.where(mask, fallback)


def _merge_mapping_and_trajectories(mapping: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    if "trajectory_id" not in mapping.columns:
        raise RuntimeError("mapping parquet must contain trajectory_id")
    if "trajectory_id" not in trajectories.columns:
        raise RuntimeError("trajectory parquet missing trajectory_id")

    keep_cols = [c for c in ("trajectory_id", "messages", "problem_statement", "base_problem_statement") if c in trajectories.columns]
    if len(keep_cols) <= 1:
        return mapping

    tsub = trajectories[keep_cols].drop_duplicates(subset=["trajectory_id"], keep="first")
    merged = mapping.merge(tsub, on="trajectory_id", how="left", suffixes=("", "_traj"))
    for col in ("messages", "problem_statement", "base_problem_statement"):
        traj_col = f"{col}_traj"
        if traj_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = _coalesce_strings(merged[col], merged[traj_col])
            merged = merged.drop(columns=[traj_col])
        else:
            merged = merged.rename(columns={traj_col: col})
    return merged


def _first_user_message_text(messages_field: Any) -> str:
    if _is_missing(messages_field):
        return ""
    if isinstance(messages_field, str):
        try:
            messages = json.loads(messages_field)
        except json.JSONDecodeError:
            return messages_field.strip()
    elif isinstance(messages_field, list):
        messages = messages_field
    else:
        return ""

    if not isinstance(messages, list):
        return ""

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return ""


def _prompt_text_or_empty(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    return text if text.strip() else ""


def extract_prompt_text_from_row(row: Mapping[str, Any]) -> str:
    """Return the verbatim first user message used for prompt-side features."""
    problem_statement = _prompt_text_or_empty(row.get("problem_statement"))
    if problem_statement:
        return problem_statement

    base_problem_statement = _prompt_text_or_empty(row.get("base_problem_statement"))
    if base_problem_statement:
        return base_problem_statement

    messages_text = _first_user_message_text(row.get("messages"))
    if messages_text:
        return messages_text

    return ""


def _token_count_bpe(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


def _sentence_spans(doc: Any) -> list[Any]:
    try:
        sents = [s for s in doc.sents if s.text.strip()]
    except Exception:
        sents = []
    if sents:
        return sents
    if getattr(doc, "text", "").strip():
        return [doc[:]]
    return []


def _token_depth(token: Any) -> int:
    depth = 0
    seen: set[int] = set()
    current = token
    while current.head is not current:
        if id(current) in seen:
            break
        seen.add(id(current))
        depth += 1
        current = current.head
        if depth > 512:
            break
    return depth


def _subtree_size(token: Any, sent_start: int, sent_end: int) -> int:
    return sum(1 for child in token.subtree if sent_start <= child.i < sent_end and not child.is_space)


def _cosine_similarity(left: Any, right: Any) -> float:
    left_vec = list(left)
    right_vec = list(right)
    if len(left_vec) != len(right_vec) or not left_vec:
        return 0.0
    dot = sum(a * b for a, b in zip(left_vec, right_vec))
    left_norm = math.sqrt(sum(a * a for a in left_vec))
    right_norm = math.sqrt(sum(b * b for b in right_vec))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def _lemma_tokens(span: Any) -> set[str]:
    out: set[str] = set()
    for token in span:
        if token.is_space or token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_.strip().lower()
        if lemma:
            out.add(lemma)
    return out


def _mention_key(span: Any) -> str:
    if getattr(span, "root", None) is not None and getattr(span.root, "pos_", "") == "PRON":
        return ""
    parts = [token.lemma_.lower() for token in span if token.is_alpha and not token.is_stop]
    if not parts:
        parts = [token.text.lower() for token in span if token.is_alpha]
    if not parts:
        return ""
    return " ".join(parts).strip()


def _mention_spans(doc: Any) -> list[Any]:
    spans: list[Any] = []
    seen: set[tuple[int, int]] = set()
    for span in getattr(doc, "ents", ()):
        key = (int(span.start), int(span.end))
        if key not in seen and span.text.strip():
            spans.append(span)
            seen.add(key)
    try:
        noun_chunks = list(doc.noun_chunks)
    except Exception:
        noun_chunks = []
    for span in noun_chunks:
        key = (int(span.start), int(span.end))
        if key not in seen and span.text.strip():
            spans.append(span)
            seen.add(key)
    return spans


def _empty_prompt_features() -> dict[str, float]:
    return {name: 0.0 for name in PROMPT_FEATURE_COLUMNS}


def _prompt_features_from_doc(doc: Any, raw_text: str) -> dict[str, float]:
    if not getattr(doc, "text", "").strip() and not raw_text.strip():
        return _empty_prompt_features()

    sentences = _sentence_spans(doc)
    sentence_count = len(sentences)
    tokens = [token for token in doc if not token.is_space]
    content_tokens = [token for token in tokens if not token.is_punct]
    noun_tokens = [token for token in tokens if token.pos_ in {"NOUN", "PROPN"}]
    pron_tokens = [token for token in tokens if token.pos_ == "PRON"]

    clause_heads = [token for token in tokens if token.dep_ in _CLAUSE_DEPS]
    coord_tokens = [token for token in tokens if token.dep_ in _CONJ_DEPS]
    prep_tokens = [token for token in tokens if token.dep_ == "prep"]

    clause_lengths = [float(_subtree_size(token, token.sent.start, token.sent.end)) for token in clause_heads]
    dep_depths = [float(max((_token_depth(token) for token in sentence), default=0)) for sentence in sentences]
    dep_distances = [float(abs(token.i - token.head.i)) for token in tokens if token.head is not token]
    branching_counts: list[float] = []
    for sentence in sentences:
        sent_tokens = [token for token in sentence if not token.is_space]
        sent_set = set(sent_tokens)
        for token in sent_tokens:
            child_count = sum(1 for child in token.children if child in sent_set and not child.is_space)
            if child_count > 0:
                branching_counts.append(float(child_count))

    clause_ancestor_depths: list[float] = []
    clause_child_counts: list[float] = []
    for clause in clause_heads:
        ancestors = 0
        current = clause.head
        seen: set[int] = set()
        while current is not current.head and id(current) not in seen:
            seen.add(id(current))
            if current.dep_ in _CLAUSE_DEPS:
                ancestors += 1
            current = current.head
            if ancestors > 512:
                break
        clause_ancestor_depths.append(float(ancestors))
        clause_child_counts.append(float(sum(1 for child in clause.children if child.dep_ in _CLAUSE_DEPS)))

    conj_chains: dict[int, set[int]] = {}
    conj_head_total = 0
    conj_head_verb = 0
    conj_head_noun = 0
    for token in coord_tokens:
        if token.dep_ != "conj":
            continue
        conj_head_total += 1
        if token.head.pos_ in {"VERB", "AUX"}:
            conj_head_verb += 1
        if token.head.pos_ in {"NOUN", "PROPN"}:
            conj_head_noun += 1
        root = token
        while root.dep_ == "conj" and root.head is not root:
            root = root.head
        chain = conj_chains.setdefault(root.i, {root.i})
        chain.add(token.i)
    conj_chain_lengths = [float(len(chain)) for chain in conj_chains.values() if chain]

    head_to_pp_children: dict[int, int] = defaultdict(int)
    pp_chain_depths: list[float] = []
    for token in prep_tokens:
        head_to_pp_children[token.head.i] += 1
        depth = 1
        current = token.head
        while current is not current.head and current.dep_ == "prep":
            depth += 1
            current = current.head
            if depth > 512:
                break
        pp_chain_depths.append(float(depth))

    competing_counts: list[int] = []
    for sentence in sentences:
        sent_tokens = [token for token in sentence if not token.is_space]
        sent_set = set(sent_tokens)
        for token in sent_tokens:
            count = sum(1 for child in token.children if child in sent_set and child.dep_ in _COMPETITION_DEPS)
            if count > 0:
                competing_counts.append(count)

    pronoun_antecedent_counts: list[int] = []
    multi_antecedent_count = 0
    for sentence in sentences:
        sent_tokens = [token for token in sentence if not token.is_space]
        sent_nouns = [token for token in sent_tokens if token.pos_ in {"NOUN", "PROPN"}]
        for token in sent_tokens:
            if token.pos_ != "PRON":
                continue
            antecedent_count = sum(1 for noun in sent_nouns if noun.i < token.i)
            pronoun_antecedent_counts.append(antecedent_count)
            if antecedent_count >= 2:
                multi_antecedent_count += 1

    adjacent_lemma_overlaps: list[float] = []
    sentence_similarities: list[float] = []
    for index in range(len(sentences) - 1):
        left = sentences[index]
        right = sentences[index + 1]
        left_lemmas = _lemma_tokens(left)
        right_lemmas = _lemma_tokens(right)
        if left_lemmas or right_lemmas:
            union = len(left_lemmas | right_lemmas)
            overlap = (len(left_lemmas & right_lemmas) / union) if union else 0.0
        else:
            overlap = 0.0
        adjacent_lemma_overlaps.append(float(overlap))
        sentence_similarities.append(_cosine_similarity(left.vector, right.vector))

    mention_spans = _mention_spans(doc)
    mention_counts: dict[str, int] = defaultdict(int)
    for span in mention_spans:
        key = _mention_key(span)
        if key:
            mention_counts[key] += 1
    mention_chain_lengths = [float(count) for count in mention_counts.values() if count > 0]
    total_mentions = float(sum(mention_counts.values()))
    reused_mentions = float(sum(count - 1 for count in mention_counts.values() if count > 1))

    return {
        "prompt_len_bpe": float(_token_count_bpe(raw_text)),
        "prompt_sent_count": float(sentence_count),
        "prompt_clauses_per_sentence": _safe_div(float(len(clause_heads)), float(max(sentence_count, 1))),
        "prompt_content_tokens_per_sentence": _safe_div(float(len(content_tokens)), float(max(sentence_count, 1))),
        "prompt_clause_length_mean": _safe_div(sum(clause_lengths), float(len(clause_lengths))),
        "prompt_dep_depth_mean": _safe_div(sum(dep_depths), float(len(dep_depths))),
        "prompt_dep_distance_mean": _safe_div(sum(dep_distances), float(len(dep_distances))),
        "prompt_dep_distance_max": max(dep_distances) if dep_distances else 0.0,
        "prompt_branching_factor_mean": _safe_div(sum(branching_counts), float(len(branching_counts))),
        "prompt_clause_nesting_depth_mean": _safe_div(sum(clause_ancestor_depths), float(len(clause_ancestor_depths))),
        "prompt_clausal_branching_factor": _safe_div(sum(clause_child_counts), float(len(clause_child_counts))),
        "prompt_coordination_density": _safe_div(float(len(coord_tokens)), float(max(len(clause_heads), 1))),
        "prompt_mean_conj_chain_len": _safe_div(sum(conj_chain_lengths), float(len(conj_chain_lengths))),
        "prompt_max_conj_chain_len": max(conj_chain_lengths) if conj_chain_lengths else 0.0,
        "prompt_verb_coordination_ratio": _safe_div(float(conj_head_verb), float(conj_head_total)),
        "prompt_noun_coordination_ratio": _safe_div(float(conj_head_noun), float(conj_head_total)),
        "prompt_pp_density_per_noun": _safe_div(float(len(prep_tokens)), float(max(len(noun_tokens), 1))),
        "prompt_pp_density_per_clause": _safe_div(float(len(prep_tokens)), float(max(len(clause_heads), 1))),
        "prompt_mean_pp_chain_depth": _safe_div(sum(pp_chain_depths), float(len(pp_chain_depths))),
        "prompt_max_pp_chain_depth": max(pp_chain_depths) if pp_chain_depths else 0.0,
        "prompt_multi_pp_head_ratio": _safe_div(
            float(sum(1 for count in head_to_pp_children.values() if count >= 2)),
            float(len([count for count in head_to_pp_children.values() if count > 0])),
        ),
        "prompt_attachment_competition_ratio": _safe_div(
            float(sum(1 for count in competing_counts if count >= 2)),
            float(len(competing_counts)),
        ),
        "prompt_mean_competing_dependents_per_head": _safe_div(sum(float(c) for c in competing_counts), float(len(competing_counts))),
        "prompt_pronoun_ratio": _safe_div(float(len(pron_tokens)), float(max(len(content_tokens), 1))),
        "prompt_pronouns_per_sentence": _safe_div(float(len(pron_tokens)), float(max(sentence_count, 1))),
        "prompt_pronoun_candidate_antecedent_density": _safe_div(
            float(sum(pronoun_antecedent_counts)), float(len(pronoun_antecedent_counts))
        ),
        "prompt_multi_antecedent_pronoun_ratio": _safe_div(
            float(multi_antecedent_count), float(len(pronoun_antecedent_counts))
        ),
        "prompt_adjacent_lemma_overlap_mean": _safe_div(
            sum(adjacent_lemma_overlaps), float(len(adjacent_lemma_overlaps))
        ),
        "prompt_adjacent_lemma_overlap_min": min(adjacent_lemma_overlaps) if adjacent_lemma_overlaps else 0.0,
        "prompt_sentence_embedding_sim_mean": _safe_div(
            sum(sentence_similarities), float(len(sentence_similarities))
        ),
        "prompt_sentence_embedding_sim_min": min(sentence_similarities) if sentence_similarities else 0.0,
        "prompt_entity_reuse_ratio": _safe_div(reused_mentions, total_mentions),
        "prompt_coref_chain_len_mean": _safe_div(sum(mention_chain_lengths), float(len(mention_chain_lengths))),
    }


def extract_prompt_features(text: str, *, doc: Any | None = None) -> dict[str, float]:
    """Extract prompt-side features for one raw prompt string."""
    if not text or not text.strip():
        return _empty_prompt_features()

    if doc is None:
        nlp = _get_prompt_parse_nlp()
        if nlp is None:
            detail = constituency_nlp_last_error() or "unknown"
            raise RuntimeError(
                "Prompt NLP parser is unavailable. Install the spaCy extra and a model, then retry.\n"
                f"Details: {detail}"
            )
        doc = nlp(text)

    return _prompt_features_from_doc(doc, text)


def extract_prompt_features_row(row: Mapping[str, Any], *, doc: Any | None = None) -> dict[str, Any]:
    """Return a feature row with prompt features and provenance columns."""
    prompt_text = extract_prompt_text_from_row(row)
    features = extract_prompt_features(prompt_text, doc=doc)
    trajectory_id = str(row.get("trajectory_id", "") or "")
    task_id = task_id_from_trajectory_id(trajectory_id)
    out: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
    }
    for key in ("cf_split", "cf_row_index", "row_uid"):
        if key in row:
            out[key] = row[key]
    out.update(features)
    return out


def extract_prompt_features_many(rows: Iterable[Mapping[str, Any]], *, batch_size: int = 32) -> pd.DataFrame:
    """Extract prompt features for many rows using one spaCy load and ``nlp.pipe``."""
    records = [dict(row) for row in rows]
    if not records:
        return pd.DataFrame(columns=[*PROMPT_META_COLUMNS, *PROMPT_FEATURE_COLUMNS])

    nlp = _get_prompt_parse_nlp()
    if nlp is None:
        detail = constituency_nlp_last_error() or "unknown"
        raise RuntimeError(
            "Prompt NLP parser is unavailable. Install the spaCy extra and a model, then retry.\n"
            f"Details: {detail}"
        )

    texts = [extract_prompt_text_from_row(row) for row in records]
    try:
        docs = list(nlp.pipe(texts, batch_size=max(1, batch_size)))
    except Exception:
        docs = []
        for text in texts:
            try:
                docs.append(nlp(text))
            except Exception:
                docs.append(None)

    rows_out: list[dict[str, Any]] = []
    for row, doc in zip(records, docs):
        if doc is not None:
            feature_row = extract_prompt_features_row(row, doc=doc)
        else:
            trajectory_id = str(row.get("trajectory_id", "") or "")
            feature_row = {
                "trajectory_id": trajectory_id,
                "task_id": task_id_from_trajectory_id(trajectory_id),
                **{key: row.get(key) for key in ("cf_split", "cf_row_index", "row_uid") if key in row},
                **_empty_prompt_features(),
            }
        feature_row["trajectory_id"] = str(row.get("trajectory_id", "") or "")
        feature_row["task_id"] = task_id_from_trajectory_id(feature_row["trajectory_id"])
        rows_out.append(feature_row)

    df = pd.DataFrame.from_records(rows_out)
    ordered_cols = [c for c in [*PROMPT_META_COLUMNS, *PROMPT_FEATURE_COLUMNS] if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract prompt-side feature tables from mapping Parquet rows and write the result under the "
            "workspace processed-data directory."
        )
    )
    p.add_argument("--mapping-parquet", type=str, required=True)
    p.add_argument(
        "--trajectory-parquet",
        type=str,
        action="append",
        default=None,
        help="Optional trajectory parquet(s) for messages/problem_statement backfill.",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on output rows after filtering.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy pipe batch size for prompt feature extraction.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard id when splitting the table across jobs (with --num-shards).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of contiguous row shards; each job should use a distinct --shard-index.",
    )
    p.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Skip reading trajectory parquet files and use mapping fields only (saves memory).",
    )
    return p.parse_args()


def _resolve_input_paths(args: argparse.Namespace) -> tuple[Path, list[Path], Path]:
    mapping_path = Path(args.mapping_parquet).expanduser().resolve()
    if not mapping_path.is_file():
        raise SystemExit(f"Mapping parquet not found: {mapping_path}")

    if args.no_trajectories:
        trajectory_paths: list[Path] = []
    elif args.trajectory_parquet:
        trajectory_paths = [Path(path).expanduser().resolve() for path in args.trajectory_parquet]
    else:
        trajectory_paths = []

    output_path = Path(args.output).expanduser().resolve()
    return mapping_path, trajectory_paths, output_path


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


def _read_trajectory_subset(path: Path, wanted_ids: set[str]) -> pd.DataFrame:
    if not wanted_ids:
        return pd.DataFrame(columns=["trajectory_id"])

    ids = sorted(wanted_ids)
    chunks = [ids[i : i + 1000] for i in range(0, len(ids), 1000)]
    parts: list[pd.DataFrame] = []
    for chunk in chunks:
        try:
            part = pd.read_parquet(path, filters=[("trajectory_id", "in", chunk)])
        except Exception:
            part = pd.read_parquet(path)
            part = part[part["trajectory_id"].astype(str).isin(wanted_ids)]

        if "trajectory_id" not in part.columns:
            raise SystemExit(f"trajectory parquet missing trajectory_id: {path}")
        if len(part):
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["trajectory_id"])

    tdf = pd.concat(parts, ignore_index=True)
    keep_cols = [
        c for c in ("trajectory_id", "messages", "problem_statement", "base_problem_statement") if c in tdf.columns
    ]
    if not keep_cols:
        return pd.DataFrame(columns=["trajectory_id"])
    return tdf[keep_cols].copy()


def main() -> int:
    args = parse_args()
    mapping_path, trajectory_paths, out_path = _resolve_input_paths(args)

    df = pd.read_parquet(mapping_path)
    if "trajectory_id" not in df.columns:
        raise SystemExit("Mapping parquet missing trajectory_id.")

    # Apply shard/max-rows before any trajectory loading to keep smoke runs cheap.
    df = _subset_mapping_rows(df, args)

    if args.no_trajectories:
        existing_traj_paths = []
    elif args.trajectory_parquet is None:
        existing_traj_paths = [path for path in trajectory_paths if path.is_file()]
    else:
        missing = [str(path) for path in trajectory_paths if not path.is_file()]
        if missing:
            raise SystemExit(f"Trajectory parquet(s) not found: {missing}")
        existing_traj_paths = trajectory_paths

    if existing_traj_paths and len(df):
        wanted_ids = set(df["trajectory_id"].dropna().astype(str).tolist())
        traj_frames: list[pd.DataFrame] = []
        for path in existing_traj_paths:
            tdf = _read_trajectory_subset(path, wanted_ids)
            if len(tdf):
                traj_frames.append(tdf)
        if traj_frames:
            traj_df = pd.concat(traj_frames, ignore_index=True)
            traj_df = traj_df.drop_duplicates(subset=["trajectory_id"], keep="first")
            df = _merge_mapping_and_trajectories(df, traj_df)

    if len(df) == 0:
        feat_df = pd.DataFrame(columns=[*PROMPT_META_COLUMNS, *PROMPT_FEATURE_COLUMNS])
    else:
        feat_df = extract_prompt_features_many(df.to_dict("records"), batch_size=max(1, int(args.batch_size)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_features_parquet(feat_df, out_path, index=False)
    print(f"Wrote {len(feat_df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())