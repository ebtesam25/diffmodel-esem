#!/usr/bin/env python3
"""Reproduce the prediction-stratified SHAP analysis in paper Section 4.3.2.

By default this **refits** ``any_success`` XGBoost from the published
``paper_run`` hyperparameters (400 trees, lr=0.03, ...) on the frozen train
split. It does not load the archived pickle weights and never writes into
``results/paper_run``.

Use ``--load-saved-model`` to analyze an already-saved pickle instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
LIB = REPLICATION_ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from replication.data import load_task_level_data, train_test_indices_from_targets
from replication.paths import (
    default_paper_run_dir,
    default_tuned_models_dir,
    load_paper_xgb_hyperparams,
    results_dir_for_rq,
)
from replication.shap_analysis import (
    _default_shap_parallel_jobs,
    _explainer_expected_scalar,
    _shap_values_matrix,
    _chunked_parallel_tree_shap,
)
from replication.tuned_evaluation import assert_replication_dataset, outcome_model_prefix

TARGET = "any_success"
TAIL_FRACTION = 0.10
MID_BAND_HALF_WIDTH = 0.05
TOP_K = 5
EXPECTED_FEATURES = 54
EXPECTED_PROMPT_FEATURES = 27
EXPECTED_STRUCTURAL_FEATURES = 27
PAPER_EXPECTED_VALUE = 0.730498520774185
PAPER_CORRECT_COUNTS = {"easy": 893, "mid_band": 575, "hard": 886}
PAPER_PROMPT_TOP5_CORRECT = {"easy": 239, "mid_band": 404, "hard": 60}
PAPER_R_MEDIAN = {"easy": 0.205, "mid_band": 0.404, "hard": 0.108}
PAPER_NAMED_MID_COUNTS = {
    "prompt_pronouns_per_sentence": 64,
    "prompt_mean_conj_chain_len": 58,
    "prompt_mean_competing_dependents_per_head": 54,
}
BAND_ORDER = ("easy", "mid_band", "hard")
STRUCTURAL_PREFIXES = ("patch_", "repo_", "ast_", "func_")
QUANTITIES = (
    "net_shap",
    "max_struct_abs_shap",
    "sum_struct_abs_shap",
    "max_prompt_abs_shap",
    "sum_prompt_abs_shap",
    "prompt_struct_max_ratio",
)
NAMED_PROMPT_FEATURES = (
    "prompt_pronouns_per_sentence",
    "prompt_mean_conj_chain_len",
    "prompt_mean_competing_dependents_per_head",
)


def _split_indices(
    outcomes: pd.DataFrame, *, test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the same plain task-level split used by RQ1/RQ3."""
    y = outcomes[TARGET].to_numpy()
    if "split" in outcomes.columns:
        return train_test_indices_from_targets(outcomes, y)
    indices = np.arange(len(outcomes))
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) >= 2 else None,
    )


def _load_model(tuned_root: Path) -> Tuple[Any, List[str], str]:
    prefix = outcome_model_prefix(TARGET)
    path = tuned_root / prefix / f"{prefix}_final_xgb_model.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing tuned model: {path}\n"
            "Run: python rq1/run_tuned_evaluation.py --verbose"
        )
    pack = joblib.load(path)
    if not isinstance(pack, dict) or "model" not in pack or "feature_names" not in pack:
        raise ValueError(f"Unexpected tuned-model payload in {path}")
    return pack["model"], list(pack["feature_names"]), prefix


def _paper_feature_names() -> List[str]:
    path = REPLICATION_ROOT / "data" / "feature_manifest" / "features_vif_selected_54.json"
    names = json.loads(path.read_text())
    if len(names) != EXPECTED_FEATURES:
        raise ValueError(f"VIF feature manifest has {len(names)} names, expected {EXPECTED_FEATURES}")
    return list(names)


def _refit_paper_xgb(
    X_train: pd.DataFrame, y_train: np.ndarray, *, random_state: int, verbose: bool
) -> Tuple[Any, Dict[str, Any]]:
    """Train a new any_success XGBoost from published hyperparameters, not archived weights."""
    import xgboost as xgb

    params = load_paper_xgb_hyperparams(TARGET)
    if verbose:
        print("Refitting any_success XGBoost from paper hyperparameters (new trees, not the pickle):")
        print(f"  {params}")
        print(f"  paper_run dir left untouched: {default_paper_run_dir()}")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train)
    n_trees = int(model.get_booster().num_boosted_rounds())
    if n_trees != int(params["n_estimators"]):
        raise ValueError(f"Refit produced {n_trees} trees, expected {params['n_estimators']}")
    return model, params


def _forbid_frozen_write(path: Path) -> None:
    resolved = path.resolve()
    frozen = (REPLICATION_ROOT / "results" / "paper_run").resolve()
    try:
        resolved.relative_to(frozen)
    except ValueError:
        return
    raise SystemExit(f"Refusing to write into frozen paper_run: {resolved}")


def _split_settings(tuned_root: Path) -> Tuple[float, int]:
    manifest_path = tuned_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        return 0.2, 42
    manifest = json.loads(manifest_path.read_text())
    cli = manifest.get("cli_args") or {}
    if cli.get("aggregate_by_repo") or cli.get("group_by_repo"):
        raise SystemExit(
            "Section 4.3.2 uses the plain stratified task-level split, not a repository-grouped run."
        )
    return float(cli.get("test_size", 0.2)), int(cli.get("random_state", 42))


def _feature_masks(feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_names) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} post-VIF features, found {len(feature_names)}"
        )
    prompt = np.asarray([name.startswith("prompt_") for name in feature_names])
    structural = ~prompt
    unexpected = [
        name
        for name in np.asarray(feature_names)[structural]
        if not name.startswith(STRUCTURAL_PREFIXES)
    ]
    if prompt.sum() != EXPECTED_PROMPT_FEATURES or structural.sum() != EXPECTED_STRUCTURAL_FEATURES:
        raise ValueError(
            "Expected a 27/27 prompt/structural split; found "
            f"{prompt.sum()}/{structural.sum()}"
        )
    if unexpected:
        raise ValueError(f"Unexpected non-prompt feature families: {unexpected}")
    return prompt, structural


def _compute_or_load_shap(
    model: Any,
    X_test: pd.DataFrame,
    cache_path: Path,
    *,
    refresh: bool,
    parallel_jobs: int,
    verbose: bool,
) -> Tuple[np.ndarray, float]:
    task_ids = X_test.index.astype(str).to_numpy()
    feature_names = X_test.columns.astype(str).to_numpy()
    if cache_path.is_file() and not refresh:
        # ``allow_pickle=True`` also supports the authors' original cache,
        # whose string metadata was saved as NumPy object arrays. Numeric SHAP
        # values are still loaded directly and validated below.
        with np.load(cache_path, allow_pickle=True) as cache:
            required = {"shap_values", "task_id", "feature_names", "expected_value"}
            missing = required - set(cache.files)
            if missing:
                raise ValueError(f"SHAP cache is missing arrays: {sorted(missing)}")
            matrix = np.asarray(cache["shap_values"], dtype=float)
            cached_ids = cache["task_id"].astype(str)
            cached_features = cache["feature_names"].astype(str)
            expected_value = float(np.asarray(cache["expected_value"]).reshape(-1)[0])
        if matrix.shape != X_test.shape:
            raise ValueError(
                f"Cached SHAP shape {matrix.shape} does not match holdout {X_test.shape}"
            )
        if not np.array_equal(cached_ids, task_ids):
            if len(set(cached_ids)) != len(cached_ids) or len(set(task_ids)) != len(task_ids):
                raise ValueError("Cannot align SHAP cache because task IDs are not unique")
            positions = {task_id: index for index, task_id in enumerate(cached_ids)}
            if set(positions) != set(task_ids):
                raise ValueError("Cached task IDs do not match the replayed holdout task set")
            matrix = matrix[[positions[task_id] for task_id in task_ids], :]
        if not np.array_equal(cached_features, feature_names):
            raise ValueError("Cached feature names do not match the saved model")
        if verbose:
            print(f"Reused SHAP cache: {cache_path}")
        return matrix, expected_value

    import shap

    explainer = shap.TreeExplainer(model)
    if parallel_jobs > 1:
        raw = _chunked_parallel_tree_shap(
            model,
            X_test,
            parallel_jobs,
            mode="values",
            verbose=verbose,
        )
    else:
        raw = explainer.shap_values(X_test)
    matrix = _shap_values_matrix(
        raw, n_samples=len(X_test), n_features=X_test.shape[1]
    )
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != X_test.shape:
        raise ValueError(f"Unexpected SHAP shape {matrix.shape}; expected {X_test.shape}")
    expected_value = _explainer_expected_scalar(explainer, is_regression=False)
    _forbid_frozen_write(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        shap_values=matrix,
        task_id=np.asarray(task_ids, dtype=str),
        feature_names=np.asarray(feature_names, dtype=str),
        expected_value=np.asarray([expected_value], dtype=float),
    )
    if verbose:
        print(f"Wrote SHAP cache: {cache_path}")
    return matrix, expected_value


def _assign_bands(scores: np.ndarray) -> Tuple[np.ndarray, float, int]:
    center = float(np.mean(scores))
    n_tail = int(round(TAIL_FRACTION * len(scores)))
    order = np.argsort(scores, kind="mergesort")
    bands = np.full(len(scores), "other", dtype=object)
    bands[order[:n_tail]] = "hard"
    bands[order[-n_tail:]] = "easy"
    mid = (bands == "other") & (
        np.abs(scores - center) <= MID_BAND_HALF_WIDTH
    )
    bands[mid] = "mid_band"
    return bands, center, n_tail


def _describe(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: np.nan for name in ("mean", "median", "p25", "p75")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def _summaries(
    per_task: pd.DataFrame, chance: float
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    subsets = (
        ("all", per_task),
        ("correctly_classified", per_task[per_task["correctly_classified"]]),
    )
    for subset_name, frame in subsets:
        for band in BAND_ORDER:
            selected = frame[frame["difficulty_band"] == band]
            hits = int((selected["n_prompt_in_top5"] >= 1).sum())
            row: Dict[str, object] = {
                "subset": subset_name,
                "band": band,
                "n": int(len(selected)),
                "mean_y_pred": float(selected["y_pred"].mean()),
                "min_y_pred": float(selected["y_pred"].min()),
                "max_y_pred": float(selected["y_pred"].max()),
                "prompt_in_top5_count": hits,
                "prompt_in_top5_rate": hits / len(selected) if len(selected) else np.nan,
                "chance_prompt_in_top5": chance,
                "deviation_from_chance": (
                    hits / len(selected) - chance if len(selected) else np.nan
                ),
            }
            for quantity in QUANTITIES:
                for stat, value in _describe(selected[quantity]).items():
                    row[f"{quantity}_{stat}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _prompt_feature_counts(
    per_task: pd.DataFrame,
    top5_indices: np.ndarray,
    feature_names: List[str],
    prompt_mask: np.ndarray,
) -> pd.DataFrame:
    mask = (
        per_task["correctly_classified"].to_numpy()
        & (per_task["difficulty_band"].to_numpy() == "mid_band")
    )
    denominator = int(mask.sum())
    counts = np.bincount(
        top5_indices[mask].reshape(-1), minlength=len(feature_names)
    )
    rows = [
        {
            "feature": feature_names[index],
            "top5_count": int(counts[index]),
            "denominator": denominator,
            "top5_rate": float(counts[index] / denominator),
        }
        for index in np.flatnonzero(prompt_mask)
    ]
    return pd.DataFrame(rows).sort_values(
        ["top5_count", "feature"], ascending=[False, True], ignore_index=True
    )


def _print_camera_ready(
    summary: pd.DataFrame,
    prompt_counts: pd.DataFrame,
    *,
    center: float,
    expected_value: float,
    additivity_error: float,
) -> None:
    correct = summary[summary["subset"] == "correctly_classified"].set_index("band")
    print("\n=== RQ3 STRATIFIED SHAP (correctly classified only) ===")
    print(
        f"mean predicted probability={center:.6f}; mid-band="
        f"[{center - MID_BAND_HALF_WIDTH:.6f}, {center + MID_BAND_HALF_WIDTH:.6f}]"
    )
    print(
        f"SHAP scale=log-odds; expected value={expected_value:.12f}; "
        f"max additivity error={additivity_error:.3e}"
    )
    for band in BAND_ORDER:
        row = correct.loc[band]
        n = int(row["n"])
        hits = int(row["prompt_in_top5_count"])
        print(
            f"{band:8s}: n={n}; median net SHAP={row['net_shap_median']:+.3f}; "
            f"prompt in top 5={hits}/{n} ({row['prompt_in_top5_rate']:.1%}); "
            f"median R(x)={row['prompt_struct_max_ratio_median']:.3f}"
        )
    mid_ratio = float(correct.loc["mid_band", "prompt_struct_max_ratio_median"])
    easy_ratio = float(correct.loc["easy", "prompt_struct_max_ratio_median"])
    hard_ratio = float(correct.loc["hard", "prompt_struct_max_ratio_median"])
    print(
        "R(x) contrast: "
        f"mid/easy={mid_ratio / easy_ratio:.2f}x; "
        f"mid/hard={mid_ratio / hard_ratio:.2f}x"
    )
    indexed = prompt_counts.set_index("feature")
    print("Named prompt-feature frequencies in the correct mid-band:")
    for feature in NAMED_PROMPT_FEATURES:
        if feature not in indexed.index:
            raise ValueError(f"Paper-named prompt feature missing: {feature}")
        row = indexed.loc[feature]
        print(
            f"  {feature}: {int(row['top5_count'])}/{int(row['denominator'])} "
            f"({float(row['top5_rate']):.1%})"
        )


def _print_paper_comparison(
    summary: pd.DataFrame,
    prompt_counts: pd.DataFrame,
    *,
    n_estimators: int,
) -> None:
    correct = summary[summary["subset"] == "correctly_classified"].set_index("band")
    indexed = prompt_counts.set_index("feature")
    print("\n=== Comparison to ESEM-CR §4.3.2 (correctly classified) ===")
    print(f"refit n_estimators={n_estimators} (paper=400)")
    for band in BAND_ORDER:
        row = correct.loc[band]
        n = int(row["n"])
        hits = int(row["prompt_in_top5_count"])
        paper_n = PAPER_CORRECT_COUNTS[band]
        paper_hits = PAPER_PROMPT_TOP5_CORRECT[band]
        paper_r = PAPER_R_MEDIAN[band]
        print(
            f"{band:8s}: n={n} (paper {paper_n}); "
            f"prompt in top 5={hits}/{n}={hits / n:.1%} "
            f"(paper {paper_hits}/{paper_n}={paper_hits / paper_n:.1%}); "
            f"median R(x)={row['prompt_struct_max_ratio_median']:.3f} (paper {paper_r:.3f})"
        )
    print("Named prompt-feature frequencies vs paper:")
    for feature, paper_count in PAPER_NAMED_MID_COUNTS.items():
        count = int(indexed.loc[feature, "top5_count"])
        denom = int(indexed.loc[feature, "denominator"])
        print(
            f"  {feature}: {count}/{denom}={count / denom:.1%} "
            f"(paper {paper_count}/{PAPER_CORRECT_COUNTS['mid_band']}="
            f"{paper_count / PAPER_CORRECT_COUNTS['mid_band']:.1%})"
        )


def _finding_holds(summary: pd.DataFrame, *, n_estimators: int) -> List[str]:
    """Return failure reasons if the layered-difficulty pattern did not come back."""
    failures: List[str] = []
    if n_estimators != 400:
        failures.append(f"n_estimators={n_estimators}, expected 400")
    correct = summary[summary["subset"] == "correctly_classified"].set_index("band")
    rates = {band: float(correct.loc[band, "prompt_in_top5_rate"]) for band in BAND_ORDER}
    ratios = {
        band: float(correct.loc[band, "prompt_struct_max_ratio_median"]) for band in BAND_ORDER
    }
    if not (rates["mid_band"] > rates["easy"] > rates["hard"]):
        failures.append(
            "prompt-in-top-5 ordering is not mid > easy > hard: "
            f"{rates['easy']:.3f} / {rates['mid_band']:.3f} / {rates['hard']:.3f}"
        )
    if rates["mid_band"] < 0.55:
        failures.append(
            f"mid-band prompt-in-top-5 rate {rates['mid_band']:.3f} is too far from paper 0.703"
        )
    if not (ratios["mid_band"] > ratios["easy"] and ratios["mid_band"] > ratios["hard"]):
        failures.append(
            "median R(x) is not highest in the mid-band: "
            f"{ratios['easy']:.3f} / {ratios['mid_band']:.3f} / {ratios['hard']:.3f}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuned-dir",
        type=Path,
        default=None,
        help="Directory of saved models. Unused when refitting paper hyperparameters.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: results/analysis/rq3_feature_importance/stratified_shap).",
    )
    parser.add_argument(
        "--shap-cache",
        type=Path,
        default=None,
        help="Optional existing SHAP NPZ; defaults to a refit-specific cache in --outdir.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--shap-parallel-n-jobs", type=int, default=None)
    parser.add_argument("--refresh-shap", action="store_true")
    parser.add_argument(
        "--load-saved-model",
        action="store_true",
        help="Load a saved XGBoost pickle instead of refitting paper hyperparameters.",
    )
    parser.add_argument(
        "--assert-paper-values",
        action="store_true",
        help="Fail unless expected value and correct-only band counts match the camera-ready paper.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tuned_root = (
        args.tuned_dir.resolve() if args.tuned_dir else default_tuned_models_dir()
    )
    test_size, random_state = _split_settings(tuned_root)

    outdir = (
        args.outdir.resolve()
        if args.outdir
        else (results_dir_for_rq("rq3") / "stratified_shap")
    )
    _forbid_frozen_write(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assert_replication_dataset()
    X, outcomes = load_task_level_data()
    if TARGET not in outcomes.columns:
        raise ValueError(f"Dataset has no {TARGET!r} outcome")
    idx_train, test_indices = _split_indices(
        outcomes, test_size=test_size, random_state=random_state
    )
    prefix = outcome_model_prefix(TARGET)
    paper_params: Dict[str, Any] = {}
    if args.load_saved_model:
        model, feature_names, prefix = _load_model(tuned_root)
        n_estimators = int(model.get_params().get("n_estimators") or 0)
    else:
        feature_names = _paper_feature_names()
        missing = [name for name in feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"VIF features missing from dataset: {missing}")
        X_train = X.iloc[idx_train][feature_names]
        y_train = outcomes[TARGET].iloc[idx_train].to_numpy(dtype=int)
        model, paper_params = _refit_paper_xgb(
            X_train, y_train, random_state=random_state, verbose=args.verbose
        )
        n_estimators = int(paper_params["n_estimators"])
        model_path = outdir / f"{prefix}_refit_paper_hyperparams.pkl"
        _forbid_frozen_write(model_path)
        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "hyperparams": paper_params,
                "source": "refit_paper_hyperparams",
            },
            model_path,
        )
        if args.verbose:
            print(f"Wrote refit model (not paper_run): {model_path}")
    missing = [name for name in feature_names if name not in X.columns]
    if missing:
        raise ValueError(f"Saved model features missing from dataset: {missing}")
    prompt_mask, structural_mask = _feature_masks(feature_names)
    X_test = X.iloc[test_indices][feature_names].copy()
    y_true = outcomes[TARGET].iloc[test_indices].to_numpy(dtype=int)
    y_pred = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    jobs = (
        max(1, args.shap_parallel_n_jobs)
        if args.shap_parallel_n_jobs is not None
        else _default_shap_parallel_jobs(None)
    )
    cache_path = (
        args.shap_cache.resolve()
        if args.shap_cache
        else outdir / f"{prefix}_shap_values_holdout_refit.npz"
    )
    if args.shap_cache is None:
        _forbid_frozen_write(cache_path)
    shap_values, expected_value = _compute_or_load_shap(
        model,
        X_test,
        cache_path,
        refresh=args.refresh_shap,
        parallel_jobs=jobs,
        verbose=args.verbose,
    )
    # Use the Booster path explicitly: this is the same raw-margin call used
    # for the paper artifacts and avoids sklearn-wrapper version differences.
    import xgboost as xgb

    margins = np.asarray(
        model.get_booster().predict(
            xgb.DMatrix(X_test, feature_names=feature_names),
            output_margin=True,
        ),
        dtype=float,
    )
    net_shap = shap_values.sum(axis=1)
    additivity_error = float(
        np.max(np.abs(expected_value + net_shap - margins))
    )
    # The frozen paper artifact has a small (~0.006 log-odds), nearly constant
    # TreeExplainer/XGBoost base-value offset across package versions. This is
    # far below the error obtained by treating SHAP values as probabilities
    # and does not affect rankings, bands, or per-feature magnitudes.
    if additivity_error > 1e-2:
        raise ValueError(
            "SHAP values are not additive on the raw-margin scale; "
            f"maximum error={additivity_error:.6g}"
        )

    bands, center, n_tail = _assign_bands(y_pred)
    abs_shap = np.abs(shap_values)
    prompt_abs = abs_shap[:, prompt_mask]
    structural_abs = abs_shap[:, structural_mask]
    top5_indices = np.argsort(-abs_shap, axis=1, kind="mergesort")[:, :TOP_K]
    n_prompt_top5 = prompt_mask[top5_indices].sum(axis=1).astype(int)
    max_prompt = prompt_abs.max(axis=1)
    max_structural = structural_abs.max(axis=1)
    ratio = np.divide(
        max_prompt,
        max_structural,
        out=np.full(len(max_prompt), np.nan),
        where=max_structural > 0,
    )
    correct = (y_pred >= args.threshold).astype(int) == y_true
    per_task = pd.DataFrame(
        {
            "task_id": X_test.index.astype(str),
            "row_index_in_shap_frame": np.arange(len(X_test)),
            "y_true": y_true,
            "y_pred": y_pred,
            "correctly_classified": correct,
            "difficulty_band": bands,
            "net_shap": net_shap,
            "max_struct_abs_shap": max_structural,
            "sum_struct_abs_shap": structural_abs.sum(axis=1),
            "max_prompt_abs_shap": max_prompt,
            "sum_prompt_abs_shap": prompt_abs.sum(axis=1),
            "prompt_struct_max_ratio": ratio,
            "n_prompt_in_top5": n_prompt_top5,
        }
    )
    chance = 1.0 - comb(EXPECTED_STRUCTURAL_FEATURES, TOP_K) / comb(
        EXPECTED_FEATURES, TOP_K
    )
    summary = _summaries(per_task, chance)
    prompt_counts = _prompt_feature_counts(
        per_task, top5_indices, feature_names, prompt_mask
    )

    correct_counts = {
        band: int(
            (
                per_task["correctly_classified"]
                & (per_task["difficulty_band"] == band)
            ).sum()
        )
        for band in BAND_ORDER
    }
    if args.assert_paper_values:
        if abs(expected_value - PAPER_EXPECTED_VALUE) > 1e-9:
            raise SystemExit(
                f"Expected-value mismatch: {expected_value} != {PAPER_EXPECTED_VALUE}"
            )
        if correct_counts != PAPER_CORRECT_COUNTS:
            raise SystemExit(
                f"Correct-only band counts {correct_counts} != paper {PAPER_CORRECT_COUNTS}"
            )

    per_task_path = outdir / f"{prefix}_shap_stratified_per_task.csv"
    summary_path = outdir / f"{prefix}_shap_stratified_summary.csv"
    prompt_path = outdir / f"{prefix}_prompt_top5_mid_band.csv"
    per_task.to_csv(per_task_path, index=False)
    summary.to_csv(summary_path, index=False)
    prompt_counts.to_csv(prompt_path, index=False)
    run_manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "tuned_models_dir": str(tuned_root),
        "hyperparameter_source": "paper_run_json_refit" if paper_params else "saved_model",
        "xgb_hyperparams": paper_params or None,
        "n_estimators": n_estimators,
        "target": TARGET,
        "n_holdout": len(per_task),
        "n_tail_before_correctness_filter": n_tail,
        "mean_predicted_probability": center,
        "mid_band_half_width": MID_BAND_HALF_WIDTH,
        "classification_threshold": args.threshold,
        "correct_only_band_counts": correct_counts,
        "expected_value_log_odds": expected_value,
        "max_additivity_error_log_odds": additivity_error,
        "prompt_in_top5_chance": chance,
        "feature_family_counts": {
            "prompt": int(prompt_mask.sum()),
            "structural": int(structural_mask.sum()),
        },
        "outputs": {
            "per_task": str(per_task_path),
            "summary": str(summary_path),
            "prompt_feature_counts": str(prompt_path),
            "shap_cache": str(cache_path),
        },
    }
    (outdir / "stratified_shap_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )

    _print_camera_ready(
        summary,
        prompt_counts,
        center=center,
        expected_value=expected_value,
        additivity_error=additivity_error,
    )
    _print_paper_comparison(summary, prompt_counts, n_estimators=n_estimators)
    finding_failures = _finding_holds(summary, n_estimators=n_estimators)
    if finding_failures:
        raise SystemExit(
            "Refit did not recover the Finding 3 pattern:\n  - " + "\n  - ".join(finding_failures)
        )
    print(f"\nOutputs written under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
