#!/usr/bin/env python3
"""
Plot SHAP waterfall decompositions for selected holdout tasks (RQ3 figures).

Uses saved tuned XGBoost models from ``run_model_shap.py`` plus the published task-level dataset.
Does **not** re-run hyperparameter tuning or full-test SHAP export.

Typical use (paper: borderline-difficulty tasks, :math:`\\hat{p} \\approx 0.68`):

- ``--average-n 10`` — tasks with predicted probability closest to the holdout mean
- ``--plot-suffix _avg10`` — output filename tag

Also writes easiest/hardest single-task waterfalls unless you only request averages.

Example::

    python rq3/plot_shap_waterfalls.py \\
      --outdir results/analysis/rq3_feature_importance/model_shap \\
      --targets any_success --average-n 10 --plot-suffix _avg10 --verbose
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
_LIB = REPLICATION_ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from replication.data import load_task_level_data, train_test_indices_from_targets
from replication.paths import default_dataset_path, results_dir_for_rq

PRIMARY_SLUG = "xgb"


def _load_comparison_module():
    path = REPLICATION_ROOT / "rq1" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("_task_level_model_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shap_values_matrix(
    shap_values: Any,
    *,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            arr = np.asarray(shap_values[1])
        else:
            arr = np.asarray(shap_values[0])
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 1 and n_samples is not None and n_features is not None:
        if arr.size == n_samples * n_features:
            arr = arr.reshape(n_samples, n_features)
        elif arr.size == n_samples and n_features == 1:
            arr = arr.reshape(n_samples, 1)
    return arr


def _explainer_expected_scalar(explainer: Any, *, is_regression: bool) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple)):
        ev = ev[1] if len(ev) > 1 and not is_regression else ev[0]
    a = np.asarray(ev, dtype=float).ravel()
    if not is_regression and a.size >= 2:
        return float(a[1])
    return float(a[0]) if a.size > 0 else 0.0


def _one_row_shap_vector(explainer: Any, X_shap: pd.DataFrame, ridx: int) -> np.ndarray:
    Xi = X_shap.iloc[[ridx]]
    raw = explainer.shap_values(Xi)
    mat = _shap_values_matrix(raw, n_samples=1, n_features=X_shap.shape[1])
    return np.asarray(mat, dtype=float).reshape(-1)


def _position_for_task_id(X_shap: pd.DataFrame, tid: str) -> int:
    """Integer iloc position of ``tid`` in ``X_shap.index`` (exact string match)."""
    want = str(tid).strip()
    labels = np.asarray(X_shap.index.astype(str))
    matches = np.flatnonzero(labels == want)
    if matches.size == 0:
        sample = list(X_shap.index[:5])
        raise ValueError(
            f"task_id {want!r} not in this run's X_shap rows (n={len(X_shap)}). "
            f"If the main job used --shap-sample, the task must fall in that subsample. "
            f"First index values (sample): {sample!r}"
        )
    if matches.size > 1:
        raise ValueError(f"task_id {want!r} matched {int(matches.size)} duplicate index rows")
    return int(matches[0])


def _score_center_and_anchor(
    scores: np.ndarray,
    average_pick: str,
    average_prob: Optional[float],
) -> Tuple[float, str]:
    if average_prob is not None:
        p = float(average_prob)
        return p, f"anchor_prob:{p}"
    if average_pick == "median":
        return float(np.median(scores)), "anchor_median_of_y_pred"
    if average_pick != "mean":
        raise ValueError(f"unknown average_pick: {average_pick!r}")
    return float(np.mean(scores)), "anchor_mean_of_y_pred"


def _build_average_scenarios(
    scores: np.ndarray,
    X_shap: pd.DataFrame,
    *,
    average_n: int,
    average_task_id: Optional[str],
    average_pick: str,
    average_prob: Optional[float],
) -> List[Tuple[str, int, str]]:
    """
    Return list of (scenario_label, row_index, selection_rule) for each "average" waterfall.
    ``average_n`` must be >= 1.
    """
    if average_n < 1:
        raise ValueError("average_n must be >= 1")
    if average_task_id is not None:
        if average_n != 1:
            raise ValueError("When --average-task-id is set, use --average-n 1 (single fixed row).")
        r = _position_for_task_id(X_shap, average_task_id)
        return [("average", r, f"fixed_task_id:{average_task_id.strip()}")]

    center, anchor = _score_center_and_anchor(scores, average_pick, average_prob)
    dist = np.abs(scores.astype(float) - center)
    order = np.argsort(dist, kind="mergesort")
    picked: List[int] = []
    seen: set = set()
    for j in order.flat:
        ij = int(j)
        if ij in seen:
            continue
        seen.add(ij)
        picked.append(ij)
        if len(picked) >= average_n:
            break

    slots: List[Tuple[str, int, str]] = []
    n_got = len(picked)
    for k, ridx in enumerate(picked):
        rank = k + 1
        rule = f"{anchor}_distance_rank{rank:02d}_of_{n_got}_requested_{average_n}"
        if average_n == 1:
            slots.append(("average", ridx, rule))
        else:
            slots.append((f"average_{rank:02d}", ridx, rule))
    return slots


def render_waterfall_scenarios(
    model: Any,
    X_shap: pd.DataFrame,
    explainer: Any,
    *,
    artifact_prefix: str,
    out_path: Path,
    is_regression: bool,
    y_holdout: Optional[pd.Series] = None,
    plot_filename_suffix: str = "",
    verbose: bool = False,
    average_n: int = 1,
    average_task_id: Optional[str] = None,
    average_pick: str = "mean",
    average_prob: Optional[float] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Optional[str]]:
    import matplotlib.pyplot as plt
    import shap

    pfx = artifact_prefix
    if is_regression:
        scores = np.clip(model.predict(X_shap), 0.0, 1.0).astype(float)
    else:
        scores = model.predict_proba(X_shap)[:, 1].astype(float)
    ev_s = _explainer_expected_scalar(explainer, is_regression=is_regression)

    avg_slots = _build_average_scenarios(
        scores,
        X_shap,
        average_n=average_n,
        average_task_id=average_task_id,
        average_pick=average_pick,
        average_prob=average_prob,
    )
    if len(avg_slots) < average_n and average_task_id is None:
        if verbose:
            print(
                f"  waterfall: requested {average_n} average rows but only "
                f"{len(avg_slots)} distinct rows in X_shap (n={len(X_shap)})"
            )

    scenarios: List[Tuple[str, int, str]] = [
        ("easiest", int(np.argmax(scores)), ""),
        ("hardest", int(np.argmin(scores)), ""),
    ]
    scenarios.extend(avg_slots)

    y_vec: Optional[np.ndarray] = None
    if y_holdout is not None:
        aligned = y_holdout.reindex(X_shap.index)
        n_miss = int(aligned.isna().sum())
        if n_miss and verbose:
            print(f"  waterfall: {n_miss} rows missing y_holdout after reindex to X_shap")
        y_vec = aligned.to_numpy()

    wf: Dict[str, str] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for label, ridx, avg_rule in scenarios:
        fname = f"{pfx}_shap_waterfall_{label}{plot_filename_suffix}.png"
        try:
            vals = _one_row_shap_vector(explainer, X_shap, ridx)
            row_doc: Dict[str, Any] = {
                "scenario": label,
                "task_id": str(X_shap.index[ridx]),
                "row_index_in_shap_frame": int(ridx),
                "y_pred": float(scores[ridx]),
                "expected_value_base": float(ev_s),
                "diagram_path": str((out_path / fname).resolve()),
            }
            if avg_rule:
                row_doc["average_selection_rule"] = avg_rule
            if y_vec is not None:
                yt = y_vec[ridx]
                if isinstance(yt, (float, np.floating)) and np.isnan(yt):
                    row_doc["y_true"] = None
                else:
                    row_doc["y_true"] = float(yt) if isinstance(yt, (np.floating, float)) else yt

            exp_row = shap.Explanation(
                values=vals,
                base_values=ev_s,
                data=X_shap.iloc[ridx].to_numpy(dtype=float),
                feature_names=list(X_shap.columns),
            )
            plt.close("all")
            shap.plots.waterfall(exp_row, max_display=15, show=False)
            plt.gcf().savefig(out_path / fname, dpi=150, bbox_inches="tight")
            plt.close("all")
            wf[label] = str((out_path / fname).resolve())
            manifest_rows.append(row_doc)
        except Exception as e:
            if verbose:
                print(f"  waterfall {label} failed: {e}")
            try:
                plt.close("all")
            except Exception:
                pass

    csv_path: Optional[str] = None
    if manifest_rows:
        csv_name = f"{pfx}_shap_waterfall_scenarios{plot_filename_suffix}.csv"
        csv_full = out_path / csv_name
        pd.DataFrame(manifest_rows).to_csv(csv_full, index=False)
        csv_path = str(csv_full.resolve())
    return wf, manifest_rows, csv_path


def _assert_dataset() -> None:
    path = default_dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Task-level dataset missing: {path}")


def _split_indices(
    n_rows: int,
    y_strat: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    strat = y_strat if len(np.unique(y_strat)) >= 2 else None
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=strat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Same --outdir as the finished task_level_model_shap run (contains experiment_manifest.json).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to experiment_manifest.json (default: <outdir>/experiment_manifest.json).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated targets (default: targets_run from manifest).",
    )
    parser.add_argument(
        "--plot-suffix",
        type=str,
        default="_annotated",
        help="Suffix inserted before .png (default: _annotated). Use empty string to overwrite originals.",
    )
    parser.add_argument("--random-state", type=int, default=None, help="Override manifest random_state.")
    parser.add_argument("--test-size", type=float, default=None, help="Override manifest test_size.")
    parser.add_argument(
        "--average-task-id",
        default=None,
        help=(
            'Use this exact task_id (DataFrame index string) for the "average" waterfall only '
            "(requires --average-n 1). Must exist in the replayed X_shap."
        ),
    )
    parser.add_argument(
        "--average-pick",
        choices=("mean", "median"),
        default="mean",
        help='When --average-task-id is unset: pick test row with y_pred closest to mean or median over X_shap (default: mean).',
    )
    parser.add_argument(
        "--average-prob",
        type=float,
        default=None,
        metavar="P",
        help="Center P for picking average row(s): smallest |y_pred - P| (e.g. 0.5). Overrides --average-pick.",
    )
    parser.add_argument(
        "--average-n",
        type=int,
        default=1,
        metavar="N",
        help=(
            'How many distinct "average" waterfalls (N rows with smallest |y_pred - center|; '
            "center from mean, median, or --average-prob). Filenames use average_01 … average_N when N>1."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.average_n < 1:
        parser.error("--average-n must be >= 1")

    out_root = Path(args.outdir).resolve()
    man_path = Path(args.manifest).resolve() if args.manifest else out_root / "experiment_manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {man_path}")

    manifest = json.loads(man_path.read_text())
    cli = manifest.get("cli_args") or {}
    if cli.get("aggregate_by_repo") or cli.get("group_by_repo"):
        raise SystemExit(
            "This helper only supports plain stratified task-level splits (no repo grouping / aggregation)."
        )
    if not cli.get("use_task_loader", True):
        raise SystemExit("Regeneration with --no-use-task-loader is not implemented in this script.")

    test_size = float(args.test_size if args.test_size is not None else cli.get("test_size", 0.2))
    random_state = int(args.random_state if args.random_state is not None else cli.get("random_state", 42))
    shap_sample = cli.get("shap_sample")
    if shap_sample is not None:
        shap_sample = int(shap_sample)

    targets_list: List[str]
    if args.targets:
        targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets_list = list(manifest.get("targets_run") or [])
    if not targets_list:
        raise SystemExit("No targets (set --targets or ensure manifest has targets_run).")

    _assert_dataset()
    X, task_outcomes = load_task_level_data()

    import shap  # noqa: WPS433

    y_any = task_outcomes["any_success"].values
    if "split" in task_outcomes.columns:
        idx_train, idx_test = train_test_indices_from_targets(task_outcomes, y_any)
    else:
        y_strat = np.asarray(y_any)
        if y_strat.dtype.kind == "f":
            y_strat = (y_strat >= 0.5).astype(int)
        else:
            y_strat = y_strat.astype(int)
        idx_train, idx_test = _split_indices(
            len(X), y_strat, test_size=test_size, random_state=random_state
        )
    X_test_full = X.iloc[idx_test]

    if args.verbose:
        print(f"Replayed holdout: n_test={len(idx_test)} (test_size={test_size}, random_state={random_state})")

    for target in targets_list:
        if target not in task_outcomes.columns:
            raise ValueError(f"target {target!r} not in outcomes columns")
        pfx = f"{target}_{PRIMARY_SLUG}"
        target_dir = out_root / pfx
        pkl_path = target_dir / f"{pfx}_final_xgb_model.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(f"Missing saved model: {pkl_path}")

        pack = joblib.load(pkl_path)
        model = pack["model"]
        feat_names: List[str] = list(pack["feature_names"])
        missing = [c for c in feat_names if c not in X_test_full.columns]
        if missing:
            raise ValueError(f"{pfx}: {len(missing)} model features missing from X_test: {missing[:8]}...")

        X_test_model = X_test_full[feat_names].copy()
        y_all = task_outcomes[target].values
        y_test = y_all[idx_test]
        y_series = pd.Series(y_test, index=X_test_model.index)

        y_train_vals = y_all[idx_train]
        try:
            ttype = type_of_target(y_train_vals)
        except Exception:
            ttype = "continuous"
        is_regression = ttype not in ("binary", "multiclass")

        if shap_sample is not None and shap_sample < len(X_test_model):
            X_shap = X_test_model.sample(n=shap_sample, random_state=42)
            if args.verbose:
                print(f"  {pfx}: SHAP subsample replay n={shap_sample} (matches manifest shap_sample)")
        else:
            X_shap = X_test_model

        explainer = shap.TreeExplainer(model)
        suffix = args.plot_suffix or ""
        wf, rows, csvp = render_waterfall_scenarios(
            model,
            X_shap,
            explainer,
            artifact_prefix=pfx,
            out_path=target_dir,
            is_regression=is_regression,
            y_holdout=y_series,
            plot_filename_suffix=suffix,
            verbose=args.verbose,
            average_n=args.average_n,
            average_task_id=args.average_task_id,
            average_pick=args.average_pick,
            average_prob=args.average_prob,
        )
        print(f"{pfx}: wrote {len(wf)} waterfall PNGs; manifest -> {csvp}")
        if args.verbose and rows:
            for r in rows:
                print("  ", r)


if __name__ == "__main__":
    main()
