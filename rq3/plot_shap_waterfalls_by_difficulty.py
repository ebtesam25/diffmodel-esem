#!/usr/bin/env python3
"""
Plot SHAP waterfalls for **easy**, **hard**, and **borderline** holdout tasks (RQ3 figures).

Reuses saved models from ``run_model_shap.py`` (same split and pickles as ``plot_shap_waterfalls.py``).
Writes:

  ``<target_dir>/shap_waterfalls_easy/``     — top N rows by the ranking score (highest = easiest)
  ``<target_dir>/shap_waterfalls_hard/``     — bottom N rows (lowest = hardest)
  ``<target_dir>/shap_waterfalls_average/``  — N rows with smallest ``|y_pred - center|`` (default
  ``center`` = mean of ``y_pred`` on ``X_shap``, i.e. near the typical holdout prediction; same rule
  as ``regenerate_task_level_shap_waterfalls.py`` and aligned with SHAP waterfall baselines
  ``E[f(X)]`` over the explanation set)

With ``--beeswarm``, also writes **one beeswarm per bucket** (same N rows per bucket), using batched
Tree SHAP on easy / hard / **average** subsets (average rows match the waterfall average pick).
Feature **y-order** is shared across the three beeswarms: same as ``order=shap.Explanation.abs.mean(0)``
evaluated on SHAP values for the **full** replayed ``X_shap`` (full holdout test frame, or the
manifest ``shap_sample`` subsample when present), under:

  ``<target_dir>/shap_beeswarm_buckets/``

Waterfalls and beeswarms (unless you pass ``--beeswarm-max-display K``) show **all** model features so
plots match the full fitted feature set.

Default ranking matches the existing **easiest** / **hardest** waterfalls: model ``y_pred`` on the
replayed ``X_shap`` frame (probability for classifiers, clipped regression preds otherwise).

Optional ``--rank-by pass_rate`` ranks by empirical task pass rate (1 − pass_rate = harder), aligned
to ``X_shap`` by ``task_id``.

Example::

    python rq3/plot_shap_waterfalls_by_difficulty.py \\
      --outdir results/analysis/rq3_feature_importance/model_shap \\
      --n-each 10 --plot-suffix _top10 --verbose

    python rq3/plot_shap_waterfalls_by_difficulty.py \\
      --outdir ... --n-each 10 --beeswarm --plot-suffix _buckets10

Limitations: same as ``plot_shap_waterfalls.py`` (task loader only; no repo grouping).
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

PRIMARY_SLUG = "xgb"


def _load_plot_waterfalls_module():
    path = REPLICATION_ROOT / "rq3" / "plot_shap_waterfalls.py"
    spec = importlib.util.spec_from_file_location("_plot_shap_waterfalls", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_comparison_module():
    path = REPLICATION_ROOT / "rq1" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("_task_level_model_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _y_pred_scores(model: Any, X_shap: pd.DataFrame, *, is_regression: bool) -> np.ndarray:
    if is_regression:
        return np.clip(model.predict(X_shap), 0.0, 1.0).astype(float)
    return model.predict_proba(X_shap)[:, 1].astype(float)


def _rank_row_indices(scores: np.ndarray, *, n: int, high_first: bool) -> List[int]:
    """Return up to ``n`` distinct iloc positions: highest scores first if ``high_first`` else lowest."""
    s = np.asarray(scores, dtype=float)
    if high_first:
        order = np.argsort(-s, kind="mergesort")
    else:
        order = np.argsort(s, kind="mergesort")
    out: List[int] = []
    seen: set = set()
    for j in order.flat:
        ij = int(j)
        if ij in seen:
            continue
        seen.add(ij)
        out.append(ij)
        if len(out) >= n:
            break
    return out


def render_extreme_bucket(
    rtw: Any,
    model: Any,
    X_shap: pd.DataFrame,
    explainer: Any,
    *,
    artifact_prefix: str,
    out_path: Path,
    is_regression: bool,
    row_indices: List[int],
    bucket_label: str,
    rank_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    y_holdout: Optional[pd.Series],
    plot_filename_suffix: str,
    selection_rule: str,
    verbose: bool,
) -> Tuple[int, Optional[str]]:
    import matplotlib.pyplot as plt
    import shap

    pfx = artifact_prefix
    out_path.mkdir(parents=True, exist_ok=True)
    ev_s = rtw._explainer_expected_scalar(explainer, is_regression=is_regression)

    y_vec: Optional[np.ndarray] = None
    if y_holdout is not None:
        aligned = y_holdout.reindex(X_shap.index)
        y_vec = aligned.to_numpy()

    manifest_rows: List[Dict[str, Any]] = []
    n_ok = 0
    for rank, ridx in enumerate(row_indices, start=1):
        scenario = f"{bucket_label}_{rank:02d}"
        fname = f"{pfx}_shap_waterfall_{scenario}{plot_filename_suffix}.png"
        try:
            vals = rtw._one_row_shap_vector(explainer, X_shap, ridx)
            row_doc: Dict[str, Any] = {
                "scenario": scenario,
                "bucket": bucket_label,
                "task_id": str(X_shap.index[ridx]),
                "row_index_in_shap_frame": int(ridx),
                "y_pred": float(y_pred_scores[ridx]),
                "rank_score": float(rank_scores[ridx]),
                "expected_value_base": float(ev_s),
                "selection_rule": selection_rule,
                "diagram_path": str((out_path / fname).resolve()),
            }
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
            shap.plots.waterfall(exp_row, max_display=int(X_shap.shape[1]), show=False)
            plt.gcf().savefig(out_path / fname, dpi=150, bbox_inches="tight")
            plt.close("all")
            manifest_rows.append(row_doc)
            n_ok += 1
        except Exception as e:
            if verbose:
                print(f"  {bucket_label} waterfall {scenario} failed: {e}")
            try:
                plt.close("all")
            except Exception:
                pass

    csv_path: Optional[str] = None
    if manifest_rows:
        csv_name = f"{pfx}_shap_waterfall_{bucket_label}_manifest{plot_filename_suffix}.csv"
        csv_full = out_path / csv_name
        pd.DataFrame(manifest_rows).to_csv(csv_full, index=False)
        csv_path = str(csv_full.resolve())
    return n_ok, csv_path


def _explanation_from_shap_matrix_rows(
    rtw: Any,
    explainer: Any,
    X_shap: pd.DataFrame,
    shap_matrix: np.ndarray,
    row_indices: List[int],
    *,
    is_regression: bool,
) -> Any:
    import shap

    if len(row_indices) < 2:
        raise ValueError("beeswarm needs at least 2 rows (SHAP beeswarm does not support a single instance)")
    vals = np.asarray(shap_matrix[row_indices, :], dtype=float)
    if vals.shape != (len(row_indices), X_shap.shape[1]):
        raise ValueError(
            f"unexpected SHAP slice shape {vals.shape}, expected ({len(row_indices)}, {X_shap.shape[1]})"
        )
    X_sub = X_shap.iloc[row_indices]
    ev = rtw._explainer_expected_scalar(explainer, is_regression=is_regression)
    base = np.full(len(row_indices), ev, dtype=float)
    return shap.Explanation(
        values=vals,
        base_values=base,
        data=X_sub.to_numpy(dtype=float),
        feature_names=list(X_shap.columns),
    )


def render_beeswarm_buckets(
    rtw: Any,
    explainer: Any,
    X_shap: pd.DataFrame,
    *,
    artifact_prefix: str,
    out_path: Path,
    is_regression: bool,
    easy_idx: List[int],
    hard_idx: List[int],
    average_idx: List[int],
    plot_filename_suffix: str,
    max_display: Optional[int],
    verbose: bool,
) -> Optional[str]:
    import matplotlib.pyplot as plt
    import shap
    from shap.plots._utils import convert_ordering

    pfx = artifact_prefix
    out_path.mkdir(parents=True, exist_ok=True)

    raw_full = explainer.shap_values(X_shap)
    full_mat = np.asarray(
        rtw._shap_values_matrix(
            raw_full,
            n_samples=len(X_shap),
            n_features=X_shap.shape[1],
        ),
        dtype=float,
    )
    # Same rule as default beeswarm order=Explanation.abs.mean(0), but evaluated on the full SHAP
    # frame (replay of test / shap_sample) so easy / hard / average plots share one feature ordering.
    feature_order = convert_ordering(shap.Explanation.abs.mean(0), shap.Explanation(np.abs(full_mat)))

    buckets: List[Tuple[str, List[int]]] = [
        ("easy", easy_idx),
        ("hard", hard_idx),
        ("average", average_idx),
    ]
    manifest_rows: List[Dict[str, Any]] = []
    for label, indices in buckets:
        fname = f"{pfx}_shap_beeswarm_{label}{plot_filename_suffix}.png"
        fpath = out_path / fname
        try:
            exp = _explanation_from_shap_matrix_rows(
                rtw, explainer, X_shap, full_mat, indices, is_regression=is_regression
            )
            plt.close("all")
            shap.plots.beeswarm(exp, max_display=max_display, show=False, order=feature_order)
            plt.gcf().savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close("all")
            manifest_rows.append(
                {
                    "bucket": label,
                    "n_tasks": len(indices),
                    "row_indices_in_shap_frame": indices,
                    "task_ids": [str(X_shap.index[i]) for i in indices],
                    "diagram_path": str(fpath.resolve()),
                    "feature_order_rule": "Explanation.abs.mean(0) on full X_shap SHAP matrix",
                }
            )
        except Exception as e:
            if verbose:
                print(f"  beeswarm {label} failed: {e}")
            try:
                plt.close("all")
            except Exception:
                pass

    csv_path: Optional[str] = None
    if manifest_rows:
        csv_full = out_path / f"{pfx}_shap_beeswarm_buckets_manifest{plot_filename_suffix}.csv"
        flat: List[Dict[str, Any]] = []
        for row in manifest_rows:
            for pos, tid in enumerate(row["task_ids"]):
                flat.append(
                    {
                        "bucket": row["bucket"],
                        "rank_in_bucket": pos + 1,
                        "task_id": tid,
                        "row_index_in_shap_frame": row["row_indices_in_shap_frame"][pos],
                        "diagram_path": row["diagram_path"],
                        "feature_order_rule": row.get("feature_order_rule", ""),
                    }
                )
        pd.DataFrame(flat).to_csv(csv_full, index=False)
        csv_path = str(csv_full.resolve())
    return csv_path


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
        "--n-each",
        type=int,
        default=10,
        metavar="N",
        help="How many easy and how many hard waterfalls (default: 10).",
    )
    parser.add_argument(
        "--rank-by",
        choices=("pred", "pass_rate"),
        default="pred",
        help="Rank tasks by model y_pred (default, matches easiest/hardest) or empirical pass_rate.",
    )
    parser.add_argument(
        "--plot-suffix",
        type=str,
        default="_easy_hard_10",
        help="Suffix inserted before .png (default: _easy_hard_10).",
    )
    parser.add_argument("--random-state", type=int, default=None, help="Override manifest random_state.")
    parser.add_argument("--test-size", type=float, default=None, help="Override manifest test_size.")
    parser.add_argument(
        "--beeswarm",
        action="store_true",
        help="Write 3 beeswarm plots (easy / hard / average), each using the same N tasks as the waterfalls.",
    )
    parser.add_argument(
        "--beeswarm-max-display",
        type=int,
        default=None,
        metavar="K",
        help="Cap beeswarm to top K features by global importance; omit for **all** features (default).",
    )
    parser.add_argument(
        "--average-pick",
        choices=("mean", "median"),
        default="mean",
        help="Center for the average bucket: mean or median of y_pred on X_shap (default: mean). Ignored if --average-prob is set.",
    )
    parser.add_argument(
        "--average-prob",
        type=float,
        default=None,
        metavar="P",
        help="Fix center P for average bucket (smallest |y_pred - P|). Overrides --average-pick.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.n_each < 1:
        parser.error("--n-each must be >= 1")
    if args.beeswarm and args.n_each < 2:
        parser.error("--beeswarm requires --n-each >= 2 (SHAP beeswarm needs multiple instances)")
    if args.beeswarm and args.beeswarm_max_display is not None and args.beeswarm_max_display < 1:
        parser.error("--beeswarm-max-display must be >= 1 when set (or omit for all features)")

    from replication.paths import default_dataset_path

    if not default_dataset_path().is_file():
        raise FileNotFoundError(f"Task-level dataset missing: {default_dataset_path()}")

    rtw = _load_plot_waterfalls_module()

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
    pass_rate_all = task_outcomes["pass_rate"].values

    if args.verbose:
        print(f"Replayed holdout: n_test={len(idx_test)} (test_size={test_size}, random_state={random_state})")

    suffix = args.plot_suffix or ""
    n_req = int(args.n_each)

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

        pr_test = pass_rate_all[idx_test]
        pass_rate_series = pd.Series(pr_test, index=X_test_model.index).reindex(X_shap.index)

        if args.rank_by == "pass_rate":
            if pass_rate_series.isna().any():
                n_bad = int(pass_rate_series.isna().sum())
                raise ValueError(f"{pfx}: {n_bad} X_shap rows missing pass_rate after reindex")
            rank_scores = pass_rate_series.to_numpy(dtype=float)
            rule = "pass_rate_desc_is_easier"
        else:
            rank_scores = _y_pred_scores(model, X_shap, is_regression=is_regression)
            rule = "y_pred_desc_is_easier_model_view"

        y_pred_scores = _y_pred_scores(model, X_shap, is_regression=is_regression)

        easy_dir = target_dir / "shap_waterfalls_easy"
        hard_dir = target_dir / "shap_waterfalls_hard"
        average_dir = target_dir / "shap_waterfalls_average"

        explainer = shap.TreeExplainer(model)
        easy_idx = _rank_row_indices(rank_scores, n=n_req, high_first=True)
        hard_idx = _rank_row_indices(rank_scores, n=n_req, high_first=False)
        avg_slots = rtw._build_average_scenarios(
            y_pred_scores,
            X_shap,
            average_n=n_req,
            average_task_id=None,
            average_pick=args.average_pick,
            average_prob=args.average_prob,
        )
        average_idx = [int(r) for _, r, _ in avg_slots]
        avg_selection_rule = avg_slots[0][2] if avg_slots else "anchor_mean_of_y_pred_distance"

        if len(easy_idx) < n_req and args.verbose:
            print(f"  {pfx}: only {len(easy_idx)} distinct easy rows available (requested {n_req})")
        if len(hard_idx) < n_req and args.verbose:
            print(f"  {pfx}: only {len(hard_idx)} distinct hard rows available (requested {n_req})")
        if len(average_idx) < n_req and args.verbose:
            print(f"  {pfx}: only {len(average_idx)} distinct average rows available (requested {n_req})")

        ne, ce = render_extreme_bucket(
            rtw,
            model,
            X_shap,
            explainer,
            artifact_prefix=pfx,
            out_path=easy_dir,
            is_regression=is_regression,
            row_indices=easy_idx,
            bucket_label="easy",
            rank_scores=rank_scores,
            y_pred_scores=y_pred_scores,
            y_holdout=y_series,
            plot_filename_suffix=suffix,
            selection_rule=rule,
            verbose=args.verbose,
        )
        nh, ch = render_extreme_bucket(
            rtw,
            model,
            X_shap,
            explainer,
            artifact_prefix=pfx,
            out_path=hard_dir,
            is_regression=is_regression,
            row_indices=hard_idx,
            bucket_label="hard",
            rank_scores=rank_scores,
            y_pred_scores=y_pred_scores,
            y_holdout=y_series,
            plot_filename_suffix=suffix,
            selection_rule=rule,
            verbose=args.verbose,
        )
        na, ca = render_extreme_bucket(
            rtw,
            model,
            X_shap,
            explainer,
            artifact_prefix=pfx,
            out_path=average_dir,
            is_regression=is_regression,
            row_indices=average_idx,
            bucket_label="average",
            rank_scores=y_pred_scores,
            y_pred_scores=y_pred_scores,
            y_holdout=y_series,
            plot_filename_suffix=suffix,
            selection_rule=avg_selection_rule,
            verbose=args.verbose,
        )
        msg = (
            f"{pfx}: easy -> {ne} PNGs, manifest {ce}; hard -> {nh} PNGs, manifest {ch}; "
            f"average -> {na} PNGs, manifest {ca} "
            f"(rank_by={args.rank_by}, average_pick={args.average_pick!r}, average_prob={args.average_prob!r}; "
            f"folders {easy_dir.name}/ {hard_dir.name}/ {average_dir.name}/)"
        )
        if args.beeswarm:
            bees_dir = target_dir / "shap_beeswarm_buckets"
            bm = render_beeswarm_buckets(
                rtw,
                explainer,
                X_shap,
                artifact_prefix=pfx,
                out_path=bees_dir,
                is_regression=is_regression,
                easy_idx=easy_idx,
                hard_idx=hard_idx,
                average_idx=average_idx,
                plot_filename_suffix=suffix,
                max_display=args.beeswarm_max_display,
                verbose=args.verbose,
            )
            msg += f"; beeswarm manifest -> {bm} ({bees_dir.name}/)"
        print(msg)


if __name__ == "__main__":
    main()
