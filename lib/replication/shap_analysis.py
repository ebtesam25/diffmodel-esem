"""SHAP explainability helpers (TreeExplainer, interactions, parallel chunks)."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def _shap_values_matrix(
    shap_values,
    *,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
) -> np.ndarray:
    """Normalize SHAP return value to (n_samples, n_features)."""
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


def _default_shap_parallel_jobs(cli_val: Optional[int]) -> int:
    """Workers for row-chunked Tree SHAP; respects ``SLURM_CPUS_PER_TASK`` when unset on CLI."""
    if cli_val is not None and cli_val >= 1:
        return int(cli_val)
    env = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit():
        return max(1, int(env))
    return max(1, min(16, (os.cpu_count() or 8)))


def _row_chunk_index_lists(n_rows: int, n_chunks: int) -> List[np.ndarray]:
    n_chunks = max(1, min(int(n_chunks), n_rows))
    parts = np.array_split(np.arange(n_rows), n_chunks)
    return [p for p in parts if len(p) > 0]


def _merge_shap_values_chunks(parts: List[Any]) -> Any:
    if not parts:
        return parts
    first = parts[0]
    if isinstance(first, list):
        return [
            np.concatenate([np.asarray(parts[i][j], dtype=float) for i in range(len(parts))], axis=0)
            for j in range(len(first))
        ]
    return np.concatenate([np.asarray(p, dtype=float) for p in parts], axis=0)


def _merge_shap_interaction_chunks(parts: List[Any]) -> Any:
    if not parts:
        return parts
    first = parts[0]
    if isinstance(first, list):
        return [
            np.concatenate([np.asarray(parts[i][j], dtype=float) for i in range(len(parts))], axis=0)
            for j in range(len(first))
        ]
    return np.concatenate([np.asarray(p, dtype=float) for p in parts], axis=0)


def _worker_tree_shap_chunk(model: Any, X_chunk: pd.DataFrame, mode: str) -> Any:
    """Picklable worker: rebuild ``TreeExplainer`` per chunk (safe with joblib loky)."""
    import shap

    ex = shap.TreeExplainer(model)
    if mode == "values":
        return ex.shap_values(X_chunk)
    return ex.shap_interaction_values(X_chunk)


def _chunked_parallel_tree_shap(
    model: Any,
    X_shap: pd.DataFrame,
    n_jobs: int,
    *,
    mode: str,
    verbose: bool,
    min_rows_parallel: int = 256,
    min_chunk_rows: int = 48,
) -> Any:
    """
    Split ``X_shap`` by rows across workers; each runs ``TreeExplainer`` + SHAP on its chunk.
    Results are concatenated in **original row order** (same as a single call).
    """
    import shap

    n = len(X_shap)
    if n_jobs <= 1 or n < min_rows_parallel:
        ex = shap.TreeExplainer(model)
        if mode == "values":
            return ex.shap_values(X_shap)
        return ex.shap_interaction_values(X_shap)

    max_chunks = n // min_chunk_rows
    if max_chunks < 2:
        ex = shap.TreeExplainer(model)
        if mode == "values":
            return ex.shap_values(X_shap)
        return ex.shap_interaction_values(X_shap)

    n_chunks = min(max(2, n_jobs), max_chunks)
    chunks = _row_chunk_index_lists(n, n_chunks)
    if len(chunks) < 2:
        ex = shap.TreeExplainer(model)
        if mode == "values":
            return ex.shap_values(X_shap)
        return ex.shap_interaction_values(X_shap)

    if verbose:
        est = n // len(chunks)
        print(f"  SHAP: {mode} row-parallel ({len(chunks)} chunks, ~{est} rows/chunk, joblib loky)...")

    from joblib import Parallel, delayed

    parts: List[Any] = Parallel(n_jobs=len(chunks), backend="loky")(
        delayed(_worker_tree_shap_chunk)(model, X_shap.iloc[idx], mode) for idx in chunks
    )
    if mode == "values":
        return _merge_shap_values_chunks(parts)
    return _merge_shap_interaction_chunks(parts)


def _safe_plot_filename_part(s: str) -> str:
    import re

    return re.sub(r"[^\w\-.]+", "_", str(s))[:120]


def _explainer_expected_scalar(explainer, *, is_regression: bool) -> float:
    """Scalar baseline f(E[X]) for TreeExplainer (regression or binary positive margin)."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple)):
        ev = ev[1] if len(ev) > 1 and not is_regression else ev[0]
    a = np.asarray(ev, dtype=float).ravel()
    if not is_regression and a.size >= 2:
        return float(a[1])
    return float(a[0]) if a.size > 0 else 0.0


def _shap_interaction_tensor(int_vals) -> np.ndarray:
    """Mean absolute interaction across samples; multiclass -> mean over class tensors."""
    if isinstance(int_vals, list):
        stacked = [np.mean(np.abs(np.asarray(v)), axis=0) for v in int_vals]
        return np.mean(np.stack(stacked, axis=0), axis=0)
    return np.mean(np.abs(np.asarray(int_vals)), axis=0)


def compute_shap_and_interactions(
    model,
    X: pd.DataFrame,
    outdir: str,
    *,
    artifact_prefix: str,
    sample_n: Optional[int] = None,
    topk_interactions: int = 50,
    no_interactions: bool = False,
    verbose: bool = False,
    is_regression: bool = False,
    row_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    """
    TreeExplainer SHAP + optional ``shap_interaction_values`` on matrix ``X``
    (must match training inputs for the tree model).

    ``sample_n`` (CLI ``--shap-sample``) may subsample rows **inside this function only** for
    faster SHAP artifacts; callers should pass the full held-out ``X_test`` for metrics elsewhere.

    With ``row_parallel_jobs`` > 1 and enough rows, ``shap_values`` / ``shap_interaction_values``
    are computed on **disjoint row chunks** in parallel (``joblib`` / ``loky``) and concatenated
    in original order — same result as one call, faster on many CPUs (e.g. ``SLURM_CPUS_PER_TASK``).

    All saved files are prefixed with ``artifact_prefix`` (e.g. ``any_success_xgb_``) so
    each artifact encodes outcome and model family.
    """
    results: Dict[str, Any] = {"dependence_interaction_plots": [], "waterfall_plots": {}}
    pfx = artifact_prefix
    pj = max(1, int(row_parallel_jobs))
    results["shap_row_parallel_jobs"] = pj
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import shap
    except ImportError as e:
        results["shap_error"] = f"missing dependency: {e}"
        return results

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    if sample_n is not None and sample_n < X.shape[0]:
        if verbose:
            print(f"  SHAP: subsampling {sample_n} / {X.shape[0]} rows")
        X_shap = X.sample(n=sample_n, random_state=42)
    else:
        X_shap = X

    results["n_holdout_test_rows"] = int(X.shape[0])
    results["n_shap_rows"] = int(X_shap.shape[0])

    explainer = None
    try:
        if verbose:
            print("  SHAP: TreeExplainer")
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        results["shap_error"] = f"TreeExplainer failed: {e}"
        return results

    try:
        if verbose:
            print("  SHAP: computing shap_values...")
        if pj > 1:
            shap_raw = _chunked_parallel_tree_shap(
                model, X_shap, pj, mode="values", verbose=verbose
            )
        else:
            shap_raw = explainer.shap_values(X_shap)
        arr = _shap_values_matrix(
            shap_raw,
            n_samples=len(X_shap),
            n_features=X_shap.shape[1],
        )
        if arr.ndim == 2 and arr.size == len(X_shap) * X_shap.shape[1] and arr.shape != (
            len(X_shap),
            X_shap.shape[1],
        ):
            arr = arr.reshape(len(X_shap), X_shap.shape[1])

        mean_abs = np.mean(np.abs(arr), axis=0)
        feat_imp = pd.DataFrame({"feature": X_shap.columns.tolist(), "mean_abs_shap": mean_abs})
        feat_imp = feat_imp.sort_values("mean_abs_shap", ascending=False)
        imp_csv = out_path / f"{pfx}_shap_feature_importance.csv"
        feat_imp.to_csv(imp_csv, index=False)
        results["feature_importance_path"] = str(imp_csv)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            top_h = feat_imp.head(min(25, len(feat_imp))).iloc[::-1]
            plt.barh(top_h["feature"], top_h["mean_abs_shap"], color="#4C72B0")
            plt.xlabel("Mean |SHAP|")
            plt.title("Top features by mean |SHAP|")
            plt.tight_layout()
            bar_path = out_path / f"{pfx}_shap_mean_abs_bar.png"
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close()
            results["mean_abs_bar_plot"] = str(bar_path)
        except Exception:
            try:
                import matplotlib.pyplot as plt

                plt.close()
            except Exception:
                pass

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(9, 7))
            shap.summary_plot(arr, X_shap, show=False, max_display=min(30, X_shap.shape[1]))
            plt.tight_layout()
            spath = out_path / f"{pfx}_shap_summary.png"
            plt.savefig(spath, dpi=150, bbox_inches="tight")
            plt.close()
            results["summary_plot"] = str(spath)
        except Exception:
            try:
                import matplotlib.pyplot as plt

                plt.close()
            except Exception:
                pass

        if not no_interactions:
            try:
                if verbose:
                    print("  SHAP: shap_interaction_values (slow on large n × p; row-parallel if pj>1)...")
                if pj > 1:
                    int_vals = _chunked_parallel_tree_shap(
                        model, X_shap, pj, mode="interactions", verbose=verbose
                    )
                else:
                    int_vals = explainer.shap_interaction_values(X_shap)
                int_arr = _shap_interaction_tensor(int_vals)
                topk = min(topk_interactions, feat_imp.shape[0])
                top_feats = feat_imp.head(topk)["feature"].tolist()
                idx = [X_shap.columns.get_loc(c) for c in top_feats if c in X_shap.columns]
                int_df_rows = []
                for ii, fi in enumerate(top_feats):
                    for jj, fj in enumerate(top_feats):
                        if jj <= ii:
                            continue
                        int_df_rows.append(
                            {
                                "f1": fi,
                                "f2": fj,
                                "mean_abs_interaction": float(int_arr[idx[ii], idx[jj]]),
                            }
                        )
                int_df = pd.DataFrame(int_df_rows).sort_values("mean_abs_interaction", ascending=False)
                int_csv = out_path / f"{pfx}_shap_interactions_topk.csv"
                int_df.to_csv(int_csv, index=False)
                results["interactions_path"] = str(int_csv)

                try:
                    import matplotlib.pyplot as plt

                    sub = int_arr[np.ix_(idx, idx)]
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        sub,
                        xticklabels=top_feats,
                        yticklabels=top_feats,
                        vmax=np.percentile(sub, 95) if sub.size else None,
                    )
                    plt.title("SHAP interaction (top features, mean |·|)")
                    plt.tight_layout()
                    hpath = out_path / f"{pfx}_shap_interaction_heatmap.png"
                    plt.savefig(hpath, dpi=150, bbox_inches="tight")
                    plt.close()
                    results["interaction_heatmap"] = str(hpath)
                except Exception:
                    try:
                        import matplotlib.pyplot as plt

                        plt.close()
                    except Exception:
                        pass

                dep_paths: List[str] = []
                for _, irow in int_df.head(5).iterrows():
                    try:
                        import matplotlib.pyplot as plt

                        f1, f2 = str(irow["f1"]), str(irow["f2"])
                        if f1 not in X_shap.columns or f2 not in X_shap.columns:
                            continue
                        j1 = int(X_shap.columns.get_loc(f1))
                        xvals = X_shap[f1].to_numpy(dtype=float)
                        yvals = arr[:, j1]
                        cvals = X_shap[f2].to_numpy(dtype=float)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sc = ax.scatter(xvals, yvals, c=cvals, cmap="viridis", alpha=0.65, s=18, edgecolors="none")
                        plt.colorbar(sc, ax=ax, label=f2)
                        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
                        ax.set_xlabel(f1)
                        ax.set_ylabel(f"SHAP value for {f1}")
                        ax.set_title(f"SHAP dependence: {f1} (color = {f2})")
                        plt.tight_layout()
                        fn = f"{pfx}_shap_dependence_{_safe_plot_filename_part(f1)}_x_{_safe_plot_filename_part(f2)}.png"
                        dp = out_path / fn
                        plt.savefig(dp, dpi=150, bbox_inches="tight")
                        plt.close()
                        dep_paths.append(str(dp))
                    except Exception:
                        try:
                            import matplotlib.pyplot as plt

                            plt.close()
                        except Exception:
                            pass
                results["dependence_interaction_plots"] = dep_paths
            except Exception as e:
                results["interaction_warning"] = str(e)

        try:
            import matplotlib.pyplot as plt

            if is_regression:
                scores = np.clip(model.predict(X_shap), 0.0, 1.0).astype(float)
            else:
                scores = model.predict_proba(X_shap)[:, 1].astype(float)
            ev_s = _explainer_expected_scalar(explainer, is_regression=is_regression)
            scenarios = [
                ("easiest", int(np.argmax(scores)), f"{pfx}_shap_waterfall_easiest.png"),
                ("hardest", int(np.argmin(scores)), f"{pfx}_shap_waterfall_hardest.png"),
                ("average", int(np.argmin(np.abs(scores - float(np.mean(scores))))), f"{pfx}_shap_waterfall_average.png"),
            ]
            wf: Dict[str, str] = {}
            for label, ridx, fname in scenarios:
                try:
                    exp_row = shap.Explanation(
                        values=arr[ridx, :],
                        base_values=ev_s,
                        data=X_shap.iloc[ridx].to_numpy(dtype=float),
                        feature_names=list(X_shap.columns),
                    )
                    plt.close("all")
                    shap.plots.waterfall(exp_row, max_display=15, show=False)
                    plt.gcf().savefig(out_path / fname, dpi=150, bbox_inches="tight")
                    plt.close("all")
                    wf[label] = str(out_path / fname)
                except Exception:
                    try:
                        plt.close("all")
                    except Exception:
                        pass
            if wf:
                results["waterfall_plots"] = wf
        except Exception:
            try:
                import matplotlib.pyplot as plt

                plt.close()
            except Exception:
                pass
    except Exception as e:
        results["error"] = str(e)
    return results
