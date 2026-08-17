#!/usr/bin/env python3
"""Download Hugging Face datasets used by the paper preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import DownloadConfig, get_dataset_config_names, get_dataset_split_names, load_dataset

DEFAULT_DATASETS = (
    "togethercomputer/CoderForge-Preview",
    "R2E-Gym/R2E-Gym-V1",
    "SWE-bench/SWE-smith",
    "nebius/SWE-rebench",
)


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


def _discover_configs(dataset_id: str) -> tuple[str, ...]:
    configs = tuple(get_dataset_config_names(dataset_id))
    return configs if configs else ("default",)


def load_one_dataset(
    dataset_id: str,
    num_proc: int,
    arrow_root: Path | None,
    parquet_root: Path | None,
) -> tuple[str, int]:
    configs = _discover_configs(dataset_id)
    download_config = DownloadConfig(resume_download=True, max_retries=5)
    loaded = 0

    for config in configs:
        config_name = None if config == "default" else config
        splits = get_dataset_split_names(dataset_id, config_name=config_name)
        for split in splits:
            ds = load_dataset(
                path=dataset_id,
                name=config_name,
                split=split,
                download_config=download_config,
                num_proc=num_proc,
            )
            loaded += len(ds)
            if arrow_root is not None:
                target = arrow_root / _safe_name(dataset_id) / _safe_name(config) / _safe_name(split)
                target.parent.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(target))
            if parquet_root is not None:
                target = parquet_root / _safe_name(dataset_id) / _safe_name(config) / f"{_safe_name(split)}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                ds.to_parquet(str(target))
    return dataset_id, loaded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory; writes hf_arrow/ and hf_parquet/ beneath it.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="HF dataset id (repeatable). Default: CoderForge-Preview + upstream benchmark tables.",
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--num-proc-per-dataset", type=int, default=1)
    p.add_argument("--no-arrow", action="store_true")
    p.add_argument("--no-parquet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    arrow_root = None if args.no_arrow else out / "hf_arrow"
    parquet_root = None if args.no_parquet else out / "hf_parquet"
    if arrow_root is not None:
        arrow_root.mkdir(parents=True, exist_ok=True)
    if parquet_root is not None:
        parquet_root.mkdir(parents=True, exist_ok=True)

    datasets_to_load = tuple(dict.fromkeys(args.dataset)) if args.dataset else DEFAULT_DATASETS
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                load_one_dataset,
                ds_id,
                args.num_proc_per_dataset,
                arrow_root,
                parquet_root,
            ): ds_id
            for ds_id in datasets_to_load
        }
        for fut in as_completed(futs):
            ds_id = futs[fut]
            try:
                loaded_id, rows = fut.result()
                print(f"[OK] {loaded_id}: {rows:,} rows")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {ds_id}: {exc}")

    manifest = out / "download_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": list(datasets_to_load),
                "arrow_root": str(arrow_root) if arrow_root else None,
                "parquet_root": str(parquet_root) if parquet_root else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
