# Dataset

This folder contains the **analysis dataset** for the paper. Use `./run_paper.sh` or `./run_all.sh`; the shipped parquet is the intended artifact.

## Main file


| File                         | Description                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `task_level_dataset.parquet` | One row per task (45,769 tasks after excluding R2E-Gym; SWE-Smith + SWE-Rebench only). |


## Identifiers and mapping (repo + patch)

Each row links a task to its source benchmark instance and gold patch metadata (from CoderForge-Preview mapping):


| Column             | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| `task_id`          | Stable task identifier (CoderForge task key).                                       |
| `trajectory_id`    | Representative trajectory id (features are identical across runs of the same task). |
| `cf_split`         | Source split: `SWE_Smith` or `SWE_Rebench`.                                         |
| `cf_row_index`     | Row index within the CoderForge-Preview export.                                     |
| `base_repo`        | Repository slug (e.g. `owner/name`).                                                |
| `base_patch`       | Gold patch text (unified diff).                                                     |
| `base_base_commit` | Base commit hash for the task checkout.                                             |
| `base_instance_id` | Upstream benchmark instance id when available.                                      |




## Outcomes (agent success)

Generated from Qwen3-Coder-480B trajectories in CoderForge-Preview (OpenHands v0.52.1, up to 8 runs per task):


| Column        | Description                                       |
| ------------- | ------------------------------------------------- |
| `pass_rate`   | Fraction of runs that passed all tests ∈ [0, 1].  |
| `any_success` | 1 if ≥1 run succeeded (pass@k).                   |
| `maj_success` | 1 if majority of runs succeeded (≥50% pass rate). |
| `n_runs`      | Number of trajectories aggregated into this task. |




## Train/test split

80/20 holdout, stratified on `any_success`, `random_state=42` (36,615 train / 9,154 test):


| Path                          | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| `splits/train_test_split.csv` | `task_id`, `split`, `any_success`                   |
| `make_train_test_split.py`    | Regenerate the CSV (see `lib/replication/split.py`) |


RQ scripts load this table when the parquet has no `split` column.

## Features (63 numeric predictors)

Columns prefixed with `patch_`, `repo_`, or `prompt_` are the static task features described in the paper (Tables patch / repository / prompt). Nine collinear features were dropped during VIF selection (see `feature_manifest/`); models use the remaining **54** features after VIF on the training split.

## Supporting files


| Path                                             | Description                                 |
| ------------------------------------------------ | ------------------------------------------- |
| `dataset_summary.json`                           | Row counts and outcome rates.               |
| `splits/train_test_split.csv`                    | Task-level train/test assignment.           |
| `make_train_test_split.py`                       | Regenerate `splits/train_test_split.csv`.   |
| `feature_manifest/features_vif_selected_54.json` | Features retained after VIF (threshold 10). |
| `feature_manifest/features_vif_dropped_9.json`   | Features removed by VIF.                    |
