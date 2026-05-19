# Train/test split

| File | Description |
|------|-------------|
| `train_test_split.csv` | `task_id`, `split` (`train` / `test`), `any_success` |

**Protocol:** 80/20 holdout, stratified on `any_success`, `random_state=42`, task-level (36,615 train / 9,154 test).

Regenerate with:

```bash
cd diffmodel_esem_replication
export PYTHONPATH="$(pwd)/lib:$PYTHONPATH"
export CODERFORGE_REPLICATION_ROOT="$(pwd)"
python data/make_train_test_split.py
```

Implementation: `lib/replication/split.py`.
