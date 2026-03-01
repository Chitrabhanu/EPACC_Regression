from __future__ import annotations

from pathlib import Path

from epacc_mle.config import load_config
from epacc_mle.data.io import load_split_data
from epacc_mle.data.folds import make_fold_indices
from epacc_mle.paths import ensure_run_dirs, make_run_id, save_resolved_config


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))

    run_id = make_run_id("dev")
    run_dir = ensure_run_dirs(Path("artifacts") / "runs" / run_id)["root"]
    save_resolved_config(cfg, run_dir)

    train_df, test_df = load_split_data(cfg, split_id=1)
    folds = make_fold_indices(cfg, split_id=1, train_df=train_df)

    print(f"Run: {run_id}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("Fold sizes (train/val rows):")
    for f in folds:
        print(f"  fold {f.fold_id}: {len(f.train_idx)} / {len(f.val_idx)}")


if __name__ == "__main__":
    main()