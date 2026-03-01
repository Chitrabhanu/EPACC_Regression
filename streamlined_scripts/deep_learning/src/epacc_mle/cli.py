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

    for f in folds:
        print(f"fold {f.fold_id}: {len(f.train_idx)} / {len(f.val_idx)}")

    # ===============================
    # SMOKE TEST (INSIDE main)
    # ===============================
    from epacc_mle.data.dataset import WaveletDataset
    from epacc_mle.eval.metrics import (
        make_wavelet_pred_df,
        compute_wavelet_and_bolus_metrics,
    )
    import numpy as np

    ds = WaveletDataset(cfg, train_df.head(200))

    ids_dataset, ids_pig, ids_bolus = [], [], []
    y_true = []

    for i in range(len(ds)):
        _, y, ids = ds[i]
        ids_dataset.append(ids["dataset"])
        ids_pig.append(ids["pig_id"])
        ids_bolus.append(ids["bolus"])
        y_true.append(float(y.item()))

    y_true = np.array(y_true)
    y_pred = y_true.copy()

    pred_df = make_wavelet_pred_df(
        ids_dataset,
        ids_pig,
        ids_bolus,
        y_true,
        y_pred,
    )

    wave_m, bolus_m, bolus_df = compute_wavelet_and_bolus_metrics(pred_df)

    print(
        f"[SMOKE] wavelet MAE={wave_m.mae:.6f}, "
        f"bolus MAE={bolus_m.mae:.6f}, "
        f"boluses={len(bolus_df)}"
    )


if __name__ == "__main__":
    main()