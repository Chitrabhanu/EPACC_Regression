from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from epacc_mle.config import load_config
from epacc_mle.data.io import load_split_data
from epacc_mle.data.folds import make_fold_indices
from epacc_mle.paths import ensure_run_dirs, make_run_id, save_resolved_config


def main() -> None:
    # ===============================
    # MODE TOGGLES
    # ===============================
    # Set exactly one of these to True at a time.
    RUN_DEV_SINGLE_FOLD = False          # split 1 fold 1, 2 epochs
    RUN_FULL_CV_SPLIT1 = False           # split 1, folds 1-5 (uses cfg.train.epochs unless overridden)
    RUN_HOLDOUT_SINGLE_SPLIT = True      # train on full split-1 train set, evaluate on split-1 test set

    # If you want fast wiring tests, override epochs here (applies to whichever mode you run)
    FAST_DEBUG = False  # set True to force 2 epochs + smaller batch for quick runs

    # ===============================
    # LOAD CONFIG + INIT RUN DIRS
    # ===============================
    cfg = load_config(Path("configs/default.yaml"))

    if FAST_DEBUG:
        cfg = replace(cfg, train=replace(cfg.train, epochs=2, print_interval=1, batch_size=64))

    run_id = make_run_id("dev")
    run_dirs = ensure_run_dirs(Path("artifacts") / "runs" / run_id)
    save_resolved_config(cfg, run_dirs["root"])

    # ===============================
    # LOAD ONE SPLIT (split 1 for now)
    # ===============================
    split_id = 1
    train_df, test_df = load_split_data(cfg, split_id=split_id)
    folds = make_fold_indices(cfg, split_id=split_id, train_df=train_df)

    print(f"Run: {run_id}")
    print(f"Split: {split_id}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("Fold sizes (train/val rows):")
    for f in folds:
        print(f"  fold {f.fold_id}: {len(f.train_idx)} / {len(f.val_idx)}")

    # ===============================
    # SMOKE TEST (IDs + bolus aggregation)
    # ===============================
    from epacc_mle.data.dataset import WaveletDataset
    from epacc_mle.eval.metrics import make_wavelet_pred_df, compute_wavelet_and_bolus_metrics
    import numpy as np

    ds = WaveletDataset(cfg, train_df.head(200))

    ids_dataset, ids_pig, ids_bolus = [], [], []
    y_true = []
    for i in range(len(ds)):
        _, y, ids = ds[i]
        ids_dataset.append(ids["dataset"])
        ids_pig.append(ids["pig"])      # bolus identifier (original logic)
        ids_bolus.append(ids["bolus"])  # batch
        y_true.append(float(y.item()))

    y_true = np.array(y_true, dtype=float)
    y_pred = y_true.copy()
    pred_df = make_wavelet_pred_df(ids_dataset, ids_pig, ids_bolus, y_true, y_pred)
    wave_m, bolus_m, bolus_df = compute_wavelet_and_bolus_metrics(pred_df)

    print(
        f"[SMOKE] wavelet MAE={wave_m.mae:.6f}, "
        f"bolus MAE={bolus_m.mae:.6f}, "
        f"boluses={len(bolus_df)}"
    )

    # ===============================
    # MODE: DEV SINGLE FOLD
    # ===============================
    if RUN_DEV_SINGLE_FOLD:
        from epacc_mle.models import build_model
        from epacc_mle.train.loop import train_one_fold

        cfg_dev = replace(cfg, train=replace(cfg.train, epochs=2, print_interval=1, batch_size=64))

        fold_id = 1
        f0 = next(ff for ff in folds if ff.fold_id == fold_id)

        train_fold_df = train_df.iloc[f0.train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[f0.val_idx].reset_index(drop=True)

        model = build_model(cfg_dev)

        hist_df = train_one_fold(
            cfg=cfg_dev,
            model=model,
            train_df=train_fold_df,
            val_df=val_fold_df,
            run_dirs=run_dirs,
            split_id=split_id,
            fold_id=fold_id,
        )

        print("[DEV TRAIN] last epoch summary:")
        print(hist_df.tail(1).to_string(index=False))
        return

    # ===============================
    # MODE: FULL CV FOR SPLIT 1
    # ===============================
    if RUN_FULL_CV_SPLIT1:
        from epacc_mle.train.cv import run_cv_for_split

        summary_df = run_cv_for_split(
            cfg=cfg,
            split_id=split_id,
            train_df=train_df,
            folds=folds,
            run_dirs=run_dirs,
        )

        print("[CV SUMMARY]")
        print(summary_df.to_string(index=False))
        return

    # ===============================
    # MODE: HOLDOUT (single split)
    # Train on FULL training split, then evaluate on test split
    # ===============================
    if RUN_HOLDOUT_SINGLE_SPLIT:
        from epacc_mle.models import build_model
        from epacc_mle.train.full_train import train_full_trainset
        from epacc_mle.eval.holdout import evaluate_holdout_split

        model = build_model(cfg)
        ckpt_path = train_full_trainset(
            cfg=cfg,
            model=model,
            train_df=train_df,
            run_dirs=run_dirs,
            split_id=split_id,
        )
        print(f"[HOLDOUT] trained full-train model saved: {ckpt_path}")

        _ = evaluate_holdout_split(
            cfg=cfg,
            model=model,
            test_df=test_df,
            run_dirs=run_dirs,
            split_id=split_id,
            save_preds=True,
        )
        return

    print("No mode selected. Set one of the RUN_* toggles to True.")


if __name__ == "__main__":
    main()