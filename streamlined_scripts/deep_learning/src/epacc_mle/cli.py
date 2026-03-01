from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from epacc_mle.config import load_config
from epacc_mle.data.io import load_split_data
from epacc_mle.data.folds import make_fold_indices
from epacc_mle.paths import ensure_run_dirs, make_run_id, save_resolved_config


def main() -> None:
    # ===============================
    # MODE TOGGLES (set ONE True)
    # ===============================
    RUN_DEV_SINGLE_FOLD = False           # split 1 fold 1, 2 epochs
    RUN_FULL_CV_SPLIT1 = False            # split 1, folds 1-5
    RUN_TRAIN_FULL_SINGLE_SPLIT = False   # train full train split -> checkpoint only
    RUN_EVAL_HOLDOUT_FROM_CKPT = True     # evaluate holdout using an existing checkpoint
    RUN_HOLDOUT_SINGLE_SPLIT = False      # train full train split then evaluate holdout (end-to-end)
    RUN_HOLDOUT_ALL_SPLITS = False        # train+eval for split 1..N using experiments/holdout_runner.py

    # ===============================
    # RUN FOLDER CONTROL (better fix)
    # ===============================
    # If you want TRAIN and EVAL to use the SAME run folder, set this to an existing run id.
    # Example: "dev_20260301_122315"
    RUN_ID_OVERRIDE = "dev_20260301_122315"

    # ===============================
    # FAST DEBUG OVERRIDE
    # ===============================
    FAST_DEBUG = True
    FAST_EPOCHS = 5
    FAST_BATCH_SIZE = 64

    # ===============================
    # LOAD CONFIG
    # ===============================
    cfg = load_config(Path("configs/default.yaml"))

    if FAST_DEBUG:
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=FAST_EPOCHS, print_interval=1, batch_size=FAST_BATCH_SIZE),
        )

    # ===============================
    # INIT RUN DIRS (reuse run id if provided)
    # ===============================
    run_id = RUN_ID_OVERRIDE or make_run_id("dev")
    run_dirs = ensure_run_dirs(Path("artifacts") / "runs" / run_id)
    save_resolved_config(cfg, run_dirs["root"])

    # ===============================
    # SPLIT ID (for single-split modes)
    # ===============================
    split_id = 1

    # Load data for single split modes
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
    # MODE: DEV SINGLE FOLD (2 epochs)
    # ===============================
    if RUN_DEV_SINGLE_FOLD:
        from epacc_mle.models import build_model
        from epacc_mle.train.loop import train_one_fold

        cfg_dev = replace(cfg, train=replace(cfg.train, epochs=2, print_interval=1, batch_size=FAST_BATCH_SIZE))

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
    # MODE: TRAIN FULL SINGLE SPLIT (checkpoint only)
    # ===============================
    if RUN_TRAIN_FULL_SINGLE_SPLIT:
        from epacc_mle.models import build_model
        from epacc_mle.train.full_train import train_full_trainset

        model = build_model(cfg)
        ckpt_path = train_full_trainset(
            cfg=cfg,
            model=model,
            train_df=train_df,
            run_dirs=run_dirs,
            split_id=split_id,
        )
        print(f"[TRAIN_FULL] saved checkpoint: {ckpt_path}")
        return

    # ===============================
    # MODE: EVAL HOLDOUT FROM CHECKPOINT (no training)
    # ===============================
    if RUN_EVAL_HOLDOUT_FROM_CKPT:
        # This now correctly uses the run_id folder you set above (RUN_ID_OVERRIDE),
        # so it will find the checkpoint created in that run.
        EVAL_CKPT_PATH = Path(run_dirs["checkpoints"] / f"split_{split_id:02d}_full_train_last.pth")

        from epacc_mle.eval.holdout_from_ckpt import evaluate_holdout_from_checkpoint

        metrics = evaluate_holdout_from_checkpoint(
            cfg=cfg,
            ckpt_path=EVAL_CKPT_PATH,
            split_id=split_id,
            run_dirs=run_dirs,
            save_preds=True,
        )
        print("[EVAL_FROM_CKPT] metrics:")
        print(metrics)
        return

    # ===============================
    # MODE: HOLDOUT SINGLE SPLIT (train + eval end-to-end)
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

    # ===============================
    # MODE: HOLDOUT ALL SPLITS (train+eval loop)
    # ===============================
    if RUN_HOLDOUT_ALL_SPLITS:
        from epacc_mle.experiments.holdout_runner import run_holdout_all_splits

        _ = run_holdout_all_splits(cfg=cfg, run_dirs=run_dirs, save_preds=False)
        return

    print("No mode selected. Set one of the RUN_* toggles to True.")


if __name__ == "__main__":
    main()