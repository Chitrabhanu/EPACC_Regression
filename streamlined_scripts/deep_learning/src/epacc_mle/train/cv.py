from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from epacc_mle.config import Config
from epacc_mle.models import build_model
from epacc_mle.train.loop import train_one_fold


@dataclass(frozen=True)
class FoldResult:
    split_id: int
    fold_id: int
    best_epoch: int
    best_val_bolus_mae: float
    best_val_wavelet_mae: float


def run_cv_for_split(
    cfg: Config,
    split_id: int,
    train_df: pd.DataFrame,
    folds,
    run_dirs: Dict[str, Path],
) -> pd.DataFrame:
    """
    Runs 5-fold CV for a single split.
    Returns a summary DF with one row per fold (best epoch by val_bolus_mae).
    """
    fold_rows: List[FoldResult] = []

    for f in folds:
        fold_id = f.fold_id
        print(f"=== CV split {split_id} fold {fold_id} ===")

        train_fold_df = train_df.iloc[f.train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[f.val_idx].reset_index(drop=True)

        model = build_model(cfg)

        hist_df = train_one_fold(
            cfg=cfg,
            model=model,
            train_df=train_fold_df,
            val_df=val_fold_df,
            run_dirs=run_dirs,
            split_id=split_id,
            fold_id=fold_id,
        )

        # choose best epoch by val_bolus_mae (your real objective)
        best_i = int(hist_df["val_bolus_mae"].astype(float).idxmin())
        best_epoch = int(hist_df.loc[best_i, "epoch"])
        best_val_bolus = float(hist_df.loc[best_i, "val_bolus_mae"])
        best_val_wave = float(hist_df.loc[best_i, "val_wavelet_mae"])

        fold_rows.append(
            FoldResult(
                split_id=split_id,
                fold_id=fold_id,
                best_epoch=best_epoch,
                best_val_bolus_mae=best_val_bolus,
                best_val_wavelet_mae=best_val_wave,
            )
        )

    summary = pd.DataFrame([r.__dict__ for r in fold_rows])
    out_csv = run_dirs["cv"] / f"split_{split_id:02d}_cv_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    # optional: print mean/std to console (reviewer-friendly)
    print(
        f"[CV split {split_id}] best val bolus MAE: "
        f"mean={summary['best_val_bolus_mae'].mean():.4f} "
        f"std={summary['best_val_bolus_mae'].std(ddof=1):.4f}"
    )

    return summary