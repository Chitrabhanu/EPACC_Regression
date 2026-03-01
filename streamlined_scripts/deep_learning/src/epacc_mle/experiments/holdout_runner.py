from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from epacc_mle.config import Config
from epacc_mle.data.io import load_split_data
from epacc_mle.eval.holdout import evaluate_holdout_split
from epacc_mle.models import build_model
from epacc_mle.train.full_train import train_full_trainset


def run_holdout_all_splits(
    cfg: Config,
    run_dirs: Dict[str, Path],
    save_preds: bool = False,
) -> pd.DataFrame:
    """
    For split_id in [1..cfg.data.num_splits]:
      - train on full train split
      - evaluate on holdout test split
      - collect metrics
    Writes:
      - holdout/holdout_summary.csv
      - holdout/holdout_overall.json
    """
    num_splits = int(getattr(cfg.data, "num_splits", 26))
    rows: List[dict] = []

    for split_id in range(1, num_splits + 1):
        print(f"\n==============================")
        print(f"RUN HOLDOUT split {split_id}/{num_splits}")
        print(f"==============================")

        train_df, test_df = load_split_data(cfg, split_id=split_id)

        model = build_model(cfg)
        _ = train_full_trainset(cfg, model, train_df=train_df, run_dirs=run_dirs, split_id=split_id)

        metrics = evaluate_holdout_split(
            cfg=cfg,
            model=model,
            test_df=test_df,
            run_dirs=run_dirs,
            split_id=split_id,
            save_preds=save_preds,
        )
        rows.append(metrics)

    summary = pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)

    out_dir = run_dirs["holdout"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "holdout_summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall = {
        "num_splits": int(len(summary)),
        "wavelet_mae_mean": float(summary["wavelet_mae"].mean()),
        "wavelet_mae_std": float(summary["wavelet_mae"].std(ddof=1)),
        "bolus_mae_mean": float(summary["bolus_mae"].mean()),
        "bolus_mae_std": float(summary["bolus_mae"].std(ddof=1)),
        "wavelet_rmse_mean": float(summary["wavelet_rmse"].mean()),
        "wavelet_rmse_std": float(summary["wavelet_rmse"].std(ddof=1)),
        "bolus_rmse_mean": float(summary["bolus_rmse"].mean()),
        "bolus_rmse_std": float(summary["bolus_rmse"].std(ddof=1)),
    }
    with open(out_dir / "holdout_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n[HOLDOUT OVERALL]")
    print(
        f"bolus MAE mean={overall['bolus_mae_mean']:.4f} std={overall['bolus_mae_std']:.4f} | "
        f"wavelet MAE mean={overall['wavelet_mae_mean']:.4f} std={overall['wavelet_mae_std']:.4f}"
    )
    print(f"Saved: {summary_csv}")

    return summary