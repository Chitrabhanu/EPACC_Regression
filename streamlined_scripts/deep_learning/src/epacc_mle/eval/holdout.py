from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from epacc_mle.config import Config
from epacc_mle.data.collate import collate_wavelet_batch
from epacc_mle.data.dataset import WaveletDataset
from epacc_mle.eval.metrics import make_wavelet_pred_df, compute_wavelet_and_bolus_metrics


@torch.no_grad()
def predict_wavelets(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    model.eval()
    dev = next(model.parameters()).device

    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    ids_all: Dict[str, List] = {}

    for xb, yb, ids in loader:
        xb = xb.to(dev)
        yhat = model(xb).detach().cpu().numpy().astype(float)

        y_true_all.extend(yb.numpy().astype(float).tolist())
        # flatten (handles (B,1) or (B,) cleanly)
        y_pred_all.extend(np.asarray(yhat).reshape(-1).tolist())

        for k, vs in ids.items():
            ids_all.setdefault(k, []).extend(vs)

    return np.array(y_true_all, dtype=float), np.array(y_pred_all, dtype=float), ids_all


def evaluate_holdout_split(
    cfg: Config,
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    run_dirs: Dict[str, Path],
    split_id: int,
    save_preds: bool = True,
) -> dict:
    """
    Evaluates a trained model on the HOLDOUT test split.
    Computes:
      - wavelet-level MAE/RMSE
      - bolus-level MAE/RMSE (grouped by dataset + pig + batch)
    Saves:
      - holdout/split_XX_metrics.json
      - (optional) holdout/split_XX_preds_wavelet.csv and split_XX_preds_bolus.csv
    """
    ds = WaveletDataset(cfg, test_df)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        collate_fn=collate_wavelet_batch,
    )

    y_true, y_pred, ids = predict_wavelets(model, loader)

    # Bolus identity uses (dataset, pig, bolus=batch)
    pred_df = make_wavelet_pred_df(
        ids_dataset=ids["dataset"],
        ids_pig=ids["pig"],
        ids_bolus=ids["bolus"],
        y_true=y_true,
        y_pred=y_pred,
        ids_subject=ids.get("subject"),  # optional; helpful for debugging
    )
    wave_m, bolus_m, bolus_df = compute_wavelet_and_bolus_metrics(pred_df)

    metrics = {
        "split_id": int(split_id),
        "n_test_wavelets": int(len(pred_df)),
        "n_test_boluses": int(len(bolus_df)),
        "wavelet_mae": float(wave_m.mae),
        "wavelet_rmse": float(wave_m.rmse),
        "bolus_mae": float(bolus_m.mae),
        "bolus_rmse": float(bolus_m.rmse),
    }

    out_dir = run_dirs["holdout"]
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"split_{split_id:02d}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_preds:
        pred_df.to_csv(out_dir / f"split_{split_id:02d}_preds_wavelet.csv", index=False)
        bolus_df.to_csv(out_dir / f"split_{split_id:02d}_preds_bolus.csv", index=False)

    print(
        f"[HOLDOUT split {split_id}] wavelet MAE={metrics['wavelet_mae']:.4f} "
        f"| bolus MAE={metrics['bolus_mae']:.4f} "
        f"(boluses={metrics['n_test_boluses']})"
    )
    return metrics