from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epacc_mle.config import Config
from epacc_mle.data.collate import collate_wavelet_batch
from epacc_mle.data.dataset import WaveletDataset
from epacc_mle.eval.metrics import make_wavelet_pred_df, compute_wavelet_and_bolus_metrics
from epacc_mle.viz.curves import save_two_line_curve


@dataclass
class EpochRow:
    epoch: int
    train_loss: float
    val_loss: float
    train_wavelet_mae: float
    train_bolus_mae: float
    val_wavelet_mae: float
    val_bolus_mae: float


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_full(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    model.eval()
    dev = next(model.parameters()).device

    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    ids_all: Dict[str, List] = {}

    for xb, yb, ids in loader:
        xb = xb.to(dev)
        yhat = model(xb).detach().cpu().numpy().astype(float)

        y_true_all.extend(yb.numpy().astype(float).tolist())
        y_pred_all.extend(yhat.tolist())

        for k, vs in ids.items():
            ids_all.setdefault(k, []).extend(vs)

    return np.array(y_true_all, dtype=float), np.array(y_pred_all, dtype=float), ids_all


def train_one_fold(
    cfg: Config,
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dirs: Dict[str, Path],
    split_id: int,
    fold_id: int,
) -> pd.DataFrame:
    # Make randomness stable per split/fold
    set_seed(cfg.project.seed + split_id * 100 + fold_id)

    dev = get_device()
    model = model.to(dev)

    train_ds = WaveletDataset(cfg, train_df)
    val_ds = WaveletDataset(cfg, val_df)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_wavelet_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_wavelet_batch
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    best_val_bolus_mae = float("inf")
    best_ckpt = run_dirs["checkpoints"] / f"split_{split_id:02d}_fold_{fold_id:02d}_best.pth"

    rows: List[EpochRow] = []

    for epoch in range(1, cfg.train.epochs + 1):
        # ---- train ----
        model.train()
        train_losses: List[float] = []
        for xb, yb, _ in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)

            opt.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ---- eval train/val (MAE wavelet+bolus) ----
        train_eval_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_wavelet_batch
        )
        ytr, ptr, idtr = predict_full(model, train_eval_loader)
        pred_tr_df = make_wavelet_pred_df(idtr["dataset"], idtr["pig"], idtr["bolus"], ytr, ptr)
        tr_wave_m, tr_bolus_m, _ = compute_wavelet_and_bolus_metrics(pred_tr_df)

        yva, pva, idva = predict_full(model, val_loader)
        pred_va_df = make_wavelet_pred_df(idva["dataset"], idva["pig"], idva["bolus"], yva, pva)
        va_wave_m, va_bolus_m, _ = compute_wavelet_and_bolus_metrics(pred_va_df)

        # ---- val loss ----
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                yhat = model(xb)
                val_losses.append(float(criterion(yhat, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        row = EpochRow(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_wavelet_mae=tr_wave_m.mae,
            train_bolus_mae=tr_bolus_m.mae,
            val_wavelet_mae=va_wave_m.mae,
            val_bolus_mae=va_bolus_m.mae,
        )
        rows.append(row)

        if row.val_bolus_mae < best_val_bolus_mae:
            best_val_bolus_mae = row.val_bolus_mae
            torch.save(
                {
                    "split_id": split_id,
                    "fold_id": fold_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_bolus_mae": best_val_bolus_mae,
                    "config_model_name": cfg.model.name,
                },
                best_ckpt,
            )

        if epoch == 1 or epoch % cfg.train.print_interval == 0:
            print(
                f"[split {split_id} fold {fold_id}] epoch {epoch}/{cfg.train.epochs} "
                f"train_loss={row.train_loss:.4f} val_loss={row.val_loss:.4f} "
                f"val_bolus_mae={row.val_bolus_mae:.4f}"
            )

    hist = pd.DataFrame([r.__dict__ for r in rows])
    out_csv = run_dirs["cv"] / f"split_{split_id:02d}_fold_{fold_id:02d}_history.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_csv, index=False)

    # curves
    xs = hist["epoch"].tolist()
    save_two_line_curve(
        xs, hist["train_loss"].tolist(), hist["val_loss"].tolist(),
        "train_loss", "val_loss",
        f"Loss split {split_id} fold {fold_id}",
        run_dirs["curves"] / f"split_{split_id:02d}" / f"fold_{fold_id:02d}_loss.png",
    )
    save_two_line_curve(
        xs, hist["train_bolus_mae"].tolist(), hist["val_bolus_mae"].tolist(),
        "train_bolus_mae", "val_bolus_mae",
        f"Bolus MAE split {split_id} fold {fold_id}",
        run_dirs["curves"] / f"split_{split_id:02d}" / f"fold_{fold_id:02d}_bolus_mae.png",
    )

    return hist