from __future__ import annotations

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
from epacc_mle.viz.curves import save_two_line_curve


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_full_trainset(
    cfg: Config,
    model: nn.Module,
    train_df: pd.DataFrame,
    run_dirs: Dict[str, Path],
    split_id: int,
) -> Path:
    """
    Train a model on the FULL training split (no validation), save checkpoint.
    Returns the checkpoint path.
    """
    set_seed(cfg.project.seed + split_id * 1000)

    dev = get_device()
    model = model.to(dev)

    ds = WaveletDataset(cfg, train_df)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_wavelet_batch)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    losses: List[float] = []

    ckpt_path = run_dirs["checkpoints"] / f"split_{split_id:02d}_full_train_last.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb, _ in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        ep_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(ep_loss)

        if epoch == 1 or epoch % cfg.train.print_interval == 0:
            print(f"[full-train split {split_id}] epoch {epoch}/{cfg.train.epochs} loss={ep_loss:.4f}")

    # save checkpoint (last)
    torch.save(
        {
            "split_id": split_id,
            "epoch": cfg.train.epochs,
            "model_state_dict": model.state_dict(),
            "config_model_name": cfg.model.name,
        },
        ckpt_path,
    )

    # save loss curve (train only vs itself; we’ll just plot train twice for a 2-line helper)
    xs = list(range(1, cfg.train.epochs + 1))
    save_two_line_curve(
        xs=xs,
        ys_a=losses,
        ys_b=losses,
        label_a="train_loss",
        label_b="train_loss",
        title=f"Full-train loss split {split_id}",
        out_path=run_dirs["curves"] / f"split_{split_id:02d}" / "full_train_loss.png",
    )

    # save losses csv
    out_csv = run_dirs["cv"] / f"split_{split_id:02d}_full_train_loss.csv"
    pd.DataFrame({"epoch": xs, "train_loss": losses}).to_csv(out_csv, index=False)

    return ckpt_path