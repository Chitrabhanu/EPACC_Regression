from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from epacc_mle.config import Config
from epacc_mle.data.io import waveform_columns


@dataclass(frozen=True)
class RowIds:
    dataset: str
    pig_id: str
    bolus: str  # batch
    # optional: keep original pig string if you want
    pig_raw: str | None = None


class WaveletDataset(Dataset):
    """
    Wavelet-level dataset with per-row IDs for safe aggregation later.
    Returns: (x, y, ids_dict)
      x: FloatTensor shape (1, seq_len)
      y: FloatTensor scalar
      ids_dict: python dict with dataset/pig_id/bolus (+ optional)
    """

    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)

        feat_cols = waveform_columns(cfg)
        missing = [c for c in feat_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing waveform columns (showing up to 5): {missing[:5]}")

        # Features (N, seq_len) -> we'll unsqueeze channel in __getitem__
        self.X = self.df[feat_cols].to_numpy(dtype=np.float32)
        self.y = self.df[cfg.data.target].to_numpy(dtype=np.float32)

        ds_col = cfg.data.key_dataset
        pig_col = cfg.data.key_pig           # should be pig_id now
        bolus_col = cfg.data.key_bolus       # batch

        if ds_col not in self.df.columns or pig_col not in self.df.columns or bolus_col not in self.df.columns:
            raise ValueError(
                f"Missing ID columns. Need: {ds_col}, {pig_col}, {bolus_col}. "
                f"Have: {list(self.df.columns)[:20]}..."
            )

        self.ids_dataset = self.df[ds_col].astype(str).tolist()
        self.ids_pig = self.df[pig_col].astype(str).tolist()
        self.ids_bolus = self.df[bolus_col].astype(str).tolist()

        # raw bolus identifier column used in your original logic
        if "pig" not in self.df.columns:
            raise ValueError("Expected column 'pig' to exist (used as bolus identifier in original logic).")
        self.ids_pig_raw = self.df["pig"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # (1, seq_len)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

            ids: Dict[str, str] = {
            "dataset": self.ids_dataset[idx],
            "pig_id": self.ids_pig[idx],     # for fold splitting
            "pig": self.ids_pig_raw[idx],    # for bolus identity/aggregation
            "bolus": self.ids_bolus[idx],    # batch
        }
        return x, y, ids
        if self.ids_pig_raw is not None:
            ids["pig_raw"] = self.ids_pig_raw[idx]
        return x, y, ids