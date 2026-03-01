from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from epacc_mle.config import Config
from epacc_mle.data.io import waveform_columns


@dataclass(frozen=True)
class RowIds:
    """
    Canonical identifiers carried per wavelet row.

    - dataset: dataset namespace
    - subject: subject identifier used for CV folds (pig_id)
    - pig: bolus identity component (pig)
    - bolus: bolus instance within pig (batch)
    """
    dataset: str
    subject: str
    pig: str
    bolus: str


class WaveletDataset(Dataset):
    """
    Wavelet-level dataset with per-row IDs for safe aggregation later.

    Returns: (x, y, ids_dict)
      x: FloatTensor shape (1, seq_len)
      y: FloatTensor scalar
      ids_dict: python dict with dataset/subject/pig/bolus
    """

    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)

        # Waveform feature columns
        feat_cols = waveform_columns(cfg)
        missing_feat = [c for c in feat_cols if c not in self.df.columns]
        if missing_feat:
            raise ValueError(f"Missing waveform columns (showing up to 5): {missing_feat[:5]}")

        # IDs / keys
        ds_col = cfg.data.key_dataset
        subject_col = cfg.data.key_subject  # pig_id (for fold assignment)
        pig_col = cfg.data.key_pig          # pig (for bolus identity)
        bolus_col = cfg.data.key_bolus      # batch

        missing_id = [c for c in [ds_col, subject_col, pig_col, bolus_col] if c not in self.df.columns]
        if missing_id:
            raise ValueError(
                f"Missing required ID columns: {missing_id}. "
                f"Needed: dataset={ds_col}, subject={subject_col}, pig={pig_col}, bolus={bolus_col}. "
                f"Available (first 25): {list(self.df.columns)[:25]}"
            )

        target_col = cfg.data.target
        if target_col not in self.df.columns:
            raise ValueError(f"Missing target column '{target_col}'. Available (first 25): {list(self.df.columns)[:25]}")

        # Features (N, seq_len) -> unsqueeze channel in __getitem__
        self.X = self.df[feat_cols].to_numpy(dtype=np.float32)
        self.y = self.df[target_col].to_numpy(dtype=np.float32)

        # Store ids as string lists for fast indexing
        self.ids_dataset = self.df[ds_col].astype(str).tolist()
        self.ids_subject = self.df[subject_col].astype(str).tolist()
        self.ids_pig = self.df[pig_col].astype(str).tolist()
        self.ids_bolus = self.df[bolus_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # (1, seq_len)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        ids: Dict[str, str] = {
            "dataset": self.ids_dataset[idx],
            "subject": self.ids_subject[idx],  # for fold splitting (pig_id)
            "pig": self.ids_pig[idx],          # bolus identity component (pig)
            "bolus": self.ids_bolus[idx],      # batch
        }
        return x, y, ids