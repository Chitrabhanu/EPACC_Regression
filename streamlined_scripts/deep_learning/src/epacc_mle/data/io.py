from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from epacc_mle.config import Config


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    test_csv: Path


def get_split_paths(cfg: Config, split_id: int) -> SplitPaths:
    base = Path(cfg.data.base_dir)
    train_csv = base / cfg.data.train_dir / f"SV_PS_train_{split_id}.csv"
    test_csv = base / cfg.data.test_dir / f"SV_PS_test_{split_id}.csv"
    return SplitPaths(train_csv=train_csv, test_csv=test_csv)


def waveform_columns(cfg: Config) -> list[str]:
    return [f"{cfg.data.feature_prefix}{i+1}" for i in range(cfg.data.seq_len)]


def compute_global_outlier_bounds_from_df(cfg: Config, df: pd.DataFrame) -> Tuple[float, float]:
    z = cfg.data.outlier_filter.z
    y = df[cfg.data.target].to_numpy(dtype=float)
    mu = float(np.mean(y))
    sd = float(np.std(y))
    lo = mu - z * sd
    hi = mu + z * sd
    return lo, hi


def apply_outlier_filter(cfg: Config, df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    y = df[cfg.data.target].to_numpy(dtype=float)
    keep = (y > lo) & (y < hi)
    return df.loc[keep].reset_index(drop=True)


def load_split_data(cfg: Config, split_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_split_paths(cfg, split_id)
    if not paths.train_csv.exists():
        raise FileNotFoundError(f"Missing train csv: {paths.train_csv}")
    if not paths.test_csv.exists():
        raise FileNotFoundError(f"Missing test csv: {paths.test_csv}")

    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)

    if cfg.data.outlier_filter.enabled:
        # Strict option later: train-only bounds. For now: bounds from TRAIN to avoid holdout leakage.
        lo, hi = compute_global_outlier_bounds_from_df(cfg, train_df)
        train_df = apply_outlier_filter(cfg, train_df, lo, hi)
        test_df = apply_outlier_filter(cfg, test_df, lo, hi)

    return train_df, test_df