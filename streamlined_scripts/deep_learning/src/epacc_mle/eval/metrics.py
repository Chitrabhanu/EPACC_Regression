from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred))


def make_wavelet_pred_df(
    ids_dataset: List[str],
    ids_pig: List[str],      # pig (bolus id)
    ids_bolus: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    if not (len(ids_dataset) == len(ids_pig) == len(ids_bolus) == len(y_true) == len(y_pred)):
        raise ValueError("Length mismatch when building prediction dataframe.")
    return pd.DataFrame(
        {
            "dataset": ids_dataset,
            "pig": ids_pig,          # bolus identifier column
            "bolus": ids_bolus,      # batch
            "y_true": y_true.astype(float),
            "y_pred": y_pred.astype(float),
        }
    )


def aggregate_to_bolus(pred_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "pig", "bolus", "y_true", "y_pred"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing columns: {missing}")

    nunique = (
        pred_df.groupby(["dataset", "pig", "bolus"])["y_true"]
        .nunique(dropna=False)
        .reset_index(name="nunique_y_true")
    )
    bad = nunique[nunique["nunique_y_true"] != 1]
    if not bad.empty:
        ex = bad.head(5).to_dict(orient="records")
        raise ValueError(f"Found boluses with non-constant y_true (showing up to 5): {ex}")

    bolus_df = (
        pred_df.groupby(["dataset", "pig", "bolus"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), n_wavelets=("y_pred", "size"))
    )
    return bolus_df

def compute_wavelet_and_bolus_metrics(pred_df: pd.DataFrame) -> tuple[RegressionMetrics, RegressionMetrics, pd.DataFrame]:
    wave_m = compute_regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    bolus_df = aggregate_to_bolus(pred_df)
    bolus_m = compute_regression_metrics(bolus_df["y_true"].to_numpy(), bolus_df["y_pred"].to_numpy())
    return wave_m, bolus_m, bolus_df