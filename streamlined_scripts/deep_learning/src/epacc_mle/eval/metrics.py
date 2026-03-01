from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred))


def make_wavelet_pred_df(
    ids_dataset: List[str],
    ids_pig: List[str],           # bolus identity component (cfg.data.key_pig, typically "pig")
    ids_bolus: List[str],         # bolus instance (cfg.data.key_bolus, typically "batch")
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ids_subject: Optional[List[str]] = None,  # optional: fold/subject id (cfg.data.key_subject, pig_id)
) -> pd.DataFrame:
    """
    Build a wavelet-level prediction dataframe.

    Columns:
      - dataset
      - pig      (bolus identity component)
      - bolus    (batch)
      - y_true
      - y_pred
      - subject  (optional, for debugging/fold attribution; NOT used for aggregation)
    """
    n = len(ids_dataset)
    if not (len(ids_pig) == len(ids_bolus) == len(y_true) == len(y_pred) == n):
        raise ValueError(
            "Length mismatch when building prediction dataframe: "
            f"dataset={len(ids_dataset)}, pig={len(ids_pig)}, bolus={len(ids_bolus)}, "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if ids_subject is not None and len(ids_subject) != n:
        raise ValueError(f"ids_subject length {len(ids_subject)} does not match n={n}")

    df = pd.DataFrame(
        {
            "dataset": pd.Series(ids_dataset, dtype="string"),
            "pig": pd.Series(ids_pig, dtype="string"),
            "bolus": pd.Series(ids_bolus, dtype="string"),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    if ids_subject is not None:
        df["subject"] = pd.Series(ids_subject, dtype="string")
    return df


def aggregate_to_bolus(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate wavelet-level predictions to bolus-level predictions.

    Bolus identity is defined as:
      (dataset, pig, bolus)

    Aggregation:
      - y_true: first (after verifying constant within bolus)
      - y_pred: mean across wavelets
      - n_wavelets: number of wavelets
    """
    required = {"dataset", "pig", "bolus", "y_true", "y_pred"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing columns: {missing}")

    # Verify y_true constant within each bolus group
    nunique = (
        pred_df.groupby(["dataset", "pig", "bolus"])["y_true"]
        .nunique(dropna=False)
        .reset_index(name="nunique_y_true")
    )
    bad = nunique[nunique["nunique_y_true"] != 1]
    if not bad.empty:
        ex = bad.head(5).to_dict(orient="records")

        # Attach small sample of offending rows for debugging
        sample = (
            pred_df.merge(bad[["dataset", "pig", "bolus"]], on=["dataset", "pig", "bolus"], how="inner")
            .sort_values(["dataset", "pig", "bolus"])
            .head(10)
        )
        raise ValueError(
            "Found boluses with non-constant y_true (showing up to 5): "
            f"{ex}\nSample offending rows (up to 10):\n{sample}"
        )

    bolus_df = (
        pred_df.groupby(["dataset", "pig", "bolus"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            n_wavelets=("y_pred", "size"),
        )
    )
    return bolus_df


def compute_wavelet_and_bolus_metrics(
    pred_df: pd.DataFrame,
) -> Tuple[RegressionMetrics, RegressionMetrics, pd.DataFrame]:
    """
    Compute wavelet-level and bolus-level regression metrics.

    Returns:
      - wavelet metrics
      - bolus metrics
      - bolus_df (aggregated predictions)
    """
    wave_m = compute_regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    bolus_df = aggregate_to_bolus(pred_df)
    bolus_m = compute_regression_metrics(bolus_df["y_true"].to_numpy(), bolus_df["y_pred"].to_numpy())
    return wave_m, bolus_m, bolus_df