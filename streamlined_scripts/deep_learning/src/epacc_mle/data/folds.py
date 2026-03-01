from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from epacc_mle.config import Config


# Fold metadata encodes subject identifiers as pairs: (dataset, subject_id)
# In your project, subject_id corresponds to cfg.data.key_subject (typically "pig_id").
SubjectKey = Tuple[str, str]  # (dataset, subject_id)


@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    train_idx: list[int]
    val_idx: list[int]


def _parse_fold_cell(cell: str) -> list[SubjectKey]:
    """
    cell format example:
      "[['datasetA','Pig_1'], ['datasetB','Pig_2'], ...]"
    Returns list of (dataset, subject_id) pairs.
    """
    pairs = ast.literal_eval(cell)
    return [(str(ds), str(subject_id)) for ds, subject_id in pairs]


def load_split_folds(cfg: Config, split_id: int) -> list[list[SubjectKey]]:
    """
    Load predefined folds for a given split_id from fold_pigs_SV.csv.
    The CSV must have:
      - a column 'split'
      - fold columns named fold_1..fold_K
    Each fold cell contains a Python-literal list of [dataset, subject_id] pairs.
    """
    path = Path(cfg.data.base_dir) / cfg.data.fold_pigs_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing fold pigs csv: {path}")

    pigs_df = pd.read_csv(path)
    row = pigs_df[pigs_df["split"] == split_id]
    if row.empty:
        raise ValueError(f"split_id={split_id} not found in fold pigs csv: {path}")

    row = row.iloc[0]
    folds: list[list[SubjectKey]] = []
    for k in range(1, int(cfg.experiment.n_folds) + 1):
        col = f"fold_{k}"
        if col not in pigs_df.columns:
            raise ValueError(f"Fold pigs csv is missing required column '{col}' (n_folds={cfg.experiment.n_folds})")
        folds.append(_parse_fold_cell(row[col]))
    return folds


def make_fold_indices(cfg: Config, split_id: int, train_df: pd.DataFrame) -> list[FoldIndices]:
    """
    Build row indices for K-fold CV on the training dataframe.

    IMPORTANT:
      - Fold membership is determined by (dataset, subject_id), where subject_id is cfg.data.key_subject (pig_id).
      - This is intentionally separate from bolus identity (cfg.data.key_pig + cfg.data.key_bolus),
        which is used for bolus-level aggregation and metrics.
    """
    folds = load_split_folds(cfg, split_id)

    ds_col = cfg.data.key_dataset
    subject_col = cfg.data.key_subject  # <-- FIX: use subject id for folds

    # Validate required columns exist
    missing = [c for c in [ds_col, subject_col] if c not in train_df.columns]
    if missing:
        raise KeyError(f"Training dataframe is missing required columns for folds: {missing}. "
                       f"Available columns: {list(train_df.columns)}")

    # Map each (dataset, subject_id) key to row indices
    idx_by_key: dict[SubjectKey, list[int]] = {}
    ds_vals = train_df[ds_col].astype(str).tolist()
    subj_vals = train_df[subject_col].astype(str).tolist()

    for i, (ds, subject_id) in enumerate(zip(ds_vals, subj_vals)):
        idx_by_key.setdefault((ds, subject_id), []).append(i)

    out: list[FoldIndices] = []
    for fold_id, val_keys in enumerate(folds, start=1):
        val_set = set(val_keys)

        val_idx: list[int] = []
        train_idx: list[int] = []

        for key, idxs in idx_by_key.items():
            if key in val_set:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

        out.append(FoldIndices(fold_id=fold_id, train_idx=train_idx, val_idx=val_idx))

    return out