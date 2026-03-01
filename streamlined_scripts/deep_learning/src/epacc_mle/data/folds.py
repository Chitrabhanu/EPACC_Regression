from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from epacc_mle.config import Config


PigKey = Tuple[str, str]  # (dataset, pig)


@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    train_idx: list[int]
    val_idx: list[int]


def _parse_fold_cell(cell: str) -> list[tuple[str, str]]:
    # cell format: "[['datasetA','Pig_1'], ['datasetB','Pig_2'], ...]"
    pairs = ast.literal_eval(cell)
    return [(ds, pig) for ds, pig in pairs]


def load_split_folds(cfg: Config, split_id: int) -> list[list[PigKey]]:
    path = Path(cfg.data.base_dir) / cfg.data.fold_pigs_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing fold pigs csv: {path}")

    pigs_df = pd.read_csv(path)
    row = pigs_df[pigs_df["split"] == split_id]
    if row.empty:
        raise ValueError(f"split_id={split_id} not found in fold pigs csv")

    row = row.iloc[0]
    folds: list[list[PigKey]] = []
    for k in range(1, cfg.experiment.n_folds + 1):
        folds.append(_parse_fold_cell(row[f"fold_{k}"]))
    return folds


def make_fold_indices(cfg: Config, split_id: int, train_df: pd.DataFrame) -> list[FoldIndices]:
    folds = load_split_folds(cfg, split_id)

    ds_col = cfg.data.key_dataset
    pig_col = cfg.data.key_pig

    # map each pig key to row indices
    idx_by_key: dict[PigKey, list[int]] = {}
    for i, (ds, pig) in enumerate(zip(train_df[ds_col].tolist(), train_df[pig_col].tolist())):
        idx_by_key.setdefault((ds, pig), []).append(i)

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