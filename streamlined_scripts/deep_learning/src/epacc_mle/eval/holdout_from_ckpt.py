from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from epacc_mle.config import Config
from epacc_mle.data.io import load_split_data
from epacc_mle.eval.holdout import evaluate_holdout_split
from epacc_mle.models.io import load_model_from_checkpoint


def evaluate_holdout_from_checkpoint(
    cfg: Config,
    ckpt_path: Path,
    split_id: int,
    run_dirs: Dict,
    save_preds: bool = True,
):
    _, test_df = load_split_data(cfg, split_id=split_id)
    model, ckpt = load_model_from_checkpoint(cfg, ckpt_path)
    return evaluate_holdout_split(cfg, model, test_df, run_dirs, split_id, save_preds=save_preds)