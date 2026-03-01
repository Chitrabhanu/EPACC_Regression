from __future__ import annotations

from pathlib import Path
import torch

from epacc_mle.config import Config
from epacc_mle.models import build_model


def load_model_from_checkpoint(cfg: Config, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt