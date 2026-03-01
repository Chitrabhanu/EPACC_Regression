from __future__ import annotations

from typing import Callable, Dict

from epacc_mle.config import Config
from epacc_mle.models.cnn1d import (
    CNN1D3LWithSEBN_REG,
    # add others as you port them
)

_MODEL_BUILDERS: Dict[str, Callable[[], object]] = {
    "cnn1d_sebn_reg": lambda: CNN1D3LWithSEBN_REG(),
    "1dcnn_3_layer_se_bn_reg": lambda: CNN1D3LWithSEBN_REG(),  # alias
}

def build_model(cfg: Config):
    key = cfg.model.name.strip().lower()
    if key not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model name: {cfg.model.name}. Available: {sorted(_MODEL_BUILDERS)}")
    return _MODEL_BUILDERS[key]()