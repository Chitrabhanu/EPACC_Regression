from __future__ import annotations

from epacc_mle.config import Config
from epacc_mle.models.cnn1d import CNN1D3LWithSEBN_REG


def build_model(cfg: Config):
    name = cfg.model.name.lower()
    if name in {"cnn1d_sebn_reg", "cnn_sebn_reg", "1dcnn_3_layer_se_bn_reg"}:
        return CNN1D3LWithSEBN_REG()
    raise ValueError(f"Unknown model name: {cfg.model.name}")