from __future__ import annotations

from typing import Callable, Dict, List

import torch.nn as nn

from epacc_mle.models.cnn1d import (
    CNN1D3L,
    CNN1D3LWithDO,
    CNN1D3LWithSE,
    CNN1D3LWithSEBN,
    CNN1D3LWithSEBN_REG,
    CNN1D7LWithDO,
)
from epacc_mle.models.resnet1d import ResNet1D
from epacc_mle.models.transformer1d import Transformer1D
from epacc_mle.models.wavenet1d import WaveNet1D


# Canonical constructors
_MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "cnn1d_3l": lambda: CNN1D3L(),
    "cnn1d_3l_se": lambda: CNN1D3LWithSE(),
    "cnn1d_3l_se_bn": lambda: CNN1D3LWithSEBN(),
    "cnn1d_3l_se_bn_reg": lambda: CNN1D3LWithSEBN_REG(),
    "cnn1d_3l_do": lambda: CNN1D3LWithDO(),
    "cnn1d_7l_do": lambda: CNN1D7LWithDO(),
    "resnet1d": lambda: ResNet1D(),
    "wavenet1d": lambda: WaveNet1D(),
    "transformer1d": lambda: Transformer1D(),
}

# Aliases (backwards compatibility)
_ALIASES: Dict[str, str] = {
    #Default:
    "cnn1d_sebn_reg": "cnn1d_3l_se_bn_reg",

    # Optional: match old script names too
    "1DCNN_basic_3_layer": "cnn1d_3l",
    "1DCNN_3_layer_SE": "cnn1d_3l_se",
    "1DCNN_3_layer_SE_BN": "cnn1d_3l_se_bn",
    "1DCNN_3_layer_SE_BN_reg": "cnn1d_3l_se_bn_reg",
    "1DCNN_3_layer_DO": "cnn1d_3l_do",
    "1DCNN_7_layer_DO": "cnn1d_7l_do",
}


def resolve_model_name(name: str) -> str:
    return _ALIASES.get(name, name)


def build_from_name(name: str) -> nn.Module:
    key = resolve_model_name(name)
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}' (resolved='{key}'). Available: {available_models()}")
    return _MODEL_REGISTRY[key]()


def available_models() -> List[str]:
    return sorted(set(list(_MODEL_REGISTRY.keys()) + list(_ALIASES.keys())))


MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = _MODEL_REGISTRY
