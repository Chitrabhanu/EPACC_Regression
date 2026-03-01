from __future__ import annotations

from typing import Callable, Dict

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

# Constructor signature: () -> nn.Module
MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    # CNN family (original names preserved conceptually)
    "cnn1d_3l": lambda: CNN1D3L(),
    "cnn1d_3l_se": lambda: CNN1D3LWithSE(),
    "cnn1d_3l_se_bn": lambda: CNN1D3LWithSEBN(),
    "cnn1d_3l_se_bn_reg": lambda: CNN1D3LWithSEBN_REG(),
    "cnn1d_3l_do": lambda: CNN1D3LWithDO(),
    "cnn1d_7l_do": lambda: CNN1D7LWithDO(),

    # Other architectures you had
    "resnet1d": lambda: ResNet1D(),
    "wavenet1d": lambda: WaveNet1D(),
    "transformer1d": lambda: Transformer1D(),
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())