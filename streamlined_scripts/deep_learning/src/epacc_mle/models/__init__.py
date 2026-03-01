from __future__ import annotations

from epacc_mle.models.registry import MODEL_REGISTRY, available_models


def build_model(cfg):
    """
    Build a model from cfg.model.name using the registry.

    Expected config:
      model:
        name: cnn1d_3l_se_bn_reg
    """
    name = getattr(getattr(cfg, "model", None), "name", None)
    if not name:
        raise ValueError(
            "Config is missing model.name. Example:\n"
            "model:\n  name: cnn1d_3l_se_bn_reg\n"
            f"Available: {available_models()}"
        )

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")

    return MODEL_REGISTRY[name]()