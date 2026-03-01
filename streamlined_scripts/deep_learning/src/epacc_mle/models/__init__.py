from __future__ import annotations

from epacc_mle.models.registry import available_models, build_from_name


def build_model(cfg):
    name = getattr(getattr(cfg, "model", None), "name", None)
    if not name:
        raise ValueError(
            "Config missing model.name. Example:\n"
            "model:\n  name: cnn1d_sebn_reg\n"
            f"Available: {available_models()}"
        )
    return build_from_name(name)