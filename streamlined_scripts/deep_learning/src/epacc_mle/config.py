from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class OutlierFilterConfig:
    enabled: bool = True
    method: str = "zscore_global"  # or "zscore_train"
    z: float = 3.0


@dataclass(frozen=True)
class DataConfig:
    base_dir: str
    train_dir: str
    test_dir: str
    fold_pigs_csv: str

    key_dataset: str = "dataset"
    key_pig: str = "pig"
    key_bolus: str = "batch"
    target: str = "label"

    seq_len: int = 224
    feature_prefix: str = "bit_"

    outlier_filter: OutlierFilterConfig = OutlierFilterConfig()


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 16
    learning_rate: float = 5e-4
    print_interval: int = 25


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cnn1d_sebn_reg"
    pretrained: bool = False
    pretrained_weights_path: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    n_splits: int = 26
    n_folds: int = 5


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "epacc_waveforms"
    seed: int = 42


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    experiment: ExperimentConfig
    train: TrainConfig
    model: ModelConfig


def _deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_raw = _deep_get(raw, "project", {})
    data_raw = _deep_get(raw, "data", {})
    exp_raw = _deep_get(raw, "experiment", {})
    train_raw = _deep_get(raw, "train", {})
    model_raw = _deep_get(raw, "model", {})

    out_raw = _deep_get(data_raw, "outlier_filter", {})
    out_cfg = OutlierFilterConfig(
        enabled=bool(_deep_get(out_raw, "enabled", True)),
        method=str(_deep_get(out_raw, "method", "zscore_global")),
        z=float(_deep_get(out_raw, "z", 3.0)),
    )

    data_cfg = DataConfig(
        base_dir=str(data_raw["base_dir"]),
        train_dir=str(data_raw["train_dir"]),
        test_dir=str(data_raw["test_dir"]),
        fold_pigs_csv=str(data_raw["fold_pigs_csv"]),
        key_dataset=str(_deep_get(data_raw, "key_dataset", "dataset")),
        key_pig=str(_deep_get(data_raw, "key_pig", "pig")),
        key_bolus=str(_deep_get(data_raw, "key_bolus", "batch")),
        target=str(_deep_get(data_raw, "target", "label")),
        seq_len=int(_deep_get(data_raw, "seq_len", 224)),
        feature_prefix=str(_deep_get(data_raw, "feature_prefix", "bit_")),
        outlier_filter=out_cfg,
    )

    cfg = Config(
        project=ProjectConfig(
            name=str(_deep_get(project_raw, "name", "epacc_waveforms")),
            seed=int(_deep_get(project_raw, "seed", 42)),
        ),
        data=data_cfg,
        experiment=ExperimentConfig(
            n_splits=int(_deep_get(exp_raw, "n_splits", 26)),
            n_folds=int(_deep_get(exp_raw, "n_folds", 5)),
        ),
        train=TrainConfig(
            epochs=int(_deep_get(train_raw, "epochs", 150)),
            batch_size=int(_deep_get(train_raw, "batch_size", 16)),
            learning_rate=float(_deep_get(train_raw, "learning_rate", 5e-4)),
            print_interval=int(_deep_get(train_raw, "print_interval", 25)),
        ),
        model=ModelConfig(
            name=str(_deep_get(model_raw, "name", "cnn1d_sebn_reg")),
            pretrained=bool(_deep_get(model_raw, "pretrained", False)),
            pretrained_weights_path=str(_deep_get(model_raw, "pretrained_weights_path", "")),
        ),
    )
    return cfg