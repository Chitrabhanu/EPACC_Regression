from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from epacc_mle.config import Config


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def get_run_dir(run_id: str) -> Path:
    return Path("artifacts") / "runs" / run_id


def ensure_run_dirs(run_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": run_dir,
        "cv": run_dir / "cv",
        "holdout": run_dir / "holdout",
        "curves": run_dir / "curves",
        "checkpoints": run_dir / "checkpoints",
        "preds": run_dir / "preds",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_resolved_config(cfg: Config, run_dir: Path) -> None:
    out = run_dir / "config_resolved.yaml"
    # dataclasses -> dict
    cfg_dict: Dict[str, Any] = asdict(cfg)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)