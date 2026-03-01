from __future__ import annotations

from pathlib import Path

from epacc_mle.config import load_config


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    print("Loaded config:")
    print(cfg)


if __name__ == "__main__":
    main()