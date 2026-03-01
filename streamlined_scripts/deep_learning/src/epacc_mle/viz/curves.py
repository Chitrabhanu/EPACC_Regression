from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def save_two_line_curve(
    xs: List[float],
    ys_a: List[float],
    ys_b: List[float],
    label_a: str,
    label_b: str,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(xs, ys_a, label=label_a)
    plt.plot(xs, ys_b, label=label_b)
    plt.legend()
    plt.title(title)
    plt.savefig(out_path)
    plt.close()