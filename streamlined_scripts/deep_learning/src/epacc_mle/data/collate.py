from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def collate_wavelet_batch(batch):
    """
    Batch is list of (x, y, ids_dict).
    Returns:
      xb: (B,1,T) float
      yb: (B,) float
      ids: dict[str, list]  (e.g., ids["pig_id"][i])
    """
    xs, ys, ids_list = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.stack([y.view(()) for y in ys], dim=0).float()

    # turn list[dict] into dict[list]
    ids: Dict[str, List[Any]] = {}
    for d in ids_list:
        for k, v in d.items():
            ids.setdefault(k, []).append(v)
    return xb, yb, ids