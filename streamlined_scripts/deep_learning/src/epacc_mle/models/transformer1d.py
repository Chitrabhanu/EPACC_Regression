from __future__ import annotations

import torch
import torch.nn as nn


class Transformer1D(nn.Module):
    """
    Original: Transformer1D concept from your models.py.

    Input: (B, 1, T)
      - project to model_dim via 1x1 conv
      - permute to (T, B, C)
      - TransformerEncoder
      - permute back to (B, C, T)
      - adaptive average pool to (B, C)
      - linear regression head -> (B, 1)
    """

    def __init__(
        self,
        input_dim: int = 1,
        model_dim: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=False,  # expects (T, B, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x = self.input_proj(x)        # (B, C, T)
        x = x.permute(2, 0, 1)        # (T, B, C)
        x = self.transformer(x)       # (T, B, C)
        x = x.permute(1, 2, 0)        # (B, C, T)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.fc(x)             # (B, 1)