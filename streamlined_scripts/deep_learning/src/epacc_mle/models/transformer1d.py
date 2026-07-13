from __future__ import annotations

import math
import torch
import torch.nn as nn


class Transformer1D(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, n_heads=4, num_layers=2,
                 dim_feedforward=256, max_len=4096):
        super().__init__()
        self.max_len = max_len

        # Per-timestep linear embedding: input_dim -> model_dim
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)

        # Precompute sinusoidal table, shape (max_len, 1, C) to broadcast over batch
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(max_len).unsqueeze(1)                       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))                         # (max_len, 1, C)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x: (B, input_dim, T)
        T = x.size(-1)
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len={self.max_len}. "
                "Increase max_len."
            )

        x = self.input_proj(x)      # (B, C, T)
        x = x.permute(2, 0, 1)      # (T, B, C)
        x = x + self.pe[:T]         # slice table to actual length, then broadcast
        x = self.transformer(x)     # (T, B, C)
        x = x.permute(1, 2, 0)      # (B, C, T)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.fc(x)           # (B, 1)
