from __future__ import annotations

import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    """
    Original: WaveNetBlock
    Residual gated dilated conv block:
      tanh(filter) * sigmoid(gate) -> 1x1 residual conv -> add residual
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
        )
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tanh_out = torch.tanh(self.filter_conv(x))
        sig_out = torch.sigmoid(self.gate_conv(x))
        out = tanh_out * sig_out
        res = self.res_conv(out)
        return res + x


class WaveNet1D(nn.Module):
    """Original: WaveNet1D"""

    def __init__(self) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(1, 32, kernel_size=1)
        self.wavenet_blocks = nn.Sequential(
            WaveNetBlock(32, 32, kernel_size=2, dilation=1),
            WaveNetBlock(32, 32, kernel_size=2, dilation=2),
            WaveNetBlock(32, 32, kernel_size=2, dilation=4),
            WaveNetBlock(32, 32, kernel_size=2, dilation=8),
            WaveNetBlock(32, 32, kernel_size=2, dilation=16),
        )
        self.output_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.wavenet_blocks(x)
        x = self.output_pool(x).squeeze(-1)  # (B, 32)
        return self.fc(x)  # (B, 1)