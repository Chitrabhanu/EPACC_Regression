from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """
    Original: ResNet1D from your monolithic models.py.
    Note: this is a fairly custom progression; kept faithful to your implementation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.initial = nn.Conv1d(1, 64, kernel_size=7, padding=3)

        self.layer1 = ResidualBlock1D(64)
        self.layer2 = ResidualBlock1D(64)
        self.layer3 = ResidualBlock1D(64)

        self.transition1 = nn.Conv1d(64, 128, kernel_size=1)
        self.layer4 = ResidualBlock1D(128)
        self.layer5 = ResidualBlock1D(128)
        self.layer6 = ResidualBlock1D(128)

        self.transition2 = nn.Conv1d(128, 256, kernel_size=1)
        self.layer7 = ResidualBlock1D(256)
        self.layer8 = ResidualBlock1D(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.initial(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.transition1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.transition2(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.global_pool(x).squeeze(-1)  # (B, 256)
        return self.fc(x)  # (B, 1)