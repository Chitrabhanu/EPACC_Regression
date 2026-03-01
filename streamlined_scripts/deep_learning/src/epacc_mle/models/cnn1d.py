from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.mean(x, dim=-1)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # (B, C, 1)
        return x * s


class CNN1D3LWithSEBN_REG(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.se1 = SEBlock(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.se2 = SEBlock(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.se3 = SEBlock(256)

        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)

        x = torch.mean(x, dim=-1)  # global avg pool over time -> (B, 256)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # (B,)