from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D3L(nn.Module):
    """
    Original: CNN1D3L
    Conv1d(1->64)->Conv1d(64->128)->Conv1d(128->256)
    global mean pool over time -> FC -> regression
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)  # (B, 256)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # (B, 1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block operating on (B, C, T)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels // reduction < 1:
            raise ValueError(f"SE reduction too large for channels={channels}, reduction={reduction}")
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.mean(x, dim=-1)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # (B, C, 1)
        return x * s


class CNN1D3LWithSE(CNN1D3L):
    """Original: CNN1D3LWithSE"""

    def __init__(self) -> None:
        super().__init__()
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = F.relu(self.conv3(x))
        x = self.se3(x)
        x = torch.mean(x, dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN1D3LWithSEBN(CNN1D3LWithSE):
    """Original: CNN1D3LWithSEBN"""

    def __init__(self) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = torch.mean(x, dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNN1D3LWithSEBN_REG(CNN1D3LWithSEBN):
    """Original: CNN1D3LWithSEBN_REG (adds dropout)"""

    def __init__(self, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class CNN1D3LWithDO(CNN1D3L):
    """Original: CNN1D3LWithDO"""

    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class CNN1D7LWithDO(nn.Module):
    """Original: CNN1D7LWithDO"""

    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)