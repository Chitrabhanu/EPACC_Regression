# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        bs, channels, _ = x.size()
        squeeze = self.avg_pool(x).view(bs, channels)
        excite = F.relu(self.fc1(squeeze))
        excite = torch.sigmoid(self.fc2(excite)).view(bs, channels, 1)
        return x * excite

# Basic 3-layer 1D CNN
class CNN1D3L(nn.Module):
    def __init__(self):
        super(CNN1D3L, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(128 * 224, 2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3-layer CNN with SE
class CNN1D3LWithSE(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSE, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1)
        self.se1 = SEBlock(32, reduction_ratio)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.se2 = SEBlock(64, reduction_ratio)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.se1(self.conv1(x)))
        x = F.relu(self.se2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.fc(x)

# 3-layer CNN with SE and BatchNorm
class CNN1D3LWithSEBN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSEBN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SEBlock(32, reduction_ratio)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64, reduction_ratio)

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.se1(F.relu(self.bn1(self.conv1(x))))
        x = self.se2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.fc(x)

# 3-layer CNN with SE, BN and regression output
class CNN1D3LWithSEBN_REG(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, reduction_ratio=16):
        super(CNN1D3LWithSEBN_REG, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SEBlock(32, reduction_ratio)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64, reduction_ratio)

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.se1(F.relu(self.bn1(self.conv1(x))))
        x = self.se2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 7-layer CNN with dropout
class CNN1D7LWithDO(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.25):
        super(CNN1D7LWithDO, self).__init__()
        layers = []
        channels = [input_channels, 32, 64, 128, 256, 256, 128, 64]
        for i in range(7):
            layers.append(nn.Conv1d(channels[i], channels[i+1], 3, padding=1))
            layers.append(nn.BatchNorm1d(channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.fc(x)

# 3-layer CNN with dropout
class CNN1D3LWithDO(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.25):
        super(CNN1D3LWithDO, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.do1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.do2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.do3 = nn.Dropout(dropout_rate)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.do1(F.relu(self.bn1(self.conv1(x))))
        x = self.do2(F.relu(self.bn2(self.conv2(x))))
        x = self.do3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.fc(x)

# Model lookup dictionary
model_catalog = {
    '1DCNN_basic_3_layer': CNN1D3L(),
    '1DCNN_3_layer_SE': CNN1D3LWithSE(),
    '1DCNN_3_layer_SE_BN': CNN1D3LWithSEBN(),
    '1DCNN_3_layer_DO': CNN1D3LWithDO(),
    '1DCNN_7_layer_DO': CNN1D7LWithDO(),
    '1DCNN_3_layer_SE_BN_reg': CNN1D3LWithSEBN_REG()
}
