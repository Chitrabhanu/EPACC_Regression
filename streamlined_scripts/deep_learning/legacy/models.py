import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D3L(nn.Module):
    def __init__(self):
        super(CNN1D3L, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        s = torch.mean(x, dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)
        return x * s

class CNN1D3LWithSE(CNN1D3L):
    def __init__(self):
        super().__init__()
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)

    def forward(self, x):
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
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
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
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
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
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class CNN1D7LWithDO(nn.Module):
    def __init__(self):
        super(CNN1D7LWithDO, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.layer1 = ResidualBlock1D(64)
        self.layer2 = ResidualBlock1D(64)
        self.layer3 = ResidualBlock1D(64)
        self.layer4 = ResidualBlock1D(128)
        self.transition1 = nn.Conv1d(64, 128, kernel_size=1)
        self.layer5 = ResidualBlock1D(128)
        self.layer6 = ResidualBlock1D(128)
        self.layer7 = ResidualBlock1D(256)
        self.transition2 = nn.Conv1d(128, 256, kernel_size=1)
        self.layer8 = ResidualBlock1D(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
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
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
        

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                     padding=dilation * (kernel_size - 1), dilation=dilation)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=dilation * (kernel_size - 1), dilation=dilation)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        tanh_out = torch.tanh(self.filter_conv(x))
        sig_out = torch.sigmoid(self.gate_conv(x))
        out = tanh_out * sig_out
        res = self.res_conv(out)
        return res + x  # residual connection

class WaveNet1D(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.input_conv(x)
        x = self.wavenet_blocks(x)
        x = self.output_pool(x).squeeze(-1)
        return self.fc(x)

class Transformer1D(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, n_heads=4, num_layers=2, seq_len=100):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)  # (B, C, T)
        x = x.permute(2, 0, 1)  # (T, B, C) for transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (B, C, T)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
