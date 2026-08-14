import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 🔹 Attention Mechanism
# ------------------------------
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = self.attention(x)          # (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)  # Normalize across seq_len
        context = torch.sum(weights * x, dim=1)  # Weighted sum
        return context, weights


# ------------------------------
# 🔹 Convolutional Block with Spatial Dropout
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        # ✅ Spatial Dropout (Dropout2d works as Dropout1d in 1D conv feature maps)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.dropout(x.unsqueeze(2)).squeeze(2)  # apply spatial dropout
        return x


# ------------------------------
# 🔹 Residual Block
# ------------------------------
class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ------------------------------
# 🔹 Final Model (Reduced GRU + FC sizes)
# ------------------------------
class CNN1DAttentionEnhanced(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1DAttentionEnhanced, self).__init__()

        self.block1 = nn.Sequential(
            ConvBlock(12, 16),
            ResidualUnit(16),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            ConvBlock(16, 32),
            ResidualUnit(32),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            ConvBlock(32, 64),
            ResidualUnit(64),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block4 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=7),
            ResidualUnit(128),
            nn.MaxPool1d(kernel_size=2)
        )

        # ✅ GRU with smaller hidden size (64 → 32)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.2
        )

        # ✅ Attention updated for new GRU output size
        self.attention = Attention(input_dim=32)

        # ✅ Smaller FC layers (64 → 32)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Input: (batch_size, 12, seq_len)
        x = self.block1(x)      # → (batch_size, 16, L/2)
        x = self.block2(x)      # → (batch_size, 32, L/4)
        x = self.block3(x)      # → (batch_size, 64, L/8)
        x = self.block4(x)      # → (batch_size, 128, L/16)

        # Prepare for GRU
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, 128)

        # Temporal modeling
        x, _ = self.gru(x)      # (batch_size, seq_len, 32)

        # Attention
        x, _ = self.attention(x)  # (batch_size, 32)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



