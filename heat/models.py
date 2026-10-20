"""The three architectures. Each maps (B, lookback, channels) to one prediction per hold."""
import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, channels, holds, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(channels, hidden, layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(4 * hidden, holds)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(torch.cat([h[:, -1], h.mean(1)], 1))


class _Block(nn.Module):
    def __init__(self, width, kernel, dropout):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel, padding=pad), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel, padding=pad), nn.BatchNorm1d(width))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResNet1D(nn.Module):
    def __init__(self, channels, holds, width=64, blocks=3, kernel=3, dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(channels, width, 1)
        self.blocks = nn.Sequential(*[_Block(width, kernel, dropout) for _ in range(blocks)])
        self.head = nn.Linear(width, holds)

    def forward(self, x):
        z = self.blocks(self.stem(x.transpose(1, 2)))
        return self.head(z.mean(2))


class Transformer(nn.Module):
    def __init__(self, channels, holds, lookback, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, holds)

    def forward(self, x):
        z = self.encoder(self.embed(x) + self.pos)
        return self.head(z.mean(1))


def make_model(arch, channels, holds, lookback, size):
    if arch == "bilstm":
        return BiLSTM(channels, holds, hidden=size)
    if arch == "resnet1d":
        return ResNet1D(channels, holds, width=size)
    if arch == "transformer":
        return Transformer(channels, holds, lookback, d_model=size)
    raise ValueError(f"unknown arch {arch!r}; use bilstm, resnet1d or transformer")
