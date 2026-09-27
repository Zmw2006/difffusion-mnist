"""Small time-conditioned U-Net for 28×28 grayscale images."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        angles = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.time_proj(time_emb)[:, :, None, None]
        return self.conv2(h) + self.skip(x)


class UNet(nn.Module):
    """Predict additive Gaussian noise; input and output are [B,1,28,28]."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a multiple of 8 and at least 8")
        c = base_channels
        time_dim = c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(c), nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.pool1 = nn.Conv2d(c, c, 4, stride=2, padding=1)
        self.down2 = ResBlock(c, c * 2, time_dim)
        self.pool2 = nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.middle = ResBlock(c * 2, c * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.decode2 = ResBlock(c * 4, c * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.decode1 = ResBlock(c * 2, c, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t)
        first = self.down1(self.input(x), emb)
        second = self.down2(self.pool1(first), emb)
        middle = self.middle(self.pool2(second), emb)
        up = self.decode2(torch.cat((self.up2(middle), second), dim=1), emb)
        up = self.decode1(torch.cat((self.up1(up), first), dim=1), emb)
        return self.output(up)

