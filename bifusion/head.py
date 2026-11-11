"""Detection heads: spatial attention for P2 and the decoupled anchor-free head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .modules import ConvBNAct


class SpatialAttention(nn.Module):
    """Channel-pooled 7x7 attention map, 99 parameters at any width."""

    def __init__(self, k: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, k // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        return torch.sigmoid(self.conv(pooled))


class DecoupledHead(nn.Module):
    """Two 3x3 convolutions per branch, one class logit and 4 x bins outputs."""

    def __init__(self, cin: int, hid: int = 64, bins: int = 16, nc: int = 1) -> None:
        super().__init__()
        self.bins = bins
        self.nc = nc
        self.cls = nn.Sequential(
            ConvBNAct(cin, hid, 3),
            ConvBNAct(hid, hid, 3),
            nn.Conv2d(hid, nc, 1, bias=True),
        )
        self.box = nn.Sequential(
            ConvBNAct(cin, hid, 3),
            ConvBNAct(hid, hid, 3),
            nn.Conv2d(hid, 4 * bins, 1, bias=True),
        )
        prior = 0.01
        nn.init.constant_(self.cls[-1].bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.box[-1].bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.cls(x), self.box(x)), dim=1)


class P2Head(nn.Module):
    """Spatial attention followed by the decoupled head at 1/4 resolution."""

    def __init__(
        self, cin: int = 32, hid: int = 64, bins: int = 16, nc: int = 1
    ) -> None:
        super().__init__()
        self.sa = SpatialAttention()
        self.head = DecoupledHead(cin, hid, bins, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x * self.sa(x))
