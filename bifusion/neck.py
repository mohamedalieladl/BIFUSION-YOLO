"""Neck: lateral adapters, weighted BiFPN and CCFM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvBNAct, DWConvBlock


def resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="nearest")


class FusionNode(nn.Module):
    """Weighted sum of 2 or 3 inputs followed by a depthwise separable conv.

    The weights are nonnegative and divided by their sum plus epsilon, which is
    the fast normalized fusion written in Eqs. (28) and (29) of the paper.
    """

    def __init__(self, c: int, n_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs))
        self.conv = DWConvBlock(c)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-4)
        y = sum(wi * xi for wi, xi in zip(w, inputs))
        return self.conv(y)


class BiFPNBlock(nn.Module):
    """One block with the seven fused nodes counted in Table 3."""

    def __init__(self, c: int = 64) -> None:
        super().__init__()
        self.td4 = FusionNode(c, 2)
        self.td3 = FusionNode(c, 2)
        self.td2 = FusionNode(c, 2)
        self.out2 = FusionNode(c, 2)
        self.out3 = FusionNode(c, 3)
        self.out4 = FusionNode(c, 3)
        self.out5 = FusionNode(c, 3)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p2, p3, p4, p5 = feats
        td4 = self.td4([resize_to(p5, p4.shape[-2:]), p4])
        td3 = self.td3([resize_to(td4, p3.shape[-2:]), p3])
        td2 = self.td2([resize_to(td3, p2.shape[-2:]), p2])
        out2 = self.out2([p2, td2])
        out3 = self.out3([p3, td3, resize_to(out2, p3.shape[-2:])])
        out4 = self.out4([p4, td4, resize_to(out3, p4.shape[-2:])])
        out5 = self.out5([p5, p5, resize_to(out4, p5.shape[-2:])])
        return [out2, out3, out4, out5]


class SharedLevelProjection(nn.Module):
    """One 64 to 256 projection shared by the four CCFM levels.

    Each level keeps the prefix of the projection that its head consumes: 32
    channels at P2, 64 at P3, 128 at P4 and 256 at P5. Sharing the matrix, and
    computing only the rows a level needs, is what makes the CCFM entry of
    Table 3 (52,480 parameters) exact.
    """

    def __init__(self, cin: int = 64, cmax: int = 256) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cmax, cin, 1, 1))
        self.bias = nn.Parameter(torch.zeros(cmax))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(cin)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, width: int) -> torch.Tensor:
        return F.conv2d(x, self.weight[:width], self.bias[:width])


class CCFM(nn.Module):
    """Four CCFM units, one per level, with a shared output projection."""

    def __init__(
        self, c: int = 64, cmax: int = 256, widths: tuple[int, ...] = (32, 64, 128, 256)
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c, c, 1) for _ in range(4)])
        self.rep = nn.ModuleList([DWConvBlock(c) for _ in range(4)])
        self.out = SharedLevelProjection(c, cmax)
        self.widths = widths

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for i, x in enumerate(feats):
            neighbour = feats[i + 1] if i + 1 < len(feats) else feats[i - 1]
            a = self.proj[i](x)
            b = self.proj[i](resize_to(neighbour, x.shape[-2:]))
            y = self.rep[i](a + b) + a
            outputs.append(self.out(y, self.widths[i]))
        return outputs


class LateralAdapters(nn.Module):
    """1x1 adapters that map F2 to F5 onto the 64-channel fusion width."""

    def __init__(
        self, channels: tuple[int, ...] = (32, 64, 128, 256), c: int = 64
    ) -> None:
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(ch, c, 1) for ch in channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [proj(x) for proj, x in zip(self.proj, feats)]


class Neck(nn.Module):
    """Lateral adapters, two weighted BiFPN blocks and one CCFM pass."""

    def __init__(
        self,
        c: int = 64,
        blocks: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.lateral = LateralAdapters(channels, c)
        self.bifpn = nn.ModuleList([BiFPNBlock(c) for _ in range(blocks)])
        self.ccfm = CCFM(c)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        x = self.lateral(feats)
        for block in self.bifpn:
            x = block(x)
        return self.ccfm(x)
