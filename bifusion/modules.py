"""Building blocks used by BiFusion-YOLO.

Every block is written so that, after :meth:`BiFusionYOLO.deploy` folds batch
normalization and RepVGG branches, the number of parameters equals the row of
Table 3 in the manuscript. The counts in that table are deployed counts: a
convolution that is followed by batch normalization contributes its weights and
one folded bias, not a separate scale and shift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Keep the spatial size for odd kernels."""
    if p is not None:
        return p
    return d * (k - 1) // 2


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a bias-carrying convolution equivalent to conv followed by bn."""
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    ).to(conv.weight.device, conv.weight.dtype)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data.copy_(conv.weight.data * scale.reshape(-1, 1, 1, 1))
    if conv.bias is None:
        fused.bias.data.copy_(bn.bias.data - bn.running_mean.data * scale)
    else:
        fused.bias.data.copy_(
            (conv.bias.data - bn.running_mean.data) * scale + bn.bias.data
        )
    return fused


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, activation. Folds to one convolution."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fused(self) -> nn.Sequential:
        return nn.Sequential(fuse_conv_bn(self.conv, self.bn), self.act)


class GhostConv(nn.Module):
    """Ghost convolution with s = 2 cheap transforms.

    The primary convolution produces half of the output channels; a depthwise
    operation manufactures the other half, and the two are concatenated.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 2, d: int = 3) -> None:
        super().__init__()
        if cout % 2:
            raise ValueError("GhostConv expects an even output width")
        mid = cout // 2
        self.primary = ConvBNAct(cin, mid, k, s)
        self.cheap = ConvBNAct(mid, mid, d, 1, g=mid)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        return self.act(torch.cat((y, self.cheap(y)), dim=1))


class RepVGGBlock(nn.Module):
    """Three-branch training block with an equivalent single-convolution deploy form."""

    def __init__(self, c: int, s: int = 1) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(c, c, 3, s, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 1, s, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn_id = nn.BatchNorm2d(c) if s == 1 else None
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    def fused(self) -> nn.Sequential:
        k3 = fuse_conv_bn(self.conv3, self.bn3)
        k1 = fuse_conv_bn(self.conv1, self.bn1)
        kernel = k3.weight.data.clone()
        kernel += F.pad(k1.weight.data, (1, 1, 1, 1))
        bias = k3.bias.data.clone() + k1.bias.data
        if self.bn_id is not None:
            scale = self.bn_id.weight / torch.sqrt(
                self.bn_id.running_var + self.bn_id.eps
            )
            identity = torch.zeros_like(kernel)
            for i in range(kernel.shape[0]):
                identity[i, i, 1, 1] = scale[i]
            kernel += identity
            bias += self.bn_id.bias.data - self.bn_id.running_mean.data * scale
        conv = nn.Conv2d(
            kernel.shape[1], kernel.shape[0], 3, self.conv3.stride, 1, bias=True
        ).to(kernel.device, kernel.dtype)
        conv.weight.data.copy_(kernel)
        conv.bias.data.copy_(bias)
        return nn.Sequential(conv, self.act)


class CoordAttention(nn.Module):
    """Coordinate Attention with a max(C/32, 8) bottleneck."""

    def __init__(self, c: int, reduction: int = 32, min_channels: int = 8) -> None:
        super().__init__()
        mid = max(c // reduction, min_channels)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = ConvBNAct(2 * c, mid, 1)
        self.conv_h = nn.Conv2d(mid, c, 1, bias=True)
        self.conv_w = nn.Conv2d(mid, c, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        xh = self.pool_h(x)  # (B, C, H, 1)
        xw = self.pool_w(x)  # (B, C, 1, W)
        xw = F.interpolate(xw, size=(h, 1), mode="bilinear", align_corners=False)
        y = self.conv1(torch.cat((xh, xw), dim=1))  # (B, mid, H, 1)
        gh = torch.sigmoid(self.conv_h(y))  # (B, C, H, 1)
        gw = torch.sigmoid(self.conv_w(y))  # (B, C, H, 1), width descriptor
        gw = F.interpolate(gw, size=(1, w), mode="bilinear", align_corners=False)
        return x * gh * gw


class HWD(nn.Module):
    """Haar wavelet downsampling: four sub-bands fused by a 1x1 convolution.

    The analysis uses the orthonormal Haar pair scaled by 1/2, as in the
    manuscript, and the learned 1x1 fusion restores whatever scale the network
    needs.
    """

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = ConvBNAct(4 * cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 + x01 - x10 - x11)
        hl = 0.5 * (x00 - x01 + x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return self.conv(torch.cat((ll, lh, hl, hh), dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling fast with a 128-channel intermediate width."""

    def __init__(self, c: int, hidden: int = 128, k: int = 5) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(c, hidden, 1)
        self.cv2 = ConvBNAct(4 * hidden, c, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((y, y1, y2, y3), dim=1))


class DWConvBlock(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1, used by the neck."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(c, c, 3, 1, g=c)
        self.pw = ConvBNAct(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))
