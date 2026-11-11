"""The BiFusion-YOLO detector.

The layer order follows Table 3 of the manuscript. ``deploy()`` fuses every
batch-normalization layer and the RepVGG branches, after which the parameter
count is the deployed count reported in the paper (1,783,110).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .head import DecoupledHead, P2Head
from .modules import ConvBNAct, CoordAttention, GhostConv, HWD, RepVGGBlock, SPPF
from .neck import Neck


def _deploy_children(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, ConvBNAct):
            setattr(module, name, child.fused())
        elif isinstance(child, RepVGGBlock):
            setattr(module, name, child.fused())
        else:
            _deploy_children(child)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class BiFusionYOLO(nn.Module):
    """P2-enhanced YOLO26 variant with a weighted BiFPN and CCFM neck."""

    strides: tuple[int, int, int, int] = (4, 8, 16, 32)
    head_strides: tuple[int, int, int] = (4, 8, 16)

    def __init__(
        self,
        nc: int = 1,
        width: int = 64,
        blocks: int = 2,
        bins: int = 16,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.bins = bins

        self.stem1 = GhostConv(3, 16, 3, 2)
        self.stem2 = GhostConv(16, 32, 3, 2)
        self.stage2 = nn.Sequential(RepVGGBlock(32), CoordAttention(32))
        self.down3 = HWD(32, 64)
        self.stage3 = nn.Sequential(RepVGGBlock(64), CoordAttention(64))
        self.down4 = HWD(64, 128)
        self.stage4 = nn.Sequential(RepVGGBlock(128), CoordAttention(128))
        self.down5 = HWD(128, 256)
        self.stage5 = nn.Sequential(RepVGGBlock(256), CoordAttention(256))
        self.sppf = SPPF(256)

        self.neck = Neck(width, blocks, channels)
        self.head_p2 = P2Head(channels[0], width, bins, nc)
        self.head_p3 = DecoupledHead(channels[1], width, bins, nc)
        self.head_p4 = DecoupledHead(channels[2], width, bins, nc)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem2(self.stem1(x))
        f2 = self.stage2(x)
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        f5 = self.sppf(self.stage5(self.down5(f4)))
        p2, p3, p4, _ = self.neck([f2, f3, f4, f5])
        return [self.head_p2(p2), self.head_p3(p3), self.head_p4(p4)]

    def deploy(self) -> "BiFusionYOLO":
        """Return an inference copy with BN and RepVGG branches fused."""
        model = copy.deepcopy(self).eval()
        _deploy_children(model)
        return model

    def parameter_report(self) -> dict[str, int]:
        """Deployed parameter counts for the backbone, neck and heads."""
        model = self.deploy()
        backbone = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.split(".")[0]
            in {
                "stem1",
                "stem2",
                "stage2",
                "down3",
                "stage3",
                "down4",
                "stage4",
                "down5",
                "stage5",
                "sppf",
            }
        )
        neck = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("neck.")
        )
        heads = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("head_")
        )
        fusion = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("neck.bifpn.") and name.endswith(".weights")
        )
        total = backbone + neck + heads
        return {
            "backbone": backbone,
            "neck": neck,
            "heads": heads,
            "fusion_weights": fusion,
            "total": total,
            "table3_total": total - fusion,
        }


def build_model(**kwargs) -> BiFusionYOLO:
    return BiFusionYOLO(**kwargs)
