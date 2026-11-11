import torch

from bifusion.head import DecoupledHead, P2Head
from bifusion.model import _deploy_children, count_parameters
from bifusion.modules import CoordAttention, GhostConv, HWD, RepVGGBlock, SPPF
from bifusion.neck import CCFM, BiFPNBlock, LateralAdapters


def deployed(module):
    _deploy_children(module)
    return count_parameters(module)


def test_ghost():
    assert deployed(GhostConv(3, 16)) == 304
    assert deployed(GhostConv(16, 32)) == 2_480


def test_repvgg():
    assert deployed(RepVGGBlock(32)) == 9_248
    assert deployed(RepVGGBlock(256)) == 590_080


def test_coordinate_attention():
    assert deployed(CoordAttention(32)) == 1_096
    assert deployed(CoordAttention(64)) == 2_184
    assert deployed(CoordAttention(128)) == 4_360
    assert deployed(CoordAttention(256)) == 8_712


def test_hwd():
    assert deployed(HWD(32, 64)) == 8_256
    assert deployed(HWD(64, 128)) == 32_896
    assert deployed(HWD(128, 256)) == 131_328


def test_sppf():
    assert deployed(SPPF(256)) == 164_224


def test_neck():
    assert deployed(LateralAdapters()) == 30_976
    assert deployed(BiFPNBlock(64)) == 33_600
    assert deployed(CCFM(64)) == 52_480


def test_heads():
    assert deployed(P2Head(32)) == 115_172
    assert deployed(DecoupledHead(64)) == 151_937
    assert deployed(DecoupledHead(128)) == 225_665


def test_shapes():
    neck = CCFM(64)
    feats = [
        torch.zeros(1, 64, 160, 160),
        torch.zeros(1, 64, 80, 80),
        torch.zeros(1, 64, 40, 40),
        torch.zeros(1, 64, 20, 20),
    ]
    out = neck(feats)
    assert [o.shape[1] for o in out] == [32, 64, 128, 256]
