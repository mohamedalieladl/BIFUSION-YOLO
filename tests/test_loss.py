import torch

from bifusion.loss import DetectionLoss
from bifusion.model import BiFusionYOLO
from bifusion.ops import bbox2dist, dist2bbox, make_anchors


def test_geometry_roundtrip():
    anchors = torch.tensor([[10.0, 10.0], [20.0, 30.0]])
    stride = torch.tensor([4.0, 8.0])
    boxes = torch.tensor([[6.0, 6.0, 14.0, 16.0], [12.0, 22.0, 30.0, 44.0]])
    dist = bbox2dist(boxes, anchors, stride)
    assert torch.allclose(dist2bbox(dist, anchors, stride), boxes, atol=1e-5)


def test_anchor_count():
    feats = [torch.zeros(1, 65, 20, 20), torch.zeros(1, 65, 10, 10)]
    anchors, stride = make_anchors(feats, (4, 8))
    assert anchors.shape == (500, 2)
    assert stride.shape == (500,)


def test_loss_is_finite_and_backward():
    model = BiFusionYOLO().train()
    outputs = model(torch.zeros(1, 3, 320, 320))
    targets = [torch.tensor([[100.0, 120.0, 112.0, 132.0, 0.0]])]
    loss, parts = DetectionLoss()(outputs, targets)
    assert torch.isfinite(loss)
    assert parts["positives"] > 0
    loss.backward()
