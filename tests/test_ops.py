import numpy as np
import torch

from bifusion.data import letterbox, xywhn2xyxy, xyxy2xywhn
from bifusion.metrics import DetectionMetrics, ap_101
from bifusion.ops import nms


def test_letterbox_square():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(image, 640)
    assert out.shape == (640, 640, 3)
    assert ratio == 6.4
    assert pad == (0, 160)


def test_box_conversion_roundtrip():
    xywhn = np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32)
    boxes = xywhn2xyxy(xywhn, 100, 50)
    assert np.allclose(boxes, [[40.0, 15.0, 60.0, 35.0]])
    assert np.allclose(xyxy2xywhn(boxes, 100, 50), xywhn, atol=1e-6)


def test_nms_suppresses_overlap():
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [50.0, 50.0, 60.0, 60.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_ap_perfect():
    tp = np.array([True, True, True])
    scores = np.array([0.9, 0.8, 0.7])
    assert ap_101(tp, scores, 3) == 1.0


def test_metrics_perfect_detection():
    metrics = DetectionMetrics()
    box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    metrics.update(box, torch.tensor([0.9]), box)
    result = metrics.compute()
    assert result["precision"] == 100.0
    assert result["recall"] == 100.0
    assert result["map50"] == 100.0
