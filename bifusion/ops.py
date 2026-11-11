"""Geometry and decoding helpers shared by the loss, validation and export paths."""

from __future__ import annotations

import torch


def make_anchors(
    feats: list[torch.Tensor], strides: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return anchor centres and their strides for a list of feature maps."""
    points = []
    stride_t = []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sy = (torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5) * stride
        sx = (torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5) * stride
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((gx, gy), dim=-1).reshape(-1, 2))
        stride_t.append(
            torch.full((h * w,), float(stride), device=feat.device, dtype=feat.dtype)
        )
    return torch.cat(points, dim=0), torch.cat(stride_t, dim=0)


def dist2bbox(
    dist: torch.Tensor, anchors: torch.Tensor, stride: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert (left, top, right, bottom) distances to xyxy boxes."""
    if stride is not None:
        dist = dist * stride.reshape(-1, 1)
    x1y1 = anchors - dist[:, 0:2]
    x2y2 = anchors + dist[:, 2:4]
    return torch.cat((x1y1, x2y2), dim=1)


def bbox2dist(
    boxes: torch.Tensor, anchors: torch.Tensor, stride: torch.Tensor
) -> torch.Tensor:
    """Convert xyxy boxes to distances in stride units."""
    dist = torch.cat((anchors - boxes[:, 0:2], boxes[:, 2:4] - anchors), dim=1)
    return (dist / stride.reshape(-1, 1)).clamp(min=0.0)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU, returns shape (len(box1), len(box2))."""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(
        min=0
    )
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(
        min=0
    )
    lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Greedy non-maximum suppression, returns kept indices."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i : i + 1], boxes[order[1:]]).reshape(-1)
        order = order[1:][iou <= iou_threshold]
    return torch.stack(keep)


def decode_predictions(
    outputs: list[torch.Tensor],
    strides: tuple[int, ...],
    bins: int,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.70,
    max_det: int = 300,
    nc: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Decode raw head outputs into per-image (boxes, scores)."""
    anchors, stride_t = make_anchors(outputs, strides)
    cls = torch.cat([o[:, :nc].reshape(o.shape[0], nc, -1) for o in outputs], dim=2)
    box = torch.cat(
        [o[:, nc:].reshape(o.shape[0], 4 * bins, -1) for o in outputs], dim=2
    )
    scores_all = cls.sigmoid()
    results = []
    for i in range(outputs[0].shape[0]):
        cls_i = scores_all[i].reshape(nc, -1)
        box_i = box[i].reshape(4, bins, -1)
        dist_i = (
            box_i.softmax(dim=1)
            * torch.arange(bins, device=box.device, dtype=box.dtype).view(1, bins, 1)
        ).sum(dim=1)
        boxes_i = dist2bbox(dist_i.permute(1, 0), anchors, stride_t)
        scores_i = cls_i.max(dim=0).values if nc > 1 else cls_i[0]
        mask = scores_i > conf_threshold
        boxes_i, scores_i = boxes_i[mask], scores_i[mask]
        keep = nms(boxes_i, scores_i, iou_threshold)[:max_det]
        results.append((boxes_i[keep], scores_i[keep]))
    return results
