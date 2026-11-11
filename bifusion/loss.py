"""Task-aligned assignment and the composite detection loss.

The objective follows Table 5 and Eq. (35) of the manuscript: binary
cross-entropy for classification, CIoU for the box, and distribution focal loss
for the four discretized sides. There is no objectness term.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import bbox2dist, box_iou, dist2bbox, make_anchors


class TaskAlignedAssigner(nn.Module):
    """Top-k assignment by t = score^alpha * IoU^beta (Eq. 34)."""

    def __init__(self, alpha: float = 1.0, beta: float = 6.0, topk: int = 10) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = topk

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = pred_boxes.shape[0]
        target_scores = torch.zeros(n, device=pred_boxes.device)
        target_boxes = torch.zeros(n, 4, device=pred_boxes.device)
        target_labels = torch.zeros(n, dtype=torch.long, device=pred_boxes.device)
        if gt_boxes.numel() == 0:
            return target_labels, target_boxes, target_scores, target_scores > 0

        ious = box_iou(gt_boxes, pred_boxes)
        align = pred_scores.sigmoid()[None, :].pow(self.alpha) * ious.clamp(min=0).pow(
            self.beta
        )
        k = min(self.topk, n)
        values, indices = align.topk(k, dim=1)
        for gi in range(gt_boxes.shape[0]):
            for j in range(k):
                idx = int(indices[gi, j])
                value = float(values[gi, j])
                if value <= 0:
                    continue
                if value > float(target_scores[idx]):
                    target_scores[idx] = value
                    target_boxes[idx] = gt_boxes[gi]
                    target_labels[idx] = gt_labels[gi]
        return target_labels, target_boxes, target_scores, target_scores > 0


def ciou_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """1 - CIoU, averaged over the batch of boxes."""
    iou = box_iou(pred, target).diagonal()
    px, py = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
    tx, ty = (target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2
    rho2 = (px - tx) ** 2 + (py - ty) ** 2
    cw = torch.maximum(pred[:, 2], target[:, 2]) - torch.minimum(
        pred[:, 0], target[:, 0]
    )
    ch = torch.maximum(pred[:, 3], target[:, 3]) - torch.minimum(
        pred[:, 1], target[:, 1]
    )
    c2 = cw**2 + ch**2 + eps
    pw = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    tw = (target[:, 2] - target[:, 0]).clamp(min=eps)
    th = (target[:, 3] - target[:, 1]).clamp(min=eps)
    v = (4 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    alpha = v / (1 - iou + v + eps)
    return (1 - (iou - rho2 / c2 - alpha * v)).mean()


def dfl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distribution focal loss over the two bins adjacent to the target."""
    bins = pred.shape[-1]
    target = target.clamp(0, bins - 1 - 1e-4)
    left = target.floor().long().clamp(0, bins - 1)
    right = (left + 1).clamp(max=bins - 1)
    wl = right.float() - target
    wr = target - left.float()
    logp = pred.log_softmax(dim=-1)
    loss_l = logp.gather(-1, left.unsqueeze(-1)).squeeze(-1) * wl
    loss_r = logp.gather(-1, right.unsqueeze(-1)).squeeze(-1) * wr
    return -(loss_l + loss_r).mean()


class DetectionLoss(nn.Module):
    """Composite loss normalized by the number of assigned positives."""

    def __init__(
        self,
        bins: int = 16,
        nc: int = 1,
        strides: tuple[int, ...] = (4, 8, 16),
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        alpha: float = 1.0,
        beta: float = 6.0,
        topk: int = 10,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.nc = nc
        self.strides = strides
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner = TaskAlignedAssigner(alpha, beta, topk)

    def forward(
        self, outputs: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = outputs[0].device
        batch = outputs[0].shape[0]
        anchors, stride_t = make_anchors(outputs, self.strides)
        cls = torch.cat(
            [o[:, : self.nc].reshape(batch, self.nc, -1) for o in outputs], dim=2
        )
        box = torch.cat(
            [o[:, self.nc :].reshape(batch, 4 * self.bins, -1) for o in outputs], dim=2
        )

        probs = box.view(batch, 4, self.bins, -1).softmax(dim=2)
        offset = torch.arange(self.bins, device=device, dtype=box.dtype).view(
            1, 1, self.bins, 1
        )
        dist = (probs * offset).sum(dim=2).permute(0, 2, 1)

        loss_cls = torch.zeros((), device=device)
        loss_box = torch.zeros((), device=device)
        loss_dfl = torch.zeros((), device=device)
        num_pos = 0.0
        score_sum = 0.0

        for b in range(batch):
            scores_b = cls[b, 0] if self.nc == 1 else cls[b].max(dim=0).values
            boxes_b = dist2bbox(dist[b], anchors, stride_t)
            gt = targets[b]
            gt_boxes = gt[:, :4] if gt.numel() else gt.reshape(0, 4)
            gt_labels = gt[:, 4].long() if gt.numel() else gt.reshape(0).long()
            _, target_boxes, target_scores, positive = self.assigner(
                boxes_b.detach(), scores_b.detach(), gt_boxes, gt_labels
            )
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                scores_b, target_scores, reduction="sum"
            )
            score_sum += float(target_scores.sum())
            if positive.any():
                pb = boxes_b[positive]
                gb = target_boxes[positive]
                loss_box = loss_box + ciou_loss(pb, gb)
                d = bbox2dist(gb, anchors[positive], stride_t[positive])
                dfl_pred = box[b][:, positive].view(4, self.bins, -1).permute(2, 0, 1)
                loss_dfl = loss_dfl + dfl_loss(dfl_pred, d)
                num_pos += int(positive.sum())

        norm = max(num_pos, 1.0)
        total = (
            self.box_weight * loss_box / norm
            + self.cls_weight * loss_cls / max(score_sum, 1.0)
            + self.dfl_weight * loss_dfl / norm
        )
        parts = {
            "box": float((loss_box / norm).detach()),
            "cls": float((loss_cls / max(score_sum, 1.0)).detach()),
            "dfl": float((loss_dfl / norm).detach()),
            "positives": num_pos,
        }
        return total, parts
