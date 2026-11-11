"""Detection metrics: fixed operating point, AP and width-stratified AP."""

from __future__ import annotations

import numpy as np
import torch

from .ops import box_iou

WIDTH_BINS = [(0, 10), (10, 20), (20, 32), (32, 64), (64, 96), (96, 10**6)]
WIDTH_LABELS = ["<10", "10-20", "20-32", "32-64", "64-96", ">96"]


def match_image(
    pred_boxes: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching. Returns (tp flags, matched gt index)."""
    n = pred_boxes.shape[0]
    tp = np.zeros(n, dtype=bool)
    matched_gt = np.full(n, -1, dtype=int)
    if n == 0 or gt_boxes.numel() == 0:
        return tp, matched_gt
    ious = box_iou(gt_boxes, pred_boxes).cpu().numpy()
    used = np.zeros(gt_boxes.shape[0], dtype=bool)
    order = torch.argsort(scores, descending=True).cpu().numpy()
    for j in order:
        i = int(np.argmax(ious[:, j]))
        if ious[i, j] >= iou_threshold and not used[i]:
            used[i] = True
            tp[j] = True
            matched_gt[j] = i
    return tp, matched_gt


def ap_101(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or tp.size == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    return float(np.interp(grid, recall, precision).mean())


class DetectionMetrics:
    """Accumulates per-image predictions and computes the val protocol."""

    def __init__(self) -> None:
        self.preds: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.gts: list[torch.Tensor] = []

    def update(
        self, pred_boxes: torch.Tensor, scores: torch.Tensor, gt_boxes: torch.Tensor
    ) -> None:
        self.preds.append((pred_boxes.detach().cpu(), scores.detach().cpu()))
        self.gts.append(gt_boxes.detach().cpu())

    def _ap_at(self, threshold: float, keep, n_gt: int) -> float:
        tps, scores = [], []
        for (boxes, sc), gt in zip(self.preds, self.gts):
            if keep is not None:
                gt = gt[keep(gt)]
            tp, _ = match_image(boxes, sc, gt, threshold)
            tps.append(tp)
            scores.append(sc.numpy())
        if not tps:
            return float("nan")
        return ap_101(np.concatenate(tps), np.concatenate(scores), n_gt)

    def compute(self) -> dict:
        thresholds = np.arange(0.5, 0.96, 0.05)
        aps = [
            self._ap_at(float(t), None, sum(len(g) for g in self.gts))
            for t in thresholds
        ]
        ap50 = aps[0]
        ap = float(np.nanmean(aps))

        tps, scores = [], []
        for (boxes, sc), gt in zip(self.preds, self.gts):
            tp, _ = match_image(boxes, sc, gt, 0.5)
            tps.append(tp)
            scores.append(sc.numpy())
        tp_all = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
        sc_all = np.concatenate(scores) if scores else np.zeros(0)
        n_gt = sum(len(g) for g in self.gts)
        order = np.argsort(-sc_all)
        tp_sorted = tp_all[order].astype(np.float64)
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(1 - tp_sorted)
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best = int(np.argmax(f1)) if f1.size else 0
        p = float(precision[best]) if f1.size else 0.0
        r = float(recall[best]) if f1.size else 0.0
        f1_best = float(f1[best]) if f1.size else 0.0
        conf = float(sc_all[order][best]) if f1.size else 0.0

        bins = {}
        for (lo, hi), label in zip(WIDTH_BINS, WIDTH_LABELS):

            def keep(gt, lo=lo, hi=hi):
                width = gt[:, 2] - gt[:, 0]
                return (width >= lo) & (width < hi)

            n_bin = sum(int(keep(g).sum()) for g in self.gts)
            if n_bin == 0:
                bins[label] = {"instances": 0, "ap50": 0.0, "ap": 0.0}
                continue
            bins[label] = {
                "instances": n_bin,
                "ap50": self._ap_at(0.5, keep, n_bin),
                "ap": float(
                    np.nanmean([self._ap_at(float(t), keep, n_bin) for t in thresholds])
                ),
            }

        return {
            "precision": 100 * p,
            "recall": 100 * r,
            "f1": 100 * f1_best,
            "confidence": conf,
            "map50": 100 * ap50,
            "map": 100 * ap,
            "instances": n_gt,
            "width_bins": bins,
        }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"P {metrics['precision']:.1f}  R {metrics['recall']:.1f}  "
        f"F1 {metrics['f1']:.1f}  mAP@0.5 {metrics['map50']:.1f}  mAP@0.5:0.95 {metrics['map']:.1f}"
    ]
    for label, row in metrics["width_bins"].items():
        lines.append(
            f"  width {label:>6} px  n={row['instances']:5d}  mAP {row['ap']:.1f}"
        )
    return "\n".join(lines)
