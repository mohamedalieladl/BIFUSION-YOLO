"""Validation and test evaluation for BiFusion-YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import YoloDataset, detection_collate
from .metrics import DetectionMetrics, format_metrics
from .model import BiFusionYOLO
from .ops import decode_predictions
from .utils import load_yaml


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf: float = 0.001,
    iou: float = 0.70,
    max_det: int = 300,
    bins: int = 16,
    nc: int = 1,
) -> dict:
    model.eval()
    metrics = DetectionMetrics()
    for images, targets in loader:
        outputs = model(images.to(device, non_blocking=True))
        results = decode_predictions(
            outputs,
            getattr(model, "head_strides", (4, 8, 16)),
            bins,
            conf,
            iou,
            max_det,
            nc,
        )
        for (boxes, scores), target in zip(results, targets):
            metrics.update(boxes, scores, target[:, :4])
    return metrics.compute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BiFusion-YOLO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="configs/dut.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state.get("config") or load_yaml("configs/bifusion_yolo.yaml")
    data_cfg = load_yaml(args.data)
    model = BiFusionYOLO(
        nc=data_cfg.get("nc", 1),
        width=cfg["model"]["width"],
        blocks=cfg["model"]["bifpn_blocks"],
        bins=cfg["model"]["bins"],
    )
    model.load_state_dict(state["model"])
    model.to(device)
    if args.deploy:
        model = model.deploy()
    dataset = YoloDataset(data_cfg["path"], args.split, args.imgsz, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate,
    )
    metrics = evaluate(
        model,
        loader,
        device,
        args.conf,
        args.iou,
        args.max_det,
        cfg["model"]["bins"],
        data_cfg.get("nc", 1),
    )
    print(format_metrics(metrics))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
