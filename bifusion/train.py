"""Training entry point for BiFusion-YOLO.

The recipe follows Table 5 of the manuscript: AdamW, cosine schedule, three
warm-up epochs, EMA, mixed precision, mosaic disabled for the last ten epochs,
and the loss weights 7.5 / 0.5 / 1.5.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps --help working without the extra

    def tqdm(iterable, **kwargs):
        return iterable


from .data import YoloDataset, detection_collate
from .loss import DetectionLoss
from .model import BiFusionYOLO
from .utils import ModelEMA, increment_path, load_yaml, save_checkpoint, seed_everything
from .val import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiFusion-YOLO")
    parser.add_argument("--config", default="configs/bifusion_yolo.yaml")
    parser.add_argument("--data", default="configs/dut.yaml")
    parser.add_argument("--weights", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="bifusion_yolo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, lr: float, momentum: float, decay: float):
    grouped = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            grouped[2].append(parameter)
        elif "weight" in name:
            grouped[1].append(parameter)
        else:
            grouped[0].append(parameter)
    return torch.optim.AdamW(
        [
            {"params": grouped[0], "weight_decay": decay},
            {"params": grouped[1], "weight_decay": decay},
            {"params": grouped[2], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(momentum, 0.999),
    )


def lr_lambda(epoch: int, epochs: int, warmup: int, lrf: float) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    aug_cfg = cfg["augment"]
    val_cfg = cfg["val"]

    epochs = args.epochs or train_cfg["epochs"]
    batch = args.batch or train_cfg["batch"]
    imgsz = args.imgsz or train_cfg["imgsz"]
    workers = args.workers if args.workers is not None else train_cfg["workers"]
    seed_everything(args.seed or train_cfg["seed"], args.deterministic)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    out_dir = increment_path(Path(args.project) / args.name, exist_ok=bool(args.resume))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = YoloDataset(
        data_cfg["path"],
        "train",
        imgsz,
        augment=True,
        mosaic=aug_cfg["mosaic"],
        hsv=(aug_cfg["hsv_h"], aug_cfg["hsv_s"], aug_cfg["hsv_v"]),
        flip=aug_cfg["flip"],
        scale=aug_cfg["scale"],
        translate=aug_cfg["translate"],
        copy_paste=aug_cfg["copy_paste"],
    )
    val_set = YoloDataset(data_cfg["path"], "val", imgsz, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )

    model = BiFusionYOLO(
        nc=data_cfg.get("nc", 1),
        width=cfg["model"]["width"],
        blocks=cfg["model"]["bifpn_blocks"],
        bins=cfg["model"]["bins"],
    ).to(device)
    if args.weights:
        state = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state))
    ema = ModelEMA(model, train_cfg["ema_decay"])

    optimizer = build_optimizer(
        model, train_cfg["lr0"], train_cfg["momentum"], train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: lr_lambda(e, epochs, train_cfg["warmup_epochs"], train_cfg["lrf"]),
    )
    criterion = DetectionLoss(
        bins=cfg["model"]["bins"],
        nc=data_cfg.get("nc", 1),
        strides=model.head_strides,
        box_weight=loss_cfg["box"],
        cls_weight=loss_cfg["cls"],
        dfl_weight=loss_cfg["dfl"],
        alpha=loss_cfg["tal_alpha"],
        beta=loss_cfg["tal_beta"],
        topk=loss_cfg["topk"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_cfg["amp"]) and device.type == "cuda"
    )

    start_epoch = 0
    best = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", -1.0)

    log_path = out_dir / "results.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "lr",
                    "box",
                    "cls",
                    "dfl",
                    "positives",
                    "map50",
                    "map",
                    "precision",
                    "recall",
                    "f1",
                ]
            )

    for epoch in range(start_epoch, epochs):
        if epoch >= epochs - train_cfg["close_mosaic"]:
            train_set.mosaic_p = 0.0
        model.train()
        running = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "positives": 0.0}
        start = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = [t.to(device) for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss, parts = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            for key in running:
                running[key] += parts[key]
            progress.set_postfix(loss=f"{float(loss):.4f}", pos=int(parts["positives"]))
        scheduler.step()
        count = max(len(train_loader), 1)
        metrics = evaluate(
            ema.ema,
            val_loader,
            device,
            conf=val_cfg["conf"],
            iou=val_cfg["iou"],
            max_det=val_cfg["max_det"],
            bins=cfg["model"]["bins"],
        )
        lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    f"{lr:.6g}",
                    f"{running['box'] / count:.4f}",
                    f"{running['cls'] / count:.4f}",
                    f"{running['dfl'] / count:.4f}",
                    f"{running['positives'] / count:.1f}",
                    f"{metrics['map50']:.2f}",
                    f"{metrics['map']:.2f}",
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1']:.2f}",
                ]
            )
        improved = metrics["map"] > best
        if improved:
            best = metrics["map"]
        print(
            f"epoch {epoch + 1:3d}  box {running['box'] / count:.3f}  "
            f"cls {running['cls'] / count:.3f}  dfl {running['dfl'] / count:.3f}  "
            f"mAP50 {metrics['map50']:.1f}  mAP {metrics['map']:.1f}  "
            f"P {metrics['precision']:.1f}  R {metrics['recall']:.1f}  ({time.time() - start:.0f}s)"
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best,
            "config": cfg,
            "data": data_cfg,
        }
        save_checkpoint(
            out_dir / "last.pt",
            model,
            **{k: v for k, v in state.items() if k != "model"},
        )
        if improved:
            save_checkpoint(
                out_dir / "best.pt",
                ema.ema,
                epoch=epoch,
                best=best,
                config=cfg,
                data=data_cfg,
            )

    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
