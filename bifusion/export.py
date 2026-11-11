"""Export a trained checkpoint to ONNX, and print the TensorRT build command."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import BiFusionYOLO
from .utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BiFusion-YOLO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="bifusion_yolo.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config") or load_yaml("configs/bifusion_yolo.yaml")
    model = BiFusionYOLO(
        nc=cfg["model"].get("nc", 1),
        width=cfg["model"]["width"],
        blocks=cfg["model"]["bifpn_blocks"],
        bins=cfg["model"]["bins"],
    )
    model.load_state_dict(state["model"])
    model = model.deploy().eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    dynamic_axes = {
        "images": {0: "batch"},
        "p2": {0: "batch"},
        "p3": {0: "batch"},
        "p4": {0: "batch"},
    }
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["p2", "p3", "p4"],
        dynamic_axes=dynamic_axes if args.dynamic else None,
        do_constant_folding=True,
    )
    print(f"wrote {output}")

    if args.format == "engine":
        precision = "--fp16" if args.fp16 else ""
        print(
            "build the TensorRT engine with:\n"
            f"  trtexec --onnx={output} --saveEngine={output.with_suffix('.engine')} "
            f"--shapes=images:1x3x{args.imgsz}x{args.imgsz} {precision}".rstrip()
        )


if __name__ == "__main__":
    main()
