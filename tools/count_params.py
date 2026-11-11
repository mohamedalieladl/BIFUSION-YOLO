"""Print the deployed parameter count and check it against Table 3.

Run from the repository root:

    python tools/count_params.py --check

Table 3 itemizes convolution weights and biases. The 34 scalar fusion weights of
the BiFPN are reported separately, so both conventions are visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bifusion.model import BiFusionYOLO  # noqa: E402

PAPER = {
    "backbone": 1_139_680,
    "neck_conv": 150_656,
    "heads": 492_774,
    "table3_total": 1_783_110,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    model = BiFusionYOLO(width=args.width, blocks=args.blocks)
    raw = sum(p.numel() for p in model.parameters())
    report = model.parameter_report()
    print(f"training parameters (BN and RepVGG branches included): {raw:,}")
    for key in ("backbone", "neck", "heads"):
        print(f"deployed {key:>9}: {report[key]:,}")
    print(f"deployed fusion scalars: {report['fusion_weights']:,}")
    print(f"deployed total          : {report['total']:,}")
    print(f"Table 3 convention      : {report['table3_total']:,}")
    if args.check:
        ok = (
            report["backbone"] == PAPER["backbone"]
            and report["heads"] == PAPER["heads"]
            and report["table3_total"] == PAPER["table3_total"]
            and report["neck"] - report["fusion_weights"] == PAPER["neck_conv"]
        )
        print("matches Table 3" if ok else "does not match Table 3")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
