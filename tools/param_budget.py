"""Pure-Python recomputation of the layer budget in Table 3.

This file needs no PyTorch. It exists so the arithmetic in the paper can be
checked on any machine, and so the origin of 1,783,110 is visible in one place.
"""

from __future__ import annotations


def conv(cin: int, cout: int, k: int = 3, bias: bool = True) -> int:
    return k * k * cin * cout + (cout if bias else 0)


def ghost(cin: int, cout: int, k: int = 3, s: int = 2, d: int = 3) -> int:
    mid = cout // s
    return conv(cin, mid, k) + d * d * mid + mid


def ca(c: int, divisor: int = 32, minimum: int = 8) -> int:
    mid = max(c // divisor, minimum)
    return (2 * c) * mid + mid + mid * c + c + mid * c + c


def dwsep(c: int = 64) -> int:
    return (9 * c + c) + (c * c + c)


def head(cin: int, hid: int = 64, bins: int = 16, nc: int = 1) -> int:
    cls = conv(cin, hid) + conv(hid, hid) + conv(hid, nc, 1)
    box = conv(cin, hid) + conv(hid, hid) + conv(hid, 4 * bins, 1)
    return cls + box


BACKBONE = sum(
    [
        ghost(3, 16),
        ghost(16, 32),
        conv(32, 32),
        ca(32),
        4 * 32 * 64 + 64,
        conv(64, 64),
        ca(64),
        4 * 64 * 128 + 128,
        conv(128, 128),
        ca(128),
        4 * 128 * 256 + 256,
        conv(256, 256),
        ca(256),
        conv(256, 128, 1) + conv(512, 256, 1),
    ]
)

LATERAL = conv(32, 64, 1) + conv(64, 64, 1) + conv(128, 64, 1) + conv(256, 64, 1)
BIFPN = 2 * 7 * dwsep()
CCFM = 4 * (conv(64, 64, 1) + dwsep()) + (64 * 256 + 256)
NECK = LATERAL + BIFPN + CCFM
HEADS = head(32) + 99 + head(64) + head(128)
TOTAL = BACKBONE + NECK + HEADS


if __name__ == "__main__":
    print(f"backbone {BACKBONE:,}")
    print(f"neck     {NECK:,} (lateral {LATERAL:,}, BiFPN {BIFPN:,}, CCFM {CCFM:,})")
    print(f"heads    {HEADS:,}")
    print(f"total    {TOTAL:,}")
    assert BACKBONE == 1_139_680
    assert NECK == 150_656
    assert HEADS == 492_774
    assert TOTAL == 1_783_110
    print("Table 3 arithmetic verified")
