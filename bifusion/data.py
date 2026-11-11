"""DUT Anti-UAV dataset, letterboxing and training-time augmentation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(
    im: np.ndarray, new_shape: int | tuple[int, int] = 640, color: int = 114
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio and pad to a square."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (w, h) != (nw, nh):
        im = np.asarray(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    left, top = dw // 2, dh // 2
    im = np.pad(
        im,
        ((top, dh - top), (left, dw - left), (0, 0)),
        mode="constant",
        constant_values=color,
    )
    return im, r, (left, top)


def xywhn2xyxy(
    x: np.ndarray, w: float, h: float, padw: float = 0, padh: float = 0
) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y


def xyxy2xywhn(x: np.ndarray, w: float, h: float) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-6), 0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-6), 0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-6), 0)
    hue = np.where(r == maxc, bc - gc, hue)
    hue = np.where(g == maxc, 2.0 + rc - bc, hue)
    hue = np.where(b == maxc, 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0)
    return np.stack((hue, sat, maxc), axis=-1)


def hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    h, s, v = arr[..., 0] * 6.0, arr[..., 1], arr[..., 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack((r, g, b), axis=-1)


def augment_hsv(im: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    if hgain == 0 and sgain == 0 and vgain == 0:
        return im
    hsv = rgb_to_hsv(im.astype(np.float32) / 255.0)
    gains = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
    hsv[..., 0] = (hsv[..., 0] + (gains[0] - 1)) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 1)
    return np.clip(hsv_to_rgb(hsv) * 255.0, 0, 255).astype(np.uint8)


def random_affine(
    im: np.ndarray, boxes: np.ndarray, scale: float, translate: float
) -> tuple[np.ndarray, np.ndarray]:
    if scale <= 0 and translate <= 0:
        return im, boxes
    h, w = im.shape[:2]
    s = random.uniform(1 - scale, 1 + scale)
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h
    a, c, f = 1.0 / s, -tx / s, -ty / s
    pil = Image.fromarray(im).transform(
        (w, h),
        Image.AFFINE,
        (a, 0, c, 0, a, f),
        resample=Image.BILINEAR,
        fillcolor=(114, 114, 114),
    )
    if boxes.size:
        boxes = boxes * s
        boxes[:, [0, 2]] += tx
        boxes[:, [1, 3]] += ty
    return np.asarray(pil), boxes


def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes


def load_image(path: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    im = np.asarray(Image.open(path).convert("RGB"))
    h0, w0 = im.shape[:2]
    text = str(path)
    cut = text.rfind("images")
    label_path = Path(text[:cut] + "labels" + text[cut + 6 :]).with_suffix(".txt")
    if label_path.exists():
        raw = np.loadtxt(label_path, ndmin=2)
        labels = raw[:, 0].astype(np.float32)
        boxes = xywhn2xyxy(raw[:, 1:5], w0, h0)
    else:
        labels = np.zeros((0,), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
    im, ratio, (left, top) = letterbox(im, imgsz)
    boxes = boxes * ratio + np.array([left, top, left, top], dtype=np.float32)
    return im, boxes, labels


class YoloDataset(Dataset):
    """YOLO-format detection dataset with the augmentation recipe of Table 5."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 640,
        augment: bool = False,
        mosaic: float = 0.0,
        hsv: tuple[float, float, float] = (0.0, 0.0, 0.0),
        flip: float = 0.0,
        scale: float = 0.0,
        translate: float = 0.0,
        copy_paste: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic_p = mosaic
        self.hsv = hsv
        self.flip_p = flip
        self.scale = scale
        self.translate = translate
        self.copy_paste_p = copy_paste
        self.files = sorted(
            p
            for p in (self.root / "images" / split).rglob("*")
            if p.suffix.lower() in IMG_EXTS
        )
        if not self.files:
            raise FileNotFoundError(f"no images under {self.root / 'images' / split}")
        self.labels = [self._label_path(p) for p in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def _label_path(self, image_path: Path) -> Path:
        parts = list(image_path.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    def _raw(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        im = np.asarray(Image.open(self.files[index]).convert("RGB"))
        h0, w0 = im.shape[:2]
        path = self.labels[index]
        if path.exists():
            raw = np.loadtxt(path, ndmin=2)
            labels = raw[:, 0].astype(np.float32)
            boxes = xywhn2xyxy(raw[:, 1:5], w0, h0)
        else:
            labels = np.zeros((0,), dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)
        return im, boxes, labels

    def _mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.imgsz
        half = s // 2
        indices = [index] + [random.randrange(len(self)) for _ in range(3)]
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        boxes_all, labels_all = [], []
        for tile, idx in enumerate(indices):
            im, boxes, labels = self._raw(idx)
            im, ratio, (left, top) = letterbox(im, half)
            boxes = boxes * ratio + np.array([left, top, left, top], dtype=np.float32)
            y0 = 0 if tile < 2 else half
            x0 = 0 if tile % 2 == 0 else half
            canvas[y0 : y0 + half, x0 : x0 + half] = im
            if boxes.size:
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                boxes_all.append(boxes)
                labels_all.append(labels)
        if not boxes_all:
            return canvas, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return canvas, np.concatenate(boxes_all), np.concatenate(labels_all)

    def _copy_paste(
        self, im: np.ndarray, boxes: np.ndarray, labels: np.ndarray, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() > self.copy_paste_p:
            return im, boxes, labels
        src_im, src_boxes, src_labels = self._raw(random.randrange(len(self)))
        small = np.where((src_boxes[:, 2] - src_boxes[:, 0]) < 32)[0]
        if small.size == 0:
            return im, boxes, labels
        for j in small[:2]:
            x1, y1, x2, y2 = src_boxes[j].astype(int)
            crop = src_im[max(y1, 0) : max(y2, 1), max(x1, 0) : max(x2, 1)]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if ch >= im.shape[0] or cw >= im.shape[1]:
                continue
            nx = random.randint(0, im.shape[1] - cw)
            ny = random.randint(0, im.shape[0] - ch)
            im[ny : ny + ch, nx : nx + cw] = crop
            boxes = np.concatenate(
                (boxes, np.array([[nx, ny, nx + cw, ny + ch]], np.float32))
            )
            labels = np.concatenate((labels, src_labels[j : j + 1]))
        return im, boxes, labels

    def __getitem__(self, index: int) -> dict:
        if self.augment and self.mosaic_p > 0 and random.random() < self.mosaic_p:
            im, boxes, labels = self._mosaic(index)
        else:
            im, boxes, labels = load_image(self.files[index], self.imgsz)
        if self.augment:
            if self.copy_paste_p > 0:
                im, boxes, labels = self._copy_paste(im, boxes, labels, index)
            im, boxes = random_affine(im, boxes, self.scale, self.translate)
            if random.random() < self.flip_p:
                im = np.ascontiguousarray(im[:, ::-1])
                if boxes.size:
                    x1 = boxes[:, 0].copy()
                    boxes[:, 0] = im.shape[1] - boxes[:, 2]
                    boxes[:, 2] = im.shape[1] - x1
            im = augment_hsv(im, *self.hsv)
        if boxes.size:
            boxes = clip_boxes(boxes, im.shape[0], im.shape[1])
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes, labels = boxes[keep], labels[keep]
        target = (
            torch.from_numpy(np.concatenate((boxes, labels[:, None]), axis=1))
            if boxes.size
            else torch.zeros((0, 5))
        )
        image = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        return {"image": image, "target": target, "path": str(self.files[index])}


def detection_collate(batch: list[dict]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    return images, targets
