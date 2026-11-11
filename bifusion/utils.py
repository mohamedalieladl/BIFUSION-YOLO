"""Small utilities: configuration, seeding, EMA and checkpoint handling."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def increment_path(path: str | Path, exist_ok: bool = False) -> Path:
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot find a free name for {path}")


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.9999, updates: int = 0
    ) -> None:
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.updates = updates

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - np.exp(-self.updates / 2000.0))
        state = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(state[key].detach(), alpha=1 - decay)
            else:
                value.copy_(state[key])


def save_checkpoint(path: str | Path, model: torch.nn.Module, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
