"""Reproducibility and PNG grid export."""

import math
import random
from pathlib import Path

import torch
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_grid(samples: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = ((samples.float().cpu() + 1) / 2).clamp(0, 1)
    save_image(images, path, nrow=math.ceil(math.sqrt(len(images))), padding=2)

