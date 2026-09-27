"""Exponential moving average of model parameters for smoother sampling."""

from copy import deepcopy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(current.detach(), 1 - self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)
