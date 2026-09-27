"""Forward diffusion, epsilon prediction loss, and ancestral DDPM sampling."""

import math

import torch
from torch.nn import functional as F


class GaussianDiffusion:
    def __init__(self, steps: int = 200):
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.steps = steps
        # Cosine schedule leaves the final image close to pure Gaussian noise,
        # including when steps is small (e.g. a 20-step smoke run).
        times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        alpha_bar_curve = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        alpha_bar_curve /= alpha_bar_curve[0]
        betas = (1 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]).clamp(1e-8, 0.999).float()
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = torch.cat((torch.ones(1), alpha_bars[:-1]))

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for name in ("betas", "alphas", "alpha_bars", "alpha_bars_prev"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values[t].view(-1, *([1] * (x.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, t, x0)
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def loss(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(self.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        prediction = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(prediction, noise)

    @torch.inference_mode()
    def sample(self, model: torch.nn.Module, count: int, device: torch.device) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        x = torch.randn(count, 1, 28, 28, device=device)
        for step in reversed(range(self.steps)):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (x - beta / (1 - alpha_bar).sqrt() * model(x, t)) / alpha.sqrt()
            if step:
                # Posterior variance β̃_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t).
                variance = beta * (1 - self.alpha_bars_prev[step]) / (1 - alpha_bar)
                x = mean + variance.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
