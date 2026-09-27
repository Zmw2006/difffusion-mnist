"""Focused shape, gradient, schedule and sampling tests; no dataset download."""

import unittest

import torch

from ddpm.diffusion import GaussianDiffusion
from ddpm.model import UNet


class TestDDPM(unittest.TestCase):
    def test_forward_and_backward(self):
        torch.set_num_threads(1)
        model = UNet(base_channels=8)
        diffusion = GaussianDiffusion(steps=4)
        images = torch.rand(2, 1, 28, 28) * 2 - 1
        loss = diffusion.loss(model, images)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_schedule_and_sample(self):
        torch.set_num_threads(1)
        model = UNet(base_channels=8).eval()
        diffusion = GaussianDiffusion(steps=4)
        self.assertTrue(torch.all(diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1]))
        self.assertLess(diffusion.alpha_bars[-1].item(), 0.01)
        generated = diffusion.sample(model, 2, torch.device("cpu"))
        self.assertEqual(generated.shape, (2, 1, 28, 28))
        self.assertTrue(torch.isfinite(generated).all())
        self.assertGreaterEqual(generated.min().item(), -1)
        self.assertLessEqual(generated.max().item(), 1)


if __name__ == "__main__":
    unittest.main()
