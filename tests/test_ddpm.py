"""Focused gradient, schedule, checkpoint, and sampling tests; no dataset download."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from ddpm.diffusion import GaussianDiffusion
from ddpm.ema import EMA
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

    def test_ema_updates_and_freezes(self):
        model = UNet(base_channels=8)
        ema = EMA(model, decay=0.5)
        before = next(ema.model.parameters()).clone()
        with torch.no_grad():
            next(model.parameters()).add_(2)
        ema.update(model)
        self.assertTrue(torch.allclose(next(ema.model.parameters()), before + 1))
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.model.parameters()))

    def test_training_resume_and_cli_sampling_without_network(self):
        from sample import main as sample_main
        from train import main as train_main

        torch.set_num_threads(1)
        dataset = TensorDataset(torch.rand(10, 1, 28, 28) * 2 - 1, torch.zeros(10, dtype=torch.long))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            arguments = ["train.py", "--epochs", "1", "--steps", "4", "--base-channels", "8",
                         "--batch-size", "2", "--val-size", "2", "--max-batches", "2",
                         "--sample-every", "0", "--device", "cpu", "--out", str(directory)]
            with patch("train.datasets.MNIST", return_value=dataset), patch.object(sys, "argv", arguments):
                train_main()
            self.assertTrue((directory / "last.pt").exists())
            self.assertTrue((directory / "best.pt").exists())
            checkpoint = torch.load(directory / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("ema", checkpoint)

            resumed = arguments.copy()
            resumed[resumed.index("--epochs") + 1] = "2"
            resumed.extend(("--resume", str(directory / "last.pt")))
            with patch("train.datasets.MNIST", return_value=dataset), patch.object(sys, "argv", resumed):
                train_main()
            rows = [json.loads(line) for line in (directory / "loss.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2])

            sample_args = ["sample.py", "--checkpoint", str(directory / "last.pt"), "--out",
                           str(directory / "samples.png"), "--count", "3", "--batch-size", "2",
                           "--device", "cpu"]
            with patch.object(sys, "argv", sample_args):
                sample_main()
            self.assertTrue((directory / "samples.png").exists())


if __name__ == "__main__":
    unittest.main()
