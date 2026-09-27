"""Generate a grid of new handwritten digits from a trained checkpoint."""

import argparse
from pathlib import Path

import torch

from ddpm.diffusion import GaussianDiffusion
from ddpm.model import UNet
from ddpm.utils import save_grid, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="用训练好的 DDPM 生成手写数字")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mnist-ddpm/last.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-ddpm/samples.png"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count 须为正整数")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(checkpoint["steps"]).to(device)
    save_grid(diffusion.sample(model, args.count, device), args.out)
    print(f"已生成 {args.count} 张图片：{args.out}")


if __name__ == "__main__":
    main()
