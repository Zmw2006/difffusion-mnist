"""Generate new digit grids from stored raw or EMA model weights."""

import argparse
from pathlib import Path

import torch

from ddpm.diffusion import GaussianDiffusion
from ddpm.model import UNet
from ddpm.utils import save_grid, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="用训练好的 DDPM 生成手写数字")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mnist-ddpm/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-ddpm/samples.png"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32, help="每批采样张数，避免显存不足")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--raw", action="store_true", help="用原模型权重，默认使用 EMA 权重")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error("count 和 batch-size 须为正整数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("未检测到 CUDA；可使用 --device cpu 或 --device auto")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"] if args.raw else checkpoint.get("ema", checkpoint["model"]))
    model.eval()
    diffusion = GaussianDiffusion(checkpoint["steps"]).to(device)
    batches = []
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        batches.append(diffusion.sample(model, count, device).cpu())
        print(f"已生成 {start + count}/{args.count}", flush=True)
    save_grid(torch.cat(batches), args.out)
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
