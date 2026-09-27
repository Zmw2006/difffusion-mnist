"""Show one real digit becoming noisy across the forward diffusion process."""

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化正向加噪过程")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--index", type=int, default=0, help="MNIST 训练集中的图像编号")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/forward-process.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps 须至少为 2")
    set_seed(args.seed)
    dataset = datasets.MNIST(args.data_root, train=True, download=True,
                             transform=transforms.Compose((transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,)))))
    if not 0 <= args.index < len(dataset):
        parser.error("--index 超出训练集范围")
    image, label = dataset[args.index]
    image = image.unsqueeze(0)
    diffusion = GaussianDiffusion(args.steps)
    noise = torch.randn_like(image)
    times = torch.linspace(0, args.steps - 1, 8).round().long()
    frames = [image]
    frames.extend(diffusion.q_sample(image, t.reshape(1), noise) for t in times)
    grid = torch.cat(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(((grid + 1) / 2).clamp(0, 1), args.out, nrow=9, padding=2)
    print(f"原始数字：{label}；从左到右：原图、t={times.tolist()}；已保存：{args.out}")


if __name__ == "__main__":
    main()
