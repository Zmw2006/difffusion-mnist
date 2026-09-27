"""Train an unconditional DDPM on MNIST; no test-set data are used."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddpm.diffusion import GaussianDiffusion
from ddpm.model import UNet
from ddpm.utils import save_grid, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="在 MNIST 上训练 DDPM")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-ddpm"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size, lr 须为正数；num-workers 须非负")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = UNet(args.base_channels).to(device)
    diffusion = GaussianDiffusion(args.steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["steps"] != args.steps or checkpoint["base_channels"] != args.base_channels:
            parser.error("续训时 --steps 和 --base-channels 必须与权重文件一致")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
    args.out.mkdir(parents=True, exist_ok=True)
    history_file = args.out / "loss.jsonl"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_images = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            total_images += images.shape[0]
        mean_loss = total_loss / total_images
        print(f"epoch {epoch + 1}/{args.epochs} | loss {mean_loss:.6f} | device {device}", flush=True)
        with history_file.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch + 1, "loss": mean_loss}) + "\n")
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch + 1, "steps": args.steps, "base_channels": args.base_channels}
        torch.save(checkpoint, args.out / "last.pt")
        model.eval()
        save_grid(diffusion.sample(model, 16, device), args.out / f"epoch_{epoch + 1:03d}.png")


if __name__ == "__main__":
    main()
