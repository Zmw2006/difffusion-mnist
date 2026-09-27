"""Train an unconditional DDPM on the official MNIST training split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ddpm.diffusion import GaussianDiffusion
from ddpm.ema import EMA
from ddpm.model import UNet
from ddpm.utils import save_grid, set_seed


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MNIST 上训练 DDPM")
    parser.add_argument("--epochs", type=int, default=20, help="总训练轮数，包括已经完成的轮数")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=200, help="扩散与采样步数")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000, help="从 60000 张训练图中划给验证集；0 表示不用验证")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=1, help="每隔几轮保存预览，0 表示关闭")
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0, help="每轮最多训练几个 batch；仅用于冒烟测试")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-ddpm"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.grad_clip <= 0:
        parser.error("epochs、batch-size、lr、grad-clip 须为正数")
    if args.num_workers < 0 or args.sample_every < 0 or args.max_batches < 0 or args.preview_count < 1:
        parser.error("num-workers、sample-every、max-batches 须非负；preview-count 须为正")
    if not 0 <= args.val_size < 60000 or not 0 <= args.ema_decay < 1:
        parser.error("val-size 须在 [0,60000)；ema-decay 须在 [0,1)")
    if args.steps < 2 or args.base_channels < 8 or args.base_channels % 8:
        parser.error("steps 须至少为 2；base-channels 须为不小于 8 的 8 的倍数")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("未检测到 CUDA；可使用 --device cpu 或 --device auto")
    return args


@torch.no_grad()
def validation_loss(model: UNet, diffusion: GaussianDiffusion, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        total += diffusion.loss(model, images).item() * len(images)
        count += len(images)
    return total / count


def main() -> None:
    args = options()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))
    dataset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    if args.val_size:
        train_set, val_set = random_split(dataset, (len(dataset) - args.val_size, args.val_size),
                                          generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = dataset, None
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, generator=shuffle_generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common) if val_set is not None else None

    model = UNet(args.base_channels).to(device)
    ema = EMA(model, args.ema_decay)
    diffusion = GaussianDiffusion(args.steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        for name in ("steps", "base_channels", "seed", "val_size", "ema_decay"):
            if name in checkpoint and checkpoint[name] != getattr(args, name):
                raise ValueError(f"续训参数 --{name.replace('_', '-')} 必须与检查点一致")
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if "cuda_rng" in checkpoint and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        if "shuffle_rng" in checkpoint:
            shuffle_generator.set_state(checkpoint["shuffle_rng"].cpu())
        if args.out != args.resume.parent:
            raise ValueError("--out 须与续训检查点所在目录相同，以保留日志和最佳权重")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    history_file = args.out / "loss.jsonl"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)
            total += loss.item() * len(images)
            count += len(images)
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break
        train_loss = total / count
        val_loss = validation_loss(ema.model, diffusion, val_loader, device) if val_loader else None
        if val_loss is None:
            print(f"epoch {epoch + 1}/{args.epochs} | train {train_loss:.6f} | no validation", flush=True)
        else:
            print(f"epoch {epoch + 1}/{args.epochs} | train {train_loss:.6f} | val {val_loss:.6f}", flush=True)
        with history_file.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                                  "train_images": count}) + "\n")

        improved = val_loss is not None and val_loss < best_loss
        if improved:
            best_loss = val_loss
        if args.sample_every and (epoch + 1) % args.sample_every == 0:
            ema.model.eval()
            save_grid(diffusion.sample(ema.model, args.preview_count, device),
                      args.out / f"epoch_{epoch + 1:03d}.png")
        checkpoint = {"model": model.state_dict(), "ema": ema.model.state_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                      "steps": args.steps, "base_channels": args.base_channels,
                      "seed": args.seed, "val_size": args.val_size, "ema_decay": args.ema_decay,
                      "best_loss": best_loss, "torch_rng": torch.get_rng_state(),
                      "shuffle_rng": shuffle_generator.get_state()}
        if device.type == "cuda":
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, args.out / "last.pt")
        if improved:
            torch.save(checkpoint, args.out / "best.pt")


if __name__ == "__main__":
    main()
