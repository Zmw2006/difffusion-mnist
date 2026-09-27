"""Plot train and validation epsilon-MSE from JSONL logs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制训练与验证损失曲线")
    parser.add_argument("--log", type=Path, default=Path("runs/mnist-ddpm/loss.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-ddpm/loss.png"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("日志为空")
    epochs = [item["epoch"] for item in records]
    train = [item.get("train_loss", item.get("loss")) for item in records]
    val = [item.get("val_loss") for item in records]
    if any(loss is None for loss in train):
        parser.error("日志中存在缺少训练损失的行")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, "o-", label="Train noise MSE")
    if any(loss is not None for loss in val):
        ax.plot([e for e, loss in zip(epochs, val) if loss is not None],
                [loss for loss in val if loss is not None], "o-", label="Validation noise MSE")
    ax.set(xlabel="Epoch", ylabel="Noise prediction MSE", title="MNIST DDPM training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
