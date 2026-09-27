"""Inspect official MNIST counts and export a small training-image grid."""

import argparse
from collections import Counter
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.utils import save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 MNIST 数据与类别分布")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/mnist-preview.png"))
    args = parser.parse_args()
    train = datasets.MNIST(args.data_root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(args.data_root, train=False, download=True, transform=transforms.ToTensor())
    print(f"训练图像：{len(train)}；测试图像：{len(test)}；图像形状：{tuple(train[0][0].shape)}")
    print("训练类别计数：", dict(sorted(Counter(int(label) for label in train.targets).items())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image([train[index][0] for index in range(64)], args.out, nrow=8, padding=2)
    print(f"训练图像网格：{args.out}")


if __name__ == "__main__":
    main()
