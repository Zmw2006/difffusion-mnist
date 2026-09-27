# MNIST Diffusion：用 DDPM 生成手写数字

一个从零编写的 PyTorch 小项目：用 **DDPM（去噪扩散概率模型）** 学习 MNIST 训练集的分布，再从高斯噪声生成新的 28×28 灰度手写数字。这是**无条件生成**；不能指定生成哪个数字，也不是数字分类器。仓库不包含预训练权重或冒充训练结果的样例图片。

## 快速开始

建议 Python 3.10 或更新版本；有 CUDA 时自动使用 GPU，没有时使用 CPU。安装 PyTorch 时，如需指定 CUDA 版本，请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择适合本机的命令，然后安装 torchvision。以下是一组通用命令：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --epochs 20 --batch-size 128
python sample.py --checkpoint runs/mnist-ddpm/last.pt --count 64
```

首次训练通过 torchvision 下载 MNIST 至 `data/`，仅使用 `train=True` 的训练集。每轮保存 `runs/mnist-ddpm/last.pt`、`epoch_001.png` 等生成预览，并将平均噪声预测 MSE 逐行写入 `loss.jsonl`。最后一条命令保存 `runs/mnist-ddpm/samples.png`。`data/`、`runs/`、模型权重不会被提交到 Git。

在普通 CPU 上完整训练与每轮 200 步采样可能耗时较长。可以先运行一个功能验证：

```bash
python train.py --epochs 1 --batch-size 64 --steps 20 --base-channels 8 --out runs/smoke
python sample.py --checkpoint runs/smoke/last.pt --count 4 --out runs/smoke/test.png
```

功能验证使用较小模型与较少步数，**生成质量不代表正式训练**。训练更久通常有助于成像，但具体效果依设备、随机种子及超参数而变化。

## 原理对应到代码

对原图像 $x_0\in[-1,1]^{1\times28\times28}$，在随机时刻 $t$ 直接加噪：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

其中 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=0}^{t}\alpha_s$；$\beta_t$ 根据余弦噪声日程计算，使最后一步接近纯噪声。U-Net 接收 $x_t$ 和时刻 $t$，预测噪声 $\epsilon_\theta(x_t,t)$；训练目标是 $\|\epsilon-\epsilon_\theta\|_2^2$ 的均值。`ddpm/model.py` 是时间嵌入与 U-Net；`ddpm/diffusion.py` 是前向加噪、损失和反向采样。采样从标准正态噪声出发，逐步按 DDPM 后验方差还原，最后映射到灰度图。

## 常用选项

```bash
python train.py --epochs 30 --batch-size 128 --lr 0.0002 --steps 200 --base-channels 32 --seed 42 --out runs/experiment
python train.py --epochs 30 --steps 200 --base-channels 32 --out runs/experiment --resume runs/experiment/last.pt
python sample.py --checkpoint runs/experiment/last.pt --count 100 --seed 7 --out runs/experiment/grid.png
```

续训时 `--epochs` 是**目标总轮数**，不是额外轮数；`--steps` 和 `--base-channels` 必须与检查点匹配。每次采样使用检查点存储的这两项配置。权重用 `weights_only=True` 加载；仍建议只使用可信来源的检查点。

## 目录

| 路径 | 内容 |
| --- | --- |
| `train.py` | 下载训练集、训练、记录损失与每轮预览 |
| `sample.py` | 从检查点生成图片网格 |
| `ddpm/model.py` | 时间条件 U-Net |
| `ddpm/diffusion.py` | 前向扩散与 DDPM 反向采样 |
| `ddpm/utils.py` | 随机种子与图片导出 |

这是教学用的小型实现，默认不计算 FID，也不保证达到某篇论文的发表指标。如需严谨比较生成质量，应固定训练轮数、模型规模、随机种子和评价数据，再报告定量指标。
