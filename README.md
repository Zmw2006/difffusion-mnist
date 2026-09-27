# MNIST Diffusion：从原理到生成手写数字

使用 PyTorch 实现一个小型 **DDPM（Denoising Diffusion Probabilistic Model，去噪扩散概率模型）**。模型从 MNIST 的真实手写数字学习如何去除噪声；训练后从随机噪声出发，逐步生成新的 28×28 灰度数字图像。

> 这是**无条件生成**项目：模型不会按照指定标签生成某个数字，也不识别用户上传的数字。仓库没有预训练权重或伪造的实验指标；图片要在实际训练后生成。

## 你能在这个仓库做什么

- 下载并检查 MNIST，输出类别数量与原图网格。
- 查看同一张数字从清晰到模糊的正向扩散过程。
- 用时间条件 U-Net 预测噪声，训练 DDPM；将官方训练集固定拆分为训练集和验证集。
- 用 EMA（参数指数滑动平均）生成每轮预览，保存最后一次及验证损失最小的检查点。
- 从检查点续训、分批生成图片、切换原始参数或 EMA 参数。
- 画出训练与验证损失曲线；运行不下载数据的单元测试和 GitHub Actions。

## 快速开始：三条命令

先进入仓库目录，安装 Python 3.10 或更新版本，然后执行：

~~~bash
python -m pip install -r requirements.txt
python train.py --epochs 20 --batch-size 128
python sample.py --checkpoint runs/mnist-ddpm/best.pt --count 64
~~~

默认按有无 CUDA 自动选择 GPU 或 CPU。第一次训练时会从 torchvision 支持的地址下载 MNIST 到 <code>data/</code>。训练完成后查看 <code>runs/mnist-ddpm/samples.png</code>。如果安装 PyTorch 时需要选特定 CUDA 版本，请先按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 的指令装好 torch 和 torchvision，再安装其余依赖。

建议使用独立环境：

~~~bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

~~~powershell
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

> 完整训练可能花较长时间，尤其在 CPU 上；速度取决于设备和参数。这个仓库不承诺固定训练时间，也不声称已经训练出某个精度。

## 先跑通功能，再正式训练

只验证下载、前向、反向传播、保存和加载是否可用：

~~~bash
python -m unittest discover -s tests -v
python train.py --epochs 1 --batch-size 8 --steps 20 --base-channels 8 \
  --max-batches 2 --val-size 0 --preview-count 4 --out runs/smoke
python sample.py --checkpoint runs/smoke/last.pt --count 4 \
  --batch-size 2 --out runs/smoke/samples.png
~~~

Windows PowerShell 可将训练命令写成一行，或将 Bash 行尾的反斜杠换成 PowerShell 的续行符。单元测试使用随机合成张量并模拟数据集，不下载 MNIST；冒烟训练命令才会下载数据。<code>--max-batches 2</code> 只训练两个 batch，**其图片不能用来判断模型质量**。

正式训练推荐不要指定 <code>--max-batches</code>；默认 <code>--val-size 5000</code>，从官方 60000 张训练图中固定取 55000 张训练、5000 张验证。官方的 10000 张测试图**没有参与训练、挑选最佳权重或报告分数**。

## 第一步：认识数据

~~~bash
python -m scripts.inspect_mnist
python -m scripts.forward_process --steps 200 --index 0
~~~

第一条输出训练集、测试集大小、训练集各数字数量，保存前 64 张原图至 <code>runs/mnist-preview.png</code>。第二条保存 <code>runs/forward-process.png</code>：最左是原图，后面八张是在不同扩散时刻加噪后的图。命令行同时打印对应的时刻编号。正向加噪图使用**同一份随机噪声**，便于观察噪声强度的变化。

训练前用 <code>ToTensor()</code> 将像素从整数映射到 [0,1]，再用 <code>Normalize((0.5,), (0.5,))</code> 映射到 [-1,1]。采样结果再映射回 [0,1] 保存为 PNG。

## 第二步：理解扩散公式

### 1. 正向过程：已知图片，逐步加入噪声

记真实图片为 $x_0$，某一步的噪声强度为 $\beta_t$：

$$
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=0}^{t}\alpha_s .
$$

不必真的从 0 一步一步加到 $t$；可以直接采样任意时刻：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+
    \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
$$

第一项保留原图，第二项加入标准高斯噪声。代码在 <code>ddpm/diffusion.py</code> 的 <code>q_sample</code>。本项目用余弦噪声日程生成 $\beta_t$，在末端限制 $\beta_t\leq 0.999$，使末端图像接近纯噪声。这里代码的时刻编号是 **0 到 T−1**；数学表达式按相同编号书写。

### 2. 训练过程：让模型认出加进去的噪声

对每张图随机挑一个 $t$ 和一份 $\epsilon$，算出 $x_t$；让网络输出 $\epsilon_\theta(x_t,t)$，最小化：

$$
L=\mathbb E_{x_0,t,\epsilon}\left[
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2
\right].
$$

实现用逐像素平均平方误差（MSE）。训练并没有直接告诉模型“这是一张 7”；MNIST 标签仅供数据检查，训练循环丢弃标签。网络输入是 <code>[B,1,28,28]</code> 图片和 <code>[B]</code> 时刻张量，输出仍是 <code>[B,1,28,28]</code> 的噪声预测。

<code>ddpm/model.py</code> 使用正弦时间嵌入和小型 U-Net：28×28 → 14×14 → 7×7 → 14×14 → 28×28；跳跃连接把较高分辨率的信息送回解码器。<code>--base-channels</code> 决定宽度，默认 32。

### 3. 反向过程：从随机噪声生成图片

从 $x_{T-1}\sim\mathcal N(0,I)$ 出发，用网络预测噪声，再从最后一步倒着走到第 0 步。代码采用 DDPM 的均值：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)\right)
$$

并在 $t>0$ 时加上后验方差的随机噪声：

$$
\tilde\beta_t=\beta_t
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t},\qquad
x_{t-1}=\mu_\theta(x_t,t)+\sqrt{\tilde\beta_t}\,z .
$$

$t=0$ 时不再加噪声。对应代码是 <code>GaussianDiffusion.sample</code>。这种逐步采样比一次前向预测慢：默认 **200 次网络调用/批**。

### 4. 为什么有 EMA

训练参数在每个 batch 后都会变化。EMA 维护一个平滑版本：

$$
\theta_{\mathrm{EMA}}\leftarrow
d\,\theta_{\mathrm{EMA}}+(1-d)\,\theta,\qquad d=0.995.
$$

默认预览和最终采样使用 EMA 参数；加 <code>--raw</code> 可以与原始训练参数比较。EMA 并不能保证每个实验的图片都更好，仍要看生成结果。

## 第三步：训练与续训

~~~bash
python train.py --epochs 20 --batch-size 128 --lr 0.0002 \
  --steps 200 --base-channels 32 --seed 42 --out runs/mnist-ddpm
~~~

每轮产生或更新以下文件：

| 文件 | 用途 |
| --- | --- |
| <code>config.json</code> | 本次运行的命令行配置 |
| <code>loss.jsonl</code> | 每轮的训练噪声 MSE、验证噪声 MSE、实际训练图像数 |
| <code>last.pt</code> | 最新轮次的模型、EMA、优化器、配置和随机状态 |
| <code>best.pt</code> | 验证噪声 MSE 最小的轮次；关闭验证集时不生成 |
| <code>epoch_001.png</code> 等 | EMA 模型的定期采样预览 |

训练日志里的 MSE 是**预测噪声的误差**，不是图片分类准确率、FID 或人工质量评分。验证 MSE 每次都抽新的时刻和噪声，会有随机波动；“最佳”只表示这个指标最小，仍需查看样图。

继续训练到总计 30 轮：

~~~bash
python train.py --epochs 30 --batch-size 128 --lr 0.0002 \
  --steps 200 --base-channels 32 --seed 42 --val-size 5000 \
  --out runs/mnist-ddpm --resume runs/mnist-ddpm/last.pt
~~~

<code>--epochs 30</code> 表示**累计 30 轮**。续训必须保持 <code>steps</code>、<code>base-channels</code>、<code>seed</code>、<code>val-size</code>、<code>ema-decay</code> 一致；<code>--out</code> 要指向检查点所在目录。检查点保存了 PyTorch CPU/CUDA 及 DataLoader 的随机状态，方便按轮继续训练；不同设备、工作进程数量或底层算法可能影响逐位重复性。

## 第四步：看曲线、生成图片

~~~bash
python -m scripts.plot_loss --log runs/mnist-ddpm/loss.jsonl \
  --out runs/mnist-ddpm/loss.png
python sample.py --checkpoint runs/mnist-ddpm/best.pt \
  --count 64 --batch-size 16 --seed 123 --out runs/mnist-ddpm/samples.png
python sample.py --checkpoint runs/mnist-ddpm/last.pt \
  --count 64 --seed 123 --raw --out runs/mnist-ddpm/raw.png
~~~

<code>--batch-size</code> 只控制一次生成多少张，显存不足时可以减小。相同种子在相同设备、版本、模型和批大小下更容易重现结果；改动批大小可能改变随机数消耗顺序。采样会自动读取检查点中的 <code>steps</code> 和 <code>base_channels</code>。

## 参数速查

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| <code>--epochs</code> | 20 | 总轮数 |
| <code>--batch-size</code> | 128 | 训练批大小 |
| <code>--lr</code> | 0.0002 | AdamW 学习率 |
| <code>--steps</code> | 200 | 扩散步数，训练与采样必须相同 |
| <code>--base-channels</code> | 32 | U-Net 的基础通道数，须是 8 的倍数 |
| <code>--val-size</code> | 5000 | 从官方训练集留出的验证张数；0 关闭验证 |
| <code>--ema-decay</code> | 0.995 | EMA 平滑系数 |
| <code>--grad-clip</code> | 1.0 | 梯度范数裁剪阈值 |
| <code>--sample-every</code> | 1 | 每隔几轮保存预览；0 关闭 |
| <code>--preview-count</code> | 16 | 每轮预览图片数量 |
| <code>--num-workers</code> | 0 | 数据加载工作进程数 |
| <code>--device</code> | auto | auto / cpu / cuda |
| <code>--max-batches</code> | 0 | 0 表示完整训练；其他值仅用于功能检查 |

## 代码地图

| 路径 | 作用 |
| --- | --- |
| <code>ddpm/model.py</code> | 正弦时间嵌入、残差块、小型 U-Net |
| <code>ddpm/diffusion.py</code> | 余弦日程、正向加噪、损失、反向 DDPM 采样 |
| <code>ddpm/ema.py</code> | 模型参数指数滑动平均 |
| <code>ddpm/utils.py</code> | 随机种子和图片网格导出 |
| <code>train.py</code> | 数据划分、训练、验证、断点续训、检查点 |
| <code>sample.py</code> | 加载权重并分批生成图片 |
| <code>scripts/inspect_mnist.py</code> | 数据数量和类别分布、原图网格 |
| <code>scripts/forward_process.py</code> | 正向加噪过程图 |
| <code>scripts/plot_loss.py</code> | 训练与验证 MSE 曲线 |
| <code>tests/test_ddpm.py</code> | 数值形状、反向传播、EMA、完整命令流程测试 |
| <code>.github/workflows/tests.yml</code> | 每次推送时运行 Python 测试 |

## 常见问题

**图片像雪花或看不出数字？** 先确认不是 <code>--max-batches</code> 冒烟训练；查看 <code>loss.png</code> 和多个轮次预览。增加训练轮数或模型宽度可能改善效果，但没有固定保证。

**找不到 best.pt？** 当 <code>--val-size 0</code> 时没有验证集，自然没有“验证集最佳”检查点；使用 <code>last.pt</code> 采样。

**显存不足？** 依次减小训练的 <code>--batch-size</code>、<code>--base-channels</code> 或 <code>--preview-count</code>。采样时单独减小 <code>sample.py --batch-size</code>。改变模型宽度后不能直接加载旧权重。

**为什么损失下降但图片仍一般？** 噪声预测 MSE 衡量的是给定加噪样本的误差，不能单独代表人的视觉评价。检查训练是否足够、输入是否在 [-1,1]、是否使用同一个步数，并对比不同轮次的生成样例。

**下载 MNIST 失败？** 确认网络可访问数据源；也可按 torchvision 的 MNIST 目录格式先将数据放入 <code>data/MNIST/raw/</code>。测试命令不依赖数据下载。

**能否指定生成 0 或 7？** 目前是无条件模型，不能。要指定数字，需要加入类别嵌入并按标签训练或使用 classifier-free guidance；那是后续扩展方向。

**算“论文严格复现”吗？** 不能这样宣称。这是面向 MNIST 的教学实现，保留了 DDPM 的核心训练与采样公式，使用余弦噪声日程和小型 U-Net；原论文的大规模设置、网络配置和评价协议并未逐项复刻。仓库没有声称达到论文的指标。

## 复现实验时请记录

记录 Python、PyTorch、torchvision 版本，设备、随机种子、训练轮数、模型宽度、扩散步数、验证划分、最终权重来源（best/last、EMA/raw）、生成样本数量和人工或定量评价规则。<code>config.json</code> 与 <code>loss.jsonl</code> 帮助追踪单次运行，但**没有代替**最终图片质量评估。请不要用训练集做“未见数据”评价，也不要把冒烟样例当正式成果。

参考资料：[DDPM 原论文（Ho 等，2020）](https://arxiv.org/abs/2006.11239) · [Improved DDPM 及余弦日程（Nichol 与 Dhariwal，2021）](https://proceedings.mlr.press/v139/nichol21a.html) · [Torchvision MNIST 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。

本项目基于 MIT License 发布。数据、训练日志、生成图片与权重默认被 <code>.gitignore</code> 排除。
