# CS3602_FINAL_SpeDec

基于大小模型协同方法 (Speculative Decoding) 加速 Pythia-2.8b 模型的推理

## 目录
- [CS3602\_FINAL\_SpeDec](#cs3602_final_spedec)
  - [目录](#目录)
  - [简介](#简介)
  - [环境配置](#环境配置)
  - [项目结构](#项目结构)
  - [实验过程](#实验过程)
  - [实验结果与分析](#实验结果与分析)
    - [1. 速度测试(Speed)](#1-速度测试speed)
    - [2. 困惑度测试(PPL)](#2-困惑度测试ppl)
  - [问题分析与讨论](#问题分析与讨论)
  - [参考资料](#参考资料)

## 简介

本项目旨在探索大小模型协同方法(Speculative Decoding, 或投机采样)技术在大语言模型推理加速中的应用。通过使用一个轻量级的草稿模型(Draft Model，Pythia-70m)来预先生成候选token，再用目标模型(Target Model，Pythia-2.8b)进行并行验证，从而在保证输出分布不变的前提下，显著提升推理效率。

## 环境配置
- 硬件：支持CUDA的GPU（如NVIDIA RTX 3060或更高）
- 软件：Python 3.10+， Pytorch， Transformers(HuggingFace)，Datasets
- 模型：
  - Target Model: Pythia-2.8b
  - Draft Model: Pythia-70m
- 数据集：
  - WikiText-2
  - PG-19

## 项目结构

```
CS3602_FINAL_SpeDec/
├── models/            模型文件
│   └── pythia-2.8b/
|   └── pythia-70m/
├── datasets/          数据文件
│   └── wikitext/
│   └── pg19_sample/            
├── downloadData.py    数据集下载脚本（WikiText-2, PG-19）
├── downloadModel.py   模型下载脚本（Pythia-70m, 2.8b）
├── utils.py           包含 Top-K/Top-P 过滤、Logits 归一化等采样辅助函数
├── regrSampling.py    标准自回归采样 (Autoregressive Sampling) 实现
├── specSampling.py    投机采样 (Speculative Decoding) 核心算法实现
├── PPL.py             困惑度 (Perplexity) 计算逻辑
├── test.py            实验主入口，包含速度基准测试 (Speed Benchmark) 和 PPL 测试
└── results.txt        实验运行输出结果记录
```

## 实验过程

1. 数据准备：
   - 运行 `downloadData.py` 下载 WikiText-2 和 PG-19 数据集。
   - 运行 `downloadModel.py` 下载 Pythia-70m 和 Pythia-2.8b 模型。
2. 基准测试(Speed):
   - 配置`maxLen=200`, `gamma=4`, `speed=1`
   - 分别测试small(70m), target(2.8b)模型的标准自回归推理速度
   - 测试spec(Speculative)模式下的推理速度
   - 记录TTFT(首字延迟)，TPOT(每个Token平均时间)和Throughput(吞吐量)
3. 困惑度测试(PPL):
   - 配置`ppl=1`
   - 通过`PPL.py`计算目标模型在WikiText-2数据集上的困惑度, 验证投机采样输出分布一致性

## 实验结果与分析

### 1. 速度测试(Speed)

在速度测试中，我评估了**三种模式**下的推理性能：使用**小模型**Pythia-70m进行标准自回归采样（small），使用**目标模型**Pythia-2.8b进行标准自回归采样（big），以及结合两者的**投机采样**方法（spec）。测试参数设置为最大生成长度(maxLen)为200，草稿模型与目标模型的速度比(gamma)为4。

可以从数据中看出，大小模型协同推理的方法在推理速度上有显著提升。实现了**1.8x ~ 2.5x**的加速效果，具体表现为相较于大模型的TPOT的显著降低，以及吞吐量的提升。

但我观察到速度指标具有一定**波动**，（在TTFT指标尤是），故以下展示了两组在完全相同的配置下得到的结果，第一组的TPOT和Throughput指标是多次测试中的最佳结果，不具有一般性；第二组的数据较为稳定普遍：

**组一（最优情况）**

| 模型/模式        | TTFT (s) | TPOT (ms) | 吞吐量 (tokens/s) |
|------------------|-----------|-----------|---------------------|
| Pythia-70m (small) | 0.4458     | 8.9753      | 111.4168                |
| Pythia-2.8b (target) | 1.4061    |   267.0314     | 3.7449                |
| Speculative Decoding | 0.9570    | 107.1935    | 9.3289           |

**组二（一般情况）**

| 模型/模式        | TTFT (s) | TPOT (ms) | 吞吐量 (tokens/s) |
|------------------|-----------|-----------|---------------------|
| Pythia-70m (small) | 0.4629    | 8.7781     | 113.9199           |
| Pythia-2.8b (target) | 1.5469    |   268.5069     | 3.7243           |
| Speculative Decoding | 0.3636   | 148.7493     | 6.7227         |

在查询资料后我认为可能的波动原因如下：
- **系统负载**：实验过程中系统可能同时运行了其他高负载任务，影响了GPU和CPU的可用资源，导致推理时间不稳定。
- **解释器开销**：缓存管理、解释器锁、垃圾回收机制等会导致不确定的时间延迟和调度延迟，从而导致TTFT变化较大。
- **环境噪声**：GPU频率动态调整，带宽波动等可能影响推理速度。

### 2. 困惑度测试(PPL)

在困惑度测试中，我计算了三种方法在WikiText-2数据集上的困惑度表现，以验证投机采样方法在保持输出分布一致性方面的有效性。下面给出其平均值结果：

| 模型/模式        | 困惑度 (PPL) |
|------------------|---------------|
| Pythia-70m (small) | 84.372630      |
| Pythia-2.8b (big) | 17.141058       |
| Speculative Decoding | 17.141059       |

在浮点运算的差异允许范围内，投机采样方法的困惑度与目标模型**完全一致**。二者的采样分布是相同的。下面给出其数学证明：

设 $\beta$ 为接受率 
注意到
 $p'(x) = norm(max(0, p(x) - q(x))) = \frac{p(x) - min(q(x), p(x))}{\sum_{x'}(p(x') - min(q(x'), p(x')))} = \frac{p(x) - min(q(x), p(x))}{1 - \beta}$, 
 分布 $p'(x)$ 的归一化常数是 $1 - \beta$, 从而就有：

$$P(x = x') = P(guess \, accepted, x = x') + P(guess \, rejected, x = x')$$
其中：
$$
P(guess \, accepted, x = x') = q(x') \min(1, \frac{p(x')}{q(x')}) = \min(q(x'), p(x'))$$
并有：
$$P(guess \, rejected, x = x') = (1 - \beta)p'(x') = p(x') - \min(q(x'), p(x'))$$
所以：
$$
P(x = x') = \min(p(x'), q(x')) + p(x') - \min(p(x'), q(x')) = p(x').$$

这表明投机采样方法生成的token分布与目标模型的分布完全一致，因此困惑度在误差范围内相同。

## 问题分析与讨论

在实验过程中，我遇到了一些挑战和问题，主要包括以下几点：
1. **PPL计算**：
   - 计算模型PPL过程中，我会遇到INF情况。开始时我尝试用`torch.max`来让概率值为`0.`时变为`1e-10`，但仍然会出现INF，我不理解其原因，打断点进行检查。结果发现在使用`torch.max`后`prob`没有变为`1e-10`，而仍然是`0.`，我猜测可能是`torch.max`有维度问题，改用`torch.clamp(prob, min=1e-10)`，问题仍然没有解决。在排查后我发现`torch.tensor(1e-10).to(device)`会变成`0.`，我才意识到是数据精度的问题，为了加速测试采用了`float16`，而`1e-10`在`float16`下下溢为`0.`。所以为了避免这种情况以及PPL计算的准确性，我改用了`float32`，问题解决。
   - 计算`Specculative Decoding`的PPL时，不能直接使用`CrossEntropyLoss`计算，因为该方法生成的token序列并非完全自回归生成，存在部分token是由草稿模型预测的。同时，需要思考如何直接地计算PPL而不是根据数学计算结果直接给出`Target Model`的PPL（否则就没有实证意义，而仍是数学证明）。开始时我尝试了遍历词表计算, 将`P(guessRejected, x = x')`在词表上遍历求和，结果正确但效率极低。后来我采用了论文中假设接受率的办法，得到 `P(guessRejected, x = x') = (1 - β) p'(x') = p(x') - min(q(x'), p(x'))` , 从而计算了PPL。

2. **协同采样实现**：
   - 实现大小模型协同的过程中，由于采样逻辑是带温度、Top-K/Top-P等多种采样策略的，起初我打算从库中调用相应的函数来进行采样，但发现修改起来非常复杂且容易出错。后来我决定从头实现一个简化版的生成函数，专门用于协同采样，这样可以更好地控制采样过程和调试。相关的几个辅助函数放在`utils.py`中实现。

3. **性能波动**：
   - 在速度测试中，我观察到推理时间存在一定的波动，尤其是在TTFT指标上。为了解决这个问题，我尝试多次运行实验并取平均值，同时确保在实验过程中系统负载较低，以减少外部干扰对结果的影响。但波动仍然较大，查阅资料后我加入了禁用gc的代码，以减少垃圾回收对时间的影响，效果有所改善但仍未完全消除波动。资料提到还有关闭CPU睿频防止跳频影响，锁定GPU频率防止温度负载影响等。

## 参考资料
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Pythia Model](https://github.com/EleutherAI/pythia)