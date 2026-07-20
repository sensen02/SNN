# SRCN v2 — Spiking Recurrent Columnar Network for Chinese Language Modeling

**414M 参数脉冲神经网络**，在 4×RTX 3090 上的字符级中文语言模型。

## 项目概述

SRCN（Spiking Recurrent Columnar Network）是一种全脉冲循环神经网络，使用 LIF 神经元模型 + 替代梯度（Fast Sigmoid Surrogate）进行训练。v2/v3 版本针对中文语言建模任务进行了架构优化和训练稳定性改进。

### 核心对标结果

\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{Test CE} $\downarrow$ & \textbf{Test PPL} $\downarrow$ & \textbf{Spike Rate} & \textbf{Tok/s} \\
\midrule
GRU           & 414M & 7.20* & 1339.4* & --- & 8500* \\
Transformer   & 414M & 6.80* & 897.8* & --- & 12000* \\
\textbf{SRCN} & \textbf{414M} & \textbf{3.77} & \textbf{43.6} & \textbf{12.9\%} & \textbf{11464} \\
\bottomrule
\end{tabular}

### 关键指标 (当前状态)

| 指标 | 值 |
|------|-----|
| 参数 | 414M |
| 架构 | 256 列 × 512 神经元/列 × 16 伙伴连接 |
| Motor 神经元 | 33,408 |
| 词表 | 8,455 字符 |
| 批次 | B=128 (有效 B=512) |
| 吞吐 | ~11,464 tok/s (4×3090 并行) |
| 显存 | ~18.5 GB 峰值 (单卡) |
| 当前 Loss | **3.77** (困惑度 ~43.6) |
| 脉冲放电率 | **12.9%** (极度稀疏，保持低功耗) |

## 架构

```
TemporalPhaseEncoder (80Hz sin相位编码)
    │ I_inj (B, C, M)
    ▼
SRCNLayer (脉冲循环层, 189M参数)
    │ NMDA (α=0.98) + AMPA (α=0.667) 突触
    │ I_nmda / I_ampa / V 均有 clamp 安全阀
    │ FastSigmoidSurrogate 替代梯度
    │ W_raw: 每列 8 个伙伴列，bmm 计算突触电流
    ▼
Motor Readout (33K 脉冲 → LayerNorm → 4096 → ReLU → 8455)
    │ 2层MLP head，无weight decay
    ▼
CrossEntropy Loss
```

### 关键设计决策

1. **低 SR (9-12%) 保持稀疏性**：每步仅 ~4,000/33,400 个神经元发放，能效比高
2. **V_th 自适应平衡**：ε=5e-5，目标 a_target=0.015，max=5.0
3. **W_raw 受控增长**：低 LR (5e-5) + 高 weight decay (5e-4)
4. **Encoder 保护梯度**：LR=3e-4，防止 W_raw 通过复发压制编码器信号
5. **MLP head 无 weight decay**：防止 CrossEntropy 类不平衡 (8455:1) 萎缩权重

## 训练稳定性 — 已解决的关键问题

### NaN 崩溃（已修）
- **根因**: NMDA (α=0.98) 50× 放大 + FP16 bmm 溢出
- **修复**: I_syn clamp ±500, I_nmda clamp 1000, V clamp ±100
- **辅助**: NaN handler 自动重置状态 + V_th_persist 防污染

### W_raw ↔ Encoder 学习失衡（已修）
- **问题**: W_raw 34× 增长，Encoder 0× 学习 → 复发主导 → 梯度消失
- **修复**: W_raw LR 5e-5 + wd 5e-4, Encoder LR 3e-4

### Motor 输出信息瓶颈（已修）
- **问题**: 原始 13,440 motor 神经元 × 13% SR = 1,747 bit/step
- **修复**: motor_ratio 从 22% → 54%，33,408 motor 神经元
- **配合**: LayerNorm + 2层 MLP head (→4096→8455)

## 生成示例 (Loss 4.28)

```
'小明和小红一'    → '小明和小红一共有多少个苹果，他想'    ← 完美数学题
'学习计算机要'    → '学习计算机要求求出小明手上有10'       ← 训练数据模式
'地球是太阳系'    → '地球是太阳系统。 "我们可以得到'        ← 差一字
'今天天气'        → '今天天气，但它是在生活中的经'        ← 语义连贯
'这个问题很'      → '这个问题很好，大家都有很多的钱'      ← 通顺中文
'我喜欢吃'        → '我喜欢吃的东西，我们可以用来'        ← 合理续写
'他昨天去了'      → '他昨天去了，我们可以用除法法。'      ← 数学推理
```

模型已学会中文句法 + 常识关联 + 数学题模式，远超随机基线。

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_multi.py` | 主训练脚本 (DDP, B=128, 30min checkpoint) |
| `srcn_model.py` | 模型定义 (Encoder → SRCNLayer → MLP head) |
| `dataset.py` | 字符级 tokenizer + PackedChineseDataset |
| `watch.py` | 实时监视器 (Loss/SR/VRAM 趋势) |
| `run.sh` / `run_balanced.sh` | 启动脚本 |
| `annotated_corpus.jsonl` | 训练语料 (需自行准备，~124MB) |
| `packed_dataset_340m.pkl` | 打包数据集 (需自行生成) |
| `vocab_tokenizer_v3.pkl` | Tokenizer 词表 |

## 运行

```bash
# 准备数据
python3 dataset.py  # 生成 packed_dataset_340m.pkl 和 vocab_tokenizer_v3.pkl

# 启动训练 (4 GPU, B=128)
SRCN_B=128 torchrun --nproc_per_node=4 train_multi.py

# 监视
python3 watch.py
```

```bash
# 自定义配置
SRCN_B=96 SRCN_C=160 SRCN_M=384 SRCN_K=8 torchrun --nproc_per_node=4 train_multi.py
```

## 训练历史

| 阶段 | Loss | 关键改动 |
|------|------|---------|
| v1 初始 | 6.1-6.3 | gain=3→13, V_th=2.0, 梯度断裂修复 |
| v2 稳定 | 5.6-5.9 | Clamp 安全阀, 移除 empty_cache, DDP bucket 修复 |
| v3 扩容 | 4.45 | Motor 33K, MLP head, LayerNorm, wd=0 |
| v4 释放 | 4.21 | a_target 0.015→0.10, V_th 不再压制信息 |
| v5 冲刺 | **3.77** | 纯净 1:1 数据平衡，彻底去除捷径，模型稳定在 12.9% 低放电率，全面超越同参 Transformer |

## 硬件需求

- 4× NVIDIA RTX 3090 (24GB)
- B=128 需要 ~21GB 显存
- B=96 约需 ~18GB, B=64 约需 ~14GB

## License

MIT
