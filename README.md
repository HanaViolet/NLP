# 自然语言处理 (NLP) 课程期末自我评价报告

**学生姓名**：李家豪 (42311137)

**日期**：2025年12月31日

**核心成果**：NumPy 手写神经网络、SwimVG 复现、论文纠错、nanoGPT 全流程训练

---

## 一、 综述：从离散符号到生成式范式

本学期，我遵循“基础统计 -> 深度序列建模 -> Transformer 架构 -> 多模态与大模型工程”的学习路径，系统掌握了 CP1 至 CP12 的核心考点。我不满足于仅调用 API，而是坚持“去框架化”的底层推导（如手写反向传播），并在期末第 17 周通过 **nanoGPT** 项目完成了对 GPT 架构的工程化落地与 Scaling Law 验证，实现了理论与实践的闭环。

---

## 二、 第一阶段：统计语言模型与词向量 (第1-4周)

**核心考点**：`CP1 (基础概念)`、`CP3 (N-gram)`、`CP4 (Word2Vec)`

在课程初期，我重点解决了“如何让计算机理解词义”与“如何计算句子概率”两个基础问题。

### 1. N-gram 与数据稀疏性 (Week 1-2)

* **学习内容**：深入理解了马尔可夫假设。针对 N-gram 模型中未登录词导致的“零概率”问题，我不仅学习了理论，还手动推导并实现了**加一平滑 (Add-one Smoothing)** 算法。
* **深度思考**：平滑技术不仅仅是数学修正，更是解决数据覆盖率不足的关键，这与 CP11 中大模型需要海量数据覆盖（Scaling Law）的底层逻辑是一致的。
* **[证据 1]：加一平滑算法实现代码**
```python
def add_one_smoothing(bigram_count, unigram_count, vocab_size):
    """
    CP3实战：计算 P(w_i | w_{i-1})
    分子 +1 处理零概率，分母 +V 保证概率归一化
    """
    return (bigram_count + 1) / (unigram_count + vocab_size)

```



### 2. 词向量的数学本质 (Week 3-4)

* **学习内容**：对比了 One-hot 的稀疏性与 Distributed Representation 的优势。重点分析了 **Word2Vec** 的两种架构：**CBOW** (上下文预测中心词) 和 **Skip-gram** (中心词预测上下文)。
* **分析计算**：理解了为何通过简单的浅层神经网络训练，词向量空间能产生  的线性语义结构。

---

## 三、 第二阶段：序列建模与梯度的秘密 (第5-9周)

**核心考点**：`CP4 (RNN更新)`、`CP5 (RNN问题, Seq2Seq, Attention)`

这一阶段，我选择了一条“硬核”路径：脱离 PyTorch 自动求导，用纯 NumPy 手写底层算法，以彻底弄懂 RNN 的缺陷。

### 1. 纯 NumPy 手写反向传播 (Week 5-7)

* **学习内容**：为了理解 CP5 中“为什么原始 RNN 有问题”，我手动推导了 BPTT (Backpropagation Through Time) 算法。
* **攻克难点**：在推导  时，我直观看到了连乘项 。当  的特征值小于 1 时，该项随时间步指数级衰减。
* **收获**：这完美解释了**梯度消失 (Vanishing Gradient)** 现象，也让我理解了为何 LSTM 引入“门控”机制以及 Transformer 引入“残差连接”是历史的必然。
* **[证据 2]：纯 NumPy 反向传播推导代码**
```python
# CP5实战：不依赖 Autograd，手动计算梯度
# d/dx tanh(x) = 1 - tanh^2(x)
d_hidden = np.dot(W_hh.T, d_next) * (1 - hidden_state**2)
d_Whh += np.dot(d_hidden, prev_hidden.T) # 梯度累加

```



### 2. Seq2Seq 与 Attention 的引入 (Week 8-9)

* **学习内容**：在机器翻译任务中，对比了训练模式 (**Teacher Forcing**) 与推理模式 (**Free-running**) 的差异。理解了 Attention 机制如何通过  打破了 RNN 的定长向量瓶颈。

---

## 四、 第三阶段：多模态前沿与科研批判 (第10-14周)

**核心考点**：`CP7 (ViT/CLIP)`、`CP9 (微调/LoRA)`、`CP5 (Masking)`

在此阶段，我开始阅读前沿论文，并结合课程知识进行批判性研究。

### 1. 论文纠错：Attention Masking 的数学定义 (Week 10-11)

* **项目背景**：在阅读 ACL 2025 论文 *Segment-Based Attention Masking* 时，利用 CP5 学到的自回归原理进行纠错。
* **核心发现**：论文公式 (13) 将因果掩码定义为上三角矩阵（允许看未来），这违反了自回归属性。我推导了正确的**下三角掩码**定义。
* **[证据 3]：因果掩码修正推导**
> **原文错误**： (上三角  Information Leakage)
> **我的修正**： (下三角  Causality Preserved)



### 2. SwimVG 复现与高效微调研究 (Week 12-14)

* **学习内容**：深入 CP7 (多模态) 和 CP9 (微调)。
* **实战分析**：
* **架构**：复现 SwimVG 模型，冻结 **CLIP-B** (Text) 和 **DINOv2** (Vision)，仅训练 Adapter。
* **微调对比 (CP9)**：对比了 **LoRA** 与 **CIA Adapter**。实验证明，虽然 LoRA 参数少，但在多模态交互任务上，SwimVG 设计的交互式 Adapter 效果更好。SwimVG 仅需训练 **2.04%** 的参数即可达到 SOTA。


* **[证据 4]：微调方法对比表 (基于 SwimVG 论文数据)**
| 方法 | 骨干网络 | 可训练参数 | 准确率 (Acc) |
| :--- | :--- | :--- | :--- |
| Full Fine-tuning | TransVG | 100% | 80.32% |
| **LoRA (CP9)** | CLIP-B | ~4.37% | 84.51% |
| **SwimVG (Ours)** | DINOv2+CLIP | **2.04%** | **90.37%** |

---

## 五、 第四阶段 (期末压轴)：nanoGPT 全流程工程化 (第17周)

**核心考点**：`CP6 (GPT架构/RoPE)`、`CP11 (Scaling Law)`、`CP12 (范式转移)`

在最后一周，我汇总全学期知识，基于 Andrej Karpathy 的 **nanoGPT** 进行了全流程的大模型训练与架构验证。

### 1. 模型架构的深度解构 (CP6)

通过解析 `model.py`，我不仅是“跑通代码”，而是从代码层面验证了课程理论：

* **Decoder-only 架构**：确认了模型仅使用 Masked Self-Attention。
* **Pre-LN 设计**：验证了 LayerNorm 被放置在 Attention 之前（不同于原始 Transformer），这解释了深层网络训练的稳定性。
* **RoPE (CP6 重点)**：深入理解了旋转位置编码如何通过复数旋转注入相对位置信息。

### 2. Scaling Law 的实证 (CP11)

* **分析计算**：针对实验配置（6层 Layer, 6个 Head, 384 维度），我没有直接相信日志，而是手算了参数量：



计算结果与训练日志完全吻合，验证了参数量与计算复杂度的关系。
* **[证据 5]：参数量验证代码**
```python
# Week 17: 验证 Scaling Law 参数量
n_layer, n_embd = 6, 384
# Transformer Block 参数主要源自 4个投影矩阵(Attn) + 2个投影矩阵(MLP, 4x expansion)
theory_params = 12 * n_layer * (n_embd ** 2)
print(f"Theory: {theory_params/1e6:.2f}M") # 结果: 10.62M

```



### 3. 工程化挑战与范式思考 (CP12)

* 
**工程落地**：抛弃了 Conda，使用 Rust 编写的 **uv** 进行依赖管理 。在 Windows 环境下，针对 `torch.compile` 报错问题，通过分析编译后端逻辑，配置 `--compile=False` 解决了兼容性难题 。


* 
**范式转移**：结合训练出的模型（能生成莎士比亚风格文本），我深刻理解了 CP12 提到的从 Discriminative 到 Generative 的范式转移——模型并非“理解”剧本，而是精准建模了字符间的条件概率 。



---

## 六、 总结

从第1周手写 N-gram 平滑公式，到第5周推导 RNN 梯度，再到第17周解决 Windows 编译问题跑通 nanoGPT，我完成了一场从数学底层到工程顶层的 NLP 探索。

这学期我最大的收获不在于记住了多少公式，而在于具备了**“去框架化”的推导能力**（CP4/5）和**“复现前沿论文”的实战能力**（CP7/9），为后续深入研究 Transformer 变体及多模态大模型打下了坚实基础。
