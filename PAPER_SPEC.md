# PAPER_SPEC.md — Adaptive-LoRA 论文复现规格

> 依据：*Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching*（arXiv:2610.34756v1）。本文件仅规定复现的算法与实验协议。
>
> 状态标记：**[论文]** 论文中报告；**[复现决策]** 本项目自行选定；**[待核验]** 论文没有充分明确或尚未核查。AI 不得将后两者表述为论文原设定。

## 1. 复现目标与范围

核心目标：实现 Adaptive-LoRA 的 frozen-base calibration、层重要性排序、Top-K 层选择、选层 LoRA 训练，并与 Full LoRA / Random-layer LoRA 公平对照。

第一轮**不包含** Section 3 全量 path-patching 机制分析、全部 11 个下游任务、分布式训练以及新算法改进。

## 2. 核心算法 [论文]

基础模型为冻结的 $M$；校准集为 $D_{cal}$，样本数为 $N$。对每个输入 $x$：

1. 使用未修改的基础模型，确定最高概率预测 token $t^*(x)$，以及其概率 $q(x)=P_M(t^*(x)\mid x)$。
2. 对每层 $l$，**独立**将 Attention **子模块输出**置零，重新前向并记录原 token $t^*(x)$ 的概率 $q_l^A(x)$。
3. 移除 Attention 消融后，独立将同层 MLP **子模块输出**置零，记录原 token 的概率 $q_l^{MLP}(x)$。
4. 层重要性：

$$
I_l = \frac{1}{N}\sum_{x\in D_{cal}}\left(\left|q(x)-q_l^A(x)\right|+\left|q(x)-q_l^{MLP}(x)\right|\right)
$$

5. 按 $I_l$ 降序取 Top-K 层，只对这些层添加 LoRA 适配参数。

### 算法不变量（不可改）

- 校准时模型处于 `eval()` 且无需梯度；基础权重不更新。
- Attention 和 MLP **分别消融**；不是同时消融，也不是把整层 Transformer block 置零。
- 消融后依然测量**原始模型**选择的 $t^*$；不能在每次消融后重新取 `argmax`。
- 每次干预后恢复原始状态；异常发生时也必须清理 hooks。
- 在目标模块的正确输出位置消融，处理好 tensor/tuple 等实际返回结构。
- 论文标题里的 *path patching* 与本校准算法的 *frozen-base zero-ablation* 不等同。

### 预测位置 [复现决策]

第一轮使用一致的 prompt 模板，从输入末尾的 **next-token logits** 提取 $t^*$。先逐样本实现作为正确性基准，再考虑 batch 优化。不能在 right-padding 下盲目使用 `logits[:, -1, :]`；必须验证其确实对应有效预测位置。tokenizer、padding、截断、special token、模型 revision 全部记录。

**[待核验]** 论文对所有数据集的 prompt 模板、具体 token 位置等未在此前资料中完全明确，因此这个工程决策不等于作者代码的逐 token 实现。

## 3. 校准数据

**[论文]** 论文附录 E.2 描述从训练集抽取 $N$ 个 calibration 样本；采用基础模型预测 token，因此不需要 gold-answer 标注。

| 实验 | 模型与规模 | 性质 |
|---|---|---|
| Smoke test | Qwen2.5-0.5B，N=30，Top-7 | [复现决策] 小规模验证；Top-7 参考论文 0.5B 相关分析，不等于 1.5B 主表配置 |
| Main | Qwen2.5-1.5B，28 层，Top-8，N=250 | [论文] |
| 低成本默认校准 | N=30 | [论文] |
| Smoke 数据源 | Stanford Alpaca 输入文本 | [复现决策] 仅验证算法流程 |
| 正式训练集 | commonsense_170k 抽取 10,000 条；math_10k 抽取 5,000 条 | [论文]，具体版本与抽样索引待核验 |

- 正式实验尽量从目标训练集抽样；不从评估集抽取 calibration 输入。
- 固定并保存 seed、样本 ID/索引、样本内容哈希、prompt 与模板。
- 默认不得将答案 `output` / `reference` 拼入 calibration prompt，除非任务协议明确要求并单独注明。

## 4. 训练参考配置 [论文，执行前须再次核对原 PDF]

| 项目 | Qwen2.5-1.5B 主实验参考 |
|---|---|
| LoRA | rank=16，alpha=32，target Q/K/V |
| Optimizer | AdamW，lr=1e-4，weight_decay=0.01 |
| Betas / eps | (0.9, 0.999) / 1e-8 |
| Schedule | 3 epochs，无 warmup |
| Batch | 每卡 8，gradient accumulation 8 |
| Precision / length | FP16，max length 512 |
| Seed | 42（训练配置；**不能视为论文强制的校准抽样 seed**） |

尚待明确：数据集准确版本、预处理和数据划分、完整 prompt 模板、训练 loss mask、模型权重 revision、评估实现、不同 GPU 数下有效 batch 的等价性。所有差异都需记录。

## 5. 对照实验

- **Full LoRA**：所有层的 Q/K/V 添加 rank=16 LoRA。
- **Random-K LoRA**：随机选 K 层，仅这些层添加同规格 LoRA。
- **Adaptive-K LoRA**：按 $I_l$ 选 K 层，仅这些层添加同规格 LoRA。

统一训练集、评估集、预处理、seed、优化器与评估程序。**注意 Full 与 K-layer 的可训练参数量不同**；分别报告效果、实际可训练参数数量、显存和耗时，不能声称它们“同参数预算”。

论文主表参考数值（Qwen2.5-1.5B，rank16，commonsense average）：Full LoRA 56.9%、Adaptive-LoRA 59.1%、LoRA+ 57.9%、DoRA 55.5%。这些不是本项目结果。

## 6. 必须通过的正确性测试

| 编号 | 验收项 |
|---|---|
| T01 | 校准前后基础模型参数未变化，校准不产生梯度 |
| T02 | 预测位置正确，消融始终使用原始 $t^*$ |
| T03 | Attention 消融仅置零目标 Attention 输出 |
| T04 | MLP 消融仅置零目标 MLP 输出 |
| T05 | hooks 在正常/异常退出时均被清理；原始前向恢复 |
| T06 | N=1 的重要性等于两项概率差绝对值之和 |
| T07 | N>1 的重要性等于逐样本重要性的算术平均 |
| T08 | Top-K 数量正确、唯一、合法，并列时排序稳定 |
| T09 | 只有所选层 Q/K/V LoRA 参数可训练，基础权重冻结 |
| T10 | 在相同 seed、数据与配置下可重复（说明非确定性来源） |

## 7. 开放问题及处理

| ID | 问题 | 处理原则 |
|---|---|---|
| Q01 | 数据集版本、样本索引 | 查论文/数据仓库；未查明时标记替代设置 |
| Q02 | 原始 prompt/预测 token 位置 | 明确项目自己的实现选择 |
| Q03 | 模型/依赖版本 | 保存 revision 和环境清单 |
| Q04 | 训练 loss mask、样本拼接、评估细节 | 进入正式训练前冻结协议 |
| Q05 | 作者完整代码是否已发布 | 未验证前不依赖其存在 |

**规则：当论文未给出细节时，记录不确定性和工程假设，而非默默补齐。**
