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

### Zero-ablation 工程边界 [复现决策]

- 在经 M1 模块发现器确认的 Attention/MLP 上注册临时 forward hook，而非在 block 或 Q/K/V 投影上干预。仅以新的 `zeros_like` 主输出替换返回值，避免原地修改与输入/残差共享的存储。
- M1 实测 Attention 为 `(Tensor, None)`、MLP 为 Tensor；Attention 普通 tuple/list 仅替换第 0 项 hidden states，附加项保留身份和位置。未知返回结构或非三维 hidden states 明确报错，不猜测适配。
- 调用者先执行全模型 `eval()`、冻结全部参数，再在 `no_grad()`/`inference_mode()` 内使用；helper 检查这些前提，不隐式更改模型模式或参数的 `requires_grad`。所有比较使用完整新输入、`use_cache=False`，不复用历史 KV cache。
- 同一根模型只允许一个活跃消融 scope，禁止跨层/跨组件的嵌套；只移除本 scope 的 hook，保留原有观察 hook。正常退出但目标未执行时应报错；异常路径仍清理，不能用“未命中”覆盖原异常。
- 限于串行 eager 前向；注册门禁不是共享模型的并发隔离，也不宣称支持 compiled/distributed 模型或共享子模块的别名根模型。
- 非目标模块可能因上游干预而出现不同激活，不能要求它们等于原始模型的激活。验证“仅目标直接置零”时应比较**同一次前向**各模块在消融 hook 前后的输出，同时另行验证移除 hook 后的完整前向恢复。

### 预测位置 [复现决策]

第一轮使用一致的 prompt 模板，从输入末尾的 **next-token logits** 提取 $t^*$。先逐样本实现作为正确性基准，再考虑 batch 优化。不能在 right-padding 下盲目使用 `logits[:, -1, :]`；必须验证其确实对应有效预测位置。tokenizer、padding、截断、special token、模型 revision 全部记录。

**[待核验]** 论文对所有数据集的 prompt 模板、具体 token 位置等未在此前资料中完全明确，因此这个工程决策不等于作者代码的逐 token 实现。

### M4 数值、运行与审计协议 [复现决策]

- 原始 $t^*$ 在未消融的有效末尾 logits 上取 `argmax`；模型 logits 保持加载时的 dtype。概率以 **FP32 softmax** 计算，所有消融显式读取该原始 token 的概率，不能读取新的最大概率或对整个序列维做归一化。
- 逐样本先求两项绝对差，用 Python float / `math.fsum` 双精度累积后除以 $N$；不按层数归一化，不让有符号差抵消。分别保存 Attention/MLP 的平均绝对差及合计 $I_l$。
- 使用 Section 4.4 的 **accuracy mode**：候选为全部 24 层，按重要性降序取 Top-7。分数完全相等时按零基层索引升序；并列规则不是论文给定的作者实现。
- 仅支持冻结 CPU 基础模型的逐样本串行、未编译 PyTorch 前向，`batch_size=1`、`use_cache=False`、`eval()`、`inference_mode()`；CLI 固定 inference seed=20261114、4 个 CPU intra-op 线程和确定性算法。M2 的抽样 seed=20261112 不变；这两个 seed 含义不同。这里的 eager 执行不要求 Attention backend 名为 `eager`，实际 backend 单独记录（本机实测 `sdpa`）。
- baseline 前拒绝已有的本地 forward/pre-hooks；核对完整层描述符属于当前模型。每次消融作用域仅执行一次目标前向，恢复后再进行下一独立消融。全局 hooks、共享模型并发、编译和分布式调用不在已验证边界内。
- 每样本额外运行一次未消融前向，与该样本完整 baseline logits 做 `torch.equal` 恢复检查。评分需要 $N(1+2L)$ 次前向；恢复验收额外增加 $N$ 次，因此 $L=24,N=30$ 总计 **1,500 次**，其中 1,440 次消融、30 次 baseline、30 次恢复；额外恢复不进入重要性公式。
- 整次运行前后比较全部 `named_parameters()` 的名称、dtype、shape 和原始字节 SHA-256，并检查冻结/无梯度/hook 状态。指纹不覆盖 buffers，也不是密码学来源认证。
- CLI 默认绑定已验收 M2 的内容指纹；重新核对整个清单、原生 tokenizer 配置和已保存 prompt 的分词 IDs/mask/位置。不会重新抽样或把答案加入输入；不重新读取原 parquet，来源和 user-only 构造仍依赖 M2 的独立验收。`--limit 1` 仅取固定清单首条做门禁，不改变原 30 条清单。
- 原始逐样本概率及时写 JSONL，最终导出完整 JSON、24 层 CSV/JSON、全层排名与 Top-7、SVG/离线 HTML 图表。每次使用新结果目录，成功标记最后写入；失败保留部分轨迹但不视为成功。配置、环境、Git 基线/dirty 与源码字节指纹及分项 wall-clock 耗时一并保存；不把 CPU smoke 的计时当作论文性能复现。
- 保存后另用不导入校准器/选层器的脚本从原始概率复算公式、排名及多种文件的一致性。真实 N=1 与 N=30 首样本的比较只用于有限的重复性核验，不能冒充完整 N=30 重跑；N=100、不同采样 seed 的 Jaccard 稳定性及正式结果仍须另行执行。

## 3. 校准数据

**[论文]** 附录 E.2 描述从训练集抽取 $N$ 个 calibration 样本；Section 4.3 和附录 E.4 明确使用冻结基础模型预测 token，因此不需要 gold-answer 标注。

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

### M2 smoke 数据协议 [复现决策]

以下仅冻结本项目的流程验证协议，不代表作者逐 token 的原始实现或正式主表数据：

- 数据：`tatsu-lab/alpaca`，train split，revision `dce01c9b08f87459cf36a430d809084718273017`；固定单个 parquet 的文件名、字节 SHA-256 与原始行顺序，避免浮动分支和自动数据脚本。
- 抽样：`random.Random(20261112).sample(range(population_size), 30)`，不放回、保留抽取顺序、索引零基。Python 版本也是复现条件，不能只固定 seed。
- 模板：固定 M1 的 Qwen tokenizer revision；传入一条 user 消息，内容为首尾去空白后的 instruction，input 非空时以两个换行拼接。使用 tokenizer 自带 chat template 和 `add_generation_prompt=True`，允许该模板加入默认 system；不传入答案。
- 特殊 token：先渲染模板文本，再以 `add_special_tokens=False` 分词；与原生 `apply_chat_template(tokenize=True)` 逐样本核对，保存模板原文与哈希。
- 长度：max_length=512，逐样本无 padding；超长则整次报错，**不静默截掉 assistant 起始标记、不重抽样**。通用分词辅助函数仍支持截断和左右 padding，其方向取实际 tokenizer 配置并记录。
- 位置：每条输入最后一个有效 token 的零基索引，用于该位置的 next-token logits；M2 不执行模型预测，也不产生 $t^*$ 或层重要性。
- 清单：原始整行哈希可包含答案字段，只用于来源追踪；模型输入仅来自 instruction/input。保存输入字段哈希、完整 prompt 哈希、最终 IDs/mask/有效位置和配置指纹。原始输入自身是否含答案仍需审查，哈希不是脱敏机制。
- 本次数据经 HTTPS 镜像传输并核对其固定 revision/LFS SHA-256；官方域名连通性和原作者版本一致性仍属 [待核验]。不得把“镜像字节哈希匹配”表述成独立验证了官方来源。

## 4. 训练参考配置 [论文]

M5 已按需核查 PDF 第 7 页 Section 4.5 / 5.1、第 18–19 页 Appendix C.1–C.2。正式运行前仍需核对实验所属配置；Section 4.5 说明投影目标随实验变化，Section 5.1 的主比较明确使用 Q/K/V，与本项目要求一致，不能因其他 Q/V 分析擅自改目标。

| 项目 | Qwen2.5-1.5B 主实验参考 |
|---|---|
| LoRA | rank=16，alpha=32，target Q/K/V |
| Optimizer | AdamW，lr=1e-4，weight_decay=0.01 |
| Betas / eps | (0.9, 0.999) / 1e-8 |
| Schedule | 3 epochs，无 warmup |
| Batch | 每卡 8，gradient accumulation 8；单卡有效 batch 64，4 卡 DDP 合计 256（Appendix C.2） |
| Precision | FP16 AMP；adapter 初始化为 FP32、前向/反向在 FP16 下计算，基础权重冻结于 FP16 |
| Length / preprocessing | 最大 512 tokens；超长截断，短样本 padding 到 8 的倍数 |
| Seed | 42，用于 PyTorch / NumPy / Python random（**不能视为论文强制的校准抽样 seed**） |

尚待明确：数据集准确版本与划分、逐 token 预处理/截断方向、完整 prompt 模板、训练 loss mask、模型权重 revision、评估实现。Appendix C.2 已明确不同 GPU 数下的 batch 数量，但正式对照仍需统一实际有效 batch，不能把 4 卡的 256 当作单卡 64 的等价训练。所有差异都需记录。

### M5 选择性 LoRA 与短训练协议 [复现决策]

- 只复用已验收 M4 **N=30** 的 Top-7 `[0, 1, 2, 5, 3, 8, 10]`，不重新校准，不使用 N=1 的另一组排名。默认绑定选层 JSON 的字节 SHA-256，并核对 N30 完成标记、模型版本及全部已记录产物哈希；哈希不是来源签名。
- 使用实际发现器的完整 Q/K/V 路径构造锚定正则，注入后按模块和参数对象身份核对。rank=16、alpha=32，标准 PEFT `init_lora_weights=True`、B=0、dropout=0、bias=none；不启用其他 LoRA 变体或额外可训练 embedding/MLP/输出投影。dropout 与初始化选择不是论文明确给出的作者实现。
- 基础参数全冻结，但训练必须开启 autograd 并调用 `train()`，允许梯度经过冻结的下游运算流向 LoRA；不得沿用校准的全局 no-grad/inference_mode。保留注入前基础参数的规范名称与对象引用，前后按同一名称/字节协议比较，避免 PEFT 重命名造成伪差异；指纹不覆盖 buffers。
- 当前仅为 **CPU 工程 smoke**：保留缓存基础权重 BF16、PEFT adapter FP32、cross entropy FP32、不使用 AMP；不同于论文 FP16 AMP。固定 seed=42（Python/NumPy/PyTorch）、4 个 CPU 线程和确定性算法，关闭 KV cache。不转基础 dtype，以便延续 M3/M4 的字节指纹。
- 使用仓库内定义的 3 条极短合成监督样本，固定顺序、每 batch 1 条，最多 3 个 optimizer steps；不取正式训练/评估数据，也不把答案加回校准输入。最大 96 tokens（含 EOS），不 padding、不截断、不拼接，超长明确拒绝；这与论文 512-token 预处理不同。
- prefix 为 M2 相同 tokenizer 的 user-only generation prompt；先验证 `tokenize(prefix)` 是 `tokenize(prefix+answer)` 的未变前缀，再追加明确的 `tokenizer.eos_token_id`。本机 EOS 为 `<|endoftext|>` / 151643，不冒充 `<|im_end|>` 或作者回答模板。prompt labels=-100，监督回答与 EOS；第一回答的 `labels[P]` 由 `logits[P-1]` 预测。
- AdamW 的 lr=1e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8 沿用论文参考；本 smoke 不做 accumulation、scheduler、warmup、梯度裁剪或 prefix cache。要求梯度全部存在且有限、整体非零，adapter 实际更新；标准 B=0 下首步 A 梯度为零是正常现象，不能要求每个 A/B 都非零。
- checkpoint 只含 adapter safetensors 和配置/审计元数据，显式禁止 embedding 自动保存；使用新目录，拒绝覆盖。再次从同一固定本地 revision 加载全新基础模型，核对基础字节和全部 adapter tensor，再在 eval/no-grad 下比较 3 条 user-only prefix 的完整 logits，要求精确一致。**此验收不包含 optimizer/RNG/scheduler 的训练断点续跑**。
- CLI 保存配置、合成样本 ID/IDs/labels 与边界、参数统计、每步有限 loss/梯度/更新摘要、环境、Git 基线/dirty、源码字节、初始化/恢复探针哈希和日志；最后才写成功标记。异常清理梯度并回到 eval，部分训练不回滚但不得视为成功。M5 不证明收敛、下游收益或论文速度，也不授权 M6/M7。

## 5. 对照实验

- **Full LoRA**：所有层的 Q/K/V 添加 rank=16 LoRA。
- **Random-K LoRA**：随机选 K 层，仅这些层添加同规格 LoRA。
- **Adaptive-K LoRA**：按 $I_l$ 选 K 层，仅这些层添加同规格 LoRA。

统一训练集、评估集、预处理、seed、优化器与评估程序。**注意 Full 与 K-layer 的可训练参数量不同**；分别报告效果、实际可训练参数数量、显存和耗时，不能声称它们“同参数预算”。

论文主表参考数值（Qwen2.5-1.5B，rank16，commonsense average）：Full LoRA 56.9%、Adaptive-LoRA 59.1%、LoRA+ 57.9%、DoRA 55.5%。这些不是本项目结果。

M6 按需核查 **Table 2 / Section 5.3（PDF 第 8 页）、Section 5.4（第 9 页）和 Appendix C.3（第 19 页）**：Table 2 的 0.5B 比较为 Full24 / Adaptive7、rank16、N30，但训练是 10,000 条、3 epochs。Section 5.4 的 Random 是均匀选层，同超参数比较使用 Alpaca-ZH test perplexity、3,000 条训练样本，不能与主表准确率或本工程 smoke 混同。Appendix C.3 规定无 in-context examples、greedy temperature=0；commonsense 使用答案抽取、math 使用 CoT 与任务专用解析，不是通用字符串 exact match。

### M6 统一基线与评估协议 [复现决策]

- 复用最终 M4 N30 重要性排名的 Adaptive7，Full 使用全部24层，Random7 用局部 `random.Random(layer_seed).sample(range(24), 7)` 均匀不放回；默认选层 seed=20261115。保存抽样顺序、升序层ID和零基约定，Python版本也是复现条件。选层 RNG 与训练 RNG 分离，不重新校准，不扩展 Sequential/Reverse 等额外实验。
- 三方法各自从固定本地 revision **重新加载全新基础模型**；在注入前核对与 M4 相同的全部规范参数字节指纹。共享同一不可变 `SharedConfig` 和一次构建的完整数据协议；每方法注入前、优化前分别重置 Python/NumPy/PyTorch 的 train_seed=42，CPU线程数和确定性开关一致。目标数量不同会消耗不同初始化随机数，因此不承诺跨层集合的同名 A 初始化逐层相同；不将同 seed 误称为逐投影配对初始化。
- 同用 M5 的 Q/K/V LoRA 与 AdamW 短训练器，rank16/alpha32、dropout0、bias=none、基础 BF16、adapter/loss FP32、无AMP；每方法恰好3个短合成 batch，batch1、无 accumulation/scheduler/warmup/clipping。CLI 仅允许1–3步、CPU；不支持 epochs、GPU、下载或正式数据训练开关。训练长度上限96、无padding/截断，仍沿用 M5 的答案/EOS loss mask 与 causal shift。
- 固定合成训练3条、独立合成评估3条，均使用同一 user 消息与 tokenizer 原生模板（含模板默认system）。M4选层来源绑定其实际 M2 manifest 的文件和内容哈希，并逐方法重新核对30条保存 prompt 的分词。校准可以来自训练集，这符合论文；但评估必须与训练/校准在 **sample_id、prompt字节哈希、实际prefix token列表哈希** 三个身份域均无交集，token哈希自行重算。计数是唯一标识/哈希交集，不是样本对数量；相同参考答案不视为输入泄漏。
- 生成接口不接收 reference/labels，只输入单样本、无padding的 prefix，再追加模型自己的预测；在 `eval()` / no-grad 下对 `logits[0,-1]` 做 argmax，实现 temperature=0。无KV cache、无采样/beam search；默认 max_new_tokens=4，硬限制1–8，仅以显式 `tokenizer.eos_token_id` 停止。逐条保存生成token、解码、停止原因和实际prefix指纹，正常/异常均恢复各子模块原training flag；拒绝已有梯度或本地forward/pre-hooks。
- 参考答案仅在预测完成后按唯一ID对齐评分；固定 `skip_special_tokens=True`、`clean_up_tokenization_spaces=False`，比较两端 `strip()` 后的字符串，保留大小写和内部空白。指标名为 **`smoke_exact_match_strip_only_not_paper_accuracy`**，不是论文答案抽取、CoT/regex 或 Alpaca-ZH PPL。合成问题自身可含目标词，只验证输入/评分路径，不构成无答案暗示的真实能力基准。
- 每方法记录实际投影/参数白名单、有限loss/梯度/更新、基础参数不变，以及评估前后全部基础+adapter参数字节相同。Full实测1,474,560参数，Random7/Adaptive7各430,080（Full的29.1667%），**不是同参数预算**；不调低Full rank以伪造同预算。只报告CPU wall-clock（含审计/哈希/IO），不测GPU显存、不据此宣称论文速度或效果。
- 各方法独立保存 adapter-only safetensors，复用 M5 已验收保存器；M6 不额外验证三方法 fresh-base恢复或 optimizer/RNG断点续跑。仅在三方法实际数据/配置/基础/评估身份一致、输入/源码未变后写整体完成标记；失败保留部分结果，不自动重试或覆盖。配置、seed/IDs、源码/依赖、预测/评分和JSON/CSV对照均保存在新目录。
- 另以不导入项目训练/评估/协议模块的脚本，独立核对存档来源、输入身份/拆分、评分和真实safetensors形状/数量/两种命名域指纹。该核验不重新执行前向/反向，不独立重建梯度；直接输入交集为零不代表排除了语义改写、原始题面内嵌答案或预训练污染。正式数据/模板/答案抽取和成本仍需用户确认，M6完成后不自动进入M7。

### M7 首轮协议核查（2026-11-15）

本轮已重新核查PDF第7–9页Section 4.5/5.1、Table 1及Section 5.4，第18–19页Appendix C.1–C.3，以及参考文献[14]。首轮选择Table 1的1.5B/28层/N250/Top8/rank16，而非Table 2的N30；用户已确认 [M7_PROTOCOL.md](M7_PROTOCOL.md) 中7项复现决定，机器可读配置见 [配置JSON](configs/qwen2_5_1_5b_commonsense_r16.json)。

- **[论文]** 训练10,000条commonsense、3 epochs、max512、pad到8倍数，AdamW lr1e-4/weight_decay0.01/betas(0.9,0.999)/eps1e-8，无warmup；Q/K/V LoRA rank16/alpha32；冻结FP16基础权重、FP32 adapter初始化及FP16 AMP；每卡batch8、accumulation8。单卡64不同于4卡DDP256。
- **[复现决策，用户已确认]** 第一轮仅做8项commonsense的Full/Random8/Adaptive8；规范化Alpaca纯文本前缀，不使用M2–M6的Qwen chat/system模板；只监督答案+显式EOS，完整序列右截断512，无可预测监督token即报错，不静默丢弃或重抽。
- **[复现决策，用户已确认]** 固定LLM-Adapters提交并分别以局部seed42抽10,000训练、其内250校准、各任务1,000评估（OBQA500）；实际索引与输入身份尚待下载后验收。来源路径名test.json不自动证明与原始benchmark官方test split同一。
- **[复现决策，用户已确认]** greedy最多32新tokens，仅解析生成续文；合法标签大小写归一化及词边界匹配，多个不同合法标签冲突判无效，失败保留分母，8任务等权宏平均。引用库无边界首匹配/4 beams并非论文明确协议，不照搬。
- **[复现决策，用户已确认]** 候选Qwen1.5B固定revision/LFS哈希来自镜像，保留官方端点不可达、未独立认证官方来源的局限；权重未下载，实际字节仍待核对。所选数据仓库commit不是论文公开的原始commit。
- **[工程状态，数据文本层已通过]** 九个固定JSON已按长度+Git blob+SHA256全部核验，并实际运行离线审计：训练170,420行抽10,000、校准250、八任务各1,000（OBQA500），字段统一为`answer,input,instruction,output`；跨拆分`sample_id`与规范prompt哈希交集均为0，组内重复prompt（train 107、social_i_qa 8）仅记录未处理。状态`TEXT_GATE_PASS_TOKENIZER_PENDING`，见[数据门禁通过报告](reports/m7_data_gate_pass_20261115.md)。**不含**真实BPE/512截断/token级交集、语义去重与预训练污染检查。
- **[复现决策，用户已放宽]** 模型许可前置门禁改以API元数据（`license: apache-2.0`、`gated: false`、`private: false`）作为许可依据，放弃"必须先取得固定LICENSE原文字节"的原条件；固定许可字节（预期11,343 bytes、blob `6634c8cc…`）**始终未取得**，该证据缺口如实保留，不声称已核验许可原文，也不声称其内容被篡改。此放宽仅解阻权重获取的许可前置，不等于授权下载、付费GPU或正式训练。
- **[待核验]** 权重字节/1.5B真实模块返回值、CUDA/AMP/梯度累积与GPU checkpoint/评估尚未验收；权重未下载，LFS SHA256未核对。新协议检查器及文本审计器仅标准库离线审查，不是正式训练入口；confirmed字段不提供推送/安装/下载/付费计算授权。
- **[工程状态，已通过适用Linux CPU验收]** 原候选torch2.6.0+cu124搭配Transformers5.22.2/PEFT0.21.2等11项版本已在独立Linux环境离线安装、通过pip check和导入；Linux183项CPU测试通过，2项真实tokenizer用例因缺缓存NOT RUN。原[候选直接依赖](requirements-linux-cu124-m7.in)字节未改，仍不是完整传递依赖锁；实际67项wheel官方哈希锁与freeze见[环境CPU报告](reports/m7_environment_cpu_20261115.md)所列私有证据。此状态不证明CUDA/AMP、真实1.5B模块或正式训练已通过。

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
