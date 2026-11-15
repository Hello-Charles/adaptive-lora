# M5 选择性 LoRA 与训练 smoke 验收（2026-11-15）

## 当前任务与完成内容

**M5 三项门禁全部通过。** 本阶段仅实现用户要求的选择性 LoRA、1–3 batch 工程训练和 checkpoint 恢复；先写核心测试，再实现，微型模型门禁通过后才运行固定本地 Qwen2.5-0.5B。没有进入 M6、正式微调或下游评估。

### 真实运行摘要

| 项目 | 实际结果 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B`，24 层，`Qwen2ForCausalLM` |
| requested revision | `060db6499f32faf8b98477b0a26969ef7d8b9987`，仅使用已有本地缓存 |
| resolved revision | 模型 config 未暴露，记录为 null，不编造作者版本 |
| 选层来源 | 已验收 M4 **N30**，不是 N1 的另一组 Top-7 |
| Top-7（重要性降序、零基） | **`[0, 1, 2, 5, 3, 8, 10]`** |
| 结构定位顺序 | `[0, 1, 2, 3, 5, 8, 10]` |
| LoRA | 标准 PEFT、rank16、alpha32、dropout0、bias=none、标准 A 随机/B=0 初始化 |
| 注入目标 | **21 个 Q/K/V 投影**；未给其余层、MLP、o_proj、embedding 或 lm_head 添加训练参数 |
| 可训练参数 | **430,080 个元素、42 个 A/B 权重张量** |
| 基础参数 | **494,032,768 个元素**，冻结、无梯度，训练前后字节指纹与 M3/M4 一致 |
| 全部参数 | 494,462,848 个元素（基础 + LoRA） |
| 运行方式 | CPU、4 intra-op threads、seed42（Python/NumPy/PyTorch）、确定性算法、sdpa、禁用 KV cache |
| 实际 dtype | 基础 BF16、adapter FP32、cross entropy FP32，不使用 AMP |
| 训练数据 | 仓库内定义的 3 条极短合成监督样本；固定顺序，不使用正式训练/评估数据 |
| batch / steps | batch_size=1，gradient accumulation=1，恰好 **3 个 optimizer steps** |
| 长度 | 25、27、26 tokens（含回答和显式 EOS），无 padding/截断；2 个监督 token/条 |
| 初始化检查 | 3 条 user-only prefix 的完整 logits 在 LoRA 注入前后精确一致 |
| checkpoint | adapter-only safetensors，**1,725,816 bytes**，仅上述 42 个 A/B 张量 |
| 恢复检查 | 释放训练对象后，从同一固定本地缓存加载全新基础模型；adapter tensor 完整恢复，3 条完整 prefix logits 精确一致、最大绝对差均为 0 |
| 训练段耗时 | **4.632139 s**，包含训练内的指纹/审计，不是单纯 kernel benchmark |
| 整次耗时 | **18.103995 s**，包含配置/加载/审计/探针/训练/保存/恢复，不是论文性能对照 |
| 最终回归 | **123 tests，10.264 s，无跳过**（原 92 + M5 新增 31） |

### 三步数值

| Step | 合成样本 ID | Loss | 梯度 L2 norm | 有梯度张量 | 非零梯度张量 | 更新张量 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | synthetic-m5-v1/0 | 3.573270320892334 | 38.129387389080044 | 42 | 21 | 42 |
| 2 | synthetic-m5-v1/1 | 3.0456244945526123 | 29.238013465161263 | 42 | 42 | 42 |
| 3 | synthetic-m5-v1/2 | 3.504087209701538 | 190.8409594887316 | 42 | 42 | 42 |

全部 loss、梯度和更新后 adapter 权重有限。首步标准 B=0 导致 A 梯度为零是正常情况；AdamW 的 decoupled weight decay 仍能改变非零 A，所以“更新了 42 个张量”**不等于**“42 个梯度都非零”。验收不强制每个 A/B 初始梯度非零。

三个步骤来自不同样本，loss 不能作为同一样本训练曲线，亦不能据此声称收敛、准确率提升或复现了论文收益。

### 参数量和基础权重证据

实际 Q 为 896→896，K/V 各为 896→128；使用 Appendix C.1 的低秩因子参数量：

- Q：`16*(896+896) = 28,672` / 层。
- K/V：各 `16*(896+128) = 16,384` / 层。
- 每层：61,440；7 层合计 **430,080**。

基础规范参数名、dtype、shape 与原始字节 SHA-256：

`3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3`

与 M3/M4 一致。PEFT 会重命名参数路径，因此保存注入前的规范名与同一参数对象引用后比较，而不是把包装前后整模型名字的变化当作基础权重变化。该指纹不覆盖 buffers、不是 safetensors 文件哈希或来源签名。

M4 最终选层文件字节 SHA-256：

`0f5db68e4451ade4f92bc4353ca1c8c40226f9165f81a009ab129f962c640bdb`

## 文件变更

- [trainer.py](../src/adaptive_lora/trainer.py)：完整路径锚定正则注入、模块/参数身份白名单、GQA 参数统计、规范基础参数指纹、监督样本边界、显式 FP32 causal cross entropy、1–3 batch AdamW、有限梯度/更新验证及异常清理。
- [checkpoint.py](../src/adaptive_lora/checkpoint.py)：独占新目录、adapter-only safetensors、文件/张量指纹、固定 revision 和基础字节匹配、实际 PEFT fresh-base 恢复。没有 pickle 加载或自动基础模型下载。
- [train.py](../scripts/train.py)：M4 N30 完整性/模型绑定、离线固定模型、受限合成样本训练、初始化/恢复探针、配置/环境/Git/源码/参数/逐步日志、最终成功标记和失败保留。
- [test_lora_targets.py](../tests/test_lora_targets.py)：7 项，精确层号、非方形 K/V、初始化保持 logits、错误输入前置拒绝、冻结/额外参数/字节变更和 BF16/FP32。
- [test_trainer.py](../tests/test_trainer.py)：11 项，prefix/回答/EOS 边界、真实本地 tokenizer、mask/causal shift、三步真实微型反向、NaN 梯度注入、清理及相同 seed 的微型重复性。
- [test_checkpoint.py](../tests/test_checkpoint.py)：6 项，真实微型保存/全新实例恢复、独占目录、版本/基础字节、文件/配置/张量篡改负例。
- [test_train.py](../tests/test_train.py)：7 项，N30 来源、CLI 步数/seed/revision/缓存门禁、真实微型端到端、训练或恢复失败不得标为成功。
- [PAPER_SPEC.md](../PAPER_SPEC.md)、[README.md](../README.md)、[REPRO_PLAN.md](../REPRO_PLAN.md)：核查训练附录、记录本 smoke 偏差与真实验收结果；不改 M4 算法或历史产物。
- [m5_validation.json](m5_validation.json)：由真实输出/最终测试/独立审计生成的索引，不含完整 prompt、答案文本、token IDs 或权重。

全部新增源码/测试配中文注释和 docstring，说明冻结与 autograd 区别、causal shift、分词拼接、EOS、初始化零梯度、checkpoint 范围和 mock/真实模型边界。

此前 M2/M3/M4/注释维护的未提交工作全部保留。当前分支 `repro/m5-selective-lora-smoke`，基线 commit `577527ad0ec343941bbed0b6519ff3544e301cc7` 仍是 **M1 提交，不是 M5 提交**；运行时记录 dirty 和 30 个实际源码字节哈希。未自动提交或推送，未下载模型/数据、未安装或升级依赖、未使用付费 GPU。

## 论文依据与工程假设

按需从本地 PDF 读取，而非把规格文档当作原文：

- **Section 4.5（PDF 第 7 页）**：选层低秩增量 `ΔW=BA`、不等输入/输出维度、未选层冻结；AdamW lr1e-4、rank16、alpha32、3 epochs。该节说明 projection targets 随实验变化，不能把某个 Q/V 分析强制替代项目的 Q/K/V。
- **Section 5.1（第 7 页）**：主比较 Qwen2.5-1.5B、28 层，Q/K/V、N250、Top8；不是本阶段 0.5B/Top7/合成数据的结果。
- **Appendix C.1（第 18 页）**：PyTorch/Transformers/PEFT，adapter 参数量为逐目标的 `r*(d_in+d_out)` 之和。未把本文的当前依赖版本声称为作者版本。
- **Appendix C.2（第 18–19 页）**：seed42、AdamW weight_decay0.01、betas=(0.9,0.999)、eps1e-8、无 warmup、3 epochs、每卡8并累积8；单卡有效64，4卡 DDP256；FP16 AMP、FP32 adapter 初始化、FP16 冻结基础权重；max512、超长截断、短样本 pad 到8的倍数。

| 维度 | 本阶段实际设置 / 定性 |
|---|---|
| 选层 | 复用已验收的 M4 N30 / Top7，不重新计算；0.5B smoke 是 **[复现决策]** |
| 初始化 / dropout | 标准 PEFT A 随机/B=0，dropout0，bias=none；论文未明确作者这部分完整实现，属 **[复现决策]** |
| 精度 | BF16 冻结基础权重 + FP32 adapter / loss，不使用 AMP；**不同于论文 FP16 AMP** |
| 数据 / batch | 3 条仓库内合成数据、batch1、accumulation1、3 steps；不是正式训练或论文 3 epochs |
| 序列协议 | 同一 user-only prefix + 回答 + 显式 tokenizer EOS；max96、无 padding/截断，超长拒绝；不是作者模板 |
| loss | prompt=-100，回答与 EOS 受监督，显式 shift；**作者完整 loss mask 仍待核验** |
| EOS | `<|endoftext|>` / 151643，不声称等于 `<|im_end|>` |
| 优化器 | lr/weight_decay/betas/eps 沿用已核查论文参考；本 smoke 无 scheduler、梯度裁剪、AMP 或 prefix cache |
| 恢复 | adapter 权重与推理输出，**不含 optimizer/RNG/scheduler 的断点续跑** |

正式结果之前仍需用户确认数据版本/抽样、完整训练模板、loss mask/截断/拼接、评估方法与成本。以上 smoke 决策不自动升级为正式实验协议。

## 执行命令

命令从项目根目录运行；读取模型/数据用 `-I`，UTF-8 输出用 `-X utf8`。M5 从 2026-11-14 开始准备、2026-11-15 完成；证据缓存沿用起始日目录，最终结果目录使用完成日。

```bash
# 初始环境/源码基线，不切换映射，不安装依赖。
git status --short --branch
git log -1 --format='%H %s'
git switch -c repro/m5-selective-lora-smoke
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v

# 按需提取论文训练章节；第17–18页摘录中第18页为 Appendix C。
pdftotext -enc UTF-8 -f 7 -l 7 papers/adaptive_lora.pdf .cache/m5-20261114/paper-training-main.txt
pdftotext -enc UTF-8 -f 17 -l 18 -layout papers/adaptive_lora.pdf .cache/m5-20261114/paper-implementation-details.txt
pdftotext -enc UTF-8 -f 19 -l 19 -layout papers/adaptive_lora.pdf .cache/m5-20261114/paper-implementation-details-page19.txt

# M4 输入复核，不运行模型，不重新校准。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/validate_calibration.py \
  --run-dir results/m4-20261114-n30 --expected-n 30 \
  --compare-prefix-run results/m4-20261114-n1 \
  --output .cache/m5-20261114/m4-input-audit.json

# 四组测试均先红后绿；这些命令按实现进度多次执行，日志分别保留，不覆盖。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -p test_lora_targets.py -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -p test_trainer.py -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -p test_checkpoint.py -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -p test_train.py -v

# 真实训练：仅此一次 0.5B 三步运行。目录已存在，复核时应换新目录而非覆盖。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/Scripts/python.exe -I -X utf8 scripts/train.py \
  --m4-dir results/m4-20261114-n30 --steps 3 --seed 42 --threads 4 \
  --output-dir results/m5-20261115-smoke

# 独立存档一致性核验：不导入 trainer/checkpoint，不加载基础模型、不重新训练。
.venv/Scripts/python.exe -I -X utf8 .cache/m5-20261114/audit_m5.py \
  --run-dir results/m5-20261115-smoke --output results/m5-20261115-evidence/audit.json

# 最终回归、编译、依赖和 Git 检查。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python.exe -I -X utf8 -m compileall -q src scripts tests
.venv/Scripts/python.exe -I -X utf8 -m pip check
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/train.py --help
git diff --check
git ls-files data results .cache .venv papers
```

PEFT API 和真实 Qwen 投影维度另用只读 Python 检查并保存日志；它们不被当作训练执行证据。环境实际为 Python3.12.8、torch2.15.0+cpu、transformers5.22.2、peft0.21.2、accelerate1.15.0、safetensors0.8.0、numpy2.5.4，详见运行环境 JSON。

## 测试结果（PASS / FAIL / NOT RUN）

| 检查 | 状态 | 证据 |
|---|---|---|
| M5 开始时旧测试 | PASS | 92 tests，14.268 s |
| 4 组先行红灯 | FAIL（预期） | 实现前分别缺 trainer 模块、监督 API、checkpoint 模块或 train.py；原始日志保留，非最终失败 |
| LoRA targets / trainer / checkpoint / CLI | PASS | 分别 7 / 11 / 6 / 7 项，均无跳过；不是只 mock 模型前向 |
| 真实运行前完整回归 | PASS | 123 tests，13.302 s |
| 真实 Qwen 三步训练 | PASS | 3 个有限 loss、全部梯度存在/有限、整体非零、adapter 更新、基础参数无梯度且字节未变 |
| 初始化与 fresh-base 恢复 | PASS | 各 3 条 user-only prefix，完整 logits 精确一致；真实恢复模型全部冻结 |
| safetensors / 独立存档一致性 | PASS | 42 个 A/B、430080 元素；18 个 M5 产物、13 个 M4 产物、30 个源码哈希匹配 |
| 最终回归 | PASS | **123 tests，10.264 s，无跳过** |
| compileall / pip check / CLI help | PASS | 正常退出；pip 无损坏依赖 |
| Git diff / 大文件跟踪边界 | PASS | 无空白错误；CRLF 提示非失败；data/results/.cache/.venv/papers 没有已跟踪文件 |
| 真实 0.5B 三步完整重跑 | NOT RUN | 相同 seed 的精确重复只在微型测试验证，不能冒充真实模型重复运行 |
| optimizer/RNG 断点续跑 | NOT RUN | 本阶段 checkpoint 仅验证 adapter 推理恢复 |
| 正式数据收敛/效果、Full/Random 对照、M6/M7 | NOT RUN | 未获本轮范围授权，不自动执行 |

独立存档审计只从保存记录和实际 adapter tensor 检查一致性，不能从日志重建梯度或再次证明真实前向；真实执行依据仍是 CLI 运行、运行时代码门禁、原始日志和测试。内部哈希不是来源认证。

## 运行日志和输出路径

- [真实运行目录](../results/m5-20261115-smoke/)：resolved_config、environment、provenance、git_commit、training_samples/sample_ids、trainable_parameters、post_training_audit、steps.jsonl、training_metrics、model_runtime、initialization_check、reload_check/reload_audit、run_summary 和 run.log。
- [adapter checkpoint](../results/m5-20261115-smoke/checkpoint/)：adapter_config.json、adapter_model.safetensors、README.md、checkpoint.json；没有基础权重、optimizer 或 pickle。
- [训练指标](../results/m5-20261115-smoke/training_metrics.json)、[运行日志](../results/m5-20261115-smoke/run.log)、[恢复探针](../results/m5-20261115-smoke/reload_check.json)、[完成标记](../results/m5-20261115-smoke/run_summary.json)。
- [验收证据目录](../results/m5-20261115-evidence/)：独立 audit.json、红/绿/完整测试日志、真实 CLI stdout、编译/依赖/Git 检查、PEFT API 与投影检查、论文相关摘录及审计程序副本。
- [机器可读索引](m5_validation.json)：记录上述产物、证据和源码 SHA-256；不发布输入文本、完整 IDs 或权重。

data、results、checkpoint、缓存、虚拟环境和 PDF 均被 Git 忽略，未将这些本地产物加入索引。公开发布前仍须检查实际待提交内容；忽略规则不能保护已被跟踪的敏感文件。

## 未解决问题

1. 正式数据版本/索引、作者完整 prompt、训练 loss mask/样本拼接与逐 token 截断/答案协议、评估实现仍待核验；本 smoke 的合成样本和显式 EOS 只服务工程门禁。
2. 真实模型未做完整相同 seed 重跑；确定性设置和微型重复测试不构成跨设备/依赖版本的数值保证。
3. 未验证 optimizer/RNG/scheduler 的训练恢复；当前 checkpoint 恢复范围已明确限制。
4. 不提供收敛、下游准确率、GPU 显存或论文速度结论。CPU 短序列计时与作者 A100/FP16 配置不可直接比较。
5. 当前源码处于 dirty 工作树，HEAD 仍是 M1。未创建 M5 commit 或公开推送，源码身份由运行时哈希补充。

## 下一步建议（不自动执行）

等待用户验收和另行指令后进入 **M6：统一 Full LoRA / Random-K / Adaptive-K 训练与评估接口**。先统一正式数据和评估协议、可控超参数及数据隔离，再讨论成本和实验规模；Full 与 K-layer 的参数量不同，不能宣称同参数预算。M7 正式训练、GPU/数据成本和公开推送仍需单独授权。
