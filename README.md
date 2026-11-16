# Adaptive-LoRA 复现工程

按阶段复现论文 *Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching*。算法事实与工程决策分别标注在 [PAPER_SPEC.md](PAPER_SPEC.md)，阶段门禁见 [REPRO_PLAN.md](REPRO_PLAN.md)，协作规范见 [CLAUDE.md](CLAUDE.md)。论文原文可从 [arXiv:2610.34756](https://arxiv.org/abs/2610.34756) 查阅；本地 `papers/adaptive_lora.pdf` 不上传至公开仓库。

## 当前范围：M0–M6 已验收，M7 准备中

M0 的工程骨架与环境、M1 的真实模型结构、M2 的固定校准输入和 M3 的独立子模块 zero-ablation 均已验收。M4 在30条输入上完成固定token校准、24层重要性和Top-7，最终零基排名为 **`[0, 1, 2, 5, 3, 8, 10]`**；M5完成选择性Q/K/V LoRA的真实三步CPU训练与fresh-base checkpoint恢复。M6已完成 **Full / Random7 / Adaptive7统一训练与评估接口**：每方法从相同基础权重新加载，统一数据/配置各训练3个短合成batch，共9步；实际可训练参数为 **1,474,560 / 430,080 / 430,080**，非同参数预算。最终 **162个测试通过**，三方法合成smoke exact-match均0/3，不能当作正式任务准确率。M7已开始首项协议核查并由用户确认，尚未执行GPU pilot或正式训练/效果实验。证据见 [阶段门禁](REPRO_PLAN.md)、[M1](reports/m1_validation.md)、[M2](reports/m2_validation.md)、[M3](reports/m3_validation.md)、[M4](reports/m4_validation.md)、[M5](reports/m5_validation.md) 和 [M6验收报告](reports/m6_validation.md)。

## M7：协议已确认，尚未执行 GPU 实验

用户已确认第一轮采用Qwen2.5-1.5B、N250、Top8、rank16/alpha32，比较Full/Random8/Adaptive8的8项commonsense任务。固定模型/数据来源、Alpaca纯文本前缀、答案loss mask、抽样、解析和单卡batch差异见 [M7协议](M7_PROTOCOL.md) 与 [配置JSON](configs/qwen2_5_1_5b_commonsense_r16.json)。这些复现决定不是论文未披露细节的还原声明。

[离线检查器](scripts/check_m7_protocol.py) 和 [协议/评分模块](src/adaptive_lora/m7_protocol.py) 只使用标准库，可在未安装PyTorch等项目依赖时运行。它不是CUDA训练入口；已确认配置返回REVIEWED_NOT_EXECUTABLE，始终execution_authorized=false、runtime_validated=false。使用新输出目录：

```bash
.venv/Scripts/python.exe -I -S -B -X utf8 scripts/check_m7_protocol.py \
  --config configs/qwen2_5_1_5b_commonsense_r16.json \
  --output-dir results/m7-protocol-review-new
```

[Linux/CUDA候选直接依赖](requirements-linux-cu124-m7.in) 的固定版本现已在独立Linux环境安装并通过适用CPU验收；该.in仍不是完整传递依赖锁，其初始“未安装”注释保留原批准字节。实际67项wheel哈希锁、安装报告和Linux freeze另存私有证据。协议核查结果见 [M7首项报告](reports/m7_preflight_20261115.md)。

2026-11-15最新环境状态：另获一次最多30分钟的缓存续接授权后，**118.766秒内完成全67个缓存wheel官方核验、离线依赖/项目安装、pip check、固定版本导入与适用Linux CPU验收**，新下载wheel为0。Linux单测 **183项通过、2项真实tokenizer测试NOT RUN**（缺少固定缓存，未下载）；本地Windows185项回归通过。范围状态为 **PASS_APPLICABLE_CPU，不是185项Linux全通过、CUDA通过或M7完成**。最新证据见 [CPU环境验收报告](reports/m7_environment_cpu_20261115.md) 和 [机器索引](reports/m7_environment_cpu_20261115.json)。

环境仓库保留已发布完整SHA `9051fdfe86379aa3502e66bf1a586d4eb00565ba`，见 [repro/m7-run-preflight](https://github.com/Hello-Charles/adaptive-lora/tree/repro/m7-run-preflight)；未用新的数据审计代码替换其editable安装。此前官方源超时、核验器vendor误判及工具拒绝记录保留在 [历史恢复报告](reports/m7_environment_resume_20261115.md)，旧截止未重置、未绕过权限或覆盖失败证据。[首轮SSH阻塞](reports/m7_environment_20261115.md)和[较早盘点](reports/autodl_postrestart_20261115.md)也保留历史。

2026-11-15数据门禁续轮：用户另行批准一次新的9000秒/120,000,000 bytes窗口，并明确批准在SSH别名 `autodl` 执行。复用已部署完整SHA `2df16eb…`、只读复核两份既有许可、跳过README，从上一窗口13,631,488字节部分文件续传（先核对已记录摘要，再用远端尾部4,096字节@offset 13,627,392核验对齐）。**实际1,320.791秒完成九个固定文件的长度+Git blob+SHA256三重核验并运行离线审计**，状态 `TEXT_GATE_PASS_TOKENIZER_PENDING`。实测行数：训练170,420抽10,000、校准250、八任务各1,000（OBQA 500），字段统一为 `answer,input,instruction,output`；**跨拆分 sample_id 与规范prompt SHA256 交集均为0**。组内重复prompt如实记录未处理（train 107、social_i_qa 8）。这**不是**完整数据验收：真实tokenizer/512截断/padding/token级交集、语义去重与预训练污染均未检查，ODC-By只覆盖数据库层。详见[数据门禁通过报告](reports/m7_data_gate_pass_20261115.md)与[机器索引](reports/m7_data_gate_pass_20261115.json)。

2026-11-16模型门禁通过：另获批准的固定revision权重下载与单次CPU结构核验**实际完成**。1.5B权重3,087,467,144 bytes、LFS SHA256 `a961db72…c796` 双重核验通过（流式重算+远端blob独立复核）；pinned revision由HF快照目录名 `8faed761…4323` 实证；`config.json` 的Git blob SHA1 `169d606d…aaab` 与配置记录逐字一致（此前缺字节级证据，本轮补齐）。实测 **28层** `model.layers.0`–`27`，索引连续、Attention/MLP路径各28个互不重复、每层Q/K/V齐全；每层 `self_attn` 输出为 `(Tensor[1,4,1536], None)`、`mlp` 为 `Tensor[1,4,1536]`，dtype bfloat16，56个hook全部触发；单次CPU前向 logits `[1,4,151936]`、预测位置3。下载约194秒（约15.9 MB/s，由mtime推定），较上一轮数据窗口快两个数量级。状态 **`STRUCTURE_GATE_PASS_TOKENIZER_RUN_PENDING`**：真实1.5B tokenizer端到端（BPE边界/512截断/mask）、token级交集、CUDA/AMP/累积、pilot与正式训练/评估均**未运行**，**M7仍未完成**。GPU全程0 MiB、未训练、未push，权重与原始数据未传回本地。详见[结构门禁报告](reports/m7_structure_1_5b_20261116.md)与[机器索引](reports/m7_structure_1_5b_20261116.json)。

2026-11-16 tokenizer门禁通过：在已核验的1.5B快照上**实际端到端运行分词管线**，状态 **`TOKENIZATION_PASS_TOKEN_IDENTITY_DISJOINT`**。**10,000条训练样本全部构造成功、0报错**，BPE边界核验无一行失败；逐行校验mask不变量（prompt段labels=-100、监督段非负、pad段labels=-100且mask=0、长度均为8的倍数、未截断行监督段以EOS收尾）。训练侧从未触发512截断（完整序列 min57/p50 91/p95 307/max 468），监督段6–7 token；校准250条0截断且prediction_position均为末个有效token。**token级身份交集16个组合全部为空**（训练与校准分别对8任务），`sampling_sha256` 与上一轮文本门禁逐字相同、18,750条prompt身份0不符，证明分词的就是已审计的同一批行。附带结论：pinned 0.5B与pinned 1.5B的4个tokenizer文件**逐字节相同**（`tokenizer.json`/`tokenizer_config.json`/`vocab.json`/`merges.txt`，库版本亦同），而模型`config.json`有意不同作为对照，因此该比较具区分力；此结论**不**推广到权重或结构。`pad==eos==151643` 已由10,000行实测确认，有效性只能靠mask/labels判定。远端部署SHA `2df16eb…` 不含该模块（首次见于未推送的`45d6278`），故以base64内嵌已提交字节并校验SHA256后内存执行，未改远端repo、未push。CUDA/AMP/累积、pilot与正式训练/评估仍未运行。详见[tokenizer门禁报告](reports/m7_tokenization_1_5b_20261116.md)与[机器索引](reports/m7_tokenization_1_5b_20261116.json)。

2026-11-16协议修订（评估侧不截断）：上轮发现的1条piqa评估前缀521 token（`source_index=314`，instruction 2,255字符、input为空）超出512，右截断会丢掉`Answer format:`之后的`solution1/solution2`格式提示与`### Response:`标记。用户选择**评估侧不截断**，新增第8项确认决定 `evaluation_truncation = none_keep_full_prefix`；训练与校准仍右截断512，不丢样本、不重抽、不缩小评估分母。新增 `build_evaluation_input`（`truncated` 恒 False，超参考长度只标注不裁剪）与4项测试，决策数7→8同步，`M7_PROTOCOL.md` 第3节改“八项”并纠正已失效的哈希链说明。**关键结论：v1 从未真正截断评估前缀（其 `tokenize_ids` 本就 `truncation=False`，只统计超长条数），故修订前后训练/校准/评估长度分布与token级交集五类统计逐项相同，核验结论为 `IDENTICAL_NUMERIC_RESULT`——本次修订是把声明与既有行为一致化，并给出显式可测路径，不改变任何计算结果。** 配置字节哈希由 `6f715cef…` 变为 `4b0405b0…`、协议哈希由 `d60f1375…` 变为 `bcaf19a7…`，早先审计产物属历史证据、不再对应当前配置（本轮未重跑数据审计）。远端部署配置仍是修订前字节，故v2报告`config_sha256`如实为旧值。详见[协议修订报告](reports/m7_protocol_revision_20261116.md)与[机器索引](reports/m7_protocol_revision_20261116.json)。

2026-11-16训练执行层实现：新增 [m7_training.py](src/adaptive_lora/m7_training.py) 与 **28项测试**，填补"M7 精度/累积只有配置、没有可执行代码"的缺口（既有 `train_smoke` 明确不做 accumulation/AMP，且硬限制1–3步、batch=1、CPU）。三件事：**①设备解析不许静默回退**——`resolve_device("cuda")` 在 CUDA 不可用时直接报错，因为静默回退会让自称的 GPU 训练实际跑在 CPU 上、使耗时/显存/吞吐证据全部失真；**②AMP 在 CPU 上显式禁用并如实上报**（CUDA+FP16 启用 GradScaler，BF16 不需缩放，CPU 一律禁用），即使禁用也保留 device_type/dtype 以区分"请求了什么/实际是否生效"；**③梯度累积数学**——组内 loss 先除以该组实际 micro-batch 数再反向，使累积梯度等于"整组平均 loss"的梯度而非求和，且 unscale 之后才判断梯度量级，用真实梯度逐元素验证（误差 0.0）。**必须在正式训练前确定的一项事实**：10,000 条训练样本无法被有效 batch 64 整除（10,000 = 64×156 + 16），每 epoch 最后一组只有 2 个 micro-batch；`error`/`step`/`drop` 三种处置策略必须在任何前向之前选定，本实现默认 `error`（不静默少用样本、不静默改变有效 batch）。状态 **`ENGINE_IMPLEMENTED_GPU_NOT_RUN`**：**真实 CUDA/AMP 数值、真实 1.5B 前向与显存、pilot、正式训练/评估均未运行**，M7 仍未完成。全套 **256项回归通过**。详见[训练执行层报告](reports/m7_training_engine_20261116.md)与[机器索引](reports/m7_training_engine_20261116.json)。

许可依据仍为API元数据 `license: apache-2.0`，**固定LICENSE原文字节仍未取得**；配置中 `metadata_source` 保留 `mirror_only_not_independently_authenticated`（经 `hf-mirror.com` 下载，不宣称独立官方来源验证）。该放宽是项目工程决定，不构成法律结论。历史证据见[资源阻塞报告](reports/m7_assets_blocked_20261115.md)。

2026-11-15独立数据门禁：新增[标准库离线审计器](scripts/audit_m7_dataset.py)，固定Git blob/字段/seed42抽样/答案隔离与prompt文本交集，22项新增测试、本地全套207项通过。经单独批准，完整SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb`（连同自身共5个原未公开提交）已发布至[repro/m7-data-preflight](https://github.com/Hello-Charles/adaptive-lora/tree/repro/m7-data-preflight)，服务器在另一新目录从公开origin精确检出该SHA，原环境保留。固定DATA_LICENSE（ODC-By，**不覆盖每条内容的独立权利**）及代码LICENSE共31,186 bytes匹配；辅助README读超时，操作26.456秒停止，尚未发出9个JSON请求。获批跳过README的续接随后因原13:29:24.449360 UTC截止已过，在获取前拒绝；13:37 UTC只读核查九个JSON/audit/resume目录均不存在，源码与包元数据未变。**状态为BLOCKED_NETWORK_TIMEOUT_WINDOW_EXPIRED，实际行数/样本/文本隔离审计NOT RUN**，不是发现了数据重叠。详见[数据门禁报告](reports/m7_data_gate_20261115.md)与[机器索引](reports/m7_data_gate_20261115.json)。

2026-11-15独立数据门禁续轮：用户另行批准一次新的最多900秒/120,000,000 bytes窗口，并明确批准在SSH别名 `autodl` 执行。复用已部署完整SHA `2df16eb…`、只读复核两份既有许可、跳过README，在新目录启动。实际在**890.0009秒内只写入1个部分训练文件13,631,488/96,529,722 bytes**（0个完成文件），随后SIGALRM终止，**审计从未启动**；实测约15.3 KiB/s，按此速率完整九文件约需123分钟。旧结果字节、部署与环境SHA、平台153项及venv 69项元数据均未变，部分文件保留未删除。**状态为BLOCKED_TRANSFER_BUDGET_EXHAUSTED，九个JSON完整性/行数/抽样/文本交集仍全部NOT RUN**，不是hash不符或数据重叠。详见[续轮报告](reports/m7_data_gate_resume_20261115.md)与[机器索引](reports/m7_data_gate_resume_20261115.json)。本报告提交不自动push。

2026-11-16批组装层实现：新增 `collate_supervised_examples` 与 **7项测试**（全套 **263项通过**）。既有 `build_supervised_example` 只产出单条样本，而训练侧完整序列长度实测 57–468 不等，要拼成 micro-batch 8 就必须做 padding 与批内对齐。两个陷阱：**①pad 与 EOS 同为 151643**——若用 `input_ids != pad_token_id` 推断有效位置，每个真实 EOS 都会被判成 padding、label 置 -100、mask 置 0，训练照常收敛却学不会停止；故全程只用 `attention_mask` 与 `labels` 判定，并设专测逐条断言监督区间末尾 EOS 仍可见。**②两层 padding 叠加**——单条样本已按 8 补过一次，直接追加会让中间那层边界无法从 IDs 恢复（pad==eos 无法区分），故先求真实有效长度 `prompt+supervised` 再从干净区间重补到批宽（最长有效长度向上取整到 8）。前置条件校而不改：批内 `sample_id` 唯一、单条样本自带 padding 必须自成一段、prompt 全 -100、监督区间无负值、未截断样本以 EOS 收尾（用 labels 判断）。状态 **`COLLATE_IMPLEMENTED_GPU_NOT_RUN`**：真实 CUDA、batch 前向数值、pilot 与正式训练/评估均未运行。详见[批组装报告](reports/m7_collate_20261116.md)与[机器索引](reports/m7_collate_20261116.json)。

2026-11-16数据装配层实现：新增 `src/adaptive_lora/m7_batches.py` 与 **23项测试**（全套 **286项通过**）。填补"三块零件没有接起来"的缺口：`build_supervised_example` 产出单条 → `m7_batches` 按 epoch 定序并切 micro-batch → `collate_supervised_examples` 拼成批 → `run_accumulation` 消费。装配层的关键约束：训练目标取 `output` 字段不取 `answer`（否则是评估答案泄漏，有专测验证），每条样本的 `sample_id` 由本层补全，坏样本让整批失败而不跳过（否则会静默得到 9,999 条）。发现并修复一个实现 bug：`random.Random((seed, epoch))` 在 Python 3.12 抛 `TypeError`，改为整数混合派生种子。当时暴露的两处协议空白见下一条。状态 **`BATCH_ASSEMBLY_IMPLEMENTED_GPU_NOT_RUN`**，详见[装配层报告](reports/m7_batch_assembly_20261116.md)与[机器索引](reports/m7_batch_assembly_20261116.json)。

2026-11-16两项训练循环决定落地（全套 **291项通过**）：用户确认 **①训练顺序 `none`**（每 epoch 沿用 seed42 抽样顺序，论文只写"3 epochs"未规定 shuffle）；**②末组余量 `drop`**（10,000 条不能被有效 batch 64 整除，每 epoch 末组仅 2 个 micro-batch）。两项写入配置 `training` 块并纳入 `m7_protocol.py` 固定值门禁，决策数 8 → 10。**关键语义修正：`drop` 此前与 `error` 并列处理、选中即崩溃，等于没有 drop 语义**；本轮改为真正丢弃，`run_accumulation` 从不对被丢弃的 micro-batch 做前向，并上报 `micro_batches_dropped`。**丢弃必须留清单**：装配层为每个 epoch 产出 `dropped_sample_ids` 及其 SHA-256，否则"用了 10,000 条"无法事后核对。确定的论文形状：每 epoch 156 个 optimizer step、9,984 条参与训练、丢 16 条；3 epochs 共 468 步、48 个**样本-次**未参与更新。注意在 `none` 顺序下被丢的永远是**同一批 16 条**（不是 48 条不同样本），任何"已使用全部 10,000 条"的表述都是错的。另修掉一个真实守卫缺口：loss 未连到可训练参数时 `.grad is None` 会让 `count_nonzero` 抛 `TypeError`，现改为明确的 `ValueError`。状态 **`PARTIAL_GROUP_AND_ORDER_CONFIRMED_GPU_NOT_RUN`**：真实 CUDA/AMP、真实 10,000 条装配、pilot 与正式训练均未运行。详见[末组策略报告](reports/m7_partial_group_decision_20261116.md)与[机器索引](reports/m7_partial_group_decision_20261116.json)。

2026-11-16真实数据装配核验通过：状态 **`REAL_BATCH_ASSEMBLY_PASS_CPU_ONLY`**。经用户明确批准在 `autodl` 新建输出目录，单次纯 CPU 执行 **8.666657 秒**，10,000 条真实样本全部构造成功、0 截断；三个 epoch 实际检查 **3,750 个 micro-batch / 30,000 个样本-次**，右 padding、监督 labels、真实 EOS 保留及样本顺序全部通过，批宽 88–472（均为8倍数）。每 epoch **计划**使用9,984条、丢弃同一批16条、156个optimizer steps；三epoch共计划468步，但**实际执行0步，无模型前向、无训练、CUDA未初始化**。丢弃身份清单仅存远端私有目录，已由独立只读核验对照原审计后缀确认，未取回题面/答案/身份列表。使用固定本地提交 `945fc410…` 的内嵌模块和新配置字节，远端repo仍为批准的 `2df16eb…` 且clean，旧部署配置未同步。全套本地 **292项测试通过、无跳过**（日志仍有Windows子进程解码异常，完整保留）；远端与本地证据独立核验通过。详见[真实装配报告](reports/m7_real_batch_assembly_20261116.md)与[机器索引](reports/m7_real_batch_assembly_20261116.json)。M7整体未完成；optimizer/RNG checkpoint、训练入口集成与CUDA/pilot仍待完成，不自动进入训练。

2026-11-16训练断点机制：新增 [m7_checkpoint.py](src/adaptive_lora/m7_checkpoint.py)，与旧adapter-only推理恢复分离。safetensors+严格JSON保存有序可训练参数、AdamW、GradScaler、Python/NumPy/Torch默认RNG及epoch/下一批/global_step；要求完整累积组、已清梯度、同一身份/参数顺序/超参数/版本，不用pickle、不保存基础权重、不覆盖目录。**29项CPU测试通过**，含全新微型Qwen+真实PEFT恢复后下一组参数/优化器/loss与连续运行逐项一致；全套 **321项通过，无跳过**。状态 **`TRAINING_CHECKPOINT_CPU_PASS_CUDA_NOT_RUN`**：单CUDA设备RNG分支已写但未运行，固定1.5B与正式入口停跑续跑仍NOT RUN。现有`on_step`回调早于清梯度，不能直接保存；调用者还须审计基础模型与LoRA身份。详见[checkpoint报告](reports/m7_training_checkpoint_20261116.md)与[机器索引](reports/m7_training_checkpoint_20261116.json)。未远端操作、未改配置、未push；M7整体未完成。

2026-11-16训练前集成：新增[有界会话](src/adaptive_lora/m7_session.py)连接组后清梯度/安全断点、实际监督张量身份、跨epoch精确恢复及两层drop清单；10项真实PEFT微型Qwen CPU用例通过，加4项[训练前CLI](scripts/run_m7_pretraining.py)门禁测试，全套**335项通过**。固定源码 **`397df807152a19d530fc6d1a9c5c829a0f561ebb`** 已审查181个唯一Git文件对象并发布至[训练前门禁分支](https://github.com/Hello-Charles/adaptive-lora/tree/repro/m7-pretraining-gates)。**远端部署被工具权限门禁拒绝，未执行**；只读确认新部署目录不存在、旧SHA`2df16eb…`不变且clean、4090占用0MiB。因此N250/Top8尚未实际计算、三方法CUDA/AMP pilot与GPU续跑均NOT RUN；没有启动正式训练，没有改变模型/数据或配置哈希链。详见[训练前集成报告](reports/m7_pretraining_20261116.md)。

2026-11-16最新实际运行：另获批准后通过SSH增量Git bundle成功部署固定SHA **`397df807…`**，新旧repo均clean、配置/旧审计哈希未变。远端CPU **303项实际通过、30项跳过、另2项NOT RUN**（缺固定0.5B tokenizer缓存，不能称335项全通过）。随后一次N1/N250 CUDA阶段在首条N1**未消融baseline**的预测位置遇到非有限logits，13.634秒退出；N1未完成、N250/Top8/pilot/GPU续跑均未执行，optimizer steps0，GPU已释放。CPU检查全部338个存储张量及其FP16转换均有限；eager FP16 QK点积中间溢出是有合成证据的可能机制，**真实首个异常算子尚未定位**。未重跑、未换精度/backend、未启动正式训练。详见[CUDA基线失败报告](reports/m7_cuda_calibration_failure_20261116.md)与[机器索引](reports/m7_cuda_calibration_failure_20261116.json)。

## 代码阅读与注释约定

源码、脚本和测试使用中文注释/docstring，重点说明预测位置、padding/截断、答案隔离、hook 生命周期、离线加载和结果保留等隐含约束。后续新增或修改代码也必须遵循 [CLAUDE.md 的代码注释规范](CLAUDE.md#代码注释规范现有代码及后续新增修改代码均适用)；关键逻辑详细解释，简单语句不逐行复述。

Windows 终端或重定向日志出现中文乱码时，可为单次命令添加 `-X utf8`，例如 `.venv/Scripts/python.exe -I -X utf8 scripts/inspect_model.py --help`，无需修改系统编码。

## Python 环境与 M0 检查

项目声明 Python `>=3.10`；本次实际使用 Windows 上的 Python 3.12.8，符合要求。机器仅检测到 Intel Iris Xe，M0 采用 **CPU 版** PyTorch；这是本机工程选择，不是论文训练配置。使用 Git Bash 在项目根目录创建独立环境并运行测试：

```bash
python -m venv .venv
.venv/Scripts/python.exe -I -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -I -m pip install -e '.[repro]'
.venv/Scripts/python.exe -I -m pip check
.venv/Scripts/python.exe -I -m unittest discover -s tests -v
.venv/Scripts/python.exe -I scripts/record_environment.py --metadata-only --output reports/m0_venv_environment.json
```

PowerShell 下可用 `./.venv/Scripts/python.exe -I -m unittest discover -s tests -v` 运行测试。M0 的历史快照使用 `--metadata-only`，其中 `not_checked` **不代表导入成功**；本轮已实际导入依赖，见 [M1 环境快照](reports/m1_verified_environment.json)。环境脚本在目标 JSON 已存在时拒绝覆盖，重新记录时须提供新的输出路径。真实安装的版本保存在 [requirements-windows-py312.txt](requirements-windows-py312.txt)；重新创建该 Windows CPU 环境时，应先用上面的 PyTorch CPU 专用索引安装对应版本，再在虚拟环境中安装锁定的依赖和本项目。

[首次 M0 记录](reports/m0_status.md)与[首次环境快照](reports/m0_environment.json)保留安装前的历史状态（其中依赖尚未安装、目录还不是 Git 仓库），不是当前环境结论。新环境记录见 [reports/m0_venv_environment.json](reports/m0_venv_environment.json)，Git 和安装完成后的状态见 [reports/m0_git_setup.md](reports/m0_git_setup.md)。`pyproject.toml` 的可选依赖是项目安装入口；实际版本以对应平台的环境快照和锁定清单为准。

`.gitignore` 排除虚拟环境、本地 PDF、数据、结果、模型权重、checkpoint、缓存和典型密钥文件。忽略规则不能保护已被 Git 跟踪的文件；请勿把凭据放入仓库，推送前仍需人工检查待提交内容。后续阶段使用 Git 追踪变更并记录提交编号；创建新提交或公开推送仍以当次任务的授权为准。

## M1：模型结构探查（已通过真实模型门禁）

论文第 3.1 节报告机制分析使用 Qwen2.5-0.5B（24 层）。本项目选择 Hugging Face 仓库 `Qwen/Qwen2.5-0.5B` 和固定 revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，均为 **[复现决策]**，不是论文原权重版本。实际模型 `Qwen2ForCausalLM` 有 24 层 `model.layers.0`–`model.layers.23`；每层 `self_attn` 的输出为 `(Tensor, None)`，`mlp` 的输出为 Tensor。每层 Q/K/V 投影路径、tokenizer 细节、短输入 logits 与有效预测位置见 [真实结构报告](reports/m1_structure.json) 和 [M1 验收记录](reports/m1_validation.md)。M1 的观察性 hooks 不修改模型输出；独立 zero-ablation 实现及其边界见下文 M3。

脚本默认只使用本地缓存，不会静默下载模型。已有授权下载时也必须显式传入 `--allow-download`、固定 `--revision` 和**新建且不存在**的 `--cache-dir`；本次权重保存在被 Git 忽略的 `data/models/` 下。复核已下载的模型时使用不同的输出路径（不会覆盖历史报告）：

```bash
.venv/Scripts/python.exe -I scripts/inspect_model.py --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 --output reports/m1_structure_repeat.json
```

本次第一个探查调用因把 `post_attention_layernorm` 错认作 Attention 而失败；修复并补回归测试后从缓存**离线**重跑成功。模型配置未暴露 `resolved_revision`，缓存快照目录与请求的 SHA 一致；详见验收记录。

## M2：校准输入准备（已通过真实数据门禁）

使用 `tatsu-lab/alpaca` 的 train parquet，固定 revision `dce01c9b08f87459cf36a430d809084718273017`。实测共 52,002 行，seed=20261112 抽取 30 条；输入为 instruction/input 的 user 消息，Qwen chat template 加 assistant generation prompt，不拼接 output/reference。完整模板、revision、索引、三类哈希和输入 IDs/mask/有效位置保存在本地清单中；[不含原始文本的验收索引](reports/m2_validation.json)可用于核对来源。

校准逐样本无 padding；渲染模板后使用 `add_special_tokens=False`，已与 30 条原生 chat-template 分词逐一比对。max_length=512，超长时拒绝整次生成，**不截掉 assistant 标记，也不重抽样**；本次长度为 26–64 tokens、无截断。具体协议及其 **[复现决策]** 标记见 [PAPER_SPEC.md](PAPER_SPEC.md#m2-smoke-数据协议-复现决策)。真实分词与数据准备通过不代表已验证模型校准或批量 logits 数值等价。

脚本只允许显式下载一个固定 SHA 的 parquet 并检查字节 SHA-256，始终离线读取 M1 tokenizer。数据经 `hf-mirror.com` 下载，官方域名连接失败；镜像文件与其 LFS 哈希匹配，不冒充独立官方来源验证。首次处理触发 Windows 长路径限制后改用本地 PyArrow 直接读取；未安装依赖或改变系统设置。Alpaca 数据卡标注 CC-BY-NC-4.0，使用前应遵守其非商业用途等许可约束。

已有缓存的离线复核（输出路径必须尚不存在，脚本不会覆盖历史清单）：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/prepare_calibration.py \
  --dataset-cache-dir data/datasets/alpaca_dce01c9b08f87459cf36a430d809084718273017_m2_20261114 \
  --hub-endpoint https://hf-mirror.com --n 30 --seed 20261112 --max-length 512 \
  --output results/m2-20261114/manifest-recheck.json
```

数据/model revision 的默认值均已固定为完整 SHA；新机器必须先准备匹配的本地 tokenizer 缓存。若需要下载该小型数据文件，显式使用 `--allow-download` 并传入**尚不存在**的专用数据缓存目录；大型数据或模型下载仍需按 [CLAUDE.md](CLAUDE.md) 另行取得授权。完整清单位于 [results/m2-20261114/](results/m2-20261114/)，受 Git 忽略；报告中的配置/样本指纹相同，不要求带时间戳和运行命令的 JSON 整文件哈希相同。M2 阶段没有自动提交、推送或继续后续阶段；M3 由用户另行指令启动。

## M3：独立 zero-ablation（已通过真实模型门禁）

[ablation.py](src/adaptive_lora/ablation.py) 提供 `zero_ablation(model, layer, component)` 上下文管理器；layer 由 M1 的模块发现器生成，component 为 `attention` 或 `mlp`。它只用新建的 `zeros_like` 替换目标 hidden states：Attention 普通 tuple/list 的第 0 项置零，附加项保持身份；MLP 的 Tensor 整体置零。不会原地清零输入/残差，也不消融整个 block；未知结构明确报错。

调用者必须先全模型 `eval()`、冻结参数，并使用 `torch.no_grad()` / `torch.inference_mode()`；helper 检查前提，不替调用者偷偷修改模式。同一根模型只允许一个消融 scope，嵌套即拒绝，正常退出从未命中目标也报错；只清理本 scope 的 hook，保留已有观察 hook。该接口限于串行 eager 前向，调用时禁用 KV cache；不支持把共享模型的并发请求当作隔离实验。

真实验证先在第 0 层分别运行 Attention/MLP，然后在同一 4-token 输入上逐一遍历 24 层的 48 个目标。每次均核对目标为零、附加项不变、非目标输出在**同一次前向**的 hook 前后相等；移除 hook 后完整 logits 精确恢复。上游消融会改变下游输入，不能误把这种变化说成“额外模块被直接消融”。两个真实异常注入场景、完整基础参数前后字节指纹和 48 个测试也通过；详见 [M3 验收报告](reports/m3_validation.md) 和 [验收索引](reports/m3_validation.json)。

默认验证一个层，不会自动遍历所有层；以下单层命令和全层命令需要分别显式执行，输出路径必须是新文件：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/validate_ablation.py \
  --layer 0 --threads 4 --output results/m3-20261114/single-layer-repeat.json

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/validate_ablation.py \
  --all-layers --threads 4 --output results/m3-20261114/all-layers-repeat.json
```

模型 revision 和缓存默认复用 M1，脚本没有下载开关；只使用 CPU 短输入，不做训练。原始 [single-layer.json](results/m3-20261114/single-layer.json)、[all-layers.json](results/m3-20261114/all-layers.json)、[环境快照](results/m3-20261114/environment.json)与[运行日志](results/m3-20261114/run.log)保存在 Git 忽略目录。M3 本身不产生 $t^*$、概率或层重要性；M4 由用户另行指令启动，没有自动跨阶段、提交或推送。

## M4：固定 token 校准、重要性与 Top-K

[calibration.py](src/adaptive_lora/calibration.py) 先从未消融模型确定每条输入的原始 $t^*$ 和 $q$，再用 M3 的独立作用域遍历 24 层 Attention/MLP，始终读取同一 token 的概率。[selection.py](src/adaptive_lora/selection.py) 按公式 (5) 的均值降序选 Top-7，完全并列时按小层号优先。概率计算用 FP32 softmax、Python `math.fsum` 累积；这些数值约定、末尾 next-token 位置、CPU smoke 和并列规则均属 **[复现决策]**，详见 [M4 协议](PAPER_SPEC.md#m4-数值运行与审计协议-复现决策)。

[离线入口](scripts/run_calibration.py) 默认绑定已验收的 M2 内容指纹，重新验证完整清单和本地 tokenizer 的 IDs/mask/位置，要求固定 revision 的本地模型缓存。没有下载开关，也不加载答案、注入 LoRA 或训练。每条输入的 48 次消融后额外做完整 logits 恢复检查，所以 N=30 共 **1,500 次前向**（其中 30 次仅用于恢复验收），而非一次普通推理。

先用固定清单首条验收，再显式执行完整 30 条；示例路径必须是尚不存在的新目录，本机 CPU 完整校准可能耗时数十分钟：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/run_calibration.py \
  --manifest results/m2-20261114/manifest.json --limit 1 --top-k 7 --threads 4 \
  --output-dir results/m4-recheck-n1

# 仅在上一条成功、检查输出后执行；去掉 limit，仍使用原 30 条清单，不重新抽样。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/run_calibration.py \
  --manifest results/m2-20261114/manifest.json --top-k 7 --threads 4 \
  --output-dir results/m4-recheck-n30
```

每次目录保存配置/环境/Git 基线与 dirty 源码哈希、实际样本 IDs/mask、逐样本原始 token/概率、24 层 CSV/JSON、Top-K、SVG/HTML 图和日志。`selected_layers.json` 的 `selected_layer_indices` 保留重要性排名，`selected_layer_indices_ascending` 是方便后续 LoRA 显示的层号升序；索引均从 **0** 开始。仅在全部成功后才写 `run_summary.json`；失败可能留下部分 JSONL/配置，不可因为存在一个结果文件就声称校准完成。完整输入与结果继续受 Git 忽略。

[独立复核入口](scripts/validate_calibration.py) 不导入校准器/选层器，也不加载模型，从原始概率重新计算公式、排序、CSV/JSON/JSONL、文件哈希和源码身份。完整运行后可与独立 N=1 首样本比较，但这**不等于完整 N=30 重跑**：

```bash
.venv/Scripts/python.exe -I -X utf8 scripts/validate_calibration.py \
  --run-dir results/m4-recheck-n30 --manifest results/m2-20261114/manifest.json \
  --expected-n 30 --compare-prefix-run results/m4-recheck-n1 \
  --output results/m4-recheck-audit.json
```

图表使用标准库生成，不需要 matplotlib；本地打开 [最终 importance.html](results/m4-20261114-n30/importance.html) 可切换明暗主题、纹理和键盘/hover 提示，表格提供全部数值，图上的 `*` 标识 Top-K 而非另一条数值系列。纵轴是平均绝对概率变化，**不是下游准确率或训练收益**。本机 N=1 和 N=30 图表均已通过真实 Chrome 的渲染/交互检查并查看截图。

本次 N=30 校准耗时 1,107.881943 s（约 18.46 分钟，包含参数指纹、1,440 次消融、30 次恢复和日志），不是论文性能 benchmark。完整结果见 [24 层 CSV](results/m4-20261114-n30/layer_importance.csv)、[排名/选层](results/m4-20261114-n30/selected_layers.json)、[样本清单](results/m4-20261114-n30/sample_ids.json)、[独立复算](results/m4-20261114-audit/n30.json) 和 [M4 验收索引](reports/m4_validation.json)。基础参数未变，30 次完整 logits 恢复全部通过；独立 N=1 轨迹与 N=30 首样本一致，但未重复整个 N=30 或运行不同抽样 seed 的稳定性分析。M4 已验收；M5 由用户另行指令启动，见下文。

## M5：选择性 LoRA、三步短训练与 adapter 恢复

[trainer.py](src/adaptive_lora/trainer.py) 使用实际发现的完整模块路径，给 M4 **N30** 选出的 7 层 Q/K/V 注入标准 PEFT LoRA（rank16、alpha32）。实际 21 个投影、42 个 A/B 张量，共 **430,080 个可训练元素**；基础模型 494,032,768 个参数全冻结。K/V 为 896→128，并非 Q 的 896→896 方阵，因此参数量按每个投影的 `r*(d_in+d_out)` 分别计算。

训练必须启用 autograd 和 `train()`，不能把“基础参数冻结”误做成整个前向 `no_grad()`；仅初始化/恢复探针在 eval/no-grad 下比较。合成监督样本的 user prefix 不含回答，teacher-forcing 后缀才含训练答案；先核对完整分词的 prefix 边界，prompt labels=-100，回答和明确追加的 tokenizer EOS 被监督。`labels[P]` 对应 `logits[P-1]`，不会把 causal shift 多做或少做一次。

[离线 train.py](scripts/train.py) 硬限制 **1–3 步**、batch_size=1、CPU；默认固定最终选层文件哈希、模型 revision、seed42、4 线程和确定性算法。只用仓库内 3 条短合成样本，不下载、不加载正式训练数据、不改变 M2/M4。以下示例仅可写到尚不存在的新目录：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/Scripts/python.exe -I -X utf8 scripts/train.py \
  --m4-dir results/m4-20261114-n30 --steps 3 --seed 42 --threads 4 \
  --output-dir results/m5-recheck-smoke
```

真实 [M5 输出](results/m5-20261115-smoke/) 已完成，3 步 loss 为 `3.573270320892334`、`3.0456244945526123`、`3.504087209701538`；每步梯度均存在且有限，adapter 更新，基础规范参数名/原始字节指纹与 M3/M4 一致。不同步骤使用不同样本，**不能据此声称 loss 收敛或准确率提升**。本机训练段 4.632139 s、总计 18.103995 s（包含审计/探针/保存恢复），不是论文性能 benchmark。

[checkpoint.py](src/adaptive_lora/checkpoint.py) 只保存 adapter safetensors，显式禁止自动保存基础 embedding；所有结果/checkpoint 均拒绝覆盖。真实权重文件为 1,725,816 bytes，只有这 42 个 A/B 张量。训练对象释放后，重新从同一固定本地缓存加载基础模型、恢复所有 adapter；3 条 user-only prefix 的完整 logits 精确一致，最大绝对差为 0。该 checkpoint 用于 **adapter 权重/推理恢复，不包含 optimizer、RNG 或 scheduler 断点续跑状态**。

本阶段保留 BF16 基础权重、FP32 adapter/交叉熵，不启用 AMP；dropout=0、显式 `<|endoftext|>` EOS、答案-only loss mask、最大 96 tokens、无 padding/截断、合成数据和 3 steps 都是 **[复现决策]**，不同于论文 FP16 AMP、正式数据、512-token 预处理和 3 epochs。论文依据及差异见 [M5 协议](PAPER_SPEC.md#m5-选择性-lora-与短训练协议-复现决策)。

全套 123 个测试通过、无跳过；包含微型真实 Qwen2/PEFT 反向与 fresh-base 保存恢复，而非仅 mock 成功。另有独立存档核对：18 个 M5 产物、13 个 M4 输入产物、30 个源码哈希和实际 safetensors 的参数量/形状/字节指纹均一致。详见 [M5 报告](reports/m5_validation.md)、[验收索引](reports/m5_validation.json) 和本地 [audit.json](results/m5-20261115-evidence/audit.json)。以上是M5当时的验收状态；M6由用户另行指令启动，复用了这条训练路径，但没有重跑完整M5 CLI的注入探针/保存恢复流程。

## M6：统一 Full / Random-K / Adaptive-K 基线

[baselines.py](src/adaptive_lora/baselines.py) 规划三方法的层集合；[protocol.py](src/adaptive_lora/protocol.py) 一次构建共享合成训练/评估协议；[comparison.py](src/adaptive_lora/comparison.py) 复用M5训练器与 [evaluation.py](src/adaptive_lora/evaluation.py) 的prompt-only greedy生成/独立评分。每方法必须使用重新加载的同字节基础模型，不能继承前一种方法已训练的实例。

[离线比较入口](scripts/compare_baselines.py) 只允许每方法1–3个CPU batch、每评估输入1–8个新token；没有下载、GPU或epochs开关。默认绑定M4最终N30及其M2输入文件/内容指纹，[来源检查器](src/adaptive_lora/comparison_io.py) 逐方法核对实际tokenizer与30条保存prompt的分词。下例输出目录必须尚不存在：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/Scripts/python.exe -I -X utf8 scripts/compare_baselines.py \
  --m4-dir results/m4-20261114-n30 --manifest results/m2-20261114/manifest.json \
  --steps 3 --train-seed 42 --layer-seed 20261115 --rank 16 --alpha 32 \
  --max-new-tokens 4 --threads 4 --output-dir results/m6-recheck-smoke
```

本次 [真实运行](results/m6-20261115-smoke/) 的实际结果：

| 方法 | 所选层（零基；保留抽样/重要性顺序） | 可训练参数 | Full占比 | 短训练步数 | 合成smoke EM |
|---|---|---:|---:|---:|---:|
| Full | 0–23全部层 | 1,474,560 | 100% | 3 | 0/3 |
| Random7 | `[9, 0, 15, 19, 5, 11, 20]` | 430,080 | 29.1667% | 3 | 0/3 |
| Adaptive7 | `[0, 1, 2, 5, 3, 8, 10]` | 430,080 | 29.1667% | 3 | 0/3 |

Full实测72个Q/K/V投影、144个A/B张量；K-layer各21个投影、42个张量。三者均rank16/alpha32，因此 **Full与K-layer不是同参数预算**。训练seed42对各方法的注入/优化分别重置；Random独立layer_seed=20261115，不改变训练RNG，也不保证不同目标集合中同名A的初始化完全相同。

训练3条、评估3条、校准30条的ID/prompt字节/prefix-token直接交集均为0；校准来自训练侧本身允许。参考答案不传给生成接口，只在生成完成后按ID评分；不会通过修改gold改变模型输入。评估逐步argmax、temperature0、无KV cache，只在明确tokenizer EOS或新token上限停止，正常/异常均恢复模块原training flags。

所有预测均到达4-token长度上限，三方法strip-only exact-match均0/3；指标为 `smoke_exact_match_strip_only_not_paper_accuracy`。这是有意极短的接口验收，合成题面自身可含目标词，不是无答案暗示的能力基准，更不是论文commonsense答案抽取、math CoT/regex或Section5.4的Alpaca-ZH test PPL。不得从该结果声称方法等效或Adaptive收益。

三者基础参数494,032,768个，冻结/无梯度/训练前后字节均通过；评估前后全部基础+adapter参数字节相同。各方法保存adapter-only checkpoint、配置/seed/选层、训练轨迹、预测/评分与参数统计；根目录保存共享协议、拆分审计、环境/Git/41个源码哈希及 [CSV](results/m6-20261115-smoke/comparison.csv) / [JSON](results/m6-20261115-smoke/comparison.json)。仅在三方法身份一致后写整体成功标记。失败目录保留且不能复用；M6未额外验证三方法checkpoint恢复或训练断点续跑。

最终162个测试通过、无跳过；[独立存档审计](results/m6-20261115-evidence/audit.json) 的261项检查通过，包括54个运行摘要产物、15个M4/M2来源、41个运行时源码哈希、真实safetensors形状/数量/两种命名域字节和独立评分。Adaptive三步训练数值及训练后adapter与既有M5相同设置的训练核心精确一致，不等于完整M5 CLI或三基线重复实验。总wall-clock49.677969 s含加载/哈希/审计/IO，不是论文速度结果，未测GPU显存。

详见 [M6报告](reports/m6_validation.md)、[机器索引](reports/m6_validation.json) 和 [协议与偏差](PAPER_SPEC.md#m6-统一基线与评估协议-复现决策)。当前评估模块已接入比较CLI，未单独提供正式 `evaluate.py`。M6完成后停止，不自动进入M7、下载、改依赖或提交/推送；正式数据/评估协议与GPU成本须由用户确认。

## 后续阶段前的待核验清单

以下条目均为 **[待核验]**，不是论文事实；详细状态见 [PAPER_SPEC.md 第 7 节](PAPER_SPEC.md#7-开放问题及处理)：

- Q01：数据集精确版本、训练样本 ID 和抽样索引。
- Q02：原作者逐 token prompt 模板、预测位置和精确截断边界；Appendix C.2 的 512-token 截断及 pad 到 8 的倍数已核查。
- Q03：作者实际使用的模型权重 revision 仍待核验；本项目 M1 的模型 SHA、tokenizer 和依赖版本已记录。
- Q04：训练loss mask、样本拼接方式、commonsense答案抽取及math任务专用解析的具体实现；Appendix C.3的无in-context examples与greedy解码已核查。
- Q05：作者完整代码的发布状态。

M0 没有核查论文算法章节；M1 已核查第 3.1 节的 0.5B 模型与 24 层描述；M2 已核查第 4.3 节及附录 E.1–E.4 的数据依据；M3 已按第 4.3 节公式 (4)、Algorithm 1 第 5–6 行及附录 E.2 实现独立、无梯度的子模块输出消融。M4 核查 PDF 第 6 页的第 4.3 节公式 (4)–(5)、Algorithm 1 第 3–12 行和第 4.4 节，完成对应评分与全层 Top-K 实现及真实 N=30 smoke 验收。M5 核查 Section 4.5 / 5.1 及 Appendix C.1–C.2（PDF 第 7、18–19 页），完成选择性 LoRA 和受限 CPU 训练/保存恢复验收。M6核查Table2 / Section5.3、第5.4节和Appendix C.3（第8–9、19页），完成三基线统一接口与真实CPU工程smoke，不混同论文主表准确率与选层PPL实验；正式数据版本、作者完整模板和loss mask仍待核验。后续按 [CLAUDE.md 的论文原文使用规范](CLAUDE.md#论文原文使用规范) 继续按需核查，不把工程假设表述为作者原配置。
