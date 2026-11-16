# REPRO_PLAN.md — Adaptive-LoRA 分阶段复现计划

> 执行模型：Claude Code + CC Switch + Codex，**单 Agent**。算法和论文事实以 `PAPER_SPEC.md` 为准。
>
> 状态：以勾选项和真实证据为准。完成一个里程碑必须记录真实执行命令、测试输出、实验产物和 Git commit；不能只凭代码生成就勾选。

## 1. 工程目录（逐阶段创建，不必一次生成）

```text
adaptive-lora/
├── CLAUDE.md
├── PAPER_SPEC.md
├── REPRO_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── smoke_test.yaml
│   ├── qwen_0.5b.yaml
│   └── qwen_1.5b.yaml
├── src/adaptive_lora/
│   ├── model_loader.py
│   ├── data.py
│   ├── ablation.py
│   ├── calibration.py
│   ├── selection.py
│   ├── trainer.py
│   └── evaluation.py
├── scripts/
│   ├── inspect_model.py
│   ├── run_calibration.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_ablation.py
│   ├── test_calibration.py
│   ├── test_selection.py
│   └── test_lora_targets.py
├── data/                # 大文件不默认进 Git
├── results/             # 每次实验独立目录
└── reports/
```

## 2. 里程碑及验收

### M0 — 项目骨架和规格冻结

任务：建立 Python 工程、文档、`.gitignore`、基本环境探查；不训练、不下载大模型。

- [x] 项目可导入、基本测试工具可运行。
- [x] Python、PyTorch、Transformers、PEFT 等已安装版本已记录（未装则如实标记）。
- [x] 模型权重/数据/密钥/缓存已被 `.gitignore` 排除。
- [x] 提交或记录首个 Git commit；论文待核验项已列入清单。

M0 证据：`reports/m0_status.md`、`reports/m0_git_setup.md`、`reports/m0_venv_environment.json`；首个 commit 为 `1c7b0e8433c3669bcb463fc5a7ba1441efc3e0f5`。M0 初始快照曾将第三方包导入标为 `not_checked`；随后在用户明确授权下，`reports/m1_verified_environment.json` 已验证关键依赖实际导入成功，虚拟环境测试通过。

**门禁**：环境现状明确后才进入模型结构探查。

### M1 — 模型结构探查

任务：加载 Qwen2.5-0.5B；检查 24 层结构、每层 Attention / MLP 的实际模块路径、q/k/v 投影和 Attention 输出类型；执行一次短输入前向。

产出：`model_loader.py`、`inspect_model.py`、结构探查报告。

- [x] 证实模块路径，未凭经验硬编码。
- [x] 证实返回类型（tensor/tuple/其他）和有效预测位置。
- [x] 保存真实运行命令、输出和依赖版本。

M1 证据：`reports/m1_structure.json`、`reports/m1_verified_environment.json` 和 `reports/m1_validation.md`。固定模型仓库 revision `060db6499f32faf8b98477b0a26969ef7d8b9987` 是**复现决策**而非论文指定版本；Transformers 模型配置未暴露 `resolved_revision`，实际缓存快照目录与所请求 SHA 一致。首次结构探查误将 `post_attention_layernorm` 识别为 Attention，修正并补测试后从隔离缓存离线重跑成功；M1 没有实现 zero-ablation。

**门禁**：模块返回结构未确认，不写 zero-ablation hook。

### M2 — 校准数据准备

任务：先用 Stanford Alpaca 等指令数据，稳定抽样 30 条；固定 prompt、tokenizer、预测位置，保存输入 ID/哈希。

产出：`data.py`、数据 manifest、测试。

- [x] N=30、固定 seed 后抽样可重现。
- [x] reference answer 未意外混入校准输入。
- [x] 截断与 padding、有效末尾位置经过测试。
- [x] 记录数据来源、版本、样本 ID/索引和 prompt 模板。

M2 证据（2026-11-14）：[验收报告](reports/m2_validation.md)、[机器可读验收索引](reports/m2_validation.json)；完整输入清单位于 Git 忽略的 [manifest.json](results/m2-20261114/manifest.json) 与 [manifest-repeat.json](results/m2-20261114/manifest-repeat.json)。固定 `tatsu-lab/alpaca` revision `dce01c9b08f87459cf36a430d809084718273017`，实读 train 共 52,002 行，以 seed=20261112 抽取 30 条；两次独立离线运行的样本、协议和内容 SHA-256 一致，30 条均与 Qwen 原生 chat-template 分词核对一致。实际长度 26–64 tokens，无截断；真实批量右 padding 和有效末尾也通过验证。全套 28 个测试通过、无跳过；超长拒绝、不静默丢掉 generation prompt 的策略是 **[复现决策]**。

数据经镜像下载并匹配固定文件的字节哈希，官方域名直连失败；首次 parquet 处理触发 Windows 长路径限制，改用本地 PyArrow reader 后离线成功，没有重复下载或更改环境。Git 分支为 `repro/m2-calibration-data`，记录基线 commit `577527ad0ec343941bbed0b6519ff3544e301cc7`、dirty 状态及实际源码哈希；该 SHA 是 M1 基线，**不是 M2 提交**，本次未自动提交或推送。早期受阻记录保留在 [m2_progress.md](reports/m2_progress.md)，不再代表当前状态。

**门禁**：M2 仅准备输入；不得把本清单当作已运行校准、选层或模型训练的证据。完成后等待用户指令进入 M3。

### M3 — 单层 zero-ablation

任务：从**一个输入、一层、一个子模块**开始；分别实现 Attention 与 MLP 输出消融，然后扩展到全部层。

产出：[ablation.py](src/adaptive_lora/ablation.py)、[test_ablation.py](tests/test_ablation.py)、[离线验证脚本](scripts/validate_ablation.py)。

- [x] Attention 和 MLP 消融互不叠加。
- [x] 仅目标模块输出被零化，输出结构未被破坏。
- [x] 正常退出/异常退出后 hooks 清理。
- [x] 清除 hook 后正常模型输出恢复。

M3 证据（2026-11-14）：[验收报告](reports/m3_validation.md)、[机器可读验收索引](reports/m3_validation.json)；原始结果和日志位于 Git 忽略的 [results/m3-20261114/](results/m3-20261114/)。先以固定本地 Qwen 的第 0 层验证，再扩展到 24 层的 48 个独立目标；单个输入 `Hello, world!` 共 4 tokens，全部正常消融均观察 48 个子模块，目标主输出为零、其余模块在同一次前向的 hook 前后相等，所有目标 block 残差输出仍非零。每次清理后完整 logits 精确恢复，Attention 附加返回项保持不变；第 0 层的 Attention/MLP 各额外注入一次真实前向异常，清理和恢复均通过。全部 494,032,768 个基础参数的前后字节指纹一致、无梯度。

最终 48 个测试通过、无跳过（原有 28 个 + M3 新增 20 个）；覆盖同一模型嵌套拒绝、Tensor/tuple/list 结构、原地别名风险、正常/调用方异常/前向异常/错误返回结构/注册失败/未命中清理，以及 eval、冻结和禁用梯度前提。小模型测试不冒充真实 Qwen 验证。

Git 分支 `repro/m3-zero-ablation`，记录基线 commit `577527ad0ec343941bbed0b6519ff3544e301cc7`、dirty 状态和源码 SHA-256；该 SHA 不是 M3 提交，本次未自动提交或推送。未下载权重、未改环境、未训练。

**门禁**：`PAPER_SPEC.md` 的 T03–T05 已通过；本阶段只实现串行 eager、`use_cache=False` 下的消融契约，不计算 $t^*$、概率、层重要性或 Top-K。完成后等待用户指令进入 M4。

### M4 — 重要性评分与 Top-K 选层

任务：固定原始 $t^*$；实现 $q,q_l^A,q_l^{MLP}$ 及 $I_l$；N=30 聚合；Top-7 排序；导出 CSV/JSON 和重要性图。

产出：`calibration.py`、`selection.py`、`run_calibration.py`、测试和结果。

- [x] 原始 token 在所有消融中保持一致。
- [x] N=1 / N>1 公式单元测试通过。
- [x] 输出 24 层重要性、Top-7 和所用样本清单。
- [x] 记录环境、代码版本和耗时；可选 N=100 或不同 seed 检查 Jaccard 稳定性。

M4 证据（2026-11-14）：[验收报告](reports/m4_validation.md)、[机器可读验收索引](reports/m4_validation.json)；[N=1 门禁](results/m4-20261114-n1/) 通过后，在同一 M2 固定清单上完成 [N=30 全量 smoke](results/m4-20261114-n30/)。全 24 层共 1,440 次独立消融均测原始 token，另有 30 次 baseline 和 30 次完整 logits 恢复检查，总计 **1,500 次前向**；所有恢复及 494,032,768 个基础参数的前后字节指纹、冻结/无梯度/hook 检查均通过。FP32 softmax、math.fsum 累加、零基索引并列规则等已标注为复现决策。

最终 Top-7（重要性降序）为 **`[0, 1, 2, 5, 3, 8, 10]`**；按层号升序为 `[0, 1, 2, 3, 5, 8, 10]`。24 层 CSV/JSON、全部样本与原始概率、SVG/交互 HTML、配置/环境/源码指纹/日志已保存。校准用时 1,107.881943 s（约 18.46 分钟），含参数指纹、恢复和日志，不是论文性能 benchmark。独立脚本复算全部概率→公式/排名及跨文件一致性通过；N=1 完整轨迹与 N=30 首样本精确一致，但**未重跑完整 N=30，也未做 N=100/不同 seed 的 Jaccard**。全部 92 个测试通过、无跳过；明暗图表、纹理、键盘/hover、布局已用真实 Chrome 验证并查看截图。

Git 分支 `repro/m4-calibration-selection`，基线 commit `577527ad0ec343941bbed0b6519ff3544e301cc7` 仍是 M1 提交；通过 dirty 状态及源码 SHA-256 记录 M4 实际代码，未创建 M4 提交或推送。M4 不含 LoRA 注入、训练或下游效果评估。

**里程碑 A：校准与选层算法完成。此前不得开始正式 LoRA 微调；M4 完成后停止，等待用户另行指令进入 M5。**

### M5 — 选择性 LoRA 与训练 smoke test

任务：PEFT 在指定 Top-K 层的 Q/K/V 添加 LoRA；检查 trainable params；仅运行几步训练与 checkpoint 加载。

产出：`trainer.py`、`train.py`、`test_lora_targets.py`。

- [x] 仅所选层 LoRA 可训练，基础模型冻结。
- [x] 训练 1–3 个 batch，梯度与 loss 无异常、无 NaN。
- [x] checkpoint 可保存并恢复。

M5 证据（2026-11-15）：[验收报告](reports/m5_validation.md)、[机器可读验收索引](reports/m5_validation.json)；真实运行保存在 Git 忽略的 [results/m5-20261115-smoke/](results/m5-20261115-smoke/)。复用最终 M4 N30 Top-7 `[0, 1, 2, 5, 3, 8, 10]`，只在这 7 层的 21 个 Q/K/V 投影注入 rank16/alpha32 的标准 PEFT LoRA；实测 **430,080 个可训练元素、42 个 A/B 张量**，494,032,768 个基础参数冻结且无梯度，前后规范参数名/原始字节指纹与 M3/M4 一致。

CPU 上以 seed42、batch_size=1、无 accumulation 执行恰好 3 个极短合成 batch，loss 分别为 `3.573270320892334`、`3.0456244945526123`、`3.504087209701538`，梯度和更新均有限，adapter 指纹改变。首步 21 个非零梯度张量符合 B=0 初始化，A 的零梯度不是失败；AdamW weight decay 仍可改变 A。3 条 user-only prefix 的注入前后完整 logits 精确一致，训练后 adapter-only safetensors 保存为 1,725,816 bytes，再从固定本地缓存加载全新基础模型恢复；3 条完整 logits 再次精确一致、最大绝对差为 0。

已核查 Section 4.5 / 5.1 和 Appendix C.1–C.2（PDF 第 7、18–19 页）。BF16 冻结基础权重、FP32 adapter/loss、不使用 AMP、3 条合成数据、显式 EOS 与答案 loss mask、96-token 超长拒绝等均为 **[复现决策]**，不是论文 FP16 AMP / 3 epochs / 正式数据实验。训练段 4.632139 s、总计 18.103995 s，包含审计/探针/保存恢复，不是论文性能 benchmark。

最终全套 **123 个测试通过、无跳过**（原 92 + M5 新增 31）；覆盖精确层匹配、GQA 参数量、mask/causal shift、异常梯度/清理、短训练重复性、独占 checkpoint、fresh-base 恢复、CLI 范围与失败标记。独立存档核验通过：18 个 M5 产物、13 个 M4 输入产物、30 个运行时源码哈希及实际 safetensors 的形状/参数量/前后命名域指纹一致；该核验不重新执行反向。

Git 分支 `repro/m5-selective-lora-smoke`，基线 commit `577527ad0ec343941bbed0b6519ff3544e301cc7` 仍是 M1 提交，通过 dirty 和源码字节记录当前实现，未自动提交/推送，未改变依赖或下载。

**门禁**：T09 及 M5 三项通过。checkpoint 恢复仅指 adapter 权重/推理输出，不含 optimizer/RNG 训练断点续跑；未验证真实 0.5B 三步完整重跑、收敛或下游收益。完成后停止，等待用户另行指令进入 M6。

### M6 — 统一基线

任务：建立 Full LoRA / Random-K LoRA / Adaptive-K LoRA 的统一训练与评估接口。

- [x] 相同训练数据、评估协议和可控训练超参数。
- [x] Random-K 保存 seed 和层 ID。
- [x] 报告实际可训练参数量差异（不能说 Full 与 K-layer 是同预算）。
- [x] 无训练/评估直接输入重叠：ID、prompt字节、实际prefix token三个身份域均隔离；不将此门禁扩大为语义或预训练污染保证。

M6 证据（2026-11-15）：[验收报告](reports/m6_validation.md)、[机器可读验收索引](reports/m6_validation.json)；真实运行位于 Git 忽略的 [results/m6-20261115-smoke/](results/m6-20261115-smoke/)。[统一比较入口](scripts/compare_baselines.py) 为 Full / Random7 / Adaptive7 分别从同一固定本地 revision 重新加载基础模型，复用 M5 训练器与同一 prompt-only greedy evaluator；训练/评估协议、seed42、rank16/alpha32、AdamW和CPU执行设置一致，各做3个短合成batch，共 **9 个 optimizer steps**。

Full 的24层实测 **1,474,560 个可训练元素、144个A/B张量**；Random7/Adaptive7各 **430,080 个元素、42个张量**，为Full的29.1667%，不是同参数预算。Random layer_seed=20261115、抽样顺序 `[9, 0, 15, 19, 5, 11, 20]`，Adaptive仍为最终M4 N30 `[0, 1, 2, 5, 3, 8, 10]`；21/72个实际Q/K/V投影均按完整路径审计。三方法的494,032,768个基础参数冻结、无梯度且前后字节相同，评估前后全部基础+adapter字节相同。

训练3条、评估3条、校准30条的直接ID/prompt/prefix-token交集均为0；校准来自训练侧本身允许，不作为错误拒绝。生成不接收参考答案，只追加自身预测；实际三者 strip-only smoke exact-match 均 **0/3**，全部触及4-token上限，不能声称方法等效、收敛或复现论文准确率/PPL。保存每方法checkpoint、轨迹、预测/评分和JSON/CSV比较；总耗时49.677969 s（包括加载/指纹/审计/IO），不是论文速度测量，未测GPU显存。

最终 **162 个测试通过、无跳过**（原123 + M6新增39）。独立存档核验261项通过：54个运行摘要产物、15个M4/M2来源、41个运行时Python源码哈希，全部实际safetensors形状/参数量/字节和评分一致；只读tokenizer复核分词/解码，不执行基础模型前反向。Adaptive三步数值和训练后adapter字节与既有M5同设置训练核心精确一致（样本ID重新命名），不冒充完整M5 CLI重跑或三基线重复实验。

已核查 Table2 / Section5.3、第5.4节和Appendix C.3；合成评分、短CPU训练和Random seed均为复现决策。Git分支 `repro/m6-unified-baselines`，基线commit `577527ad0ec343941bbed0b6519ff3544e301cc7`仍为M1，通过dirty和源码指纹记录实际M6实现；未自动提交/推送，未下载或改变依赖。

**门禁**：M6四项在上述工程smoke边界内通过；未执行正式commonsense/math/PPL评估、三方法完整重复、三方法fresh-base checkpoint恢复或GPU性能实验。当前评估接口复用在比较CLI中，尚无独立正式 `evaluate.py`。完成后停止，M7需另行指令，并由用户批准数据/GPU成本及冻结待核验协议。

### M7 — 正式复现实验（需用户批准 GPU 与数据成本）

任务：Qwen2.5-1.5B、N=250、Top-8、rank16、alpha32、Q/K/V。先做 pilot 确认显存/吞吐，再跑完整训练与基线。

- [x] 执行前核对论文表格、数据版本、prompt、评估协议（M7首项：文献/固定来源元数据与文本评分协议已核查，7项复现决定已由用户确认；不是数据或GPU运行验收）。
- [ ] 实际数据下载后验收许可、字节、行数、抽样清单、真实分词/截断与拆分隔离；验证1.5B实际模块结构与输出。
  - [x] 实现标准库离线数据审计器及22项测试；经单独批准发布并在新数据目录检出完整SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb`；固定DATA_LICENSE/代码LICENSE字节通过，不代表数据内容已验收。
  - [ ] 完成9个JSON获取及实际离线审计（辅助README超时后停止；原窗口续接在获取前因到期拒绝；13:37 UTC核查全部JSON、audit、resume目录不存在，未改变seed/样本或重置时限）。
  - [x] **九文件获取与离线文本审计已实际完成（2026-11-15）**：9000秒窗口内实际1,320.791秒完成9/9三重核验并运行审计，`TEXT_GATE_PASS_TOKENIZER_PENDING`；训练170,420抽10,000、校准250、八任务各1,000（OBQA500），字段统一四字段；跨拆分sample_id与prompt哈希交集全为0。组内重复prompt（train 107、social_i_qa 8）记录未处理。仍不含真实tokenizer/截断/token级交集与语义去重。
  - [x] **M7 真实 tokenizer 管线实现（2026-11-15）**：新增 `src/adaptive_lora/m7_tokenization.py`，含BPE边界核验（tokenize(prefix)须为tokenize(prefix+answer)前缀，不可猜切点）、EOS显式追加、右截断+无监督token报错、pad到8倍数、loss mask与token级身份交集比较；使用固定revision Qwen2.5-0.5B真实分词器，17项含12项真实分词测试通过，全套224项回归通过。
  - [x] 续轮另获批准的新900秒 `autodl` 窗口：890.0009秒仅写入一个部分训练文件13,631,488/96,529,722 bytes，审计未启动（该部分文件后续被下一窗口接管续传）。
  - [x] 已实现并本地测试**跨窗口累积获取工具**（Range续传+长度/Git blob/SHA256三重核验+state.json只登记已核验文件），17项合成测试通过、207项回归通过。
  - [x] v2 增补前缀接管与审计可重复进入；16项v2合成测试通过。首次9000秒执行曾被工具权限分类器拦截、未执行；经用户添加项目级Bash规则后按其原文执行成功。
  - [x] **1.5B 权重与28层结构实测通过（2026-11-16）**：另获批准的固定revision下载与单次CPU结构核验实际完成。权重3,087,467,144 bytes、LFS SHA256 `a961db72…c796` 双重核验；pinned revision由HF快照目录名实证；`config.json` Git blob SHA1 `169d606d…aaab` 与配置记录一致。28层索引连续、路径唯一、Q/K/V齐全；`self_attn` 输出 `(Tensor[1,4,1536], None)`、`mlp` 输出 `Tensor[1,4,1536]`，56个hook全触发；单次前向 logits `[1,4,151936]`。依赖 torch 2.6.0+cu124 / transformers 5.22.2 / peft 0.21.2 与 `environment_candidate` 一致。状态 `STRUCTURE_GATE_PASS_TOKENIZER_RUN_PENDING`；**真实1.5B tokenizer端到端与token级交集仍未运行**。GPU 0 MiB、未训练、未push。详见[结构门禁报告](reports/m7_structure_1_5b_20261116.md)。
  - [ ] 在1.5B真实tokenizer上端到端运行分词管线（BPE边界、512右截断、pad到8、答案+EOS mask）并对实际抽样行产出token级身份交集；模块已实现且仅在0.5B分词器上验证过，共享词表不等于行为等价。
- [ ] 准备获批的独立Linux/CUDA环境，完成CUDA/AMP/梯度累积与GPU checkpoint实现及有界pilot。
  - [x] 环境准备范围另获用户批准；已公开发布代码SHA `9051fdfe86379aa3502e66bf1a586d4eb00565ba`，并核验GitHub分支引用。
  - [x] SSH恢复后在数据盘新目录检出完整批准SHA，创建不继承平台site-packages的隔离venv，确认平台包元数据与源码字节未变。
  - [x] 隔离环境依赖/项目安装、导入与适用CPU验收：另获有限缓存续接授权后，67个wheel全批官方哈希通过，离线安装/pip check/11项固定版本导入通过；Linux183项CPU单测通过、2项真实tokenizer用例NOT RUN，不含CUDA/pilot。
  - [ ] 补验Linux两项真实tokenizer用例（固定0.5B缓存缺失，本轮未下载，不将183项通过写成185项全通过）。
- [ ] 保存训练日志、GPU 耗时和峰值显存。
- [ ] 完成统一评估；不人为追求论文数值而改动结果。

M7首项进展（2026-11-15）：用户明确指令进入M7；重新读取PDF第7–9、18–19页及数据引用，形成 [已确认协议](M7_PROTOCOL.md)、[固定配置](configs/qwen2_5_1_5b_commonsense_r16.json)、[纯标准库协议与评分模块](src/adaptive_lora/m7_protocol.py) 和 [离线检查入口](scripts/check_m7_protocol.py)。首项审查当时仅确认7项复现决定，没有批准公开推送、依赖安装、模型/数据下载或付费GPU作业；后续范围授权见下述环境准备进展。预检器即使返回REVIEWED_NOT_EXECUTABLE，也始终不授予执行权限、不标记runtime通过；真实数据、CUDA训练与效果实验仍未执行。证据见 [M7首项报告](reports/m7_preflight_20261115.md)；同日较早的SSH成功与资源快照见 [重启后盘点](reports/autodl_postrestart_20261115.md)。

M7交付/环境准备首轮记录（2026-11-15，首项之后）：用户另行批准公开推送、指定SHA部署、新隔离环境依赖安装和离线CPU验收。批准SHA `9051fdfe86379aa3502e66bf1a586d4eb00565ba` 已发布至repro/m7-run-preflight，包含6个此前未发布提交；推送前审查99个唯一Git文件对象，本地185项测试18.466s通过、无跳过。当时两次SSH均在TCP阶段返回Connection refused，未进入公钥认证，未远端检出或安装；该历史记录见 [交付与环境准备报告](reports/m7_environment_20261115.md)，不再代表当前SSH状态。

M7环境恢复续轮（2026-11-15，用户“已重新打开，继续”）：08:17 UTC严格SSH与实时盘点通过；在数据盘新目录detached checkout批准SHA并新建隔离venv。官方PyPI安装540秒超时；另获单一TUNA镜像、逐wheel官方哈希及一次30分钟尝试批准，67个wheel共3,179,040,252 bytes已缓存。安装前核验因自编工具误计setuptools内嵌vendor元数据而失败，不是已发现hash不符；本地回归先红后绿，修正后11项通过。缓存续接准备被工具权限拦截，未重试安装或重置时限；09:21 UTC只读复查确认venv仍只有pip24.0、平台包元数据和远端源码字节不变。本地项目185项离线测试19.699s通过、无跳过；远端导入/CPU验收仍NOT RUN。环境验收仍BLOCKED，无模型/数据下载、CUDA或训练；详见 [恢复报告](reports/m7_environment_resume_20261115.md) 与 [机器索引](reports/m7_environment_resume_20261115.json)。

M7环境CPU验收续轮（2026-11-15）：用户另行批准一次最多1800秒的缓存核验/离线安装/CPU操作，旧09:28:32 UTC截止及失败记录保留，不更改权限或绕过拒绝。12:01 UTC预检确认67个wheel字节与此前证据相同、venv仍pip-only；12:07–12:09 UTC在118.765724秒内完成67项全批官方核验（复用52份官方JSON、仅补14份小JSON及固定Torch哈希）、离线依赖/项目安装、pip check与固定版本导入。Linux适用单测183项8.851s通过，另2项真实tokenizer用例NOT RUN；Windows本地185项21.435s通过，续接门禁14项、原核验11项、解码器4项通过。12:10 UTC只读确认HEAD仍为批准SHA、clean、源码字节与平台包元数据不变，79份证据传输及独立核验通过。状态为PASS_APPLICABLE_CPU，M7整体仍未完成；无新wheel/正式模型/数据下载，无CUDA/pilot、正式训练/评估或push。详见 [CPU环境验收报告](reports/m7_environment_cpu_20261115.md) 与 [机器索引](reports/m7_environment_cpu_20261115.json)。

M7资源/结构门禁预检（2026-11-15，用户“继续完成M7”）：另获固定1.5B/对应tokenizer及9个数据JSON、许可与一次短CPU结构前向批准（单次30分钟/3.3GB、不含CUDA/训练/push）。首个模型LICENSE resolve请求经重定向返回HTTP403；未创建资源目录或启动获取。另获120秒/最多4请求的同镜像raw/API只读诊断批准，实际2请求：raw许可403、固定revision API200（revision/LFS哈希一致、许可标签apache-2.0），但未取得固定许可原文字节，继续条件未满足即停止。12:34 UTC只读确认资源目录仍不存在、源码/Git clean/平台包/venv69项/前次CPU结果哈希未变。本地预检诊断8项0.006s和项目185项18.765s通过；实际资源/数据内容/1.5B结构/所有GPU及正式实验均NOT RUN。既有环境PASS_APPLICABLE_CPU保留，资源门禁BLOCKED_LICENSE_HTTP403，M7整体仍未完成。详见[资源阻塞报告](reports/m7_assets_blocked_20261115.md)与[机器索引](reports/m7_assets_blocked_20261115.json)。

M7独立数据门禁续轮（2026-11-15，用户“继续”）：按需重读PDF Section5.1/Table1及AppendixC.2–C.3；官方HF许可/权重HEAD本地超时、远端Network is unreachable，固定GitHub DATA_LICENSE则可达。用户另批15分钟/120MB数据门禁；实现标准库离线字节/字段/固定抽样/文本交集审计器与22项测试，本地全套207项22.336s通过。SSH Git bundle交付被工具拒绝、未执行；未换打包方式绕过。另获公开发布批准后，将完整2df16eb07bff7990cf9daf93bb189c9d91d06deb（共5个此前未公开提交）推至既有origin新分支并核验引用；远端新目录从公开origin检出精确SHA，旧repo9051fdfe…及venv不替换。13:14:24 UTC启动，固定DATA_LICENSE/代码LICENSE共31,186 bytes通过；辅助README读超时，26.455797秒停止，9个JSON请求尚未发出。用户批准跳过README、仅用原窗口剩余时间续接，但实际入口因13:29:24.449360 UTC原截止已过而拒绝，未重置预算。13:37 UTC只读确认九JSON/audit/resume目录不存在、原失败与两份通知字节不变、新旧Git clean/源码及平台与69包元数据不变；归档核验PASS_ARCHIVE_ONLY，收尾8项0.044s通过。数据状态BLOCKED_NETWORK_TIMEOUT_WINDOW_EXPIRED，实际行数/抽样/文本交集及tokenizer/1.5B结构/CUDA/训练均NOT RUN，不将超时误记为数据重叠；M7整体不勾选。详见[数据门禁报告](reports/m7_data_gate_20261115.md)与[机器索引](reports/m7_data_gate_20261115.json)。本地报告提交不自动push或替换远端批准SHA。

### M8 — 复现报告

- [ ] 记录模型/环境/数据/配置/命令/commit。
- [ ] 给出重要性曲线、Top-K 层与稳定性。
- [ ] 给出准确率、训练参数量、成本对照。
- [ ] 以表格区分“论文原设定 / 我们的实现 / 偏差与解释”。

## 3. 单任务的固定工作闭环

1. 阅读 `CLAUDE.md`、`PAPER_SPEC.md` 和当前 `REPRO_PLAN.md`。
2. 先给出**本轮仅做什么**、修改文件、设计、风险、测试计划。
3. 编写最小代码与关键单元测试。
4. **实际运行**可运行的测试；不能运行明确写 `NOT RUN`。
5. 汇报执行命令、PASS/FAIL、修改列表、日志位置、论文假设。
6. 用户验收后才继续下一任务或阶段。

统一任务报告模板：

```markdown
## 阶段和任务
## 文件变更
## 论文依据与实现假设
## 执行命令
## 测试结果（PASS / FAIL / NOT RUN）
## 关键输出和文件路径
## 待解决问题
## 下一步（不自动执行）
```

## 4. 实验输出规范

每个实验单独保存：

```text
results/<experiment_id>/
├── resolved_config.yaml
├── environment.json
├── git_commit.txt
├── sample_ids.json
├── layer_importance.csv
├── selected_layers.json
├── trainable_parameters.json
├── metrics.json
└── run.log
```

其中未产生的产物允许不存在，不能创建空指标文件假装已运行。

## 5. 第一条可执行指令

```text
请先完整阅读 CLAUDE.md、PAPER_SPEC.md 和 REPRO_PLAN.md。
当前只执行 M0：项目骨架和环境探查；如果仓库里 M0 已通过验收，则报告证据并等待指令，不擅自进入 M1。
先给出简短计划，再修改必要文件并运行可运行的测试。
不要下载大模型、安装重要依赖或开始训练。
结束时按 CLAUDE.md 固定格式汇报，并列明 PASS/FAIL/NOT RUN。
```
