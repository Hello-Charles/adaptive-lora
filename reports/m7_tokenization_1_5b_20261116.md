# M7 tokenizer 门禁通过：1.5B 实分词、mask 不变量与 token 级身份隔离（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。**真实 1.5B tokenizer 端到端运行通过，状态 `TOKENIZATION_PASS_TOKEN_IDENTITY_DISJOINT`。M7 整体仍未完成**——CUDA/AMP/梯度累积、pilot、正式训练与评估均未验收。**另有 1 项协议问题需要用户决定（见末节）。**

本轮实际完成：

1. **10,000 条训练样本全部构造成功，0 报错**。`build_supervised_example` 的 BPE 边界核验没有在任何一行失败，也没有出现截断后无监督 token 的情形。
2. **mask 不变量逐行校验**：prompt 段 `labels=-100`、mask 全 1；监督段 labels 为非负 token id、mask 全 1；pad 段 `labels=-100`、mask 全 0；序列长度全部为 8 的倍数。三段之和必须恰好等于序列长度，不留空也不重叠。
3. **训练侧从未触发 512 截断**：完整序列长度 min 57 / p50 91 / p95 307 / p99 332 / max 468。监督段长度 6–7 token（答案加 EOS），与八任务单词标签一致。
4. **校准 250 条全部未截断**：前缀长度 min 51 / p50 84 / p95 310.4 / max 340，`prediction_position` 均等于最后一个有效 token。
5. **token 级身份交集全为 0**：训练↔8 任务、校准↔8 任务共 **16 个组合**全部 `token_identity_disjoint=true`，总交集数 0。
6. **抽样回放被证明是同一批行**：重放得到的 `sampling_sha256 = 76a61e01…8c3b` 与上一轮文本门禁记录逐字相同；逐行核对 18,750 条的 `prompt_sha256` 与 `sample_id`，不一致数为 **0**。

### 执行数据

| 项目 | 实际值 |
|---|---|
| 执行时长 | 15.528 秒（总调用时间，含 SSH 建连与解释器启动） |
| 训练构造 | 10,000 / 10,000 成功，0 错误 |
| 校准 | 250 条，0 截断 |
| 评估抽样 | 7,500 条（1,000×7 + 500） |
| 评估前缀超 512 | **1 条（piqa）** |
| token 级交集组合 | 16，全部为空 |
| CUDA | 未使用（`CUDA_VISIBLE_DEVICES=''`） |
| 模型前向 | 未执行 |
| Python / 库 | 3.12.3；torch 2.6.0+cu124、transformers 5.22.2、tokenizers 0.23.3、peft 0.21.2 |

### 逐任务前缀 token 分布（评估侧）

| 任务 | 抽样 | 最小 | p95 | 最大 | 超 512 |
|---|---:|---:|---:|---:|---:|
| boolq | 1,000 | — | 56.0 | 62 | 0 |
| piqa | 1,000 | — | 197.0 | **521** | **1** |
| social_i_qa | 1,000 | — | 104.0 | 134 | 0 |
| hellaswag | 1,000 | — | 328.0 | 361 | 0 |
| winogrande | 1,000 | — | 87.0 | 94 | 0 |
| ARC-Easy | 1,000 | — | 143.0 | 194 | 0 |
| ARC-Challenge | 1,000 | — | 155.0 | 237 | 0 |
| openbookqa | 500 | — | 107.0 | 152 | 0 |

## 关键工程事实：模块如何在不推送的前提下执行

远端部署 commit 是已批准的 `2df16eb…`，而 `src/adaptive_lora/m7_tokenization.py` **首次出现在仍停留本地的 `45d6278`**，从未推送（`origin/repro/m7-data-preflight` 仍停 `2df16eb`，`origin/repro/m7-run-preflight` 停 `9051fdf`）。

处理方式：驱动脚本以 base64 内嵌该模块的**已提交字节**，执行前校验 SHA256 必须等于 `d2542550c42a8e5f775707a9f7eaa724113e7f700832edf76eb4dc0766b21d4a`，再在内存中 `exec`。因此：

- 执行的逻辑可证明等同提交 `45d6278` 的文件字节；
- 不在远端写任何副本、不修改已部署 repo、不 push；
- 远端 `git` 前后仍为 `{commit: 2df16eb…, dirty: false}`，脚本在启动时即断言这一点，不符即中止。

本地构建器 `build_tokenization_runner.py` 做了往返自检：从生成的脚本里取回字符串字面量、base64 解码、比对已提交文件，确认逐字节相同才写盘。

## 附带结论：pinned 0.5B 与 pinned 1.5B 的 tokenizer 文件逐字节相同

上一轮报告保留了"共享词表不等于行为等价"的疑虑。本轮把它实测掉了：

| 文件 | 字节 | pinned 0.5B 与 pinned 1.5B |
|---|---:|---|
| tokenizer.json | 7,031,645 | **完全相同** |
| tokenizer_config.json | 7,228 | **完全相同** |
| vocab.json | 2,776,833 | **完全相同** |
| merges.txt | 1,671,839 | **完全相同** |

两侧 `tokenizers 0.23.3` / `transformers 5.22.2` 也相同。而 `config.json` 有意不同（隐藏维、层数不同），作为该比较具有区分力的对照。

**范围限定**：这只覆盖上述 4 个分词器文件。0.5B 与 1.5B 是不同模型，本结论**不**推广到权重、隐藏维、层数或注意力头数。含义是：在 0.5B tokenizer 上验证过分词逻辑，对 1.5B 的分词行为是可迁移的；本轮仍在 1.5B 快照上实跑了全流程，没有只靠这个推论。

## 文件变更

生产代码、`src/`、`scripts/`、`tests/`、`configs/` 本轮**无改动**。远端已部署 repo 未改动、未新增文件（除写入结果 JSON）。

本地：本报告、[机器索引](m7_tokenization_1_5b_20261116.json)，以及 README/REPRO_PLAN 状态更新。驱动脚本、构建器、核验器、诊断探针与原始日志留在 Git 忽略的 `.cache/m7-20261115-assets-04/`。报告提交不自动 push。

## 论文依据与工程假设

- **[论文] Appendix C.1–C.2（第 18–19 页）：** 训练最大 512、超长截断、短样本 pad 到 8 的倍数。本轮实测训练侧最大 468，从未触发截断；pad 到 8 的倍数逐行成立。
- **[复现决策，已在协议确认]** `supervision`：prompt labels=-100，答案+显式 EOS 监督；完整序列右截断 512，无可预测监督 token 时报错。论文未规定 loss mask 与截断方向。
- **[复现决策，已在协议确认]** `sampling`：train/cal/eval 各用局部 seed 42；先抽 10,000，再从该子集抽 250。本轮重放并证明是同一批行。
- **[必须保留的局限]** token 级身份只覆盖**直接 token 序列身份**。`semantic_deduplication_checked`、`prompt_embedded_answer_checked`、`pretraining_contamination_checked` 三个 flag 均为 `false`，不因本轮通过而改变。
- **[既有缺口，未变]** 许可原文仍未取得，`metadata_source` 仍为 `mirror_only_not_independently_authenticated`。

### `pad == eos == 151643` 已实测确认

报告中 `pad_equals_eos_observed = [True]`，且逐行校验了 pad 位置必须同时满足 `labels=-100` 与 `mask=0`。**判定有效性只能靠 mask 与 labels，不能靠 token id**——这一点在 1.5B 上已由 10,000 行的实际数据确认，不再是推断。

## 执行命令

```bash
# 1) 本地：把已提交模块字节嵌入驱动脚本（含往返自检）
.venv/Scripts/python.exe -I -B -X utf8 .cache/m7-20261115-assets-04/build_tokenization_runner.py

# 2) 远端：严格 SSH + 隔离 venv 解释器 + stdin，输出独占
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  autodl '/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python -I -B -' \
  < .cache/m7-20261115-assets-04/run_tokenization_15b.py \
  > .cache/m7-20261115-assets-04/tokenization-02.json \
  2> .cache/m7-20261115-assets-04/tokenization-02.log

# 3) 取回结果 + 远端 tokenizer 文件清单（只读）
ssh … autodl 'cat …/m7_tokenization_1_5b.json' > .cache/…/tokenization-report-01.json
ssh … autodl 'for f in tokenizer.json …; do … sha256sum …; done' > .cache/…/remote-tokenizer-inventory.tsv

# 4) 本地独立核验（不导入项目模块、不重新分词）
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-04/verify_tokenization_run.py
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| 10,000 训练样本构造（BPE 边界） | **PASS** | 10,000/10,000，0 报错，无跨边界歧义 |
| mask / pad 不变量 | **PASS** | 逐行断言 prompt/supervised/pad 三段、长度 8 倍数、EOS 结尾 |
| 训练侧 512 截断 | **PASS（未触发）** | max 468 < 512，truncated_count=0 |
| 校准 250 条与 prediction_position | **PASS** | 0 截断，位置均为末个有效 token |
| 评估 7,500 条前缀分词 | **PASS** | 八任务行数与配置一致 |
| token 级身份隔离 | **PASS** | 16 组合全空，总交集 0 |
| 抽样回放与审计一致 | **PASS** | `sampling_sha256` 相同；18,750 条 prompt 身份 0 不符 |
| `pad == eos` 实测确认 | **PASS** | `pad_equals_eos_observed=[True]` |
| 0.5B 与 1.5B tokenizer 字节同一性 | **PASS** | 4 文件 SHA256 与字节数全部相同 |
| 远端代码停在批准 SHA | **PASS** | 启动断言 `2df16eb…` 且 clean |
| 本地独立核验 | PASS | `PASS_TOKENIZATION_ARCHIVE_ONLY` |
| 项目离线回归 | PASS | 224 项，无跳过 |
| 语义去重 / 题面内嵌答案 / 预训练污染 | **NOT RUN** | 三个 flag 均 false |
| CUDA / AMP / 梯度累积 / GPU checkpoint | **NOT RUN** | 未进入 |
| pilot / 正式训练 / 正式评估 | **NOT RUN** | 未进入 |

## 运行日志和输出路径

本地 [.cache/m7-20261115-assets-04/](../.cache/m7-20261115-assets-04/)：

- `build_tokenization_runner.py`、`run_tokenization_15b_template.py`、`run_tokenization_15b.py`（33,422 bytes，内嵌已校验模块）
- `tokenization-02.json` / `.log`：成功运行；stderr 为空
- `tokenization-01.json` / `.log`：**保留的第一次失败**（我自己写错的不变量检查把 pad 段算进监督区，`labels[prompt_n:]` 误判 `-100` 为负标签；第 1 行即中止，未写任何输出）
- `tokenization-report-01.json`：远端结果原样取回，8,318 bytes，SHA256 `825488439fa3e4754237ec4771ba340667b4600e7101db7af97b8515bba9d5ab`
- `verify_tokenization_run.py`、`tokenization-verification.json`
- `remote-tokenizer-inventory.tsv`：远端 tokenizer 文件清单
- `probe_long_prefix.py` / `long-prefix-probe-01.json`、`probe_truncation_effect.py` / `truncation-effect-01.json`：超长前缀只读诊断

远端 `/root/autodl-tmp/adaptive-lora-m7-data-20261115-04/`：`m7_tokenization_1_5b.json`（与本地副本逐字节相同）。既有 `audit/`、`data/`、模型缓存未改动。

**原始数据与权重均未传回本地、未进公开仓库。**

## 未解决问题

1. **1 条 piqa 评估前缀超 512，需协议决定**（下节详述）。
2. 语义去重、题面内嵌答案、预训练污染三项未检查。
3. CUDA/AMP/梯度累积与 GPU checkpoint 未实现；现有 CLI 仍有 CPU 与 1–3 步硬门禁。
4. 许可原文缺口未解。
5. 组内重复 prompt（train 107、social_i_qa 8）已记录未处理。
6. `m7_tokenization.py` 只存在于本地 `45d6278`，远端仍无法直接 import。若后续步骤要反复调用它，应优先另行申请一次推送与重新部署，而不是继续靠内嵌字节。

## 需要用户决定的协议问题：piqa 的 521 token 前缀

`M7_PROTOCOL.md` 已预先规定："超长校准/评估前缀的实际分布、保留Response边界与长度处置须在真实tokenizer数据门禁中核验；如既定规则不能处理实际样本，先报告并申请协议修订，不自行丢样本。"本轮正好命中这一条。

**事实**（`piqa`，`source_index=314`，抽样内）：

- 前缀 2,395 字符 → **521 token**，超出协议 512 上限 9 个 token
- 来源是超长 `instruction`（2,255 字符），`input` 字段为空
- 右截断到 512 会丢掉末尾 9 个 token，解码为 `" solution1/solution2\n\n### Response:\n"`
- 保留部分末尾停在 `"Answer format:"`

即：截断同时丢掉了**答案格式提示**（`solution1/solution2`）和**Response 标记**。模型将从中途的 `"Answer format:"` 之后续写，而不是在规范的 `### Response:` 之后续写。

**影响面**：7,500 条评估抽样中的 1 条（0.013%）。训练侧与校准侧均无截断，因此不影响训练与校准。

**我没有自行处理**：没有丢样本、没有重抽、没有改用别的截断方向，该行在报告中如实计数。

## 下一步建议（不自动执行）

先请你就上面这条 piqa 样本做出协议决定；确定后再进入 CUDA/AMP 与梯度累积实现——那部分与这条样本无关，可并行推进，但正式评估必须等协议定案。

本轮未使用 CUDA、未训练、未 push，未修改配置、数据或既有审计产物。
