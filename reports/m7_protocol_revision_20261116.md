# M7 协议修订：评估侧保留完整前缀不截断（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。**第 8 项复现决定 `evaluation_truncation = none_keep_full_prefix` 已由用户确认并落地**，状态 `PASS_TOKENIZATION_V2_AND_EQUIVALENCE`。M7 整体仍未完成——CUDA/AMP/累积、pilot、正式训练与评估均未运行。

### 触发原因

上一轮 tokenizer 门禁实测发现 1 条 piqa 评估前缀为 **521 token**，超出协议的 512 上限：

| 项目 | 值 |
|---|---|
| 任务 / 来源索引 | piqa / 314 |
| 前缀 token | 521 |
| 超出 | 9 |
| instruction 字符 | 2,255（`input` 字段为空） |
| 右截断到 512 会丢掉 | `" solution1/solution2\n\n### Response:\n"` |
| 保留部分末尾 | `"Answer format:"` |

右截断同时丢掉**答案格式提示**（`solution1/solution2`）与 **Response 标记**，模型将从中途的 `"Answer format:"` 之后续写。`M7_PROTOCOL.md` 对此已有明文规定："超长校准/评估前缀的实际分布、保留Response边界与长度处置须在真实tokenizer数据门禁中核验；如既定规则不能处理实际样本，先报告并申请协议修订，不自行丢样本。"

用户选择**评估侧不截断**。理由：512 是论文的训练长度上限，评估生成不受该约束；保留完整前缀也符合协议"训练、校准、生成共享同一前缀字节"的意图。实际最长 521 token 远低于模型 `max_position_embeddings = 131072`。

## 最重要的结论：本次修订不改变任何计算结果

这是必须讲清楚的一点，否则容易误以为重跑改变了数值。

**v1（修订前）从未真正截断过评估前缀。** 它调用 `tokenize_ids`，而该函数固定 `truncation=False`，只是把"超过 512 的条数"统计成 `prefix_over_max_length`。所谓"会丢掉 9 个 token"是对**若按训练规则截断**的后果分析，不是 v1 的实际行为。

因此本次修订的实质是：**让声明的策略与既有行为一致，并提供一个显式的、可测试的代码路径** `build_evaluation_input`，其中 `truncated` 恒为 `False`、超出参考长度只标注不裁剪。

v2 报告与 v1 报告逐项对比，以下全部**完全相同**：

| 对比项 | 结果 |
|---|---|
| `train.full_token_counts` | 相同 |
| `train.supervised_token_counts` | 相同 |
| `calibration.prefix_token_counts` | 相同 |
| `evaluation.*.prefix_token_counts`（八任务） | 相同 |
| `token_identity_overlap`（16 组合） | 相同 |

本地核验器以 `IDENTICAL_NUMERIC_RESULT` 作断言结论，而非仅作说明。

## 变更内容

| 位置 | 变更 |
|---|---|
| `configs/..._r16.json` → `protocol` | 新增 `"evaluation_truncation": "none_keep_full_prefix"`；`truncation`（训练）**未改** |
| `configs/..._r16.json` → `decisions` | 新增第 8 项 `evaluation_truncation`，status `confirmed` |
| `src/adaptive_lora/m7_protocol.py` | `_FIXED_PROTOCOL` 增加同一字段；`DECISION_IDS` 增加该 id |
| `src/adaptive_lora/m7_tokenization.py` | 新增常量 `M7_EVALUATION_TRUNCATION` 与函数 `build_evaluation_input` |
| `tests/test_m7_tokenization.py` | 新增 4 项测试（不截断、短前缀不标注、与校准共享字节、非法输入拒绝） |
| `tests/test_m7_protocol.py`、`tests/test_check_m7_protocol.py` | 决策数 7 → 8 |
| `M7_PROTOCOL.md` | 第 3 节标题改"八项"，新增决定行与修订记录；**并纠正已失效的配置哈希说明** |
| `PAPER_SPEC.md` | 决策数改 8，新增评估截断决定条目 |

训练与校准侧**完全未改**：仍是右截断 512、无可预测监督 token 即报错。不丢样本、不重抽、不缩小评估分母。

## 配置哈希变化（哈希链影响必须如实说明）

| 字段 | 修订前 | 修订后 |
|---|---|---|
| 配置文件**字节** SHA256 | `6f715cef71d1fbe350541bd6b45524a4134ea2d879b1bee266468c4afb30081c` | `4b0405b028abbf79d0e29cdb660d691dbbe10e96e4dc59b3792448595038e616` |
| `protocol_sha256`（规范 JSON） | `d60f13752b4b8628d0ed8c8d9bb843ce92c8221b68d17cd8338a8447efa81c82` | `bcaf19a7a5acb8a3ddcb794d8101b8cba94047adf9194ccbc023b5cb65467802` |

**影响**：早先的离线审计产物（`audit/summary.json`、`audit/audit.json`）与结构、分词报告记录的都是**修订前**哈希。它们属于历史证据，**不再对应当前配置**。`M7_PROTOCOL.md` 里原有的"配置未修改任何字节、因此不破坏哈希链"一句已随之失效，本轮已改写为如实说明。

本轮**没有**重跑数据审计，也没有覆盖任何既有审计产物——那需要单独授权，且当前数据与抽样本身未变，重跑的必要性应由用户判断。

## 一个必须记录的不一致：远端部署的配置仍是修订前字节

v2 运行使用远端部署的 `configs/..._r16.json`，其字节仍是 `6f715cef…`（实测：`evaluation_truncation` 出现次数为 0）。因此 **v2 报告里的 `config_sha256` 如实是修订前的 `d60f1375…`**。

这不是错误，而是当前分发的真实状态：修订后的配置只在本地，远端还未更新。本次修订的语义是通过**内嵌的模块字节**（`a09f80f3…`）带入执行的，而不是通过部署配置。

本地核验器显式断言了这一事实（`runs_used_deployed_config: true`），避免把"报告里是旧哈希"误读成"运行用了错配置"或反过来掩盖它。

后续若要让远端配置与新协议对齐，需要另行批准一次推送或配置同步；在此之前，任何远端运行的 `config_sha256` 都会继续是旧值。

## 执行命令

```bash
# 构建 v2 驱动（内嵌修订后模块字节，含往返自检）
.venv/Scripts/python.exe -I -B -X utf8 .cache/m7-20261115-assets-04/build_tokenization_runner.py

# 远端执行
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  autodl '/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python -I -B -' \
  < .cache/m7-20261115-assets-04/run_tokenization_15b_v2.py \
  > .cache/m7-20261115-assets-04/tokenization-v2-01.json \
  2> .cache/m7-20261115-assets-04/tokenization-v2-01.log

# 本地核验（含 v1/v2 数值等价断言）
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-04/verify_tokenization_v2.py

# 回归
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| 配置与协议校验（含新字段与新决策） | **PASS** | `load_m7_config` 通过，`review_status = REVIEWED_NOT_EXECUTABLE` |
| 决策数 7 → 8 的断言更新 | **PASS** | 两处测试已同步，无遗留失败 |
| `build_evaluation_input` 不截断 | **PASS** | 4 项新测试；`truncated` 恒 False，521 token 前缀原样保留 |
| 与校准共享前缀字节 | **PASS** | 未超限时两者 `input_ids` 与 `prefix_ids_sha256` 逐项相同 |
| v2 远端运行 | **PASS** | 10,000/10,000 构造成功、0 错；16 组合交集全空 |
| v1 与 v2 数值等价 | **PASS** | 五类统计逐项相同，`IDENTICAL_NUMERIC_RESULT` |
| v2 使用修订后模块字节 | **PASS** | 内嵌 SHA256 `a09f80f3…` 与本地文件一致 |
| 部署仍停在批准 SHA 且 clean | **PASS** | `2df16eb…`，dirty=false |
| 本地独立核验 | PASS | `PASS_TOKENIZATION_V2_AND_EQUIVALENCE` |
| 项目离线回归 | PASS | **228 项，无跳过** |
| 语义去重 / 内嵌答案 / 预训练污染 | **NOT RUN** | 三个 flag 仍为 false |
| CUDA / AMP / 累积 / GPU checkpoint | **NOT RUN** | 未进入 |
| pilot / 正式训练 / 正式评估 | **NOT RUN** | 未进入 |
| 数据审计按新配置重跑 | **NOT RUN** | 未获授权；既有产物保持修订前哈希 |

## 运行日志和输出路径

本地 [.cache/m7-20261115-assets-04/](../.cache/m7-20261115-assets-04/)：

- `run_tokenization_15b_template_v2.py` / `run_tokenization_15b_v2.py`（36,645 bytes，内嵌 `a09f80f3…`）
- `tokenization-v2-01.json` / `.log`：成功运行，stderr 为空
- `tokenization-report-v2-01.json`：远端结果原样取回，8,429 bytes，SHA256 `ee175054b1388bea89d9d8c9824910ada8d502f0275e47131a97fbb7d8f86501`
- `verify_tokenization_v2.py` / `tokenization-v2-verification.json`
- 修订前证据保留：`tokenization-report-01.json`、`tokenization-01.json`（首次失败）、`tokenization-02.json`

远端 `/root/autodl-tmp/adaptive-lora-m7-data-20261115-04/m7_tokenization_1_5b_v2.json`。修订前结果 `m7_tokenization_1_5b.json` **保留未删**。

## 未解决问题

1. **远端配置未同步**：部署仍是修订前 `6f715cef…`。要让远端运行报告新协议哈希，需另行批准推送或配置同步。
2. **既有审计产物绑定旧哈希**：`audit/*.json` 与结构、分词报告记录的都是修订前值。是否需要按新配置重跑数据审计，需用户判断。
3. 语义去重、内嵌答案、预训练污染三项未检查。
4. CUDA/AMP/累积与 GPU checkpoint 未实现。
5. 许可原文缺口未解。
6. `m7_tokenization.py` 仍只存在于本地；远端无法直接 import，继续靠内嵌字节执行。

## 下一步建议（不自动执行）

可选的两条，都需要你决定：

- **同步远端**：批准一次推送或配置同步，让远端 `config_sha256` 与新协议对齐；这也顺带解决模块无法直接 import 的问题。
- **直接推进 CUDA/AMP 与梯度累积实现**：与上述同步无关，可在 CPU 侧先实现并测试。

正式评估必须在前述任一路径明确后再进行——届时评估侧将走 `build_evaluation_input`，piqa 那条 521 token 前缀按完整长度送入生成。

本轮未使用 CUDA、未训练、未 push；未改动数据、审计产物或远端部署。
