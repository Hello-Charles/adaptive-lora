# M7 末组余量与训练顺序决定落地（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。**用户确认了两项此前被显式暴露的协议空白**，本轮把它们从"待决定"落地为受门禁保护的确定值，并把 `drop` 从"拒绝"改成"真正丢弃并记账"。

状态 `PARTIAL_GROUP_AND_ORDER_CONFIRMED_GPU_NOT_RUN`：决定已固定、代码与测试可验证；真实 CUDA、真实 10,000 条装配、pilot 与正式训练仍未运行。

本轮没有占用 GPU、没有下载、没有训练、没有 push。

### 用户的两个决定

| 问题 | 决定 | 写入位置 |
|---|---|---|
| 训练顺序（论文只写"3 epochs"） | `none`——每 epoch 沿用 seed42 抽样顺序，不打乱 | `training.training_order` |
| 末组余量（每 epoch 末组仅 2 个 micro-batch） | `drop`——丢弃末尾不足一组的 micro-batch | `training.final_partial_group` |

两项都写进配置 JSON 的 `training` 块（与 `epochs`、`micro_batch_size` 同类），并加入 `m7_protocol.py` 的 `_FIXED_TRAINING` 固定值门禁：改配置即被预检拒绝。决策数由 8 增至 10。

## 这一轮真正改动的是什么

### `drop` 原来是"报错"，现在是"丢弃"

此前 `plan_groups` 把 `drop` 与 `error` 并列处理，两者都直接抛异常。那等于**没有 drop 语义**——用户选 `drop` 会得到一次崩溃。本轮改成：

- `plan_groups(count, accumulation, "drop")` 只返回完整组，末尾余量不追加；
- `run_accumulation` 只对完整组前向与步进，被丢弃的 micro-batch 从不进入 `forward_loss`；
- 返回值新增 `micro_batches_dropped`，把丢弃量如实报出。

### 丢弃必须留清单

"用了 10,000 条"是一句无法核对的话。`iter_epoch_micro_batches` 现在为每个 epoch 产出：

| 字段 | 含义 |
|---|---|
| `samples_used` / `samples_dropped` | 本 epoch 实际训练与丢弃的样本数 |
| `dropped_sample_ids` | 被丢弃样本的完整 `sample_id` 列表 |
| `dropped_sample_ids_sha256` | 该清单的指纹，可与训练日志交叉核对 |
| `used_order_sha256` | 实际参与训练的顺序指纹（区别于全体顺序 `order_sha256`） |

### 48 个"样本-次"，不是 48 条不同样本

这是本轮最容易写错的一句话。数值上 3 epochs × 16 条 = 48，但由于 `training_order = none` 使三个 epoch 的访问顺序**完全相同**，被丢弃的永远是同一批 16 条样本。因此：

- 正确：48 个样本-次未参与更新，涉及同一批 16 条样本；
- 错误：48 条不同样本从未被使用（那只在 `per_epoch_shuffle` 下才可能成立）；
- 也错误：已使用全部 10,000 条。

`describe_batch_plan` 的字段名直接用 `samples_used_total_visits` / `samples_dropped_total_visits`，并附 `dropped_distinctness_note` 说明单位，避免后续报告把"次"读成"条"。

### 论文形状的确定数字

| 项 | 数值 |
|---|---|
| 训练样本 | 10,000 |
| micro-batch 8 × 累积 8 | 有效 batch 64 |
| micro-batch 数 / epoch | 1,250（整满，micro-batch 层不丢样本） |
| 累积组 / epoch | 156 完整组 |
| optimizer steps / epoch | 156 |
| 每 epoch 丢弃 | 2 个 micro-batch = 16 条 |
| 每 epoch 实际训练 | 9,984 条 |
| 3 epochs optimizer steps | 468 |
| 3 epochs 丢弃 | 48 个样本-次（同一批 16 条） |

## 顺带修掉的一个计量错误

`plan_epoch_accumulation` 原先把"参与训练的样本数"算成 `已用 micro-batch 数 × 名义批大小`。
在已确认的 `drop` 策略下这恰好正确（所有被使用的批次都是满批），但同一函数也服务于
`step` 策略——那里末尾是短批，10 条样本会被报成 16 条。现改为按 `micro_batch_sizes`
逐批求和，并新增 `samples_unaccounted` 做守恒校验（已用 + 已丢 + 未记账 = 总数）。
这类误差不会让训练崩溃，只会让报告里的"训练了多少条"安静地错掉，因此必须有测试钉住。

## 顺带修掉的一个真实守卫缺口

补 `drop` 测试时发现：若 `loss` 未连到可训练参数（`.grad is None`），原代码在 `torch.count_nonzero(parameter.grad)` 处以 `TypeError` 崩溃，把"loss 没接上参数"误报成张量类型错误。已在非零统计之前加入缺梯度检查并给出明确 `ValueError`，另配专测 `test_missing_gradient_is_reported_not_crashed`，同时断言失败路径仍清空梯度。

## 文件变更

- `src/adaptive_lora/m7_training.py`：新增 `CONFIRMED_FINAL_PARTIAL_GROUP`；`drop` 改为真丢弃；`describe_batch_plan` 重做计量字段；新增缺梯度守卫
- `src/adaptive_lora/m7_batches.py`：新增 `CONFIRMED_ORDER_POLICY`；`drop` 真丢弃并产出丢弃清单；`plan_epoch_accumulation` 分开报告使用/丢弃
- `src/adaptive_lora/m7_protocol.py`：`_FIXED_TRAINING` 增两项固定值；`DECISION_IDS` 8 → 10
- `configs/qwen2_5_1_5b_commonsense_r16.json`：`training` 块增两键；新增第 9、10 项 decision
- `tests/test_m7_training.py`：`drop` 语义改写、论文形状数字、缺梯度守卫；31 项
- `tests/test_m7_batches.py`：`drop` 清单、默认值、`step` 不丢样本、实际批大小计量；26 项
- `tests/test_m7_protocol.py`、`tests/test_check_m7_protocol.py`：决策数 8 → 10
- `M7_PROTOCOL.md`：第 3 节"八项"→"十项"，新增两行与说明
- 未修改数据、模型、其他模块或既有报告

## 论文依据与工程假设

- **[论文] Appendix C.1–C.2：** 训练 10,000 条、3 epochs、micro-batch 8、accumulation 8。论文**未规定**每 epoch 是否 shuffle，也**未规定**不整除时末组如何处置——这两点都是真正的协议空白，不是本实现偷懒。
- **[复现决策，2026-11-16 已确认]** `training_order = none`：最少额外假设；论文未公开，不声称是原样。
- **[复现决策，2026-11-16 已确认]** `final_partial_group = drop`：代价是每 epoch 少用 16 条，已量化并写入报告。
- **[复现决策，已确认]** 单卡 8×8=64，不同于论文 4 卡 DDP 的 256。
- **[必须保留的局限]** 本层只保证装配与累积**逻辑**正确，不证明真实 tokenizer 下 10,000 条都能构造成功，也不证明 CUDA 数值。

## 执行命令

```bash
# 定向测试
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_m7_training.py"
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_m7_batches.py"
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_m7_protocol.py"
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_check_m7_protocol.py"

# 全套回归
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| `plan_groups` 在 `drop` 下返回只含完整组 | **PASS** | 1,250 → 156 组，丢 2 |
| `drop` 是模块默认值 | **PASS** | 不传参即得 156 组 |
| `error` 策略仍可按需显式使用 | **PASS** | 显式传参后仍报错 |
| `step` 策略使用余量、零丢弃 | **PASS** | 157 组，`samples_dropped=0` |
| `run_accumulation` 在 `drop` 下不前向被丢弃批次 | **PASS** | `on_micro_batch` 只收到 offset 0,1 |
| 丢弃量与样本数如实上报 | **PASS** | `micro_batches_dropped=1` |
| 装配层输出被丢弃样本的完整清单 | **PASS** | `dropped_sample_ids == ["sample-008","sample-009"]` |
| 丢弃清单有独立指纹 | **PASS** | `used_order_sha256 != order_sha256` |
| 论文形状在 `drop` 下的确定数字 | **PASS** | 156 步/epoch、9,984 条/epoch、468 步/48 次 |
| 样本数按实际批大小求和而非名义值 | **PASS** | `step` 下 [8,2] 记 10 条，非 16 条 |
| 48 次 ≠ 48 条不同样本 | **PASS** | `dropped_distinctness_note` 断言 |
| 缺梯度报明确错误而非 `TypeError` | **PASS** | 新增守卫 + 专测 |
| 失败路径清空梯度 | **PASS** | 断言全部 `.grad is None` |
| 决策数 8 → 10 同步 | **PASS** | 协议测试与 CLI 测试 |
| 配置固定值门禁 | **PASS** | `_FIXED_TRAINING` 含两项新键 |
| 项目离线回归 | **PASS** | **292 项，16.732s，无跳过** |
| 真实 10,000 条批量装配 | **NOT RUN** | 需远端真实数据 |
| 真实 CUDA / AMP / batch 前向 | **NOT RUN** | 无 GPU |
| pilot / 正式训练 / 评估 | **NOT RUN** | 未进入 |

## 运行日志和输出路径

- `src/adaptive_lora/m7_training.py`、`src/adaptive_lora/m7_batches.py`、`src/adaptive_lora/m7_protocol.py`
- `configs/qwen2_5_1_5b_commonsense_r16.json`
- `tests/test_m7_training.py`（31 项）、`tests/test_m7_batches.py`（26 项）
- `.cache/m7-20261115-assets-04/partial-group-regression-01.log`

## 哈希链变化

| 项 | 修订前 | 修订后 |
|---|---|---|
| 配置文件字节 SHA-256 | `4b0405b0…` | `e5e8dad3…` |
| 协议 SHA-256 | `bcaf19a7…` | `230a07b8…` |

早先审计产物记录的是修订前哈希，属历史证据，不再对应当前配置。远端部署配置仍是 `6f715cef…`，未同步。

## 未解决问题

1. 远端配置仍为修订前字节（`6f715cef…`），与本轮 10 项决定不一致；需另行批准同步或推送。
2. 真实 10,000 条批量装配未运行（需远端数据 + 1.5B tokenizer，无需 GPU）。
3. 真实 CUDA/AMP 数值、1.5B 前向显存与吞吐未验证。
4. GPU checkpoint（含 optimizer/RNG 续跑）未实现。
5. Linux 两项真实 tokenizer 用例仍缺席（远端无固定 0.5B 缓存）。
6. 语义去重、prompt 内嵌答案、预训练污染未检查。

## 下一步建议（不自动执行）

两个决定已固定，装配层可在远端跑一次**真实数据的 10,000 条装配**（纯 CPU，只需已核验数据与 1.5B tokenizer），产出真实 BPE 边界通过率、micro-batch 计划摘要与被丢弃 16 条的 `sample_id` 清单。这是 pilot 之前最后一块纯 CPU 验证。

本轮未使用 CUDA、未训练、未 push。
