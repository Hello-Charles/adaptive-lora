# M7 数据→micro-batch 装配层（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。**数据装配层已实现并通过 23 项测试**（全套 286 项通过）。状态 `BATCH_ASSEMBLY_IMPLEMENTED_GPU_NOT_RUN`：装配逻辑可测试，真实 CUDA、真实 1.5B 前向、pilot 与正式训练均未运行。

本轮没有占用 GPU、没有下载、没有训练、没有 push。

### 这一层补的是什么

此前三块零件已就位，但彼此没有接起来：

| 已有 | 缺口 |
|---|---|
| `build_supervised_example` 产出**单条**样本 | 无批量入口 |
| `collate_supervised_examples` 把**已切好的一组**拼成批 | 谁决定这一组包含哪 8 条？ |
| `run_accumulation` 消费 `micro_batches` | 谁按 epoch 与累积组切出它们？ |

`m7_batches.py` 就是中间那层：按 epoch 定序 → 切 micro-batch → 保留样本身份 → 与累积组计划对齐。

## 必须显式说明的两处协议空白

### 空白一：论文没有规定训练样本顺序

`PAPER_SPEC.md` 与 `M7_PROTOCOL.md` 对训练只写了「3 epochs」。**没有**规定每个 epoch 是否打乱、如何打乱。配置里也**没有** `shuffle` 键。

我没有替它设默认值。`ORDER_POLICIES` 只有两个选项，且必须由调用者显式传入：

| 策略 | 含义 |
|---|---|
| `none` | 每个 epoch 都按抽样顺序（该顺序本身来自 seed 42 的不放回抽样），最易审计 |
| `per_epoch_shuffle` | 每个 epoch 用 `train_seed` 与 epoch 序号**派生的独立局部 RNG** 打乱 |

默认值是 `none`——**这不代表论文原样**，只是"最少额外假设"的工程选择。若你认为原实现应该是每 epoch 打乱，这个决定需要显式记录。

工程细节：用局部 `random.Random` 而非全局 `random`，因此调用方此前是否调用过 `random.seed` 都不影响结果；有专测断言这一点。两个 epoch 的打乱结果必须不同，也有专测。

### 空白二：末组余量策略仍未定

上一轮已报告：10,000 条 / micro-batch 8 = 1,250 个 micro-batch；1,250 / 累积 8 = **156 组余 2 个 micro-batch**。本层在**装配阶段**就提前校验，`error` 与 `drop` 都直接报错，不等到跑完 epoch 才发现。

`drop` 的报错信息刻意写明"拒绝静默丢弃"，因为少用 16 条/epoch 会污染复现记录。

## 实现要点

| 函数 | 职责 |
|---|---|
| `epoch_order(count, *, epoch, policy, seed)` | 返回该 epoch 的访问顺序，纯函数 |
| `prepare_supervised_examples(rows, records, tokenizer, ...)` | 批量构造监督样本并**补上 `sample_id`** |
| `iter_epoch_micro_batches(examples, *, epoch, micro_batch_size, ...)` | 按顺序切 micro-batch，返回批次与计划摘要 |
| `plan_epoch_accumulation(epoch_plan, *, accumulation, ...)` | 叠加累积组划分，统一核对两层余量 |

三个值得注意的约束：

1. **训练目标取 `output`，不取 `answer`**。`answer` 是评估参考答案，混入训练就是答案泄漏。有专测：把 `answer` 改成任意值后，分词结果必须一字不差。
2. **`sample_id` 由本层补上**。`build_supervised_example` 本身不产出该字段，缺身份会让任何后续失败无法定位到具体样本。
3. **坏样本让整批失败，不跳过、不重抽**。一条 BPE 边界不可判定或截断后无监督 token 的样本必须让数据准备整体失败——否则会静默得到 9,999 条，破坏"10,000 条"这一已确认的抽样数量。

## 文件变更

- **新增** `src/adaptive_lora/m7_batches.py`（约 190 行，含完整中文 docstring）
- **新增** `tests/test_m7_batches.py`：23 项测试
- 未修改配置、数据、其他模块或既有报告

## 论文依据与工程假设

- **[论文] Appendix C.1–C.2：** 训练 10,000 条、3 epochs、max512、pad 到 8 倍数、每卡 batch 8、accumulation 8。本层落实"3 epochs"与"batch 8"的切分部分。
- **[复现决策，已确认]** 单卡 8×8=64，不同于论文 4 卡 DDP 的 256。
- **[复现决策，本轮新增，需你认可]** 训练顺序策略 `none` / `per_epoch_shuffle` 二选一；论文未规定，当前默认 `none`。
- **[必须保留的局限]** 本层只保证**装配逻辑**正确（顺序、切分、身份、不丢样本），不证明真实 tokenizer 下的 BPE 边界对全部 10,000 条都成立——那需要真实数据批量运行。

## 执行命令

```bash
# 定向测试
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_m7_batches.py"

# 全套回归
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| `none` 策略保持抽样顺序 | **PASS** | 三个 epoch 逐一核对 |
| 打乱结果是 0..n-1 的排列且非原序 | **PASS** | 20 条样本 |
| 不同 epoch 打乱结果不同 | **PASS** | 50 条样本，epoch 1 vs 2 |
| 同 seed 同 epoch 可复现 | **PASS** | 两次调用逐项相等 |
| 不依赖全局 `random` 状态 | **PASS** | 先 seed(999) 再 seed(1)，结果相同 |
| 顺序策略与参数校验 | **PASS** | 7 项拒绝路径 |
| 偶数切分与样本不丢不重 | **PASS** | 17 条样本用 `step` 策略 |
| `error` / `drop` 在装配阶段就拒绝余量 | **PASS** | 明确报错文本 |
| `order_sha256` 反映真实访问顺序 | **PASS** | 顺序不同则指纹不同 |
| 论文形状：1,250 micro / 156 组余 2 | **PASS** | 10,000 条完整推演 |
| 累积层余量被显式暴露 | **PASS** | `plan_epoch_accumulation` |
| **训练目标取 `output` 而非 `answer`** | **PASS** | 改写 `answer` 后分词结果不变 |
| `sample_id` 补全、重复与缺失被拒绝 | **PASS** | 3 项 |
| 空训练目标被拒绝 | **PASS** | — |
| **坏样本整批失败而非跳过** | **PASS** | `max_length=4` 触发无监督 token 错误 |
| 项目离线回归 | PASS | **286 项，16.846s，无跳过** |
| 真实 10,000 条批量装配 | **NOT RUN** | 需远端真实数据 |
| 真实 CUDA / batch 前向 | **NOT RUN** | 无 GPU |
| pilot / 正式训练 / 评估 | **NOT RUN** | 未进入 |

### 实现中自己发现并修掉的一个 bug

`random.Random((seed, epoch))` 在 Python 3.12 直接抛 `TypeError`——只接受 int/float/str/bytes/bytearray，不接受 tuple。6 项测试因此报错。**这是实现缺陷**，改为整数混合派生：

```python
derived_seed = ((seed * 0x9E3779B1) ^ (epoch * 0x85EBCA6B)) & 0xFFFFFFFF
```

取黄金比例相关奇数常量做混合并掩到 `[0, 2**32)`，使相邻 epoch 的种子分散且可跨平台复现。

## 运行日志和输出路径

- `src/adaptive_lora/m7_batches.py`
- `tests/test_m7_batches.py`：23 项
- `.cache/m7-20261115-assets-04/batches-regression-01.log`：全套回归日志

## 未解决问题

1. **训练顺序策略需你认可**：`none` 还是 `per_epoch_shuffle`。论文未规定，当前默认 `none`。
2. **末组余量策略仍未定**：`error` / `step` / `drop`，仍未解决。
3. 真实 10,000 条批量装配未运行（需远端数据 + 真实 1.5B tokenizer）。
4. 真实 CUDA、batch 前向、显存与吞吐未验证。
5. GPU checkpoint（含 optimizer/RNG 续跑）未实现。
6. 语义去重、内嵌答案、预训练污染未检查。
7. 远端配置仍是修订前字节；许可原文缺口未解。

## 下一步建议（不自动执行）

两个决定都确定后，装配层可以直接在远端跑一次**真实数据的 10,000 条装配**（不需要 GPU，只需已下载的 1.5B tokenizer 与已核验数据），产出真实的 BPE 边界通过率与 micro-batch 计划摘要。这是 pilot 之前最后一块纯 CPU 验证。

本轮未使用 CUDA、未训练、未 push。
