# M2 校准数据准备进度（2026-11-12，历史受阻记录）

> 状态更新（2026-11-14）：M2 已完成，当前验收以 [m2_validation.md](m2_validation.md) 和 [m2_validation.json](m2_validation.json) 为准。本文件保留最初尝试的上下文，不代表当前权限或测试状态；2026-11-13 的注释维护任务也已运行并通过当时的 20 个测试。

## 当前任务与完成内容

M2 的三个代码文件当时已写出；Git 状态查询成功，但后续需要权限确认的数据探查/下载与测试命令受阻，单元测试未运行，Stanford Alpaca 数据集未下载，校准 manifest 未生成，Git 提交未创建。会话的自动权限检查不可用，相关 Bash 动作被拒绝，且提示重试无效。M2 的四个验收项在当时保持未勾选。

## 文件变更

- `src/adaptive_lora/data.py`：抽样（固定 seed 的标准库随机数）、prompt 模板（tokenizer 的 chat template、仅 user 消息、`add_generation_prompt=True`）、逐样本/批量的分词与有效末尾位置、SHA-256 哈希、manifest 结构。
- `scripts/prepare_calibration.py`：与 M1 脚本一致的下载门禁（下载需固定 `--dataset-revision` 和新的空 `--dataset-cache-dir`），默认复用 M1 的固定 Qwen tokenizer 缓存，输出文件拒绝覆盖。
- `tests/test_data.py`：可复现抽样、答案不混入、chat template 调用、padding/截断与有效末尾位置、manifest 无 output 字段且哈希稳定等 fixture 测试，另有真实 Qwen tokenizer 的集成测试（缓存或 transformers 缺失时跳过）。
- `REPRO_PLAN.md`：仅补充"进行中"说明，未勾选任何 M2 验收项。

## 论文依据与工程假设

依据 `PAPER_SPEC.md` 第 3 节与论文附录 E.2：校准样本取自训练集，用基础模型预测 token，不需要 gold 答案；校准输入默认不得混入 `output`/`reference`。以下均为 **[复现决策]**，待实际运行后记录：Stanford Alpaca 作为 smoke 数据源、N=30、seed=20261112、max_length=512、Qwen chat template 且 user 内容为 `instruction + "\n\n" + input`。数据集的固定 revision 尚未查得，因此脚本要求下载时显式传入，未把未验证的 SHA 写死在代码中。

## 执行命令

当时 Git 状态查询成功；后续数据探查/下载与测试未能执行（自动权限检查不可用，相关 Bash 调用被拒绝）。

## 测试结果（PASS / FAIL / NOT RUN）

- **NOT RUN**：`tests/test_data.py` 的全部测试；`prepare_calibration.py` 的数据集下载、tokenizer 加载与 manifest 生成；可复现性二次运行；Git 分支与提交。
- 没有任何测试被标记为 PASS，也没有任何验收项被勾选。

## 运行日志和输出路径

无运行日志；尚无 `reports/m2_manifest.json`、`data/datasets/` 缓存或新 Git 提交。

## 未解决问题

1. 会话的自动权限检查已停止工作；提示需要切换到其他权限模式才能执行 Bash（例如允许命令执行或由用户批准命令）。
2. Stanford Alpaca 的固定 revision、实际下载与 30 条抽样的 manifest 均未产生。

## 下一步建议（不自动执行）

用户切换权限模式或批准命令执行后，我将：创建 `repro/m2-calibration-data` 分支，查询并固定 `tatsu-lab/alpaca` 的 revision，运行单元测试，下载数据集（约 25MB）并生成 manifest，做两次离线运行验证抽样可复现，再更新验收项并本地提交。未获得可执行环境前，不勾选 M2 任何验收项。
