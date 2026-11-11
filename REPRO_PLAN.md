# REPRO_PLAN.md — Adaptive-LoRA 分阶段复现计划

> 执行模型：Claude Code + CC Switch + Codex，**单 Agent**。算法和论文事实以 `PAPER_SPEC.md` 为准。
>
> 状态：默认全部未完成。完成一个里程碑必须记录真实执行命令、测试输出、实验产物和 Git commit；不能只凭代码生成就勾选。

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

- [ ] 项目可导入、基本测试工具可运行。
- [ ] Python、PyTorch、Transformers、PEFT 等可用版本已记录（未装则如实标记）。
- [ ] 模型权重/数据/密钥/缓存已被 `.gitignore` 排除。
- [ ] 提交或记录首个 Git commit；论文待核验项已列入清单。

**门禁**：环境现状明确后才进入模型结构探查。

### M1 — 模型结构探查

任务：加载 Qwen2.5-0.5B；检查 24 层结构、每层 Attention / MLP 的实际模块路径、q/k/v 投影和 Attention 输出类型；执行一次短输入前向。

产出：`model_loader.py`、`inspect_model.py`、结构探查报告。

- [ ] 证实模块路径，未凭经验硬编码。
- [ ] 证实返回类型（tensor/tuple/其他）和有效预测位置。
- [ ] 保存真实运行命令、输出和依赖版本。

**门禁**：模块返回结构未确认，不写 zero-ablation hook。

### M2 — 校准数据准备

任务：先用 Stanford Alpaca 等指令数据，稳定抽样 30 条；固定 prompt、tokenizer、预测位置，保存输入 ID/哈希。

产出：`data.py`、数据 manifest、测试。

- [ ] N=30、固定 seed 后抽样可重现。
- [ ] reference answer 未意外混入校准输入。
- [ ] 截断与 padding、有效末尾位置经过测试。
- [ ] 记录数据来源、版本、样本 ID/索引和 prompt 模板。

### M3 — 单层 zero-ablation

任务：从**一个输入、一层、一个子模块**开始；分别实现 Attention 与 MLP 输出消融，然后扩展到全部层。

产出：`ablation.py`、`test_ablation.py`。

- [ ] Attention 和 MLP 消融互不叠加。
- [ ] 仅目标模块输出被零化，输出结构未被破坏。
- [ ] 正常退出/异常退出后 hooks 清理。
- [ ] 清除 hook 后正常模型输出恢复。

**门禁**：`PAPER_SPEC.md` 的 T03–T05 必须通过。

### M4 — 重要性评分与 Top-K 选层

任务：固定原始 $t^*$；实现 $q,q_l^A,q_l^{MLP}$ 及 $I_l$；N=30 聚合；Top-7 排序；导出 CSV/JSON 和重要性图。

产出：`calibration.py`、`selection.py`、`run_calibration.py`、测试和结果。

- [ ] 原始 token 在所有消融中保持一致。
- [ ] N=1 / N>1 公式单元测试通过。
- [ ] 输出 24 层重要性、Top-7 和所用样本清单。
- [ ] 记录环境、代码版本和耗时；可选 N=100 或不同 seed 检查 Jaccard 稳定性。

**里程碑 A：校准与选层算法完成。此前不得开始正式 LoRA 微调。**

### M5 — 选择性 LoRA 与训练 smoke test

任务：PEFT 在指定 Top-K 层的 Q/K/V 添加 LoRA；检查 trainable params；仅运行几步训练与 checkpoint 加载。

产出：`trainer.py`、`train.py`、`test_lora_targets.py`。

- [ ] 仅所选层 LoRA 可训练，基础模型冻结。
- [ ] 训练 1–3 个 batch，梯度与 loss 无异常、无 NaN。
- [ ] checkpoint 可保存并恢复。

### M6 — 统一基线

任务：建立 Full LoRA / Random-K LoRA / Adaptive-K LoRA 的统一训练与评估接口。

- [ ] 相同训练数据、评估协议和可控训练超参数。
- [ ] Random-K 保存 seed 和层 ID。
- [ ] 报告实际可训练参数量差异（不能说 Full 与 K-layer 是同预算）。
- [ ] 无训练/评估泄漏。

### M7 — 正式复现实验（需用户批准 GPU 与数据成本）

任务：Qwen2.5-1.5B、N=250、Top-8、rank16、alpha32、Q/K/V。先做 pilot 确认显存/吞吐，再跑完整训练与基线。

- [ ] 执行前核对论文表格、数据版本、prompt、评估协议。
- [ ] 保存训练日志、GPU 耗时和峰值显存。
- [ ] 完成统一评估；不人为追求论文数值而改动结果。

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
