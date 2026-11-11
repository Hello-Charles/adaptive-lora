# Adaptive-LoRA 复现工程

按阶段复现论文 *Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching*。算法事实与工程决策分别标注在 [PAPER_SPEC.md](PAPER_SPEC.md)，阶段门禁见 [REPRO_PLAN.md](REPRO_PLAN.md)，协作规范见 [CLAUDE.md](CLAUDE.md)。论文原文可从 [arXiv:2610.34756](https://arxiv.org/abs/2610.34756) 查阅；本地 `papers/adaptive_lora.pdf` 不上传至公开仓库。

## 当前范围：M0、M1 已验收

M0 的工程骨架、版本记录、基础测试和 Git 提交已验收；M1 已使用固定 revision 的 Qwen2.5-0.5B 完成 24 层模块路径与输出类型实测，以及一次短输入前向。证据分别见 [REPRO_PLAN.md](REPRO_PLAN.md#m0--项目骨架和规格冻结) 与 [M1 验收报告](reports/m1_validation.md)。尚未准备 M2 校准数据，也未实现 zero-ablation、重要性评分或训练。

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

论文第 3.1 节报告机制分析使用 Qwen2.5-0.5B（24 层）。本项目选择 Hugging Face 仓库 `Qwen/Qwen2.5-0.5B` 和固定 revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，均为 **[复现决策]**，不是论文原权重版本。实际模型 `Qwen2ForCausalLM` 有 24 层 `model.layers.0`–`model.layers.23`；每层 `self_attn` 的输出为 `(Tensor, None)`，`mlp` 的输出为 Tensor。每层 Q/K/V 投影路径、tokenizer 细节、短输入 logits 与有效预测位置见 [真实结构报告](reports/m1_structure.json) 和 [M1 验收记录](reports/m1_validation.md)。本次的观察性 hooks 不修改模型输出；zero-ablation 留待 M3。

脚本默认只使用本地缓存，不会静默下载模型。已有授权下载时也必须显式传入 `--allow-download`、固定 `--revision` 和**新建且不存在**的 `--cache-dir`；本次权重保存在被 Git 忽略的 `data/models/` 下。复核已下载的模型时使用不同的输出路径（不会覆盖历史报告）：

```bash
.venv/Scripts/python.exe -I scripts/inspect_model.py --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 --output reports/m1_structure_repeat.json
```

本次第一个探查调用因把 `post_attention_layernorm` 错认作 Attention 而失败；修复并补回归测试后从缓存**离线**重跑成功。模型配置未暴露 `resolved_revision`，缓存快照目录与请求的 SHA 一致；详见验收记录。

## 后续阶段前的待核验清单

以下条目均为 **[待核验]**，不是论文事实；详细状态见 [PAPER_SPEC.md 第 7 节](PAPER_SPEC.md#7-开放问题及处理)：

- Q01：数据集精确版本、训练样本 ID 和抽样索引。
- Q02：原作者 prompt 模板、预测 token 的位置与截断规则。
- Q03：作者实际使用的模型权重 revision 仍待核验；本项目 M1 的模型 SHA、tokenizer 和依赖版本已记录。
- Q04：训练 loss mask、样本拼接方式与评估协议。
- Q05：作者完整代码的发布状态。

M0 没有核查论文算法章节；M1 已核查论文第 3.1 节的 0.5B 模型与 24 层描述。后续实现算法/实验前仍须按 [CLAUDE.md 的论文原文使用规范](CLAUDE.md#论文原文使用规范) 核查对应章节，无法证实时明确标注工程假设。
