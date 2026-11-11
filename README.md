# Adaptive-LoRA 复现工程

按阶段复现论文 *Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching*。算法事实与工程决策分别标注在 [PAPER_SPEC.md](PAPER_SPEC.md)，阶段门禁见 [REPRO_PLAN.md](REPRO_PLAN.md)，协作规范见 [CLAUDE.md](CLAUDE.md)。论文原文可从 [arXiv:2610.34756](https://arxiv.org/abs/2610.34756) 查阅；本地 `papers/adaptive_lora.pdf` 不上传至公开仓库。

## 当前范围：M0

目前只有 Python `src/` 包骨架、环境快照脚本、标准库单元测试和 Git 忽略规则。**尚未实现或验证**模型加载、zero-ablation、重要性评分、LoRA 训练或论文结果。用户批准后，仅在 `.venv/` 内安装 M0 所需的开发和后续阶段基础依赖；不下载模型/数据，也不运行训练。

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

PowerShell 下可用 `./.venv/Scripts/python.exe -I -m unittest discover -s tests -v` 运行测试。M0 的新环境快照使用 `--metadata-only`，只读取版本元数据而不执行第三方包代码；其中 `not_checked` **不代表导入成功**。环境脚本在目标 JSON 已存在时拒绝覆盖，重新记录时须提供新的输出路径。真实安装的版本保存在 [requirements-windows-py312.txt](requirements-windows-py312.txt)；重新创建该 Windows CPU 环境时，应先用上面的 PyTorch CPU 专用索引安装对应版本，再在虚拟环境中安装锁定的依赖和本项目。

[首次 M0 记录](reports/m0_status.md)与[首次环境快照](reports/m0_environment.json)保留安装前的历史状态（其中依赖尚未安装、目录还不是 Git 仓库），不是当前环境结论。新环境记录见 [reports/m0_venv_environment.json](reports/m0_venv_environment.json)，Git 和安装完成后的状态见 [reports/m0_git_setup.md](reports/m0_git_setup.md)。`pyproject.toml` 的可选依赖是项目安装入口；实际版本以对应平台的环境快照和锁定清单为准。

`.gitignore` 排除虚拟环境、本地 PDF、数据、结果、模型权重、checkpoint、缓存和典型密钥文件。忽略规则不能保护已被 Git 跟踪的文件；请勿把凭据放入仓库，推送前仍需人工检查待提交内容。后续阶段使用 Git 追踪变更并记录提交编号；创建新提交或公开推送仍以当次任务的授权为准。

## 后续阶段前的待核验清单

以下条目均为 **[待核验]**，不是论文事实；详细状态见 [PAPER_SPEC.md 第 7 节](PAPER_SPEC.md#7-开放问题及处理)：

- Q01：数据集精确版本、训练样本 ID 和抽样索引。
- Q02：原作者 prompt 模板、预测 token 的位置与截断规则。
- Q03：模型权重 revision、tokenizer 及依赖的版本。
- Q04：训练 loss mask、样本拼接方式与评估协议。
- Q05：作者完整代码的发布状态。

M0 没有核查论文中与后续算法或实验有关的章节。进入相应阶段前，应按 [CLAUDE.md 的论文原文使用规范](CLAUDE.md#论文原文使用规范) 核查 PDF；无法证实时明确标注工程假设。
