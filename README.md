# Adaptive-LoRA 复现工程

按阶段复现论文 *Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching*。算法事实与工程决策分别标注在 [PAPER_SPEC.md](PAPER_SPEC.md)，阶段门禁见 [REPRO_PLAN.md](REPRO_PLAN.md)，协作规范见 [CLAUDE.md](CLAUDE.md)。论文原文可从 [arXiv:2610.34756](https://arxiv.org/abs/2610.34756) 查阅；本地 `papers/adaptive_lora.pdf` 不上传至公开仓库。

## 当前范围：M0 已验收，M1 结构探查进行中

M0 的工程骨架、版本记录、基础测试和 Git 提交已验收，证据见 [REPRO_PLAN.md](REPRO_PLAN.md#m0--项目骨架和规格冻结)。M1 已建立默认离线的模型加载器、结构探查脚本与模拟模块单元测试；**尚未加载真实 Qwen 权重，未确认真实模块输出类型，也未执行前向**。用户批准后仅在 `.venv/` 内安装了开发与后续阶段的基础依赖；尚未下载模型或数据，更未实现 zero-ablation、重要性评分或训练。

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

## M1：模型结构探查（尚未通过真实模型门禁）

论文第 3.1 节报告机制分析使用 Qwen2.5-0.5B（24 层）。选择 Hugging Face 仓库 `Qwen/Qwen2.5-0.5B` 是 **[复现决策]**，权重 revision 仍 **[待核验]**。`scripts/inspect_model.py` 在成功加载后会动态列出各层 Attention、MLP 与 Q/K/V 模块路径，借助仅观察、不修改输出的临时 hooks 记录实际返回结构，并在短输入前向中验证有效预测位置；无论前向成功与否，hooks 都会被清理。**模拟模块测试的通过不等于真实 Qwen 结构已确认。**

默认仅从本地缓存读取，不会静默下载模型权重。确认已获权重下载授权、具备执行第三方包代码的权限后，才可显式使用 `--allow-download`；例如已有权重时可运行：

```bash
.venv/Scripts/python.exe -I scripts/inspect_model.py --output reports/m1_structure.json
```

当前尚无 `reports/m1_structure.json`，不可把脚本存在误写为模型探查已完成。PDF 已通过文本提取核查第 3.1 节；已安装的 Transformers 源码显示 Qwen2Attention 以 tuple 返回，但只有真实模型前向才能验证本项目实际使用的返回类型。

## 后续阶段前的待核验清单

以下条目均为 **[待核验]**，不是论文事实；详细状态见 [PAPER_SPEC.md 第 7 节](PAPER_SPEC.md#7-开放问题及处理)：

- Q01：数据集精确版本、训练样本 ID 和抽样索引。
- Q02：原作者 prompt 模板、预测 token 的位置与截断规则。
- Q03：模型权重 revision、tokenizer 及依赖的版本。
- Q04：训练 loss mask、样本拼接方式与评估协议。
- Q05：作者完整代码的发布状态。

M0 没有核查论文算法章节；M1 已核查论文第 3.1 节的 0.5B 模型与 24 层描述。后续实现算法/实验前仍须按 [CLAUDE.md 的论文原文使用规范](CLAUDE.md#论文原文使用规范) 核查对应章节，无法证实时明确标注工程假设。
