# M0 后续：虚拟环境与公开 Git 仓库（2026-11-11）

## 当前任务与完成内容

- **PASS**：系统 Python 3.12.8 满足 `pyproject.toml` 的 `>=3.10`；使用该解释器建立了项目专用 `.venv/`，未修改全局环境。
- **PASS**：在 `.venv/` 安装 CPU 版 PyTorch、Transformers、PEFT、Datasets、Accelerate、pytest 和项目自身；实际安装版本见环境快照与锁定清单。
- **PASS**：初始化 Git，基础变更提交为 `1c7b0e8433c3669bcb463fc5a7ba1441efc3e0f5`（分支 `repro/m0-setup`）。GitHub 个人仓库 [Hello-Charles/adaptive-lora](https://github.com/Hello-Charles/adaptive-lora) 已创建，`gh repo view` 返回 `PUBLIC`，所有者为 `Hello-Charles`。当前完整记录所属提交可由 `git log -1 --format=%H` 查询。
- **限制**：工具的自动权限模式拒绝直接导入刚安装的第三方包，因此未验证 PyTorch/Transformers/PEFT 等运行时导入；元数据快照将其明确标为 `not_checked`，不能等同于可用性实测。

## 文件变更

- `pyproject.toml` 增加可选 `repro` 依赖组；`requirements-windows-py312.txt` 记录当前 Windows/Python 3.12 CPU 环境的实际安装版本（包含传递依赖，排除本地 editable 路径）。
- `scripts/record_environment.py` 新增 `--metadata-only`，只读取安装包版本，不执行第三方包代码；`tests/test_m0.py` 增加相应测试，并检查虚拟环境、IDE 配置与本地 PDF 被忽略。
- `README.md` 增加虚拟环境、CPU PyTorch 索引和 M0 检查说明；`.gitignore` 排除 `.venv/`、`.claude/`、`.codex/` 和本地论文 PDF。
- `reports/m0_venv_environment.json` 为安装后的版本快照；`reports/m0_environment.json` 与 `reports/m0_status.md` 保留**安装前**历史状态，不覆盖旧证据。

## 论文依据与工程假设

这是 M0 工程环境和版本控制任务，不涉及论文公式、训练配置或结果。仅检测到 Intel Iris Xe，本次选择 CPU PyTorch 属 **[复现决策]**，不是论文训练配置；未来 GPU 环境需要另行确定依赖方案。论文相关 Q01–Q05 仍为 **[待核验]**；本轮没有下载模型、数据，也没有运行训练。

## 执行命令

- `python --version`、`python -m venv .venv`、`.venv/Scripts/python.exe --version`：均为 Python 3.12.8。
- `.venv/Scripts/python.exe -I -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu`。
- `.venv/Scripts/python.exe -I -m pip install -e '.[repro]'`。
- `.venv/Scripts/python.exe -I -m pip freeze --all --exclude-editable > requirements-windows-py312.txt`。
- `.venv/Scripts/python.exe -I -m pip check`；`.venv/Scripts/python.exe -I -m pip show adaptive-lora-reproduction torch transformers peft datasets accelerate pytest`。
- `.venv/Scripts/python.exe -I scripts/record_environment.py --metadata-only --output reports/m0_venv_environment.json`。
- `PYTHONPATH=src python -m unittest discover -s tests -v`（系统 Python；未在虚拟环境中运行导入第三方包的测试）。
- `git init -b repro/m0-setup`、`git add` 明确列出的文件、`git diff --cached --check`、`git commit`；`gh repo create Hello-Charles/adaptive-lora --public --source . --remote origin`、`gh repo view`。

## 测试结果（PASS / FAIL / NOT RUN）

- **PASS**：4 个 M0 标准库测试（包通过 `PYTHONPATH=src` 导入、两种环境快照、忽略规则）。
- **PASS**：`pip check` → `No broken requirements found.`；`pip show` 显示项目为 editable 安装，PyTorch 2.15.0+cpu、Transformers 5.22.2、PEFT 0.21.2、Datasets 5.2.0、Accelerate 1.15.0、pytest 9.2.0。
- **PASS**：Git 暂存文件逐项检查、`git diff --cached --check`；`.venv/` 与 `papers/adaptive_lora.pdf` 均被忽略。常见凭据模式的暂存区扫描无匹配，不能替代完整人工审查。
- **NOT RUN**：直接导入虚拟环境的第三方包、CPU 张量计算以及虚拟环境内可能导入它们的测试（自动权限模式拒绝含第三方代码的验证命令）。
- **NOT RUN**：模型加载、模型前向、训练和论文算法正确性测试；这些属于后续阶段。

## 运行日志和输出路径

系统 Python 测试控制台：`Ran 4 tests in 0.992s` / `OK`。安装后的元数据快照：`reports/m0_venv_environment.json`。安装包精确版本：`requirements-windows-py312.txt`。Git 提交与发布可通过 `git log`、`git remote -v` 和 `gh repo view` 复核。没有模型、数据、checkpoint 或实验指标文件。

## 未解决问题

- 关键依赖虽然已安装且无元数据冲突，但**运行时导入未验证**；在执行 M1 之前需要获得允许执行已安装第三方代码的权限并完成验证。
- `wheel` 本身未安装在虚拟环境中；可编辑安装通过构建隔离成功，不应假称所有工具包都已安装。
- Python 3.12/Windows CPU 锁定清单不等于未来 Linux/GPU 环境配置；论文模型、数据和评估细节尚未核对。

## 下一步建议（不自动执行）

本轮到此停止；M1 必须另行指示，并先解决运行时导入权限及模型权重下载授权。后续阶段用 Git 跟踪变更；未来提交、向公开仓库推送仍需对应任务授权。
