# M0 项目骨架与环境探查记录（2026-11-11）

## 当前任务与完成内容

- **PASS**：建立 `src/` 布局的 Python 包与 `pyproject.toml`；使用 `PYTHONPATH=src` 可导入。当前未安装构建后端，故没有验证安装/构建。
- **PASS**：使用标准库 `unittest` 运行基本测试；保存 Python、关键依赖实际导入状态和 Git 状态到 `m0_environment.json`。
- **PASS**：增加 `.gitignore`，在临时 Git 仓库中验证模型、数据、缓存、密钥和结果路径被忽略，同时源码与环境报告未被忽略。
- **未完全验收**：项目目录尚不是 Git 仓库，没有首个 commit 可提交或记录。已记录 `git.status=not_repository`、`git.commit=null`，未擅自初始化仓库或创建提交。M0 的其余门禁（环境现状明确）满足，但 Git 记录项仍待处理。

## 文件变更

- `pyproject.toml`、`src/adaptive_lora/__init__.py`：可导入的最小包骨架；此阶段不声明 ML 运行依赖。
- `scripts/record_environment.py`：记录依赖版本、导入状态与 Git commit；已有输出不会被覆盖。
- `tests/test_m0.py`：包导入、环境快照与 `.gitignore` 行为检查。
- `.gitignore`、`README.md`：忽略规则、阶段范围与本地执行说明。
- `reports/m0_environment.json`：本次环境快照；本文件：M0 状态与证据。

## 论文依据与工程假设

M0 仅建立工程基础，不实现论文中的校准公式、模型结构探查、数据预处理或训练；因此尚未核查 `papers/adaptive_lora.pdf` 对应章节，不能声称算法或论文结果已验证。后续实验前的 **[待核验]** 问题 Q01–Q05 保留在 `PAPER_SPEC.md` 第 7 节，并汇总在 `README.md`。选择标准库测试、`src/` 布局和当前环境下不安装依赖，均属 **[复现决策]**，不是论文设定。

## 执行命令

- `python --version`：Python 3.12.8。
- `git --version`：Git 2.48.1.windows.1；`git status --short` / `git rev-parse --is-inside-work-tree`：退出码 128（当前不是 Git 仓库）。
- `python -I -`（用 `importlib.metadata` 和 `importlib.import_module` 探查依赖）：torch、transformers、peft、datasets、accelerate、pytest、setuptools、wheel 均未安装；pip 25.0.1 可导入。
- `PYTHONPATH=src python -m unittest discover -s tests -v`：3 个测试通过。
- `python -I scripts/record_environment.py --output reports/m0_environment.json`：保存一次快照。
- `python -I -`（用 `tomllib.load` 检查 `pyproject.toml` 和 `src` 包发现配置）：检查通过。

首次临时探查命令直接读取缺失的 `torch` 版本而退出码为 1；已使用逐项捕获缺失依赖的探查命令重新执行成功，未将初次异常当作已安装的证据。

## 测试结果（PASS / FAIL / NOT RUN）

- **PASS**：`test_package_imports_from_src`。
- **PASS**：`test_environment_snapshot_reports_missing_packages_and_git`。
- **PASS**：`test_gitignore_excludes_local_artifacts_but_keeps_sources_and_reports`。
- **PASS**：`pyproject.toml` 语法与 `src` 包发现配置的静态检查。
- **NOT RUN**：`pip install -e .` / wheel 构建（缺少 setuptools；未获批准安装依赖）。
- **NOT RUN**：模型加载、训练、论文算法测试（不属于 M0；ML 依赖未安装）。
- **NOT RUN**：Git commit（项目不是 Git 仓库；未初始化或提交）。

## 运行日志和输出路径

测试命令控制台输出：`Ran 3 tests in 0.849s` / `OK`。环境快照为 `reports/m0_environment.json`；本阶段没有模型、数据或实验结果，也没有创建虚假的实验指标文件。

## 未解决问题

- Git 仓库/首个提交尚未建立，不能标记 M0 的 Git 记录项已通过。
- 关键 ML 依赖未安装，未来需要先确定兼容版本并经用户批准安装；不能凭空声称可加载 Qwen。
- `PAPER_SPEC.md` 中 Q01–Q05 及论文对应章节仍待进入相关阶段前核验。

## 下一步建议（不自动执行）

先由用户决定仓库初始化和首个 commit 的处理方式；取得用户对关键依赖安装与模型下载的确认之后，才可按阶段指令开展 M1。本轮停止在 M0。
