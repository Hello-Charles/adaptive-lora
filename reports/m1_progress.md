# M0 验收与 M1 模型结构探查进度（2026-11-11）

## 当前任务与完成内容

M0 的四项门禁已有实际证据并在 `REPRO_PLAN.md` 勾选：项目源码可导入、标准库测试运行、Python/关键依赖版本已记录、忽略规则覆盖本地模型与凭据、首个 Git commit `1c7b0e8433c3669bcb463fc5a7ba1441efc3e0f5` 已记录。`pip check` 通过不等于第三方库可实际导入；该风险转入 M1 验证。

M1 已开始但**未通过验收**：新增默认离线的模型加载器、动态查找层和 Q/K/V 路径的工具，以及准备采集 Attention/MLP 返回类型和短输入 logits 的脚本。代码和单元测试均不能代替真实 Qwen 权重前向，故 M1 的三个验收框保持未勾选。

## 文件变更

- `REPRO_PLAN.md`：仅勾选 M0 四项，保留 M1 未验收说明。
- `src/adaptive_lora/model_loader.py`：默认 `Qwen/Qwen2.5-0.5B`、`local_files_only=True`、`trust_remote_code=False`、`use_safetensors=True`；运行时从实际模块树发现 24 个 block 与子模块路径；预测位置依据 mask 取有效 token。
- `scripts/inspect_model.py`：准备在模型短输入前向中仅**观察** Attention/MLP 输出类型，正常或异常退出时移除观察 hooks；默认不下载权重，输出报告只在成功后创建，不覆盖已有报告。
- `tests/test_model_loader.py`：使用模拟模块测试离线默认值、路径发现、层数与 Q/K/V 校验、right-padding 预测位置和返回值结构描述；没有伪造真实 Qwen 运行结果。
- `README.md`：标注 M0 已验收、M1 部分完成及权重下载门禁。

## 论文依据与工程假设

论文原文 `papers/adaptive_lora.pdf` 第 3.1 节（PDF 第 3 页）明确机制分析使用 **Qwen2.5-0.5B、24 层**，并涉及 Q/K/V 投影。本轮使用 `pdftotext` 成功读取该章节；PDF 页面渲染工具不可用，但文本提取可用。选取 Hugging Face `Qwen/Qwen2.5-0.5B` 为 **[复现决策]**，权重 revision 仍 **[待核验]**。已安装的 Transformers 5.22.2 源文件 `modeling_qwen2.py` 中 `Qwen2Model.layers` 为 ModuleList，`Qwen2DecoderLayer` 含 `self_attn`/`mlp`，`Qwen2Attention.forward` 源码返回 tuple；**源码结构不等于真实模型运行验证**。未实现 zero-ablation 或训练。

## 执行命令

- `git status --short --branch`；`.venv/Scripts/python.exe --version`；`.venv/Scripts/python.exe -I -m pip check`。
- 使用标准库检查默认 Hugging Face 缓存：目录存在，未找到 `models--Qwen--Qwen2.5-0.5B*`。
- `pdftotext -f 1 -l 3 -layout papers/adaptive_lora.pdf -`；`pdftotext -f 4 -l 5 -layout papers/adaptive_lora.pdf -`（仅按需阅读模型和方法相关页面）。
- `.venv/Scripts/python.exe -I -c 'import torch; print("torch", torch.__version__)'`：被自动权限模式拒绝，未执行第三方包代码。
- `git switch -c repro/m1-model-inspection`。
- `PYTHONPATH=src python scripts/inspect_model.py --help`。
- `PYTHONPATH=src python -m unittest discover -s tests -v`；`python -I -m py_compile src/adaptive_lora/model_loader.py scripts/inspect_model.py tests/test_model_loader.py`；`git diff --check`。

## 测试结果（PASS / FAIL / NOT RUN）

- **PASS**：系统 Python 中 10 个标准库单元测试（M0 四个、M1 离线模拟六个）。
- **PASS**：M1 新文件通过语法编译；脚本命令行帮助可显示；`pip check` 为 `No broken requirements found.`。
- **NOT RUN**：虚拟环境的 PyTorch/Transformers/PEFT 运行时导入（权限拒绝）；真实 Qwen2.5-0.5B 加载、实际模块路径/返回类型测量、短输入前向（默认缓存缺少权重，下载需用户明确授权）。
- **NOT RUN**：消融、校准、训练及 M2 之后的实验。

## 运行日志和输出路径

单元测试控制台结果：`Ran 10 tests in 1.746s` / `OK`。M0 历史环境见 `reports/m0_venv_environment.json`、`reports/m0_git_setup.md`，当前工作分支为 `repro/m1-model-inspection`，起点为已发布的 `599c46d38ce1f1cb63ae936eb1ac5fd45c5dbf07`。**没有**生成 `reports/m1_structure.json`、模型权重、实验指标或假定的模块输出。

## 未解决问题

1. 运行新安装第三方包的命令继续遭自动权限模式 `[Code from External]` 拒绝；不得借测试或换脚本绕过。
2. 本地默认缓存无 Qwen2.5-0.5B 权重；下载约 0.5B 级别模型需用户按 `CLAUDE.md` 单独批准。未下载。
3. 权重 revision、真实 Attention/MLP 返回结构与有效预测位置未实测，M1 门禁未通过。

## 下一步建议（不自动执行）

需要在获得执行已安装第三方包的权限及模型权重下载授权后，运行 M1 结构探查并以真实结果核对路径与返回类型。模块返回结构确认前，不进入 zero-ablation 阶段。本轮未启动 M2。
