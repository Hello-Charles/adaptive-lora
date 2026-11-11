# M1 模型结构探查验收（2026-11-11）

## 当前任务与完成内容

在用户明确授权执行已安装第三方包代码和下载大型模型权重后，使用固定 revision 的 Qwen2.5-0.5B 完成一次真实短输入前向。配置为 24 层；全部 24 层的 Attention、MLP 和 Q/K/V 路径均在实际模型上被发现，子模块输出由只观察、不改动的临时 hooks 记录。M1 三项验收已在 `REPRO_PLAN.md` 勾选；未开始 M2、消融或训练。

## 文件变更

- `src/adaptive_lora/model_loader.py`：加载器可指定独立缓存目录；修正 Attention 模块发现逻辑，避免将 `post_attention_layernorm` 错判为 Attention。
- `scripts/inspect_model.py`：下载时强制提供固定 revision 和新建的空缓存目录；本次重跑使用已下载权重，设置离线模式，不重复下载。
- `tests/test_model_loader.py`：增加缓存与下载门禁、带 layernorm 的真实结构形状回归测试；Windows 路径断言改为跨平台写法。
- `reports/m1_verified_environment.json`：实际导入的环境快照；`reports/m1_structure.json`：真实模型结构和短输入前向结果；先前的 `reports/m1_progress.md` 保留为本轮以前的进度记录。

## 论文依据与工程假设

论文第 3.1 节报告 Qwen2.5-0.5B 具有 24 层。选用 Hugging Face 的 `Qwen/Qwen2.5-0.5B`、短输入 `Hello, world!`、Windows CPU 环境以及仓库提交 `060db6499f32faf8b98477b0a26969ef7d8b9987` 均属 **[复现决策]**，不是论文指定的权重 revision、prompt 或设备。原论文准确的权重 revision 仍 **[待核验]**。模型配置没有暴露 `resolved_revision`（JSON 中为 `null`）；本次请求固定 SHA，且本地缓存快照目录名称与该 SHA 一致。M1 的观察性 hooks 不执行 zero-ablation，不能将此结果称为校准或 path patching。

## 执行命令

- `.venv/Scripts/python.exe -I -c 'import torch, transformers, peft; print(...)'`：PyTorch 2.15.0+cpu、Transformers 5.22.2、PEFT 0.21.2 均可实际导入。
- `.venv/Scripts/python.exe -I scripts/record_environment.py --output reports/m1_verified_environment.json`：记录实际导入状态和 Git 基线提交。
- 使用 Python 标准库查询 `https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B`：取得上述模型仓库 SHA；查询该 revision 的文件树，`model.safetensors` 公布大小为 988097824 字节。
- `HF_HUB_DISABLE_TELEMETRY=1 .venv/Scripts/python.exe -I scripts/inspect_model.py --model-id Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 --allow-download --output reports/m1_structure.json`：模型权重下载和加载成功，但第一次结构发现因误将 `post_attention_layernorm` 当作 Attention 而退出码为 1；未写出结果文件。
- 修复并补回归测试后：`.venv/Scripts/python.exe -I -m unittest discover -s tests -v`：11 个测试通过。
- `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/Scripts/python.exe -I scripts/inspect_model.py --model-id Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 --output reports/m1_structure.json`：从隔离缓存离线加载、短输入前向并写出结果。
- `python -I -`（标准库 JSON 与缓存文件验证）：校验 24 个索引、每层路径和输出结构、预测位置及权重文件大小。

## 测试结果（PASS / FAIL / NOT RUN）

- **PASS**：PyTorch、Transformers、PEFT 等依赖实际导入；`pip check` 无元数据冲突。
- **PASS**：11 个 M0/M1 单元测试（其中 M1 的模块发现测试使用模拟模块，不冒充真实模型测量）。首次测试运行中一个 Windows 路径断言失败，修正后全部通过。
- **PASS**：真实模型为 `Qwen2ForCausalLM`、配置层数 24。实测路径为 `model.layers.0` 至 `.23`，每层有 `.self_attn`、`.mlp` 及 `.self_attn.{q,k,v}_proj`。24 层 Attention 均返回 tuple：首项为 shape `[1,4,896]`、dtype `torch.bfloat16` 的 Tensor，第二项为 None；MLP 均返回同形状、同 dtype 的 Tensor。
- **PASS**：tokenizer 为 fast `Qwen2Tokenizer`；`padding_side=right`、`truncation_side=right`，pad/eos token ID 均为 151643。短输入 token ID `[9707,11,1879,0]`，有效预测位置为索引 3，logits shape `[1,4,151936]`，该位置最高分 token ID 1096。此处仅验证位置，不构成 M4 校准结果。
- **PASS**：缓存快照目录含固定 SHA 和 988097824 字节 `model.safetensors`；`data/` 路径被 Git 忽略。
- **NOT RUN**：M2 数据准备、zero-ablation、层评分、LoRA 训练、正式评估。额外的 CUDA 可用性查询被自动权限模式拒绝；不影响已完成的 M1 前向证据。

## 运行日志和输出路径

真实模型输出：`reports/m1_structure.json`；环境和版本：`reports/m1_verified_environment.json`；下载权重：`data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987/`（已忽略，不纳入 Git）。单元测试控制台输出：`Ran 11 tests in 8.956s`、`OK`。结果 JSON 的 `git_commit=dca6c804795e43421c99bc143f848064268a7167` 是执行时的**基线提交**，本轮修复当时尚未提交；最终代码版本应以本轮完成后的 Git 提交为准。

## 未解决问题

- 论文原始权重 revision、正式 prompt 模板与校准数据版本仍待核验；不能将本次工程选择表述为作者配置。
- 模型配置未暴露 `resolved_revision`；仅能确认请求固定 SHA 和缓存快照目录一致。
- Windows 缓存不支持 symlink，Hugging Face 提示可能多占磁盘空间；未启用管理员模式或删除缓存。

## 下一步建议（不自动执行）

M1 验收后停止，等待用户指令再开始 M2。M3 的 zero-ablation hooks 必须以本次实测的 Attention tuple 返回结构为前提，并独立通过相应正确性测试。
