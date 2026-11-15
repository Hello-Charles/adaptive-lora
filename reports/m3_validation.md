# M3 独立 zero-ablation 验收（2026-11-14）

## 当前任务与完成内容

**M3 四项门禁及 T03–T05 通过。** 本阶段从单输入、第 0 层的 Attention/MLP 分别消融开始，随后显式扩展到全部 24 层；未进入 M4，不计算 $t^*$、概率、层重要性或 Top-K。

| 验证项目 | 真实结果 |
|---|---|
| 模型 | 本地固定 `Qwen/Qwen2.5-0.5B`，`Qwen2ForCausalLM`，24 层 |
| requested revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| 输入 | `Hello, world!`；IDs `[9707, 11, 1879, 0]`，有效预测位置 3（零基） |
| 执行模式 | CPU、4 threads、seed=20261114、全模型 eval、参数冻结、inference_mode、use_cache=False |
| 先行单层验证 | 第 0 层 Attention/MLP 各 1 个正常案例，另各 1 个真实异常注入 |
| 全层验证 | 24 × 2 = **48 个独立目标**，从未联合消融 |
| 目标输出 | 主 hidden states `[1, 4, 896]` 置零，dtype/device/shape 保留；Attention 的 `(Tensor, None)` 与辅助项身份保留 |
| 非目标检查 | 每个正常案例均观察全部 48 个 Attention/MLP；同次前向的 hook 前后值相等，未被直接修改 |
| 残差路径 | 所有正常案例的目标 block 输出仍非零，未把整个 block 置零 |
| 正常/异常恢复 | 每次移除消融及观察 hook 后，完整 logits 与基线 **torch.equal 精确相等** |
| logits | `[1, 4, 151936]`，torch.bfloat16，全部有限；不提取预测 token 或概率 |
| 基础权重 | 494,032,768 个参数元素的前后指纹一致；全部冻结、无梯度 |
| 最终测试 | **48 个测试全部通过，无跳过**（原有 28 个 + M3 新增 20 个） |

参数指纹（参数名、dtype、shape 和原始字节，非 safetensors 文件哈希）：

`3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3`

单层与全层运行的前后参数指纹均为该值。指纹只覆盖 `named_parameters`，不把 buffers 也宣称为已验证不变。

## 文件变更

- [ablation.py](../src/adaptive_lora/ablation.py)：`zero_ablation(model, layer, component)` 上下文管理器；检查调用前提、模块归属及主输出结构；使用新建的零张量；禁止同根模型嵌套消融；正常/异常路径移除自己注册的 handle 并释放活跃登记；正常退出零次命中明确报错。
- [test_ablation.py](../tests/test_ablation.py)：真实 PyTorch 小模型的 17 项消融测试及 3 项验证脚本工具/CLI 测试；不加载 Qwen 权重，不用替身冒充真实模型。
- [validate_ablation.py](../scripts/validate_ablation.py)：只读本地缓存的短输入验证器；默认只做一层，`--all-layers` 才显式遍历全部独立目标。检查 hook 前后结构、输出别名风险、恢复、异常、权重字节指纹；失败不创建成功 JSON，已有结果拒绝覆盖。
- [PAPER_SPEC.md](../PAPER_SPEC.md)：记录 zero-ablation 的工程边界，不改公式与独立消融算法。
- [REPRO_PLAN.md](../REPRO_PLAN.md)、[README.md](../README.md)：更新 M3 门禁和离线复核方式。
- [m3_validation.json](m3_validation.json)：从两份真实结果校验后生成的逐目标验收索引，包含源码和原始证据文件 SHA-256。

此前 M2 与中文注释维护的未提交改动均保留。没有修改 M1 loader/观察 hook 或 M2 数据流程的执行逻辑，没有安装/升级依赖。

## 论文依据与工程假设

本轮按需从本地 PDF 提取第 6、20 页文字核对，而非把规格整理当作原文：

- **[论文] Section 4.3、公式 (4)、Algorithm 1 第 5–6 行**：在冻结基础模型上分两次独立前向，分别置零一个 Attention 输出或一个 MLP 输出。
- **[论文] 附录 E.2**：独立子模块 zero-ablation 的前向不计算梯度。
- **未实现**公式 (5) 的概率差聚合和后续选层；本阶段不是 trained-LoRA 的 path patching。

工程约定均已写入 [PAPER_SPEC.md](../PAPER_SPEC.md#zero-ablation-工程边界-复现决策)：

1. **[复现决策] hook 插入点**来自 M1 实际模块发现，而非按经验将整个 Transformer block 置零。Attention 仅替换主 Tensor，辅助项不变；MLP 返回 Tensor 时整体替换。普通 Tensor/tuple/list 在小模型上覆盖；未知 dict/namedtuple/非三维输出拒绝猜测。
2. **[复现决策] 不做原地清零**，因为输出可能与输入或残差共享存储；用 `zeros_like` 新建输出，测试专门覆盖 alias 风险。
3. **[复现决策] 显式前提**：调用者先 eval、冻结所有参数、禁用梯度；helper 校验而不偷偷修改这些状态。验证器明确执行前置冻结，并让每次前向使用完整新输入、`use_cache=False`，不传历史 KV cache。
4. **[复现决策] 作用域**：同一根模型禁止任何嵌套消融，即使换层/组件；只移除本 scope 的 hook，不清空已有 hook 字典。正常退出未命中报错，原始异常不被未命中错误遮蔽。
5. **[复现决策] 验证输入**为 M1 同款 4-token 结构探查文本，不是 M2 校准集；seed 和 CPU 线程数用于记录本地条件，不是论文设定。
6. **[验证边界] 非目标激活**可以随上游输入改变。直接作用域检查比较的是同一次前向中该模块在消融 hook 前后的值；不能误要求下游激活等于未消融基线。

M1 实测 `(Tensor, None)` / Tensor 的结构本次再次在真实模型上通过。未发现与论文独立 zero-ablation 定义的冲突。

## 执行命令

以下核心命令均实际执行；输出文件当时均不存在，未覆盖已有产物：

```bash
# Git 与环境检查：保留所有原有改动，在新本地分支工作。
git status --short --branch
git log -1 --format='%H %s'
git switch -c repro/m3-zero-ablation
.venv/Scripts/python.exe --version
.venv/Scripts/python.exe -I -m pip check

# 按需核对原文。
pdftotext -f 6 -l 6 -layout papers/adaptive_lora.pdf .cache/m3-20261114/paper-method.txt
pdftotext -f 20 -l 20 -layout papers/adaptive_lora.pdf .cache/m3-20261114/paper-calibration-cost.txt

# 修改前基线 28 PASS；新测试先写，缺模块时红灯；实现后定向测试通过。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -p 'test_ablation.py' -v

# 先在一个层上验证正常和异常恢复。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -u scripts/validate_ablation.py \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --layer 0 --threads 4 --output results/m3-20261114/single-layer.json

# 单层通过后，显式扩展到 48 个独立目标；没有启动 M4 校准。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -u scripts/validate_ablation.py \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --cache-dir data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --all-layers --threads 4 --output results/m3-20261114/all-layers.json

# 生成验收索引前核对 case 覆盖、参数/源码指纹、正常和异常恢复证据。
.venv/Scripts/python.exe -I -X utf8 .cache/m3-20261114/summarize_results.py

# 最终测试与环境证据。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/record_environment.py \
  --output results/m3-20261114/environment.json
.venv/Scripts/python.exe -I -X utf8 scripts/validate_ablation.py --help
.venv/Scripts/python.exe -I -m pip check
git diff --check
```

另检查结果与缓存的 Git 忽略状态，以及新文件语法、空白和已记录源码 SHA-256。完整终端输出归档在 [run.log](../results/m3-20261114/run.log)，原始分项日志见 [.cache/m3-20261114/](../.cache/m3-20261114/)。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 证据 |
|---|---|
| PASS | 修改前基线：28 tests，13.803 秒 |
| FAIL（预期红灯） | 先运行新测试，旧源码尚无 `adaptive_lora.ablation`，导入失败；随后实现 |
| PASS | 初版 15 项消融单元测试，0.358 秒；补全 CLI/维度/nograd 测试后，全套 48 项通过 |
| PASS | 最终全套 48 tests，12.610 秒，无跳过、无未修复失败 |
| PASS — T03 | 真实 Qwen 全 24 层 Attention 主输出分别为零，容器/辅助项保留；同次前向其他模块没有被直接替换 |
| PASS — T04 | 真实 Qwen 全 24 层 MLP Tensor 分别为零；残差 block 未被直接清零 |
| PASS — T05 | 每个正常真实 case 移除 hook 后完整 logits 精确恢复；两个第 0 层真实异常注入也恢复 |
| PASS | 小模型覆盖嵌套拒绝、已有 hook 保留、调用方异常、下游前向异常、注册失败、未知返回结构、非三维 Tensor、未命中、scope 内重新开启梯度等边界 |
| PASS | 小模型覆盖 attention Tensor/tuple/list、bfloat16、MLP dtype，以及输入/输出存储别名风险 |
| PASS | 真实运行参数前后字节指纹一致；全模型 eval、冻结，无梯度；所有本地 forward hooks 清理 |
| PASS | 两份真实结果的源码哈希一致且匹配当前文件，验收索引覆盖 48 个唯一目标 |
| PASS | CLI 输出不覆盖/参数门禁、帮助输出、依赖一致性、Git 忽略和空白检查 |
| NOT RUN | M4 的固定 $t^*$ 概率测量、N=1/N>1 重要性数学、N=30 校准、Top-K；没有运行 LoRA 训练或正式评估 |
| NOT RUN | GPU、compiled/distributed、并发共享模型、历史 KV cache 路径，不属于当前支持范围 |

单层脚本实测 8.540 秒，全层脚本 23.125 秒，均包含加载、参数指纹、恢复和断言开销；**不是论文校准耗时或性能对比**。

环境仍为 Python 3.12.8、torch 2.15.0+cpu、transformers 5.22.2，未安装/升级依赖或更改系统设置。

## 运行日志和输出路径

- [single-layer.json](../results/m3-20261114/single-layer.json)：首次第 0 层验证的 2 个正常和 2 个异常案例。
- [all-layers.json](../results/m3-20261114/all-layers.json)：48 个正常独立目标和第 0 层 2 个异常案例，含目标输出前后结构、每次恢复和完整参数指纹。
- [environment.json](../results/m3-20261114/environment.json)：当前解释器及实际导入状态。
- [run.log](../results/m3-20261114/run.log)：基线、红灯、实现后/最终测试、两次真实验证、索引校验和环境日志。
- [summarize_results.py](../results/m3-20261114/summarize_results.py)：验收索引校验脚本归档；原执行脚本位于 [.cache/m3-20261114/](../.cache/m3-20261114/)。
- [m3_validation.json](m3_validation.json)：供 Git 跟踪的机器可读索引，不含权重或激活值；两份原始结果的文件 SHA-256 及实际源码 SHA-256 均已保存。

上述原始结果/环境/日志和本地论文文本位于 Git 忽略目录。M3 没有下载数据或模型权重，也没有覆盖/删除 M1/M2 产物。

Git 分支：`repro/m3-zero-ablation`；基线 commit：`577527ad0ec343941bbed0b6519ff3544e301cc7`。工作区为 dirty，该 SHA 是 **M1 基线，不是 M3 提交**。本次未暂存、未自动提交或推送，已有未提交改动保留；实际运行源码的字节指纹用于补足代码身份。

## 未解决问题

M3 在当前环境和上述单输入契约内无阻塞，T03–T05 已通过，但仍有明确边界：

1. 只支持串行 eager、完整新输入、无历史 KV cache 的前向。活跃 scope 的登记锁不隔离外部请求；禁止共享模型并发、别名根模型和编译/分布式使用。
2. 真实异常注入仅覆盖第 0 层 Attention/MLP；其他错误路径由真实 PyTorch 小模型验证，不声称对所有架构/返回结构普适。
3. 模型 config 的 `resolved_revision` 仍为 null；记录的是请求的固定 SHA 和本地缓存，不能伪造模型配置暴露了实际 revision。
4. 正式数据/作者模板等开放问题仍见 [PAPER_SPEC.md](../PAPER_SPEC.md)，本次不改变 M2 smoke 的适用范围，也不证明批量/GPU 数值等价。
5. M3 尚未创建新的 Git 提交；基线、dirty 状态与源码哈希已如实保存。

## 下一步建议（不自动执行）

等待用户指令进入 M4：先固定未消融模型的 $t^*$ 和 $q$，实现同一个 token 在两个独立消融下的概率测量，并写 N=1/N>1 重要性及稳定 Top-K 测试。通过后才运行 N=30 校准。本阶段到此停止，不自动评分、选层或训练。
