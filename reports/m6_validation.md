# M6 统一基线与评估接口验收（2026-11-15）

## 当前任务与完成内容

**M6 四项门禁在工程 smoke 范围内全部通过。** 本阶段只实现 Full LoRA / Random-K / Adaptive-K 的统一训练与评估接口，没有进入 M7。先写测试，再实现核心模块和离线入口，完整回归通过后才运行已有固定本地 Qwen2.5-0.5B。

### 实际三方法结果

| 方法 | 所选层（零基；保留规划顺序） | Q/K/V投影 | A/B张量 | 实际可训练参数 | Full占比 | 短训练步数 | 合成smoke EM |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | 0–23全部层 | 72 | 144 | **1,474,560** | 100% | 3 | **0/3** |
| Random7 | `[9, 0, 15, 19, 5, 11, 20]` | 21 | 42 | **430,080** | 29.1667% | 3 | **0/3** |
| Adaptive7 | `[0, 1, 2, 5, 3, 8, 10]` | 21 | 42 | **430,080** | 29.1667% | 3 | **0/3** |

- 三次从同一基础缓存重新加载，不沿用前一种方法的已训练实例；模型为24层 `Qwen2ForCausalLM`，requested revision 为 `060db6499f32faf8b98477b0a26969ef7d8b9987`。config 未暴露 resolved revision，记录为 null，不能声称独立核验了作者版本。
- 每方法3个 batch，batch_size=1、无 accumulation，共 **9 个 optimizer steps**；训练样本顺序、超参数、评估输入和参考答案完全相同，只有层集合不同。
- 训练 seed=42；Random-K 独立 layer_seed=20261115，保存抽样顺序和升序层ID。Adaptive复用最终M4 **N30**，不是N1的另一组Top7，不重新校准。
- 各方法基础参数 **494,032,768** 个，全部冻结、无梯度，训练前后规范名称/dtype/shape/原始字节指纹一致；评估前后全部基础+adapter参数字节也一致。
- **Full与K-layer不是同参数预算**。实际Q为896→896、K/V为896→128，rank16下每层61,440个LoRA元素；实测24层1,474,560，7层430,080。不能把K/V视为方阵，也没有降低Full rank来伪造同预算。

### 三步训练数值

| 方法 | Step1 loss | Step2 loss | Step3 loss | 梯度L2 norm（逐步） |
|---|---:|---:|---:|---|
| Full | 3.573270320892334 | 2.9217276573181152 | 3.1625235080718994 | 50.67123335325547 / 37.652259628095294 / 204.5252474231697 |
| Random7 | 3.573270320892334 | 2.9883289337158203 | 3.106433868408203 | 34.588392541929885 / 27.738834428909822 / 149.0086079863238 |
| Adaptive7 | 3.573270320892334 | 3.0456244945526123 | 3.504087209701538 | 38.129387389080044 / 29.238013465161263 / 190.8409594887316 |

全部loss/梯度/更新后adapter有限，基础无梯度。Full每步144个有梯度/更新张量，首步72个非零；K-layer每步42个有梯度/更新张量，首步21个非零。标准B=0初始化下首步A梯度为零是正常情况；AdamW weight decay仍可能改变A，不能把“所有张量更新”说成“所有梯度均非零”。

每一步使用不同样本，三点不是同一样本的收敛曲线。Adaptive在相同token/labels、seed和配置下重复了M5训练核心：除重命名的sample_id外，三步记录及最终运行时adapter指纹完全相同；**这不是完整M5 CLI的注入探针/保存恢复重跑，也不是三方法完整重复实验**。

### 评估与拆分

- 固定合成训练3条、合成评估3条，校准30条仍取M2清单。
- 评估与训练/校准的sample_id、prompt字节哈希、实际prefix-token哈希交集分别全部为0。本次校准与合成训练的交集也为0，但算法允许校准来自训练侧，不强行禁止论文允许的关系。
- reference/labels不传入生成接口；评估输入只有prefix及身份字段，之后只追加模型自己的预测。gold仅在预测完成后按唯一ID对齐评分。
- 单样本、无padding，`logits[0,-1]`对应有效末尾next-token；argmax实现greedy / temperature0，未对零温度做除法。无KV cache，仅显式tokenizer EOS或长度上限终止。
- 本次9条预测均到达4-token长度上限，三者 **strip-only exact-match均0/3**。例如预测可能是完整提示复述或未完成句子，不能因含有目标词就算正确，也不丢弃错误样本缩小分母。
- 指标为 **`smoke_exact_match_strip_only_not_paper_accuracy`**；仅去两端空白，大小写/内部空白敏感。不是论文commonsense答案抽取、math CoT/regex或Alpaca-ZH PPL。合成题面本身可含目标词，只用于接口验证；不能宣称这是无答案暗示的能力测试或据此判断方法优劣。
- 正常/异常恢复逐模块原training flags；评估拒绝已有梯度/本地forward或pre-hooks，且不更新基础或adapter。

### checkpoint与耗时

| 方法 | adapter safetensors实际字节 | 加载基础模型/s | 统一训练+评估+审计/s | 保存checkpoint/s | 方法总计/s |
|---|---:|---:|---:|---:|---:|
| Full | 5,917,176 | 1.806885 | 14.668166 | 0.652747 | 17.160759 |
| Random7 | 1,725,832 | 0.971406 | 14.067135 | 0.622044 | 15.684157 |
| Adaptive7 | 1,725,816 | 0.902586 | 14.459060 | 0.635345 | 16.007414 |

总wall-clock **49.677969 s**，包括加载、指纹、审计、生成、序列化和IO；不是单独优化kernel时间或论文速度比较，未测GPU显存。文件含不同层名等序列化header，Random/Adaptive大小微差不能推断为不同参数量。

各checkpoint只含adapter safetensors、PEFT配置及审计元数据，复用M5已验收保存器，检查全部实际张量。**M6未额外运行三方法fresh-base恢复，不含optimizer/RNG训练断点续跑**；M5的历史恢复证据不能冒充本次三方法恢复。

基础规范参数SHA-256为：

`3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3`

共享配置内容SHA-256：

`a7b04c29014f69056e5ef93f3e0779d7cfd0cfc8f055b9bdb7ea5a9674254055`

共享完整协议内容SHA-256：

`bccee626b23a351148b22044c2c4c9206ca7904432526f978d0af7c77430554b`

文件字节哈希、规范内容哈希、运行时/存储命名域的参数指纹各有不同编码，不应相互混比；它们检查完整性而不是作者来源认证，也不覆盖buffers。

## 文件变更

- [baselines.py](../src/adaptive_lora/baselines.py)：纯Full/Random-K/Adaptive-K规划、严格零基索引、局部选层RNG、不可变层集合。
- [protocol.py](../src/adaptive_lora/protocol.py)：共享有界配置、固定训练/评估合成样本、内容指纹、监督边界、reference-free评估请求及三个身份域的直接交集门禁。
- [evaluation.py](../src/adaptive_lora/evaluation.py)：有界prompt-only greedy、EOS/长度终止、eval/no-grad模式作用域和独立ID对齐评分。
- [comparison.py](../src/adaptive_lora/comparison.py)：统一单方法runner，复用M5注入/短训练器，评估全参数字节核对，三方法数据/配置/实际参数统计汇总。
- [comparison_io.py](../src/adaptive_lora/comparison_io.py)：M4 N30→M2清单绑定、全部已记录产物哈希、实际tokenizer重新分词、环境/Git/源码身份记录；不重算校准。
- [compare_baselines.py](../scripts/compare_baselines.py)：只读本地权重、三次fresh base、有界CPU运行、独占结果目录、各方法checkpoint/轨迹/评估、CSV/JSON及最后整体成功标记。无正式epochs/GPU/下载开关。
- 五个新增测试文件：[test_baselines.py](../tests/test_baselines.py)、[test_protocol.py](../tests/test_protocol.py)、[test_evaluation.py](../tests/test_evaluation.py)、[test_comparison.py](../tests/test_comparison.py)、[test_compare_baselines.py](../tests/test_compare_baselines.py)。
- [PAPER_SPEC.md](../PAPER_SPEC.md)、[README.md](../README.md)、[REPRO_PLAN.md](../REPRO_PLAN.md)：论文边界、复现选择、命令和门禁；[m6_validation.json](m6_validation.json) 为从真实产物/测试/独立审计生成的索引。

新增源码/测试均使用中文注释/docstring说明隐含边界。未修改已验收M5训练器、checkpoint或历史M2–M5产物。当前分支 `repro/m6-unified-baselines`；基线commit仍为 `577527ad0ec343941bbed0b6519ff3544e301cc7`（M1，不是M6提交），dirty状态与41个运行时Python源码指纹记录实际实现。保留之前所有未提交工作，未自动提交或推送。

## 论文依据与工程假设

按任务读取本地论文相关页，没有整篇重读：

- **Table 2 / Section 5.3（PDF第8页）**：0.5B的Full24/Adaptive7、rank16、N30、同任务集；原训练为10,000条/3epochs。支持“同per-layer rank但不同总参数预算”，不是本次3步合成结果。
- **Section 5.4（第9页）**：Random为均匀选层；Adaptive/Random/Sequential/Reverse在相同超参数下比较，使用Alpaca-ZH test PPL、3,000训练样本。本项目M6仅实现计划要求的前三种Full/Random/Adaptive接口，不扩展额外策略，不把PPL实验说成主表准确率。
- **Appendix C.3（第19页）**：无in-context examples、greedy temperature0；commonsense答案抽取、math CoT/任务专用解析。该部分的共同greedy原则沿用，完整任务解析器尚未实现。
- **Section4.5 / 5.1、Appendix C.1–C.2**：沿用M5已核查的Q/K/V、低秩参数量与AdamW参考；M5中记录的BF16/FP32、无AMP、合成数据、答案mask、显式EOS、max96拒绝超长和3steps差异仍适用。

**[复现决策]**：本机CPU/4线程/确定性算法、固定模型revision、Random seed20261115、不放回抽样、共同seed分别重置、合成数据协议、4-token生成上限和strip-only评分、拒绝直接输入交集、独占结果目录及独立存档核验。相同seed不保证不同目标集合的同名A初始化相同，跨依赖/设备/线程配置不保证逐字节重现。正式数据版本、作者模板/截断边界、loss mask和任务专用答案解析仍为 **[待核验]**，影响正式结果的选择须由用户确认。

没有改变论文层重要性公式、Top-K规则、Q/K/V要求或把path patching与zero-ablation混同。合成smoke不需要批准正式实验假设，也不构成M7成本授权。

## 执行命令

在项目根目录、已有虚拟环境运行；下列真实结果目录不能重用，复核时必须换新名字。

```bash
# 按需核查论文基线章节；C.3复用M5已提取的第19页文本。
pdftotext -enc UTF-8 -f 8 -l 9 -layout papers/adaptive_lora.pdf .cache/m6-20261115/paper-baselines.txt

# 测试先行：按各新增文件分别运行，红灯阶段缺模块/入口，随后实现并通过。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 \
  -m unittest discover -s tests -p test_compare_baselines.py -v

# 三方法真实短CPU smoke，共9步；离线、无付费GPU或正式数据。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/Scripts/python.exe -I -X utf8 scripts/compare_baselines.py \
  --m4-dir results/m4-20261114-n30 --manifest results/m2-20261114/manifest.json \
  --steps 3 --train-seed 42 --layer-seed 20261115 --rank 16 --alpha 32 \
  --max-new-tokens 4 --threads 4 --output-dir results/m6-20261115-smoke

# 独立存档审计：不导入训练/评估/协议模块，不加载基础模型或重训。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/Scripts/python.exe -I -X utf8 .cache/m6-20261115/audit_m6.py \
  --run-dir results/m6-20261115-smoke --output results/m6-20261115-evidence/audit.json

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python.exe -I -X utf8 -m compileall -q src scripts tests
.venv/Scripts/python.exe -I -X utf8 -m pip check
.venv/Scripts/python.exe -I -X utf8 scripts/compare_baselines.py --help
git diff --check
git status --short --branch
```

原始重定向日志保存在 [.cache/m6-20261115/](../.cache/m6-20261115/)；阶段结束将已关闭日志、论文摘录和独立审计脚本归档到证据目录。未执行pip安装、升级、下载、git add/commit/push或M7。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查 | 实际结果 / 边界 |
|---|---|---|
| PASS | 阶段开始旧回归 | 123 tests，14.347s |
| PASS | baseline选层 | 6 tests；局部RNG、seed复现、完整/非法层集合、输入修改隔离 |
| PASS | 公共协议/拆分 | 8 tests；配置边界、答案与输入分离、哈希/labels/ID、允许校准属于训练、拒绝eval交集 |
| PASS | greedy评估/评分 | 11 tests；EOS/长度、最后位置、异常模式恢复、NaN/已有梯度/hooks拒绝、真实微型PEFT参数不变、gold不改变生成输入 |
| PASS | 统一runner | 6 tests；真实微型Qwen2/PEFT三方法，参数480/160/160、重复性、拒绝混合数据/配置/基础、协议回调篡改 |
| PASS | 离线CLI/来源 | 8 tests，2.804s；M4/M2绑定、重新分词、前置门禁、真实微型24层三方法端到端、第二方法失败不得整体PASS |
| PASS | 真实固定0.5B | 每方法3步、共9步；实际参数/冻结/有限训练/评估不改参数、checkpoint与统一结果导出 |
| PASS | 独立存档核验 | **261项**；54个摘要产物、15个来源、41个运行时源码、实际safetensors、输入/解码/独立评分一致 |
| PASS | 完整集成回归 | 162 tests，13.935s；通过后才运行真实比较 |
| PASS | 最终完整回归 | **162 tests，15.734s，无跳过**（123 + M6新增39） |
| PASS | compileall / pip check / CLI help | 均退出0；pip输出 `No broken requirements found.` |
| PASS | Git边界/空白 | 新数据/权重/结果受忽略，不加入Git；diff检查退出0，可能打印LF→CRLF提示，不作为失败 |
| FAIL（预期红灯） | 五组测试先行 | 核心模块或CLI尚不存在时的导入/文件错误，日志保留；实现后全部通过，无未解决失败 |
| NOT RUN | 正式训练/评估 | 无commonsense/math benchmark或Alpaca-ZH PPL、无3epochs/GPU显存/性能测量 |
| NOT RUN | 完整重复与恢复 | 未重跑三方法整次CLI，未为三方法额外fresh-base恢复，未测试optimizer/RNG训练断点续跑 |

微型集成测试使用真实Qwen2/PEFT运算但随机小权重和字符tokenizer夹具，不代表0.5B自然语言效果。CLI测试只mock权重加载及受控失败，训练/生成/保存实际执行。独立存档审计只验证已保存梯度/冻结记录一致性，不独立重建反向；其范围见审计JSON。

## 运行日志和输出路径

- [M6运行目录](../results/m6-20261115-smoke/)：共享配置/协议、样本ID、拆分审计、M4/M2来源、环境/Git/源码身份、基线规划、[comparison.csv](../results/m6-20261115-smoke/comparison.csv)、[comparison.json](../results/m6-20261115-smoke/comparison.json)、[run_summary.json](../results/m6-20261115-smoke/run_summary.json)和关闭后的[run.log](../results/m6-20261115-smoke/run.log)。
- 各方法目录 [full/](../results/m6-20261115-smoke/full/)、[random_k/](../results/m6-20261115-smoke/random_k/)、[adaptive_k/](../results/m6-20261115-smoke/adaptive_k/)：选层、配置、真实参数审计、逐步JSONL/训练指标、预测/评分、模型运行信息、checkpoint、耗时和方法完成标记。
- [独立存档审计](../results/m6-20261115-evidence/audit.json)：无完整prompt/参考答案/输入token内容，保存核验摘要和字节身份；原始完整协议和checkpoint继续只保存在Git忽略目录。
- [最终回归日志](../.cache/m6-20261115/tests-final.log)、[真实控制台日志](../.cache/m6-20261115/real-comparison-console.log)、[独立审计日志](../.cache/m6-20261115/independent-audit.log)及全部预期红灯/green日志归档到 [evidence/](../results/m6-20261115-evidence/)。
- [m6_validation.json](m6_validation.json)：机器可读验收索引，由实际产物、最终测试和审计生成，记录完整哈希，不包含完整prompt、token清单或权重。运行日志不在运行中的自散列范围，而在关闭后的独立审计/最终索引中记录。

## 未解决问题

1. 本次评估0/3是诚实smoke结果，不是成功复现论文准确率；不能据此推断学习能力、统计显著性或Adaptive收益。全部到达4-token上限，正式任务必须定义足够长度和任务专用解析。
2. 输入隔离仅覆盖精确ID/prompt/prefix tokens，不能检测语义改写、题面内含目标词或预训练污染。校准来源的远端认证仍继承M2镜像边界，不因本次本地哈希通过而消失。
3. 同seed保证受控同协议路径可测试，不保证不同目标集合中的同名A初始化相同；Full/Random只运行了一次真实训练，未有三方法完整重跑或多seed统计。
4. M6评估模块接入统一比较CLI，尚未单独提供正式 `scripts/evaluate.py`。正式commonsense/math/PPL数据适配、作者prompt/答案解析细节、模型原版revision等仍待核验。
5. M6 checkpoint仅验证保存内容，未额外三方法恢复、optimizer/RNG续跑；CPU总耗时不能外推GPU训练成本或显存。正式模型/数据/GPU成本未获本阶段授权。

## 下一步建议（不自动执行）

停在M6。用户明确要求M7并批准成本后，先冻结正式数据版本/划分、校准与训练关系、prompt/loss mask/任务评估与生成长度、有效batch和精度，再进行Qwen2.5-1.5B、N250、Top8的小型GPU pilot。不得直接运行完整训练来追逐论文数值；不自动下载、改依赖、进入M7或提交/推送。
