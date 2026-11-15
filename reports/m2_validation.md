# M2 校准数据准备验收（2026-11-14）

## 当前任务与完成内容

**M2 四项门禁通过。** 本阶段只准备校准输入，不实现 zero-ablation、不计算模型预测 token 或层重要性、不训练。

| 项目 | 本次实测/保存结果 |
|---|---|
| 数据集 | `tatsu-lab/alpaca`，train split |
| 固定 revision | `dce01c9b08f87459cf36a430d809084718273017` |
| train 大小 | 52,002 行；读取固定单文件 parquet 后实测，不来自测试夹具 |
| 下载字节数 | 24,246,638 bytes |
| 文件 SHA-256 | `06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073` |
| 校准抽样 | N=30，seed=20261112，不放回、保留抽取顺序，索引零基 |
| tokenizer | `Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| 输入长度 | 26–64 tokens；max_length=512，30 条均未截断 |
| 模板/特殊 token | user-only 消息 + tokenizer chat template + assistant generation prompt；渲染后 `add_special_tokens=False` |
| 逐样本位置 | 最后有效输入 token 的零基索引，mask 全为 1，无 padding |
| 真实批量检查 | 29 条存在右 padding；全部 30 条去掉 pad 后 IDs 与逐样本一致，预测位置正确 |
| 两次独立离线运行 | 样本、配置与内容指纹一致；时间戳和运行命令不参与一致性比较 |

内容指纹为 `7c829ccb4c332291ea432c62b3a256bf8459d5a901e4eb9a575b0818aedfedc9`。完整逐样本索引、行/输入/prompt 哈希与长度见 [机器可读验收索引](m2_validation.json)，该文件不保存原始 prompt 或答案。

## 文件变更

本阶段新增/完成：

- [data.py](../src/adaptive_lora/data.py)：确定性抽样、输入字段隔离、二值连续 mask 校验、精确截断标记、逐样本/批量有效位置、模板与内容指纹、可追溯清单。超长样本整次报错，不静默截掉回答起点或重抽样。
- [prepare_calibration.py](../scripts/prepare_calibration.py)：固定 train parquet 的 SHA/revision；数据下载与 tokenizer 离线加载分离；下载目录隔离；文件字节哈希校验；直接用本地 PyArrow 解析；结果独占创建；保存 Python/包版本、Git 基线、dirty 状态和源码字节哈希。
- [test_data.py](../tests/test_data.py)：抽样、答案字段变更、哈希/模板元数据、精确截断、左右 padding、非法输入、超长拒绝，以及固定缓存真实 Qwen tokenizer 的集成测试。
- [test_prepare_calibration.py](../tests/test_prepare_calibration.py)：CLI 参数、完整 SHA、正长度、新下载/已有离线目录、结果不覆盖、tokenizer 不联网、先校验文件哈希再读取等契约测试。
- [PAPER_SPEC.md](../PAPER_SPEC.md)：精确区分附录 E.2 的训练集抽样和 E.4 的无需答案依据，记录 M2 smoke 的工程协议。
- [REPRO_PLAN.md](../REPRO_PLAN.md)、[README.md](../README.md)：更新 M2 验收状态、离线复核方式和边界。
- [m2_progress.md](m2_progress.md)：保留早期受阻记录，增加当前状态指针，纠正“未执行任何命令”的不准确措辞（当时 Git 状态查询成功）。

此前用户要求的中文注释改动均保留；本轮未改动 M1 模型加载/观察 hook 的执行逻辑。早期注释维护记录见 [code_comments_20261113.md](code_comments_20261113.md)。

## 论文依据与工程假设

通过本地 PDF 的 `pdftotext -layout` 文本核对：

- **[论文] Section 4.3、附录 E.4**：冻结基础模型选择预测 token，不需要 gold-answer 标签。
- **[论文] 附录 E.2**：从训练集抽 N 个 calibration 输入，后续运行梯度关闭的独立子模块消融。
- **[论文] Section 4、附录 E.1/E.3**：N=30 作为低成本校准默认值；不是本项目测得的性能或稳定性结果。
- **[复现决策]** 用 Alpaca train 验证流程、固定上述数据/model revision、seed=20261112、max_length=512、user 文本拼接及 Qwen chat template；这些不是作者正式主表的逐 token 设置。
- **[复现决策]** 超长时拒绝整次清单生成，避免右截断把 assistant generation prompt 截掉；不换样本。本次 30 条都短于上限，因此没有改变固定抽样集合。
- **[复现决策]** 模板已渲染特殊标记，二次分词关闭自动添加。全部 30 条已与原生 `apply_chat_template(tokenize=True, return_dict=True)` 核对一致，不只凭 mock 断言。
- **[待核验]** 官方域名连接失败，使用 `https://hf-mirror.com` 的固定版本元数据与下载。文件大小、字节 SHA-256 和其 LFS metadata 一致，但未通过官方域名作独立来源认证，不声称这是作者采用的数据版本。

对答案泄漏的保证限于**不使用 output/reference 构建 prompt**。真实输入字段自身可能已有答案内容；不能仅用 output 是否为 prompt 子串判定泄漏，也不能把哈希当作脱敏。源数据卡声明 CC-BY-NC-4.0，后续使用必须遵守许可。

无算法冲突；此次只将原先笼统归到 E.2 的“无需 gold-answer”依据细化为 Section 4.3/E.4。

## 执行命令

下面按实际参数整理，长路径用变量减少重复；原始运行命令也保存在每份清单的 provenance 中。成功输出均为新文件，未覆盖历史结果。

```bash
# 依赖与修改前基线：PASS，无环境安装/升级。
.venv/Scripts/python.exe --version
.venv/Scripts/python.exe -I -m pip check
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v

# 按需核查原文；输出只留在被忽略的本地证据目录。
pdftotext -layout papers/adaptive_lora.pdf .cache/m2-20261114/paper.txt

# 官方域名 curl 连接失败；以下镜像 metadata 查询成功。
curl --connect-timeout 10 --max-time 30 --fail --silent --show-error --location \
  'https://hf-mirror.com/api/datasets/tatsu-lab/alpaca?blobs=true' \
  --output .cache/m2-20261114/dataset_metadata_mirror.json

CACHE=data/datasets/alpaca_dce01c9b08f87459cf36a430d809084718273017_m2_20261114
REV=dce01c9b08f87459cf36a430d809084718273017
MODEL_REV=060db6499f32faf8b98477b0a26969ef7d8b9987

# 首次调用：文件下载/字节哈希通过，旧 reader 因 Windows 长路径失败，无成功清单。
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
.venv/Scripts/python.exe -I -X utf8 scripts/prepare_calibration.py \
  --allow-download --hub-endpoint https://hf-mirror.com --dataset-cache-dir "$CACHE" \
  --dataset-revision "$REV" --model-revision "$MODEL_REV" \
  --n 30 --seed 20261112 --max-length 512 --output results/m2-20261114/manifest.json

# 修正 reader 后只用上述缓存：两条命令分别实际执行且均 PASS，没有再次下载。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/prepare_calibration.py \
  --hub-endpoint https://hf-mirror.com --dataset-cache-dir "$CACHE" \
  --dataset-revision "$REV" --model-revision "$MODEL_REV" \
  --n 30 --seed 20261112 --max-length 512 --output results/m2-20261114/manifest.json
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/prepare_calibration.py \
  --hub-endpoint https://hf-mirror.com --dataset-cache-dir "$CACHE" \
  --dataset-revision "$REV" --model-revision "$MODEL_REV" \
  --n 30 --seed 20261112 --max-length 512 --output results/m2-20261114/manifest-repeat.json

# 验证器直接读取来源行并使用原生模板分词，不调用 build_manifest 自我比较。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 .cache/m2-20261114/verify_manifest.py \
  --first results/m2-20261114/manifest.json --repeat results/m2-20261114/manifest-repeat.json \
  --parquet "$CACHE/hub/datasets--tatsu-lab--alpaca/snapshots/$REV/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet" \
  --metadata .cache/m2-20261114/dataset_metadata_mirror.json \
  --model-cache data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --output reports/m2_validation.json

# 最终测试、真实导入环境快照及 CLI 帮助。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/record_environment.py --output results/m2-20261114/environment.json
.venv/Scripts/python.exe -I -X utf8 scripts/prepare_calibration.py --help
.venv/Scripts/python.exe -I -m pip check
git diff --check
```

另用 `git status`/`git log`/`git check-ignore` 检查基线与忽略规则，并创建本地分支 `repro/m2-calibration-data`。没有暂存、提交或推送。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查及真实结果 |
|---|---|
| PASS | 修改前 20 个测试，19.230 秒 |
| FAIL（预期红灯） | 先写新测试时，旧实现缺少 `manifest_content_hash`，测试导入失败；随后实现并重新测试 |
| PASS | 初版 M2 全套 28 个测试，11.672 秒；reader 修正后的 5 个 CLI 测试也通过 |
| FAIL → 修复 | 首次真实处理报 `[WinError 206] 文件名或扩展名太长`；改用 PyArrow 直接读本地 parquet，保留下载缓存，后续离线成功 |
| PASS | 最终 28 个测试，12.417 秒，全部通过、无跳过；包括真实缓存 tokenizer，不是纯 mock |
| PASS | 源文件大小、SHA-256、revision 与镜像 metadata 一致；train 实读 52,002 行 |
| PASS | N=30 唯一索引/固定 seed；所有行、输入、prompt 哈希及样本 ID 正确 |
| PASS | 30 条 native chat-template IDs、mask、回答起始标记、有效位置均正确 |
| PASS | 30 条真实批量输入，29 条右 pad，去 pad 后 token 及有效位置与逐样本一致 |
| PASS | 两次离线清单配置/样本/content SHA 一致；原始 JSON 整文件 SHA 不同是时间戳和命令造成，不应误判不可复现 |
| PASS | 实际运行源码哈希与当前 4 个 M2 源码/测试文件一致；环境核心包导入和 `pip check` 成功 |
| PASS | CLI `--help`、输出不覆盖/新缓存门禁和 Git 忽略规则检查 |
| NOT RUN | 模型前向、冻结参数校验、zero-ablation、重要性评分、Top-K、LoRA 训练、正式评估；不属于 M2 |

环境：Python 3.12.8，torch 2.15.0+cpu，transformers 5.22.2，tokenizers 0.23.3，peft 0.21.2，datasets 5.2.0，pyarrow 26.0.1，huggingface-hub 2.4.0。没有安装、升级依赖或改变 Python 环境。

## 运行日志和输出路径

- [manifest.json](../results/m2-20261114/manifest.json)：第一份成功的完整输入清单，含模板原文、IDs/mask、索引/哈希、位置、revision、运行参数及源码哈希。
- [manifest-repeat.json](../results/m2-20261114/manifest-repeat.json)：第二次独立离线生成，不覆盖第一份。
- [environment.json](../results/m2-20261114/environment.json)：当前解释器环境快照。
- [run.log](../results/m2-20261114/run.log)：归档的真实基线/红灯、首次失败、两次成功、最终测试和验收输出。
- [verify_manifest.py](../results/m2-20261114/verify_manifest.py)：独立验收脚本副本；原调用在 [.cache/m2-20261114/](../.cache/m2-20261114/) 下，SHA 记录在机器可读验收索引中。
- [m2_validation.json](m2_validation.json)：可纳入 Git 的索引/哈希验收证据，不含原始 prompt/答案。
- [data/datasets/](../data/datasets/)：专用下载缓存；含完整固定 parquet，未重复下载，长路径处理尝试留下的目录予以保留。
- [.cache/m2-20261114/](../.cache/m2-20261114/)：原始分项日志、镜像 metadata、论文提取文本、验证脚本。

数据、完整清单、运行日志、缓存和本地论文提取文本均受 Git 忽略；没有公开发布。

Git 基线：`577527ad0ec343941bbed0b6519ff3544e301cc7`。运行时工作区为 dirty，**这是 M1 基线而不是 M2 提交**；当前 M2 的实际代码身份由清单/验收索引中的源码 SHA-256 补充。本次保留原有未提交改动，未擅自提交或推送。

## 未解决问题

M2 本地流程无阻塞。以下边界仍在，不能据本次结果宣称正式复现实验完成：

1. 官方 Hub 直连失败；目前固定的是镜像所提供内容，官方来源独立核验及与论文作者版本一致性仍待核验。
2. 作者正式训练数据版本、原始 prompt/token 位置及 loss mask/评估协议仍未冻结。本次 Alpaca smoke 选择不替代正式设置。
3. 输入字段本身含答案/敏感内容的情况不由字段隔离自动解决；未验证批量 logits 数值等价。
4. 新的 M2 Git 提交尚未创建；源码和产物的 dirty 基线及字节指纹已明确记录，不冒充已提交代码。

## 下一步建议（不自动执行）

等待用户指令进入 M3：从一个输入、一个层、一个子模块开始实现独立 zero-ablation，测试输出结构、作用范围和异常清理。当前不实现 M3、不运行训练；若要正式采用任何 smoke 数据协议，仍须先由用户确认其对正式结果的适用性。
