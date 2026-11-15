# M4 重要性评分与 Top-K 选层验收（2026-11-14）

## 当前任务与完成内容

**M4 四项门禁全部通过，里程碑 A 完成。** 本轮只执行用户指定的 M4：先通过数学/生命周期测试和真实 N=1 门禁，再完成固定 30 条输入的全部校准、独立复算及图表检查。未进入 M5，未注入 LoRA、未训练，也没有下游效果结论。

### 真实 N=30 结果

| 项目 | 已核验结果 |
|---|---|
| 模型 | 本地 `Qwen/Qwen2.5-0.5B`，24 层，CPU、4 intra-op threads |
| requested revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| resolved revision | config 未暴露，记录为 null；不编造作者版本 |
| Attention backend / logits | `sdpa` / `torch.bfloat16`；未编译的串行 PyTorch 调用 |
| 数据 | M2 已验收的 Alpaca train 固定 30 条，seed=20261112，无重新抽样 |
| 清单内容 SHA-256 | `7c829ccb4c332291ea432c62b3a256bf8459d5a901e4eb9a575b0818aedfedc9` |
| 输入长度 / 位置 | 26–64 tokens，逐条无 padding/截断；使用最后有效输入位置的 next-token logits |
| 前向 | **1,500 次**：30 次 baseline + **1,440 次独立消融** + 30 次额外恢复 |
| 固定 token | 所有消融均读取各样本原始 t* 的概率，不使用新的 argmax |
| 完整 logits 恢复 | 30/30 次 `torch.equal` 精确恢复 |
| 基础参数 | **494,032,768** 个参数元素，前后字节指纹一致，冻结且无梯度 |
| hooks | 所有独立作用域正常清理，无泄漏；异常路径由核心测试和 M3 验收覆盖 |
| 输出 | 24 层分数与两个分量、全排名、Top-7、实际样本 IDs/mask、原始逐样本 token/概率、配置/环境/源码身份、CSV/JSON/JSONL/SVG/HTML |
| N=30 时间 | 加载 **3.927790 s**；校准 **1,107.881943 s（约 18.46 分钟）**；整次 **1,113.476196 s** |
| 独立复算 | 全部公式/排名/跨文件一致性、13 个产物哈希及运行时记录的 21 个源码文件均通过 |
| 有限重复性 | 独立 N=1 全部原始概率轨迹与 N=30 首样本精确一致；**未重跑完整 N=30** |
| 最终测试 | **92 tests，20.727 s，无跳过**（48 旧 + 44 新） |

### 最终 Top-7

层索引从 **0** 开始。重要性降序为 **`[0, 1, 2, 5, 3, 8, 10]`**；若后续按层号显示/定位，则是 `[0, 1, 2, 3, 5, 8, 10]`，两种顺序不要混淆。

| 排名 | 零基层号 | I_l | Attention mean \|Δq\| | MLP mean \|Δq\| |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.673805658893 | 0.330289207952 | 0.343516450941 |
| 2 | 1 | 0.625775370801 | 0.343220757810 | 0.282554612991 |
| 3 | 2 | 0.592306367980 | 0.227070084851 | 0.365236283129 |
| 4 | 5 | 0.486686594961 | 0.205296187041 | 0.281390407920 |
| 5 | 3 | 0.460302380373 | 0.155317807818 | 0.304984572554 |
| 6 | 8 | 0.410759606608 | 0.233195277373 | 0.177564329235 |
| 7 | 10 | 0.406571174258 | 0.255757143317 | 0.150814030940 |

完整精度和其余 17 层见 [24 层 CSV](../results/m4-20261114-n30/layer_importance.csv) 与 [JSON](../results/m4-20261114-n30/layer_importance.json)。这些是本项目 smoke 的冻结基础模型 zero-ablation 分数，不是论文表格数值、训练收益或准确率。

### 先行 N=1 门禁

M2 首条 train 行 13808，53 tokens、有效位置 52；原始 token ID 785，q=`0.1498081237077713`。48 次独立消融保持该 token，50 次前向含一次额外恢复；参数/梯度/hook/完整 logits 均通过。N=1 的 Top-7 为 `[0, 1, 2, 20, 23, 16, 10]`，不替代最终 N=30 结果。

N=1 加载 4.441609 s，校准 57.185324 s，整次 63.410454 s。两次实际运行的基础参数指纹与 M3 一致：

`3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3`

指纹覆盖全部 named parameters 的名称/dtype/shape/原始字节，不覆盖 buffers，不是 safetensors 文件哈希或来源签名。不同输入长度、CPU 调度/并行的本机检查会影响 wall-clock；这些计时不能当作论文性能 benchmark。

## 文件变更

- [calibration.py](../src/adaptive_lora/calibration.py)：未消融 t*/q、固定 token 概率、独立逐层校准、公式 (5)、参数/梯度/hook 与完整 logits 恢复审计；核心结果不含时间戳，便于重复比较。
- [selection.py](../src/adaptive_lora/selection.py)：全层降序排名与 Top-K，零基索引，相等分数小索引优先。
- [artifacts.py](../src/adaptive_lora/artifacts.py)：独占 JSON/CSV、零基线单轴 SVG、明暗/纹理/键盘/hover HTML 和完整数据表，不安装 matplotlib。
- [run_calibration.py](../scripts/run_calibration.py)：离线固定清单/模型入口，原生 tokenizer 重分词、配置/环境/源码身份、逐样本 JSONL、失败保留与最终成功标记。
- [validate_calibration.py](../scripts/validate_calibration.py)：不导入校准器/排序器，从原始概率独立复算数学、排名、跨文件一致性和哈希；不执行新的模型前向。
- [test_calibration.py](../tests/test_calibration.py)、[test_selection.py](../tests/test_selection.py)、[test_artifacts.py](../tests/test_artifacts.py)、[test_run_calibration.py](../tests/test_run_calibration.py)、[test_validate_calibration.py](../tests/test_validate_calibration.py)：M4 新增 44 项测试，中文注释注明 fixture/mock 边界。
- [PAPER_SPEC.md](../PAPER_SPEC.md)、[README.md](../README.md)、[REPRO_PLAN.md](../REPRO_PLAN.md)：记录数值/运行协议、离线复核命令和已核验的 M4 门禁；不改论文算法。
- [m4_validation.json](m4_validation.json)：由真实结果/独立复核生成的验收索引，含证据与运行/后处理代码 SHA-256，不包含完整 prompt/答案。

保留此前 M2/M3/注释维护的全部未提交工作；未修改依赖、模型权重、数据或已有结果。当前分支 `repro/m4-calibration-selection`，Git 基线 `577527ad0ec343941bbed0b6519ff3544e301cc7` 是 M1 提交，**不是 M4 提交**；本次未提交、未推送。

## 论文依据与工程假设

已按需核对本地 PDF 第 6 页，而非把整理文档代替原文：

- **[论文] Section 4.3、公式 (4)、Algorithm 1 第 3–6 行**：使用冻结基础模型的原始 t*，两次独立干预分别置零 Attention 或 MLP 模块输出。
- **[论文] 公式 (5)、Algorithm 1 第 7–10 行**：逐样本取两项概率差绝对值之和，再除以 N；不能先平均有符号差再取绝对值。
- **[论文] Section 4.4、Algorithm 1 第 11–12 行**：accuracy mode 在所有层上取 Top-K；未加 efficiency-mode 的深度过滤。
- Section 4.3 支持 N=30 的低成本校准默认值，但本次 0.5B/Alpaca smoke 不等于作者正式主表配置。未发现算法冲突；本阶段不是 trained-LoRA 的 path patching。

**[复现决策]** 0.5B/Alpaca smoke、Top-7、M2 模板和有效末尾位置、FP32 softmax、math.fsum、相等分数小层号优先、CPU/4 threads/inference seed=20261114/确定性算法及每样本完整恢复检查，不冒充作者逐 token 实现。M2 抽样 seed=20261112 不变。eager 指未编译执行，不代表强制 Attention backend 为 eager；实际为 sdpa 并已记录。

评分本身需要 N(1+2L)=1,470 次前向；额外恢复是工程门禁，增加 30 次但不进入公式。基础模型冻结、eval 和 inference_mode 是三个分别满足的约束。CLI 核对完整固定清单、tokenizer 元数据/模板及已保存 prompt 的 IDs/mask/位置，不重新抽样；原 parquet 与 user-only 构造的独立来源验证仍依赖 M2，不声称本轮重新审查了所有原始答案。

## 执行命令

以下核心命令均实际执行，输出路径当时不存在，未覆盖历史结果：

```bash
# 修改前 48 tests，测试先写红灯，再实现并补全，最终 92 PASS。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python.exe -I -X utf8 -m compileall -q src scripts tests
.venv/Scripts/python.exe -I -m pip check
git diff --check
pdftotext -enc UTF-8 -f 6 -l 6 -layout papers/adaptive_lora.pdf results/m4-20261114-evidence/paper-method-utf8.txt

# 首条门禁与独立复算。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/run_calibration.py \
  --manifest results/m2-20261114/manifest.json --limit 1 --top-k 7 --threads 4 \
  --output-dir results/m4-20261114-n1
.venv/Scripts/python.exe -I -X utf8 scripts/validate_calibration.py \
  --run-dir results/m4-20261114-n1 --manifest results/m2-20261114/manifest.json \
  --expected-n 1 --output results/m4-20261114-audit/n1.json

# N=1 通过后显式运行固定的全部 30 条；后台完成，退出 0。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/Scripts/python.exe -I -X utf8 scripts/run_calibration.py \
  --manifest results/m2-20261114/manifest.json --top-k 7 --threads 4 \
  --output-dir results/m4-20261114-n30
.venv/Scripts/python.exe -I -X utf8 scripts/validate_calibration.py \
  --run-dir results/m4-20261114-n30 --manifest results/m2-20261114/manifest.json \
  --expected-n 30 --compare-prefix-run results/m4-20261114-n1 \
  --output results/m4-20261114-audit/n30.json

# 实际打开本地图表，点击主题/纹理、真实 Tab 聚焦、hover 并截图。
node .cache/m4-20261114/render_chart.cjs \
  results/m4-20261114-n1/importance.html results/m4-20261114-visual-n1
node .cache/m4-20261114/render_chart.cjs \
  results/m4-20261114-n30/importance.html results/m4-20261114-visual-n30
```

另实际检查两个 CLI 的 `--help`、结果/缓存的 Git 忽略状态，以及明/暗颜色 validator。明色 `#2a78d6` / surface `#fcfcfb`，暗色 `#3987e5` / surface `#1a1a19`，两者均 **ALL CHECKS PASS**。单系列无 CVD 相邻颜色对可比较，不能将 n/a 表述为验证多系列色盲分离。具体绝对路径/调用及输出归档在 [证据目录](../results/m4-20261114-evidence/)。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 证据/边界 |
|---|---|---|
| 旧测试基线 | PASS | 48 tests，8.474 s |
| M4 初始红灯 | FAIL（预期） | 先写测试时模块/CLI 缺失；层完整性/归属反例先失败，修复后通过，日志保留 |
| 全套回归 | PASS | **92 tests，20.727 s，无跳过**；48 旧 + 44 新 |
| 语法与依赖一致性 | PASS | compileall 成功；pip check: No broken requirements found |
| Git 空白/忽略检查 | PASS | diff --check 成功；原始数据/概率/图/结果/缓存继续被忽略；CRLF 提示不是失败 |
| T01 基础参数/梯度 | PASS | 两次真实运行均与 M3 参数指纹一致，全部冻结/无梯度；buffers 不在指纹内 |
| T02 固定 token/位置 | PASS | 对抗 fixture 覆盖 padding/argmax 改变；真实 1,440 次测量均记录原始 t* |
| T05 恢复/生命周期 | PASS | N=30 完整 logits 30/30 恢复，异常/回调失败/非有限 logits 等由真实 hook 小模型测试覆盖 |
| T06 / T07 数学 | PASS | 手算 N=1/N>1、抵消反例、边界概率；全部真实原始概率独立复算一致 |
| T08 Top-K | PASS | 恰好 7 层、唯一合法、并列稳定；全层排名及 CSV/JSON/JSONL 互相一致 |
| N=1/N=30 独立结果复核 | PASS | 两次各 13 个产物字节哈希与记录源码一致；原始概率独立复算，不依赖被测 helper |
| T10 受限重复性 | PASS（受限） | 控制模型相同输入两次结果精确一致；独立真实 N=1 与 N=30 首样本整条轨迹一致，不代表全量重跑 |
| 图表/交互/颜色 | PASS | 两份图均在 Chrome 157.0.8103.78 显示 24 柱/24 行，无横向/文字越界；明暗、纹理、真实 Tab、hover，无页面异常/外部页面请求；已查看截图 |
| 完整 N=30 重跑 / N=100 / 不同抽样 seed Jaccard | NOT RUN | 未执行，不能声称全量稳定性已验证 |
| T09 / M5 LoRA 注入与训练 | NOT RUN | 不在本轮范围 |

44 个新增测试包括：核心 19、选层 5、文件/图结构 7、CLI/清单 9、独立验收 4。小模型概率表显式使消融后 argmax 改变，padding 位置偏好另一个 token；它执行真实 nn.Module hooks，但不是 Transformer 数值模拟。独立验收工具能拒绝同步重算文件哈希后的错误分数/token/恢复/权重/选层记录，不能把哈希当作来源认证。

## 运行日志和输出路径

- [真实 N=1](../results/m4-20261114-n1/) 与 [真实 N=30](../results/m4-20261114-n30/)；各自有 run.log、run_summary.json、配置/环境/Git/源码证据、实际样本和完整概率。
- [N=30 24 层 CSV](../results/m4-20261114-n30/layer_importance.csv)、[选层](../results/m4-20261114-n30/selected_layers.json)、[样本](../results/m4-20261114-n30/sample_ids.json)、[交互图](../results/m4-20261114-n30/importance.html)及 [SVG](../results/m4-20261114-n30/importance.svg)。
- [N=1 独立复核](../results/m4-20261114-audit/n1.json)、[N=30 独立复核](../results/m4-20261114-audit/n30.json)。
- [N=1 浏览器证据](../results/m4-20261114-visual-n1/visual_checks.json)、[N=30 浏览器证据](../results/m4-20261114-visual-n30/visual_checks.json)；light-full/dark-full/texture/keyboard-focus/hover PNG 已生成，已查看两份图的明/暗全图及代表性交互截图。
- [测试、红灯、palette、驱动、命令和运行日志](../results/m4-20261114-evidence/)，[最终测试日志](../results/m4-20261114-evidence/tests-final.log)；[机器可读索引](m4_validation.json)保存这些证据的 SHA-256。

浏览器只用新建独立 profile，不使用个人会话。本机无 chromium-cli/Playwright/Puppeteer，采用 Node 原生 WebSocket 驱动已安装 Chrome，未安装包；页面 HTTP/HTTPS 请求被阻止，校验未出现外部页面请求，不把此说成整个浏览器进程的网络审计。后续若反复做图表验收，可用 `/run-skill-generator` 固化项目运行 skill；本轮不改工具配置。

## 未解决问题

1. 无阻塞 M4 的未解决错误。原作者 prompt/数据版本/权重 revision 等 Q01–Q05 保留；Alpaca 是替代 smoke 数据，不是正式主表实验。数据经镜像下载并匹配固定哈希，未独立通过官方域名认证；遵守 CC-BY-NC-4.0。
2. 未重复整个 N=30，也未做 N=100 或不同抽样 seed 的 Jaccard。首样本重现不能支持整体 Top-K 稳定性结论，N=1 与 N=30 的选层差异也不是重复运行误差。
3. 未训练/评估下游效果，不能据当前重要性声称准确率提升或训练加速。
4. 图表只验证了本机 Chrome 的上述交互，不冒充跨浏览器或完整 WCAG 审计；运行源码尚未提交，实际身份以基线/dirty/源码指纹共同界定。

## 下一步建议（不自动执行）

停止于 M4。待用户明确指令后进入 M5：仅在 `[0, 1, 2, 3, 5, 8, 10]` 的 Q/K/V 添加 LoRA，先验证可训练参数范围，再考虑 1–3 batch smoke。正式数据、训练与评估协议和成本须按后续阶段另行确认；本轮不自动进行。Git 提交/推送仍需相应请求。
