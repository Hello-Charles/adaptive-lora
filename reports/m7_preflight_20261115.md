# M7 首项验收：正式协议核查与离线准入检查（2026-11-15）

## 当前任务与完成内容

用户明确指令进入M7。本轮**只完成M7第一项验收任务**：重新核查论文正式配置、固定候选模型/数据版本、定义文本prompt与评分协议，实现可离线运行的协议检查器，并取得用户对7项复现决定的确认。

- [SSH重启后验证](autodl_postrestart_20261115.md)已通过，实际RTX4090/20 vCPU配额/90 GiB内存/50 GiB数据盘盘点已归档；对应本地提交为 `243972133088de09d64703070e543b5e8f1c7518`。这不代表实验环境已就绪。
- 以Table 1的Qwen2.5-1.5B、28层、N250、Top8、rank16/alpha32为目标；第一轮只做8项commonsense的Full/Random8/Adaptive8，不运行数学、LoRA+或DoRA。
- 已实现标准库配置门禁、规范化Alpaca前缀、合法标签抽取、按ID任务评分和完整8任务等权宏平均；只用合成夹具测试，不声称已评估真实benchmark。
- 用户通过交互选择“确认此协议（推荐）”；仅将7个decision.status从pending改为confirmed，不增加其他操作许可。
- 实际运行检查器：草案返回2/PENDING_DECISIONS；确认后最终运行返回0/REVIEWED_NOT_EXECUTABLE，**execution_authorized、runtime_validated、training_started仍全部为false**。
- 全套离线回归 **185 tests，21.408s，OK，无跳过**；其中原162项加本轮23项。Linux/CUDA依赖仅形成候选清单，未安装。

没有公开推送、远程克隆/检出、环境创建、依赖安装、大型模型/训练评估数据下载、CUDA张量或GPU训练。M7整体仍未完成；本轮结束后不自动进入下一验收任务或M8。

## 文件变更

| 文件 | 内容 |
|---|---|
| [M7_PROTOCOL.md](../M7_PROTOCOL.md) | 论文原设定、已确认复现决定、来源/资源/运行门禁 |
| [配置JSON](../configs/qwen2_5_1_5b_commonsense_r16.json) | 1.5B/28层/N250/Top8与固定文件元数据，7项已确认状态 |
| [m7_protocol.py](../src/adaptive_lora/m7_protocol.py) | 纯标准库协议校验、prompt、解析、评分及宏平均 |
| [check_m7_protocol.py](../scripts/check_m7_protocol.py) | 只写新本地目录的离线审查入口，无执行开关 |
| [候选直接依赖](../requirements-linux-cu124-m7.in) | cp312/Linux x86_64的torch2.6.0+cu124官方wheel URL/哈希及直接版本；未安装，非完整锁 |
| [协议测试](../tests/test_m7_protocol.py)、[CLI测试](../tests/test_check_m7_protocol.py) | 23项测试，含无site-packages运行、非法配置及输出不覆盖 |
| [PAPER_SPEC.md](../PAPER_SPEC.md)、[REPRO_PLAN.md](../REPRO_PLAN.md)、[README.md](../README.md) | 仅更新本项边界、协议确认和后续门禁 |
| 本报告 | 记录实际命令、结果、批准边界与未完成事项 |

未修改M0–M6既有训练、校准、checkpoint、生成模块或历史实验报告。当前分支为 `repro/m7-run-preflight`；本轮运行前基线为 `243972133088de09d64703070e543b5e8f1c7518`，M7实际检查运行如实记录dirty状态及源码字节哈希。完成后创建本地提交，最终SHA由Git日志及回复给出，不把尚未生成的提交冒充运行时版本；没有push。

## 论文依据与工程假设

本轮从本地PDF重新提取并实际核查：

- Section 4.5/5.1、Table 1及Section 5.4：PDF第7–9页。
- Appendix C.1–C.3：第18–19页。
- 参考文献[14] LLM-Adapters：第11页，本次提取第10–12页引用区。

**[论文]** 正式模型为1.5B/28层，主表N250/Top8；Q/K/V rank16/alpha32，AdamW lr1e-4、weight_decay0.01、betas(0.9,0.999)、eps1e-8、无warmup、3epochs；每卡8×累计8，冻结FP16基础、FP32 adapter初始化、FP16 AMP；max512/pad到8倍数；commonsense各1,000评估，OBQA500，无in-context examples、greedy和任务答案抽取。

**[复现决策，用户已确认]** 限定8项commonsense、Random8扩展对照、规范化Alpaca纯文本而非旧Qwen chat模板、答案+EOS loss/右截断规则、局部seed42抽样、32-token上限、大小写/词边界/歧义处理、8任务等权宏平均、单卡batch64及镜像来源局限。完整定义见 [M7_PROTOCOL.md](../M7_PROTOCOL.md)。这些不是论文未公开细节已被还原的声明。

候选模型revision为 `8faed761d45a263340a0528343f099c05c9a4323`，数据仓库commit为 `816657208af4db747803f87ba40a4c71383fed7a`；模型权重LFS SHA256和全部所选数据Git blob SHA1已固定在配置。实际仅读取小型元数据、684-byte模型配置及两份约9/13KB的参考源码文本，**未运行第三方参考脚本，未下载权重或训练/评估文件**。

官方HF本地请求超时，服务器请求报Network is unreachable；使用镜像读取候选元数据/小配置并保留未独立认证官方来源的局限。GitHub树首查TLS超时，第二次有界HTTPS读取成功。引用库默认4 beams/temperature0.1、无边界首匹配regex和旧8bit/trust_remote_code设置，不能当作论文要求直接执行。

批准前草案规范配置SHA256：`e9ab92043b66674a78aa6b00461f022dfcf7f01290fd05cec98e0071d49d8625`。

确认后规范配置SHA256：`d60f13752b4b8628d0ed8c8d9bb843ce92c8221b68d17cd8338a8447efa81c82`。

模板规范哈希：`614a1d2376a04f35e99e3517d0a2a2065f3beced8a5b9b3b159446146bf38827`。配置哈希不覆盖源码/依赖字节，也不是来源签名；它们另行存档。

预算仅作配置推导：14,250次评分forward+250次恢复=14,500次；LoRA Full28约2,981,888元素、K8约851,968元素。**不是实际forward次数、参数审计、显存或速度结果**，不把3.5倍adapter差异扩大为总显存/速度差异。

## 执行命令

以下为实际主要命令，均未安装依赖：

```bash
# 相关页重新提取；另提取18–19页和10–12页。
pdftotext -enc UTF-8 -f 7 -l 9 -layout papers/adaptive_lora.pdf \
  .cache/m7-20261115-preflight-01/paper-experiments-utf8.txt

# 最终全套离线回归。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v

# 实际确认后检查，禁用site-packages，输出目录当时不存在。
.venv/Scripts/python.exe -I -S -B -X utf8 scripts/check_m7_protocol.py \
  --config configs/qwen2_5_1_5b_commonsense_r16.json \
  --output-dir results/m7-20261115-protocol-confirmed-final

.venv/Scripts/python.exe -I -X utf8 -m pip check
.venv/Scripts/python.exe -I -X utf8 -m compileall -q src scripts tests
.venv/Scripts/python.exe -I -S -B -X utf8 scripts/check_m7_protocol.py --help
git diff --check
```

另以curl/gh有界读取GitHub、镜像模型API、PyPI版本元数据和官方cu124 wheel索引；下载文件各自隔离且不执行其中代码。对API返回的两份base64源码先核对Git blob头+字节哈希后只读解码；读取这些文件的Python使用-I。候选依赖没有执行pip install或conda创建命令。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 项目 | 实际结果 |
|---|---|---|
| FAIL（已修正） | 最早协议测试文件语法 | 一个subTest语句缺冒号，日志保留；修正后才记录真实缺模块红灯 |
| FAIL（预期红灯） | 先测协议/CLI | 缺m7_protocol模块、缺check_m7_protocol入口，实际失败后实现 |
| PASS | 初轮聚焦测试 | 22项，0.653s；随后补1项配置确认不等于授权检查 |
| PASS | 最终全部测试 | 185 tests，21.408s，无跳过；不是GPU/真实任务准确率验收 |
| PASS | 实际待确认CLI | exit2，7项pending，未授权执行；不是训练失败 |
| PASS | 用户确认后的最终实际CLI | exit0，REVIEWED_NOT_EXECUTABLE，3个运行/授权标记均false |
| PASS | 本地依赖检查 | No broken requirements found；只证明当前Windows环境依赖声明一致 |
| PASS | 语法编译与CLI帮助 | compileall和--help退出0，无安装/远程/训练开关 |
| PASS | 来源元数据与小配置一致性 | 数据Git对象元数据、模型config Git blob、官方wheel索引约束已核对；不代替真实数据/权重验收 |
| PASS | 最终存档复核 | 不导入项目模块，重算批准前后差异、配置/模板/文件哈希、预算、文档链接与变更边界；索引见下 |
| NOT RUN | Linux候选依赖组合 | 未创建环境、未安装、未导入验证 |
| NOT RUN | 真实样本/分词/1.5B模块结构 | 文件和权重尚未下载，不能声称通过 |
| NOT RUN | CUDA/AMP/累积/pilot/正式训练与评估 | 未执行，未得到本轮相应授权 |
| NOT RUN | Git公开推送及远程指定SHA部署 | 未执行 |

## 运行日志和输出路径

- [M7_PROTOCOL.md](../M7_PROTOCOL.md)：已确认协议；[重启后盘点](autodl_postrestart_20261115.md)：SSH与硬件结果。
- [草案检查输出](../results/m7-20261115-protocol-draft/)：2个JSON，保留原pending配置与exit2审查结果。
- [确认后的最终输出](../results/m7-20261115-protocol-confirmed-final/)：config.json及preflight.json，无训练成功标记、checkpoint或真实指标文件。
- [.cache/m7-20261115-preflight-01/](../.cache/m7-20261115-preflight-01/)：论文UTF-8摘录、隔离来源元数据、只读解码参考源码、红灯/聚焦/最终回归、CLI、pip/compile/help日志。
- [full-regression-final.log](../.cache/m7-20261115-preflight-01/full-regression-final.log)：185项最终回归。
- [protocol-approval.json](../.cache/m7-20261115-preflight-01/protocol-approval.json)：仅协议的用户确认记录，不是可执行权限文件。
- [verification.json](../.cache/m7-20261115-preflight-01/verification.json)：独立于项目模块的只读存档复核、文件哈希与覆盖边界；[source-snapshot](../.cache/m7-20261115-preflight-01/source-snapshot/)保存本轮最终源码/配置/文档字节。

以上结果、下载元数据、源码快照和日志受Git忽略；公开提交仅包含本轮源代码、配置、候选清单、测试和不含原始数据/凭据的报告。未把连接地址、SSH配置、密钥、原始样本、模型或checkpoint加入Git。

## 未解决问题

1. 当前已有的是**已确认的协议记录与可测试的离线门禁**，不是CUDA训练入口。旧CPU smoke训练/校准/保存/评估边界没有解除，M7后续实现仍须新任务。
2. 真实数据许可/字节/行数、实际抽样IDs、BPE边界、padding/截断和直接输入隔离尚未验收；超长输入处置若遇到实际不兼容，必须报告后再获准修订，不能静默丢样本。
3. 基础权重尚未下载，28层实际路径/返回类型、FP16加载后的参数指纹及真实可训练LoRA统计未知；配置推导不能代替这些检查。
4. 官方HF连通性仍不可用；镜像元数据不构成独立官方来源认证。候选Linux组合未实测，当前远端torch2.5.1仍不满足本地Transformers5.22.2要求。
5. 未测单卡显存、吞吐或GPU费用；完整训练和7500条/方法的评估成本不能从本轮CPU协议测试推断。协议确认未授权公开推送、安装、下载或付费GPU。

## 下一步建议（不自动执行）

下一验收任务建议先按 [REMOTE_WORKFLOW.md](../REMOTE_WORKFLOW.md) 申请公开推送/指定SHA部署及独立Linux/CUDA环境安装范围，保留平台原环境；随后另行批准固定模型/数据下载，完成真实数据与结构门禁、CUDA实现测试和有界付费pilot。

只有pilot通过、记录实际显存/吞吐并获完整成本批准后，才执行正式N250校准、3方法统一3epochs和8任务评估。本轮仅完成M7首项，停止于本地记录/测试/提交，不自动推进。
