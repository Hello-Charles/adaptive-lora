# M7 环境续接：离线安装与适用 Linux CPU 验收通过（2026-11-15）

## 当前任务与完成内容

**本轮结论：PASS_APPLICABLE_CPU，不是 M7 完成或 CUDA 验收通过。** 已完成全部67个缓存wheel的官方哈希核验、当前独立venv离线安装、pip check、固定版本导入与适用CPU测试。Linux实跑 **183项通过，另外2项真实tokenizer用例NOT RUN**；没有把它们写成185项全部通过。机器索引见 [环境CPU验收索引](m7_environment_cpu_20261115.json)。

此前 [恢复报告](m7_environment_resume_20261115.md) 中的官方源超时、vendor元数据误判和工具拒绝均保留为历史事实。本轮用户另行选择“批准有限缓存续接”，明确允许准备/测试工具及一次新的最多1800秒操作：复用旧缓存，仅补官方小JSON，全批核验后离线安装并执行CPU验收。不是沿用或重置原09:28:32.230328 UTC截止；没有修改工具权限设置或绕过拒绝。

实际过程：

1. **12:01 UTC只读预检通过**：严格SSH可达，批准SHA和clean状态一致；全部67个缓存wheel的大小/哈希与此前保存的证据完全一致。venv仍只有pip24.0，数据盘可用43,558,068,224 bytes。
2. 新自编工具在本地通过14项续接门禁测试，已有11项wheel核验回归通过；允许唯一顶层METADATA与vendor嵌套并存，不降低整wheel官方哈希门禁。
3. 唯一一次新操作始于 **12:07:36.736708 UTC**，结束于 **12:09:35.502418 UTC**，实耗 **118.765724s**，SSH exit0。1800秒总上限中预留10秒清理，active预算1790秒。
4. 全部67个wheel、3,179,040,252 bytes通过核验：Torch使用固定官方SHA256，其余66项匹配官方精确版本JSON的name/version/filename/size/SHA256。复核旧52份JSON，仅补取14份官方小JSON；**新wheel下载数为0**。
5. 生成新manifest及67项require-hashes锁，在原独立venv中完全离线安装全部依赖和批准SHA的项目。pip check输出 `No broken requirements found.`，保存真实Linux freeze。
6. 11项固定版本导入、Qwen2/PEFT类导入及极小CPU张量检查通过；Linux单测183项8.851s通过，无失败/错误/额外skip。两个依赖固定真实0.5B tokenizer缓存的用例明确NOT RUN，没有为补齐计数下载资源。
7. **12:10 UTC最终只读复查通过**：Git HEAD仍为批准完整SHA、clean、tracked源码字节和平台包元数据未变；新venv共69个发行包（67项wheel + 原pip + 本项目）。79份传回证据通过大小/哈希/字节核对，另独立核对官方记录、锁、安装报告及测试计数。

批准且实际执行的远端代码为：

`9051fdfe86379aa3502e66bf1a586d4eb00565ba`

本地报告基线为 `e05b7082f202ddfe0580f82846e9ba71c5835d1a`。新报告提交不替换远端代码，不自动push。

## 文件变更

| 文件 | 变更 |
|---|---|
| [README.md](../README.md) | 更新为安装、导入与适用CPU通过，同时明确2项NOT RUN及CUDA未验收 |
| [REPRO_PLAN.md](../REPRO_PLAN.md) | 勾选安装/适用CPU子项，单列真实tokenizer待验收项，M7父项和GPU/pilot仍未完成 |
| [PAPER_SPEC.md](../PAPER_SPEC.md) | 仅更新候选环境工程状态，不改算法或已确认正式实验协议 |
| [前次恢复报告](m7_environment_resume_20261115.md) | 顶部添加本轮后续指引，保留此前失败正文与机器索引 |
| 本报告与[机器索引](m7_environment_cpu_20261115.json) | 保存真实范围、命令、状态、计数与关键产物哈希 |
| Git忽略的本地操作脚本/测试/证据 | 新续接工具、CPU runner、只读收集、解码和离线核验；不加入公开Git |

未修改src、scripts、tests、原候选依赖文件或正式配置。远端仓库仅安装产生可忽略的构建元数据，tracked源码字节与Git clean检查均通过；没有覆盖旧失败产物。

## 论文依据与工程假设

- 本轮只做软件环境准备，不实现核心算法、数据预处理或正式评估；已重读PAPER_SPEC/REPRO_PLAN，**本轮未重读PDF**，没有增加论文事实。沿用此前核查的Section4.5/5.1、Table1与AppendixC.1–C.3，正式协议不变。
- **[复现决策]** Linux Python3.12.3上安装11项原固定版本：Torch2.6.0+cu124、Transformers5.22.2、PEFT0.21.2、Accelerate1.15.0、Safetensors0.8.0、NumPy2.5.4、Tokenizers0.23.3、Hub2.4.0、Datasets5.2.0、PyArrow26.0.1、Pytest9.2.0。实际逐包导入及版本一致，不等于正式训练兼容性已全面验收。
- **[工程边界]** 原候选.in的版本和字节保持不变，其“未安装”注释描述初次编写时状态；当前状态以本报告、真实安装报告和freeze为准。.in本身仍不是完整传递依赖锁。
- **[工程边界]** 安装全部67项显式锁定wheel时使用 `--no-index --no-deps --require-hashes --only-binary=:all:`：不解析依赖中的直接URL以防隐式联网，随后以pip check核对闭包与版本约束。本项目另以 `--no-index --no-deps --no-build-isolation -e` 安装，只执行批准仓库的构建入口。
- **[工程边界]** 下载内容验证前仅以-I下标准库读取ZIP身份/哈希，不解压执行；先全批匹配官方记录，再在独立venv安装。哈希匹配不是漏洞审计。包元数据集合不变不是全平台文件审计；venv也不是OS级容器隔离。
- **[待核验]** `torch.version.cuda == "12.4"`只是构建元数据。隐藏GPU、只创建CPU张量，不据此宣称驱动/CUDA/AMP/梯度累积/GPU checkpoint已通过。单测中的随机微型Qwen2和合成训练夹具不等于真实1.5B模块或正式训练。

## 执行命令

所有SSH使用既有别名、BatchMode、StrictHostKeyChecking，禁用密码、键盘交互和agent forwarding；不公开实际入口或密钥。新工具与下载目录分离，由平台解释器-I -B经stdin执行：

```bash
# 只读预检：复用已审阅收集器，新输出，不改远端。
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  '/root/miniconda3/bin/python -I -B -' \
  < .cache/m7-20261115-environment-02/collect_environment_state_readonly.py

# 本地门禁回归及项目离线CPU测试，stdout/stderr均保存到新私有路径。
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-03/test_continuation.py
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-02/test_install_verified_mirror.py
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-03/test_decode_final.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -v

# 同一严格SSH前缀，另加ServerAliveInterval=30/ServerAliveCountMax=3。
# 仅一次执行continue_cached_environment.py，结果另存；没有pip download命令。
# 全部wheel验证后，新venv的实际安装参数：
<隔离python> -I -B -m pip --isolated install --no-index --no-deps \
  --find-links <新verified目录> --require-hashes --only-binary=:all: \
  --no-input --report <新安装报告> -r <新哈希锁>
<隔离python> -I -B -m pip --isolated install --no-index --no-deps \
  --no-build-isolation --no-input --report <新项目报告> -e <批准repo>
<隔离python> -I -B -m pip --isolated check
<隔离python> -I -B -m pip --isolated freeze --all

# CPU探针与unittest runner由上述同一有界进程传给新venv的-I -B stdin。
# 最后仅只读运行collect_final_readonly.py，经大小/哈希校验后本地独占解码和复核。
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-03/decode_final.py
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-03/verify_evidence.py
```

新操作脚本实际文件字节SHA256为 `7e7ceaea1f9328b66708d870618aafab6995efd695c5f9fbd8d0f9bdd9b0b1f6`；与旧核验失败脚本不同，不能混写执行来源。没有修改权限设置、换索引/版本、后台无界重试、模型/数据下载或公开push。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查 | 结果与边界 |
|---|---|---|
| PASS | 新续接本地门禁 | 14项，0.113s；覆盖缓存JSON/完整清单/哈希失败前不安装/离线参数/超时组清理/不覆盖/两项NOT RUN分流，mock不代替Linux执行 |
| PASS | 原wheel核验回归 | 11项，0.102s，包含vendor嵌套与唯一顶层METADATA |
| PASS | 证据解码器 | 4项，白名单、传输哈希与不覆盖 |
| PASS | 本地Windows离线项目回归 | 185项，21.435s，无skip；Torch2.15.0+cpu，不冒充Linux环境 |
| PASS | 全67wheel官方核验及离线安装 | 66份PyPI精确JSON+固定官方Torch哈希；新下载wheel为0，完整锁/manifest已产生 |
| PASS | pip check / freeze | 无破损依赖；保存实际freeze，不是占位文件 |
| PASS | Linux导入/CPU探针 | 11项固定版本、Qwen2/PEFT类、极小CPU张量和4线程 |
| PASS（适用集合） | Linux CPU单测 | **183项，8.851s，183通过，0失败/错误/skip**；不是185全通过 |
| NOT RUN | 真实tokenizer两项 | 固定0.5B缓存缺失，按精确ID排除并在结果中记录，未偷偷下载 |
| PASS | 最终只读状态与归档 | 79份文件大小/哈希/保存字节一致；独立复核67wheel、官方JSON、安装报告、计数、旧失败哈希与源码/平台元数据 |
| NOT RUN | CUDA、AMP/累积、GPU checkpoint/pilot | 不在本轮授权，未验收 |
| NOT RUN | 正式模型/数据与训练/评估 | 未下载1.5B权重/正式数据，未执行正式训练/评估 |

两个NOT RUN的完整ID：

- `test_data.DataPreparationTests.test_real_tokenizer_template_special_tokens_truncation_and_padding`
- `test_trainer.SupervisedInputTests.test_real_qwen_tokenizer_preserves_prefix_and_explicit_eos_protocol`

## 运行日志和输出路径

远端仍使用原独立工作区；所有新产物在 `/root/autodl-tmp/adaptive-lora-m7-20261115-02/cache-continuation-01/`。旧mirror-attempt-01和artifacts内失败记录不覆盖。最终数据盘可用34,269,696,000 bytes，仅12:10 UTC快照。

本地均位于Git忽略的 [.cache/m7-20261115-environment-03/](../.cache/m7-20261115-environment-03/)：

- [新有限授权](../.cache/m7-20261115-environment-03/authorization.json)、[预检](../.cache/m7-20261115-environment-03/precheck.json)、[缓存/来源基线](../.cache/m7-20261115-environment-03/baseline.json)。
- [新操作脚本](../.cache/m7-20261115-environment-03/continue_cached_environment.py)、[远端实际结果](../.cache/m7-20261115-environment-03/remote-result.json)、[最终只读响应](../.cache/m7-20261115-environment-03/final-state.json)。
- [原样解码归档](../.cache/m7-20261115-environment-03/retrieved/)：完整manifest、哈希锁、66份官方JSON、两个pip安装报告、pip check、真实freeze、CPU导入及测试结果/日志；[传输索引](../.cache/m7-20261115-environment-03/retrieved/index.json)。
- [独立证据核验](../.cache/m7-20261115-environment-03/verification.json)、[本地185项日志](../.cache/m7-20261115-environment-03/local-regression.log)、[14项工具日志](../.cache/m7-20261115-environment-03/continuation-tests.log)、[解码器日志](../.cache/m7-20261115-environment-03/decoder-tests.log)。测试日志内的合成FAIL记录是预期门禁夹具，不是实际安装失败。

关键产物SHA256（完整表见机器索引）：

- manifest：`34c605e3d4aa3cc979924e16244ead041449130fad01f02e5bdb901c3719174b`
- require-hashes锁：`a31511bd16f3e083ea08018ed161d7536adc7361a91f706ac5c1716566a42a00`
- Linux freeze：`fc8230a63a2384b623938e346407e636f58e4ade73d604384a568f8ad2335b04`

SSH原始日志、基础设施细节、缓存和freeze不加入公开Git；本地报告完成测试/文档核查后提交，提交SHA由Git与最终回复给出，不自动push。

## 未解决问题

1. 真实0.5B tokenizer两项仍NOT RUN。此次无需为完成环境安装重新下载整批wheel；后续补验应明确缓存来源及授权，不改测试来伪造通过。
2. 当前只通过Linux适用CPU范围。真实Qwen1.5B的28层模块/返回值、实际分词、正式数据许可/字节/清单与拆分隔离仍待验收；Qwen2类能导入不替代真实模块结构探查。
3. CUDA/驱动实际计算、FP16 AMP、梯度累积、GPU checkpoint与显存/吞吐pilot均未运行。M7父项未勾选，不能因装有cu124 wheel就跳到正式训练。
4. 实际锁与freeze包含本机平台信息，不能当作跨平台通用锁；官方哈希和pip check也不保证没有漏洞或所有未来训练路径兼容。

## 下一步建议（不自动执行）

本轮在CPU环境验收与本地报告提交后停止。下一项建议先取得固定1.5B模型/所需tokenizer及正式数据下载的明确授权，按已确认revision/哈希保存到新隔离目录，验收许可、样本身份、真实分词和28层模块结构；任何GPU计算与有界pilot另外确认成本。不得自动push、下载大资源、调用CUDA、运行pilot/正式训练或进入M8。
