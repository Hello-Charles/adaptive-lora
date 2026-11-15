# M7 固定数据门禁续轮：传输预算耗尽，审计未启动

## 当前任务与完成内容

面向复现实验实现者与执行者；日期2026-11-15。**最终状态：BLOCKED_TRANSFER_BUDGET_EXHAUSTED。M7整体未完成，真实数据审计仍未运行。**

用户本轮先指令“继续”，随后两次明确批准：一次新的最多900秒/120,000,000 bytes固定数据窗口，以及在SSH主机别名 `autodl`（同一台AutoDL服务器）执行该窗口。范围只含原版本九个JSON与离线文本审计，复用已核验许可、跳过README，不改代码、不安装依赖、不下载模型、不运行CUDA/训练、不push。

已完成：

1. 复用已部署完整SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb` 的审计代码，不重新clone、不改源码。
2. 只读复核两份既有固定通知（DATA_LICENSE、LICENSE）字节，不重新下载许可、不请求README。
3. 在**新目录** `/root/autodl-tmp/adaptive-lora-m7-data-20261115-02` 启动新900秒窗口；旧操作目录与结果不变。
4. 传输在预算内未完成：**890.0009秒内仅写入1个部分文件13,631,488 bytes**，随后SIGALRM终止；**完成文件0个、审计目录未创建**。
5. 只读核验失败证据、旧结果字节、部署与环境SHA及包元数据；本地归档核验通过（仅存档一致性）。

### 实际时序与结果

| 项目 | 实际值 |
|---|---|
| 远端启动 | 2026-11-15T13:51:05.392667 UTC |
| 硬截止（启动+900秒） | 2026-11-15T14:06:05.392667 UTC |
| 结束 | 2026-11-15T14:05:55.393536 UTC |
| 实际用时 | 890.000919秒 |
| 状态 / 阶段 | FAIL / `fixed_dataset_download` |
| 错误 | `TimeoutError: data gate timeout or session termination: 14`（SIGALRM） |
| 清理复查 | `postcheck_error: data gate budget exhausted` |
| 已读资源字节 | 13,631,488 / 120,000,000（11.36%） |
| 完成文件 | 0 |
| 部分文件 | `ft-training_set/commonsense_170k.json`，13,631,488 / 96,529,722 bytes（14.12%） |
| 实测平均吞吐 | 15,316.26 B/s（约14.96 KiB/s） |
| 审计 | **未启动**；audit目录、summary、error.json均不存在 |

部分文件SHA256为 `cbc102d9f126da9ea4692fa78e9ca6638267e2fada15053b0b45fdd9c960e345`，是**不完整下载**的字节摘要，不能与固定Git blob或完整文件SHA256比较，也不构成原始数据。该文件按要求保留未删除，未覆盖任何既有数据。

按上表实测速率，完整九个JSON（固定配置文件合计112,764,504 bytes）约需**122.7分钟**，单个训练文件约需105.0分钟。因此**900秒窗口在当前网络条件下结构性不足**，不是抽样或数据内容问题。该外推只由观测字节/秒得到，未做新的网络测量，不保证后续速率相同。

### 与上一轮的区别

上一轮（13:14:24 UTC）在固定通知阶段即因README读超时停止，未发出任何JSON请求。本轮跳过README并复用许可，实际进入了JSON传输阶段，但**只取得一个不完整训练文件**，仍未完成任何文件校验，也从未进入字段/抽样/文本交集审计。两次都是网络/预算阻塞，**都不是发现了数据hash不符、标签错误或训练-评估文本重叠**。

### 工具拒绝与替代做法

本地组装了只读base64证据收集器，但其单次调用被自动模式分类器拒绝；**该命令未执行，未修改权限设置，也未改用其他批量传输方式绕过**。随后改为若干条只读SSH命令获取目录清单、字节长度、SHA256、Git状态和发行元数据；这些命令不写入、不下载数据、不导出原始JSON正文。本地证据因此来自定向只读查询而非单一证据包，报告已如实标注。

## 文件变更

生产代码、`scripts/`、`src/`、`tests/` 本轮**无改动**。远端仍执行已发布完整SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb`。

本地文档：本报告、[机器索引](m7_data_gate_resume_20261115.json)、[README](../README.md)、[计划](../REPRO_PLAN.md)、[协议](../M7_PROTOCOL.md)、[规格](../PAPER_SPEC.md)。控制/核验脚本、观测记录和原始日志只留Git忽略目录 `.cache/m7-20261115-assets-03/`，不随报告发布；报告提交不自动push，也不替换远端批准SHA。

## 论文依据与工程假设

- **[论文] Section 5.1 / Table 1（PDF第7–8页）：** commonsense训练10,000条；每任务评估1,000条、OBQA 500条；1.5B/28层/N250/Top8。**本轮未取得任何真实数据，不报告行数或准确率。**
- **[论文] Appendix C.2–C.3（第18–19页）：** seed42、max512、pad到8倍数、greedy、按任务答案抽取。论文未公开本项目的固定数据commit、逐样本索引或完整模板。
- **[复现决策，已确认且本轮未改]** 固定数据commit `816657208af4db747803f87ba40a4c71383fed7a`；train/cal/eval各用局部 `random.Random(42)`，先抽10,000再从该子集抽250；评估各抽1,000/500；种子与协议均未因传输失败而调整。
- **[工程边界]** 预算、进程组清理和超时路径按既有实现执行；传输失败后不自动重试、不改来源/版本/seed、不缩小样本、不改用其他镜像或协议替代固定GitHub路径。
- 数据许可仍为已核验的 ODC-By（`fc5b567d…`），**不覆盖每条Contents的独立权利**，未逐条清理原benchmark内容权利，不公开原始数据。代码LICENSE分开核验。

## 执行命令

```bash
# 本地控制脚本与独立核验器测试（合成夹具，不联网）。
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-03/test_data_driver.py
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-03/test_preconditions.py
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-03/test_verification.py

# 项目离线回归（未改动项目代码，用于确认基线仍通过）。
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -v

# 获批新窗口：严格SSH、平台解释器、stdin传输控制脚本，输出独占。
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  autodl '/root/miniconda3/bin/python -I -B -' < .cache/m7-20261115-assets-03/run_data_gate.py

# 失败后只读核验（目录清单/字节摘要/Git/包元数据）；不写远端、不导出数据正文。
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-03/verify_closeout.py
```

远端数据获取与审计的CLI命令（`scripts/audit_m7_dataset.py`）**本轮从未被执行**，不能写成运行证据。控制脚本SHA256为 `ab9a074f730f01f4465bc9e131ec7865a1659bc7d48ca3315887f6abc4d97c21`（生成时记录），执行字节与本机保留文件一致。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据与边界 |
|---|---|---|
| 下载控制辅助函数 | PASS | 7项，0.017s：固定URL、大小/blob/全局预算、不覆盖、不跟随重定向 |
| 新窗口前置门禁 | PASS | 5项，0.033s：许可只读复用、篡改拒绝、旧结果/新目录不覆盖 |
| 独立清单复算器 | PASS | 5项，0.198s：抽样重建、校准映射、未声明交集拒绝 |
| 本地离线回归 | PASS | 207项，14.267s，无跳过；项目代码本轮未改 |
| 归档一致性 | PASS_CLOSEOUT_ARCHIVE_ONLY | 失败阶段/预算/部分文件/环境不变/旧结果字节复核 |
| 数据窗口执行 | **FAIL** | 890.0009s后SIGALRM；0个完成文件；1个部分文件 |
| 九个JSON完整性 | **NOT RUN** | 没有任何文件通过长度+Git blob核验 |
| 真实行数/字段/抽样/文本交集 | **NOT RUN** | 审计未启动，无audit目录 |
| 真实tokenizer/截断/token隔离 | **NOT RUN** | 无对应资源或运行 |
| 1.5B结构、CUDA/AMP/累积、GPU checkpoint、pilot、训练、评估 | **NOT RUN** | 未进入 |

既有 Linux 183项适用CPU通过、2项真实tokenizer用例 NOT RUN 状态保持，不升级为全通过。

## 运行日志和输出路径

私有本地根：[.cache/m7-20261115-assets-03/](../.cache/m7-20261115-assets-03/)（Git忽略）

- `authorization.json`：新数据窗口授权与脚本摘要。
- `host-authorization.json`：`autodl` 主机明确批准，以及先前拒绝调用的未执行记录。
- `remote-result.json`：本次真实结果（FAIL、`fixed_dataset_download`、13,631,488 bytes）。
- `operation-ssh.log`：空文件——远端stdout已写入结果文件，stderr无异常文本。
- `observations.json`：只读清单、部分文件SHA256、两处SHA/clean、69项venv、剩余空间34,254,114,816 bytes。
- `environment-check.json`：平台153项元数据未变、venv 69项未变（仅发行元数据）。
- `verification.json`：归档索引；`verification-superseded-01/02.json` 保留早期版本，未删除。
- 测试日志：`driver-tests.log`、`precondition-tests.log`、`verification-tests.log`、`local-regression.log`、`closeout-verification-03.log`。

远端：新目录仅含 `result.json`、`data/ft-training_set/commonsense_170k.json`（部分）和空目录 `data/`；无audit产物。旧操作目录 `…-20261115-01` 的结果字节未变（`984edcb31c923237f991b1c224159e8d064799f461d8d1091c2397d449631835`，与既有归档一致）。部署目录 `2df16eb…`、环境目录 `9051fdf…` 均clean。

## 未解决问题

1. **窗口容量与真实带宽不匹配**：按实测约15.3 KiB/s，完整九文件约需123分钟，远超单次15分钟窗口。若要完成获取，需要更长或分段的明确授权，或先解决网络吞吐——不能在现有批准内自动延长。
2. 九个JSON仍无一个通过核验；真实行数、字段schema、固定抽样与文本交集均未知，**未发现也未排除重叠**。
3. 部分训练文件留在远端，未删除、未续传、未当有效数据使用。
4. 模型许可与官方HF端点阻塞独立未解，不受本轮影响。
5. 内容权利、真实tokenizer、1.5B结构与全部GPU/训练门禁仍未验收。

## 下一步建议（不自动执行）

建议下一轮先做**受控的吞吐诊断或分段获取批准**，而不是再次启动同样900秒窗口：例如批准按文件分段、每段独立预算与哈希核验，或是先测量固定主机多次连接的实际吞吐再决定预算规模。任何方案都应保持固定commit、固定seed与既有抽样规则，先取得并核验全部九个JSON，再进入审计；失败继续保留部分文件与记录，不重抽、不换源。

通过文本门禁后仍不自动进入真实tokenizer、模型结构、CUDA或正式训练。本轮未重试、未延长窗口、未删除远端证据，也未push任何提交。
