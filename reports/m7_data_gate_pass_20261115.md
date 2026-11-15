# M7 固定数据门禁通过：九文件核验、抽样与文本隔离审计（2026-11-15）

## 当前任务与完成内容

面向复现实验实现者与执行者。**文本层数据门禁实际通过，状态 `TEXT_GATE_PASS_TOKENIZER_PENDING`。M7 整体仍未完成**——tokenizer、1.5B 结构、CUDA 与正式实验均未验收。

用户在本轮先后批准：一次新的 9000 秒 / 120,000,000 bytes `autodl` 数据窗口，使用已本地测试的跨窗口续传工具，从既有 13,631,488 字节部分文件续传。实际执行成功：

1. **九个固定文件全部通过三重核验**：预期字节长度、Git blob SHA1（`SHA1("blob " + 长度 + NUL + 内容)`）、普通 SHA256。
2. **续传前缀被远端内容证实**：接管上一窗口 13,631,488 字节部分文件，先核对已记录长度与摘要，再取远端同一偏移的尾部 4,096 字节与本地比对一致（offset 13,627,392），随后只追加剩余字节。最终整文件 Git blob 匹配，证明续传拼接结果正确。
3. **离线审计实际运行**：训练 10,000（总体 170,420 行）、校准 250、八任务各 1,000（OBQA 500）。
4. **跨拆分直接文本交集全部为 0**：训练/校准对每个评估任务的 `sample_id` 与规范 prompt SHA256 交集均为 0。

### 实际执行数据

| 项目 | 实际值 |
|---|---|
| 窗口开始 / 结束（UTC） | 15:03:11.654515 / 15:25:12.445723 |
| 实际耗时 | 1,320.791 秒（约 22.0 分钟），批准上限 9,000 秒 |
| 本次传输字节 | 99,399,256，上限 120,000,000 |
| 实测探测速率 | 57,930.8 B/s（Range 206 支持） |
| 本窗口平均速率 | 约 75,257 B/s（含文件更大，条件比上一窗口好） |
| 完成文件 | 9 / 9，`partial_files` 为空 |
| 审计退出与状态一致 | 是（0 ↔ `TEXT_GATE_PASS_TOKENIZER_PENDING`） |
| 环境与旧结果 | 前后源码字节、平台元数据、69 项 venv、旧结果字节均未变 |

### 固定来源与实际行数

| 文件 | 字节 | 实际行数 | 字段集合 |
|---|---:|---:|---|
| ft-training_set/commonsense_170k.json | 96,529,722 | **170,420** | answer,input,instruction,output |
| dataset/boolq/test.json | 899,151 | 3,270 | 同上 |
| dataset/piqa/test.json | 913,157 | 1,838 | 同上 |
| dataset/social_i_qa/test.json | 843,748 | 1,954 | 同上 |
| dataset/hellaswag/test.json | 11,099,046 | 10,042 | 同上 |
| dataset/winogrande/test.json | 497,409 | 1,267 | 同上 |
| dataset/ARC-Easy/test.json | 1,157,216 | 2,376 | 同上 |
| dataset/ARC-Challenge/test.json | 622,510 | 1,172 | 同上 |
| dataset/openbookqa/test.json | 202,545 | **500** | 同上 |

九个文件字段完全一致，均为 `answer,input,instruction,output` 四字段，无额外或缺失字段。文件名中的 "170k" 与实际 170,420 行相符。

### 组内重复 prompt（如实报告，未过滤未重抽）

| 组 | 重复行数 |
|---|---:|
| train（10,000） | 107 |
| calibration（250） | 0 |
| social_i_qa（1,000） | 8 |
| 其余六任务 | 0 |

这些是**组内**重复，不是跨拆分泄漏。按既定协议只计数报告，不删除、不去重、不缩小样本，也未因此改变 seed 或协议。

## 文件变更

生产代码、`scripts/`、`src/`、`tests/` 本轮**无改动**。远端执行代码仍为已批准完整 SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb`，审计记录的 `git` 为 `{commit: 2df16eb…, dirty: false}`。

本地：本报告、[机器索引](m7_data_gate_pass_20261115.json)，以及 README/REPRO_PLAN/M7_PROTOCOL/PAPER_SPEC 状态更新。控制与核验脚本、窗口结果、原始日志留在 Git 忽略的 `.cache/m7-20261115-assets-04/`。报告提交不自动 push。

## 论文依据与工程假设

- **[论文] Section 5.1 / Table 1（PDF 第 7–8 页）：** commonsense 训练 10,000、每任务评估 1,000、OBQA 500。本次抽样的**数量**与论文一致；具体逐样本索引仍是已确认的复现决策。
- **[论文] Appendix C.2：** 训练 seed 42。本项目 train/cal/eval 分别用局部 `random.Random(42)`，并记录 Python 版本，均为此前已确认的复现决策。
- **[复现决策，未改]** 固定 commit `816657208af4db747803f87ba40a4c71383fed7a`；先抽 10,000 训练，再从该子集抽 250 校准；评估各抽 1,000/500。
- **[范围局限，必须保留]**
  - 名称含 `test.json` **不证明**与原始 benchmark 官方 test split 完全相同。
  - 哈希交集只覆盖**直接**身份（来源位置 ID 与规范 prompt 字节）；**未检查**语义改写、题面内嵌答案或预训练污染。
  - `tokenizer_validated=false`：尚未做真实 BPE、512 截断、padding 与 token 级身份交集。
  - `content_rights_individually_cleared=false`：DATA_LICENSE 为 ODC-By，仅覆盖数据库层，未逐条清理 benchmark 内容权利；本项目不公开原始数据。
  - `training_started=false`。

### 两个配置哈希的区别（易混淆，明确区分）

| 字段 | 值 | 含义 |
|---|---|---|
| 配置文件**字节** SHA256 | `6f715cef71d1fbe350541bd6b45524a4134ea2d879b1bee266468c4afb30081c` | 落盘原始字节；控制脚本用它锁定配置漂移 |
| `protocol_sha256`（规范 JSON） | `d60f13752b4b8628d0ed8c8d9bb843ce92c8221b68d17cd8338a8447efa81c82` | 排序键+紧凑分隔符后的规范 JSON 哈希，含决策状态 |

审计报告中的 `config_sha256` 是后者。两者不同是**预期**的序列化差异，不是配置被修改。本地已用 `load_m7_config` + `protocol_sha256` 复算，与远端报告值完全一致。

## 执行命令

```bash
# 窗口执行（严格 SSH、平台解释器 stdin、输出独占）
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  autodl '/root/miniconda3/bin/python -I -B -' \
  < .cache/m7-20261115-assets-04/run_segmented_gate_v2.py \
  > .cache/m7-20261115-assets-04/remote-window-01.json \
  2> .cache/m7-20261115-assets-04/window-01-ssh.log

# 本地独立核验（不导入项目审计器、不联网）
.venv/Scripts/python.exe -I -S -B -X utf8 .cache/m7-20261115-assets-04/verify_window.py
```

远端审计由既有部署调用：`<venv>/bin/python -I -S -B <DEPLOY>/repo/scripts/audit_m7_dataset.py --input-root <TARGET>/data --output-dir <TARGET>/audit`。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| 九文件长度 + Git blob + SHA256 | **PASS** | 9/9，逐个记录于 `completed_this_window` |
| 续传前缀远端对齐 | **PASS** | 4,096 bytes @ offset 13,627,392 一致；整文件 blob 匹配 |
| 离线文本隔离审计 | **PASS** | `TEXT_GATE_PASS_TOKENIZER_PENDING`；全部跨拆分交集为 0 |
| 抽样数量与已确认配置一致 | **PASS** | 10,000 / 250 / 1,000×7 + 500 |
| 环境与旧结果未变 | **PASS** | 源码字节、平台元数据、69 项 venv、旧结果 SHA256 |
| 控制脚本合成测试 | PASS | v2 16 项；窗口常量、Range/偏移/预算、前缀对齐、状态、审计复用 |
| 项目离线回归 | PASS | 207 项，无跳过 |
| 真实 tokenizer / 截断 / token 级交集 | **NOT RUN** | 无对应实现与运行 |
| 1.5B 真实模块结构 | **NOT RUN** | 权重未下载，许可门禁未解 |
| CUDA/AMP/累积/GPU checkpoint/pilot | **NOT RUN** | 未进入 |
| 正式训练与评估 | **NOT RUN** | 未进入 |

既有 Linux 183 项适用 CPU 通过、2 项真实 tokenizer 用例 NOT RUN 的状态保持。

## 运行日志和输出路径

私有本地根：[.cache/m7-20261115-assets-04/](../.cache/m7-20261115-assets-04/)

- `remote-window-01.json`：窗口结果与完整 `audit_summary`（30,889 bytes）
- `window-01-ssh.log`：空（无 stderr 异常）
- `window-verification.json` / `window-verification.log`：独立核验 `PASS_WINDOW_ARCHIVE_ONLY`
- `run_segmented_gate_v2.py`：SHA256 `dfa465617599a78ce5080e57998eff2817011772d8fd05be641d0b7f9dbafe47`
- `segmented-v2-tests-02.log`、`local-regression.log`、`window-denied.json`（保留先前拒绝记录）

远端 `/root/autodl-tmp/adaptive-lora-m7-data-20261115-04/`：

- `data/`：九个已核验 JSON（**原始数据不传回本地、不进公开仓库**）
- `audit/audit.json`：7,784,776 bytes，SHA256 `840fba0ca65d436587c6941b89e066689682c73bbcff8be6cd179c0eb69202f9`
- `audit/summary.json`：6,556 bytes，SHA256 `02e7b6e3a0a7a4ab57c07abf97c65098570fd20e64b20f50082e9f1b5efc863c`
- `state.json`：九文件全部 `verified`，`partial` 为空

完整抽样索引与逐样本身份（`sampling_sha256 = 76a61e01d6e0b2400b4870ed54b6d0a5afa12347389f14d19d58b05164288c3b`）保存在远端 `audit/audit.json`，本地未复制原始身份清单。

## 未解决问题

1. **真实 tokenizer 路径完全未实现**：512 右截断、pad 到 8、`add_special_tokens=False`、prefix/answer BPE 边界、答案+EOS mask、truncation 后无监督 token 报错、token 级身份交集。
2. **1.5B 权重与许可仍未解决**：官方 HF 端点不可达、镜像文件路径 403；权重未下载，LFS SHA256 未核对。
3. **CUDA/AMP 实现与 GPU checkpoint 未实现**；现有 CLI 有 CPU/1–3 步硬门禁。
4. **组内重复 prompt**（train 107、social_i_qa 8）已记录未处理；是否需要处理属协议决定，不由实现者擅自更改。

## 下一步建议（不自动执行）

数据门禁已通过，下一阶段最合理的是**真实 tokenizer 管线**：它不需要 GPU，也不依赖权重许可（可先用已缓存 0.5B tokenizer 验证逻辑，再用 1.5B tokenizer 复核），且是校准与训练的共同前置。权重下载需要单独的下载授权，并按实测速率预估时间。

本轮未下载模型、未运行 CUDA/训练、未 push 任何提交。
