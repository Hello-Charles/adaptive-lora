# M7 真实 10,000 条数据装配核验（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。状态 **`REAL_BATCH_ASSEMBLY_PASS_CPU_ONLY`**。

用户明确批准在 SSH 别名 `autodl` 上运行一次最多 600 秒的纯 CPU 核验，并新建指定输出目录、写入统计报告与私有身份清单；其他费用与数据边界不变。两次较早的 SSH 写入调用被权限门禁拒绝，均未执行；取得包含主机与写入范围的明确批准后，**实际运行一次**，耗时 **8.666657 秒**，2026-11-16 04:11:31 UTC 完成。没有运行失败后的重试、时限续接或输出覆盖。

本轮使用固定 1.5B tokenizer 对已审计抽样的 **10,000 条训练样本全部重新构造监督输入**，并按 `training_order=none`、`final_partial_group=drop` 核验三个 epoch 的切批与累积计划。逐批验证右 padding、真实 EOS 保留、监督区间、样本顺序与身份计数。

**没有加载模型、没有前向/反向、没有 optimizer step、没有初始化 CUDA、没有训练或评估。** 3 epochs 指三个 epoch 的数据装配检查，不是三轮训练。耗时是读取/核验/分词/装配与审计的 wall-clock，不是 GPU 或训练吞吐指标。

## 文件变更

- 新增本报告与 [机器可读索引](m7_real_batch_assembly_20261116.json)。
- 更新 [README](../README.md) 与 [REPRO_PLAN](../REPRO_PLAN.md) 的最新进展；历史报告保留，不回写成当时已通过。
- 核验驱动、生成器、只读校验器与完整日志保存在本地 Git 忽略的缓存目录。
- **未修改生产模块、配置、抽样 seed、数据、环境或模型。** 未 push。

远端只新建目录 `/root/autodl-tmp/adaptive-lora-m7-data-20261115-04/m7_batch_assembly_20261116_01`，包含：

| 文件 | 内容及访问边界 |
|---|---|
| `summary.json` | 聚合统计、源码/输入/清单指纹；目录权限 `700`，文件权限 `600` |
| `dropped_manifest.private.json` | 三个 epoch 计划丢弃的身份及来源索引；权限 `600`，仅留远端，不取回、不进 Git |

本地仅接收统计与指纹；原始题面、答案及私有身份列表未传回。哈希用于核对证据，不作为匿名化或来源认证保证。

## 论文依据与工程假设

- **[论文] Appendix C.2：** 训练 10,000 条、3 epochs、每卡 batch 8、累积 8、最大长度 512、padding 到 8 倍数。
- **[复现决策，已确认]** 单卡有效 batch 64，不同于论文四卡 DDP 的 256；训练顺序 `none`，末组不足一个累积组时 `drop`。后两项不是论文原样声明。
- 本轮没有修改算法或预处理规则。查阅既有 PDF 文本提取缓存 `.cache/m7-20261115-preflight-01/paper-implementation-utf8.txt` 中 C.2 对应段落；本轮直接 PDF 页渲染因缺少 `pdftoppm` 不可用，未安装依赖，不声称完成了新的 PDF 图像核查。
- 固定模型 revision `8faed761d45a263340a0528343f099c05c9a4323`；在加载前重新核对四个 tokenizer 文件及模型 `config.json` 的既有 SHA-256。
- 复用旧审计的固定抽样，不重新选择样本。训练文件实际 170,420 行、96,529,722 bytes，长度/Git blob/SHA-256 与旧审计一致；全部 10,000 个身份与 prompt 哈希复核一致。
- 本轮不重新核验八任务评估拆分或 token 级交集；其既有证据仍见 [tokenizer 门禁报告](m7_tokenization_1_5b_20261116.md)。

### 固定执行字节，不替换部署

远端仓库仍为批准提交 `2df16eb07bff7990cf9daf93bb189c9d91d06deb`，执行前后 clean。该提交不含新的装配模块，因此沿用此前批准的内存内嵌机制：

1. 本地从完整提交 `945fc4101500fe93a0da4def35ef122071d6c184` 的 Git 对象提取四个模块及配置；
2. base64 内嵌，往返核验后生成驱动；
3. 远端逐项 SHA-256 通过后只在内存执行，不替换 repo 或部署配置；
4. 本地独立 AST 还原驱动 payload，再与 Git 对象逐字节比对。

| 身份域 | 本轮证据 |
|---|---|
| 驱动 SHA-256 | `b1f63359dd154b640e418507fb50f7b9e2d0a15d3cae78d0c87cc68d7939964d` |
| 内嵌配置文件字节 SHA-256 | `e5e8dad366dc26198d41c9a0ffad3715a30a0c020a4146b02d3e77cb17e3c87f` |
| 内嵌规范协议 SHA-256 | `230a07b8314fc87faa4b59fa1b7025751ecc41ecd0d57f0837e0a8b9f72d528f` |
| 仍未同步的部署配置 SHA-256 | `6f715cef71d1fbe350541bd6b45524a4134ea2d879b1bee266468c4afb30081c` |
| 原审计抽样 SHA-256 | `76a61e01d6e0b2400b4870ed54b6d0a5afa12347389f14d19d58b05164288c3b` |

因此本轮统计确实使用新协议，但**不代表远端磁盘配置已同步**。四个模块各自的原始字节哈希、原审计文件哈希及训练文件来源哈希见机器索引。

## 执行命令

以下为本轮实际命令；输出目录已存在，不得直接重跑同一写入驱动。

```bash
# 本地生成：从固定 Git 对象嵌入已提交字节，并编译、AST 往返验证
.venv/Scripts/python.exe -I -B -X utf8 \
  .cache/m7-20261115-assets-04/build_batch_assembly_runner_01.py

# 批准后的唯一一次实际远端装配；驱动内 SIGALRM 单次上限 600 秒
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  autodl '/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python -I -B -' \
  < .cache/m7-20261115-assets-04/run_batch_assembly_01.py \
  > .cache/m7-20261115-assets-04/batch-assembly-remote-03.json \
  2> .cache/m7-20261115-assets-04/batch-assembly-remote-03.log

# 远端标准库只读归档核验：检查身份清单但不输出身份，不写远端文件
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
  -o ConnectionAttempts=1 -o ForwardAgent=no -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  autodl '/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python -I -S -B -' \
  < .cache/m7-20261115-assets-04/verify_batch_assembly_remote_01.py \
  > .cache/m7-20261115-assets-04/batch-assembly-archive-verification-01.json \
  2> .cache/m7-20261115-assets-04/batch-assembly-archive-verification-01.log

# 本地回归与标准库证据核验
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-assets-04/verify_batch_assembly_local_01.py
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 实际证据 |
|---|---|---|
| 训练来源字节、旧审计身份与固定抽样复核 | **PASS** | 全部 10,000 个 record 与原审计一致 |
| 真实 1.5B tokenizer 监督样本构造 | **PASS** | 10,000/10,000 成功；0 截断；坏样本会整体失败，不跳过 |
| 三个 epoch 真实切批与 collation | **PASS** | 3,750 个 micro-batch、30,000 个样本-次检查 |
| mask/labels、右 padding、真实 EOS 保留 | **PASS** | 每条逐项核对；batch 宽 88–472，均为 8 的倍数 |
| 固定顺序与完整组使用前缀 | **PASS** | 三个 epoch 的全序/使用序指纹一致 |
| 累积层末尾丢弃身份及计数守恒 | **PASS** | 每 epoch 1,248 批计划使用、2 批计划丢弃；9,984+16=10,000 |
| 远端私有身份清单只读复核 | **PASS** | 与原审计末尾 16 条一致；三 epoch 同一批身份 |
| 私有目录与文件权限 | **PASS** | `700` / `600` |
| Git 内嵌字节与本地证据独立复核 | **PASS** | `PASS_LOCAL_BATCH_EVIDENCE_VERIFICATION` |
| 远端归档独立复核 | **PASS** | `PASS_READ_ONLY_ARCHIVE_VERIFICATION` |
| 本地离线全套回归 | **PASS** | **292 项，16.639s，无跳过，退出码 0** |
| 模型前向中的真实 attention masking | **NOT RUN** | 本轮只校验数据 mask，无模型 |
| CUDA/AMP、训练、pilot、正式评估 | **NOT RUN** | CUDA 未初始化，实际优化步数为 0 |

全套测试日志保留负向夹具输出的 `ERROR: Protocol decisions are not all confirmed` 等信息，另有 Windows 子进程读取线程的 `UnicodeDecodeError`。最终 unittest 仍为 `OK`、退出码 0、无跳过；但**不能称日志完全无异常**。解码异常不出现在两份远端 stderr 中，本轮不扩展修复 Windows 日志读取路径，也不把它隐去。

### 必须区分的数据装配与训练计划

| 项 | 每 epoch | 三 epoch |
|---|---:|---:|
| 实际检查的 micro-batch | 1,250 | 3,750 |
| 实际检查的样本-次 | 10,000 | 30,000 |
| **计划**参与训练的 micro-batch | 1,248 | 3,744 |
| **计划**参与训练的样本-次 | 9,984 | 29,952 |
| **计划**丢弃的样本-次 | 16 | 48 |
| **计划** optimizer steps | 156 | 468 |
| **实际执行** optimizer steps | 0 | 0 |

核验对所有批次做了 collation，包括将来训练时计划丢弃的末尾两批；这只是数据检查，不代表它们被训练。`none` 顺序使三个 epoch 计划丢弃的是**同一批 16 条不同样本**，不是 48 条不同样本。

另一个重要层次区别：10,000 正好被 micro-batch 8 整除，所以 micro-batch 层的 `dropped_sample_ids` 清单为空。真正的 16 条余量在**累积组层**才被排除。本轮驱动按 `sum(plan_groups(...))` 的使用边界提取该后缀，独立写入远端私有清单，未把 micro 层空清单误报成“无丢弃”。这不等于生产装配模块已自动输出完整累积层身份清单；后续训练入口仍需显式连接这一步。

## 运行日志和输出路径

远端目录如上，本地完整统计与核验位于 `.cache/m7-20261115-assets-04/`：

- `batch-assembly-remote-03.json` / `.log`：唯一实际装配的 stdout / stderr（stderr 为空）。编号 `03` 是此前两次权限拦截后的本地日志编号，不是执行第三次。
- `batch-assembly-archive-verification-01.json`：远端只读证据核验；stderr 为空。
- `batch-assembly-local-verification-01.json`：内嵌 Git 字节、配置、统计及归档指纹核验。
- `real-batch-assembly-regression-01.log`：本地 292 项回归的完整日志。
- `run_batch_assembly_01.py`、`run_batch_assembly_template_01.py`、`build_batch_assembly_runner_01.py`：驱动与生成来源。

| 证据文件 | 原始字节 SHA-256 |
|---|---|
| 远端 `summary.json` | `c3a94077188e1f23436226b2362bc958319e3238dccbc9163834b81cdad08b47` |
| 仅远端保存的私有清单 | `0b13cdfba75ae0b581c6bbee31940feb553ebb702a1605f96a77231f71e12f12` |

stdout 是单行 JSON，远端文件是缩进 JSON；本地核验按相同序列化重建后匹配远端文件指纹，不把两种字节布局混为同一个文件哈希。

## 未解决问题

1. 远端部署配置仍为旧字节；本轮使用了经过哈希核验的内嵌新配置，没有替换部署。
2. GPU checkpoint 的 optimizer/RNG 断点续跑尚未实现。
3. 训练入口尚需将真实装配、累积计划、累积层身份清单与 checkpoint 连通；本轮不是端到端训练验收。
4. 真实 CUDA/AMP 数值、模型前向中 padding 的处理、显存/吞吐和有界 pilot 尚未验证。
5. Linux 旧的两项真实 0.5B tokenizer 测试、语义去重、题面内嵌答案及预训练污染检查仍未完成；本轮不替代这些不同范围的验收。

## 下一步建议（不自动执行）

优先完成并纯 CPU 测试 optimizer/RNG checkpoint 续跑与训练入口集成，再申请独立的 CUDA/AMP 有界 pilot 授权。真实装配这一单项已通过，但不能称为“pilot 前所有纯 CPU 工作均完成”。M7 整体继续保持未完成；不自动训练、同步部署、安装依赖或公开 push。
