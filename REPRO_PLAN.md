# Adaptive-LoRA 复现计划

## 当前工作方式（2026-11-16）

用户决定：**不再追加测试，直接准备环境、数据和正式实验代码。** M7不要求单测补齐、N1、pilot、独立结果复算或GPU恢复对照。已有报告留存，未运行的项目不记为通过；NaN/Inf、冻结权重、输入身份和hook异常清理等必要计算保护继续保留。

当前只准备，不自动启动正式GPU校准、训练或大规模评估。实际操作以[M7运行协议](M7_PROTOCOL.md)和[独立运行配置](configs/m7_formal_autodl.json)为准。

## 已完成阶段

| 阶段 | 范围 | 证据 |
|---|---|---|
| M0 | 工程/本地环境准备 | [环境记录](reports/m0_venv_environment.json) |
| M1 | 真实模型模块与返回结构 | [验收](reports/m1_validation.md) |
| M2 | 固定校准输入 | [验收](reports/m2_validation.md) |
| M3 | 独立Attention/MLP zero-ablation | [验收](reports/m3_validation.md) |
| M4 | 0.5B/N30重要性与Top7工程smoke | [验收](reports/m4_validation.md) |
| M5 | 选择性Q/K/V LoRA与短训练/恢复 | [验收](reports/m5_validation.md) |
| M6 | 三方法统一工程smoke | [验收](reports/m6_validation.md) |

上述不是1.5B正式任务结果，不外推正式准确率或GPU性能。

## M7 — 正式实验

目标：Qwen2.5-1.5B/28层、N250、Top8、Q/K/V rank16/alpha32；Full / Random8 / Adaptive8；同一训练与评估协议。

### 已有资源

- [x] 独立Linux venv已安装：Python3.12.3、torch2.6.0+cu124、Transformers5.22.2、PEFT0.21.2；重用，不重新安装。
- [x] 固定1.5B权重与tokenizer已下载，权重字节/LFS哈希与28层结构已有记录。
- [x] 九个数据JSON已下载；训练总体170,420行、固定抽10,000，其内校准250，评估7,500。
- [x] 固定抽样及真实1.5B分词/监督区间/直接输入身份隔离已有记录。
- [x] none/drop已确认：每epoch用9,984条、丢同一批16条；3epochs合计468步/方法。
- [x] 完整组checkpoint、AdamW/GradScaler/RNG与会话游标已有实现。

资源证据：[Linux环境](reports/m7_environment_cpu_20261115.md)、[模型结构](reports/m7_structure_1_5b_20261116.md)、[真实分词](reports/m7_tokenization_1_5b_20261116.md)、[真实装配](reports/m7_real_batch_assembly_20261116.md)。这些不需要重新测试作为前提。

### 当前准备任务

- [x] 精简M7协议和项目协作规则，取消复杂测试前置；冻结论文配置JSON不改。
- [x] 增加正式入口：prepare / calibrate / train / evaluate / summarize，无N1或pilot阶段。
- [x] 增加独立运行配置与启动脚本；默认只显示命令，显式start才使用GPU。
- [x] 将当前正式代码部署到新目录，沿用既有环境/原始数据，不覆盖旧结果。
      源码 15d9777，部署至 formal-ready-01/repo，配置 SHA 19dd4359。
- [x] 直接装配并保存正式训练/校准/评估输入及独立参考文件，不加载模型、不使用GPU。
      状态 DATA_PREPARED_NO_MODEL_EXECUTED；训练10000，校准250，评估7500。

### 明确启动正式实验后执行

- [ ] 一次正式N250校准并导出真实Top8（算法本身，不是测试）。
- [ ] Full28完整3epochs（468步）。
- [ ] Random8完整3epochs（468步）。
- [ ] Adaptive8完整3epochs（468步）。
- [ ] 各方法全7,500条greedy评估，8任务等权宏平均。
- [ ] 保存参数量、loss、实际显存/时间与JSON/CSV比较；不人为追求论文数值。

**数值实现说明**：旧FP16/eager首条baseline出现非有限logits，真实首个异常算子未定位。正式运行配置显式选择SDPA、保留FP16基础权重与FP16 AMP；这是复现实现选择，未运行新测试，不声称已实测修复。失败仍直接退出，不清洗NaN、不降batch、不换精度。

历史过程/失败：[训练前集成](reports/m7_pretraining_20261116.md)、[TLS部署失败](reports/m7_pretraining_deploy_failure_20261116.md)、[离线部署与CUDA首基线失败](reports/m7_cuda_calibration_failure_20261116.md)。历史中“下一步需补测/诊断/pilot”等提议已被当前用户指令替代，不再构成启动前提；历史事实本身不删除或改写为成功。

## M8 — 正式复现报告

- [ ] 固定源码/模型/环境/数据/运行配置和实际命令。
- [ ] 给出N250重要性与Top8、三方法效果/参数量/实际成本对照。
- [ ] 区分论文配置、复现决定和结果局限，不把镜像哈希或直接身份隔离扩大成来源/污染保证。

## 交付原则

- 不运行测试：明确记NOT RUN（用户要求），不制造PASS。
- 复用本地已下载资源；不自动下载、安装、公开push或泄露原始数据。
- 输出只建新目录，断点续跑显式选择源与新目标，不覆盖历史失败。
- 环境/数据/代码准备与正式GPU启动分开，本轮停在可启动状态。
