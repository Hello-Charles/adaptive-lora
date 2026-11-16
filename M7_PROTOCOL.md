# M7 正式实验运行协议

面向实验执行者。**2026-11-16用户决定：停止追加测试，直接准备环境、数据和正式代码。** 不再要求单测补跑、N1门禁、pilot、独立复算或GPU恢复对照；这些历史准备流程不再是启动条件。旧测试与失败报告保留，但不重复执行。

本轮只准备，不启动正式GPU实验。正式入口为[run_m7_experiment.py](scripts/run_m7_experiment.py)，运行配置为[m7_formal_autodl.json](configs/m7_formal_autodl.json)，启动器为[m7_formal.sh](scripts/m7_formal.sh)。旧[run_m7_pretraining.py](scripts/run_m7_pretraining.py)仅作历史工具，不再使用。

## 1. 实验目标与参数

依据论文Algorithm1、Section4.3–4.5、Section5.1/Table1及AppendixC.2–C.3：

| 项目 | 当前设置 |
|---|---|
| 模型 | Qwen/Qwen2.5-1.5B（非Instruct），28层 |
| revision | `8faed761d45a263340a0528343f099c05c9a4323` |
| 校准 | 固定训练子集中的250条；冻结/eval/no-grad；Attention、MLP模块输出分别zero-ablation |
| 选层 | 固定原始预测token的概率；两项绝对差逐样本平均；全28层Top8，并列层号升序 |
| 方法 | Full28 / Random8 / Adaptive8；Random独立seed20261115 |
| LoRA | 所选层Q/K/V，rank16、alpha32、dropout0、bias none；基础权重冻结 |
| 训练 | 同一10,000条、seed42、3 epochs；AdamW lr1e-4、wd0.01、betas(0.9,0.999)、eps1e-8，无warmup |
| 精度 | 基础FP16、adapter master FP32、CUDA FP16 AMP |
| batch | 单卡micro8×accumulation8，有效64；不同于论文4卡DDP256 |
| 文本 | 规范Alpaca纯文本；不用Qwen chat模板；答案和EOS监督，prompt/pad labels=-100 |
| 长度 | 训练/校准右截断512；pad到8倍数；评估保留完整前缀（本批最长521） |
| 顺序/余量 | none/drop；每epoch用9,984条、丢同一批16条；156步/epoch，468步/方法 |
| 评估 | 8任务各1,000，OBQA500，共7,500；greedy、最多32新token；按任务合法标签解析，8任务等权宏平均 |

以上模板、mask、采样索引、并列规则、Random对照、评估截断及解析规则是已确认的**复现决策**，不是论文逐token实现声明。Full和K8不是同参数预算；预期可训练元素分别为2,981,888与851,968，实际数量在训练结果中记录。

### 数值实现

**[复现决策] 新正式运行配置显式使用原生SDPA，基础FP16/adapter FP32/AMP FP16不变。** 原FP16/eager路径在首条未消融输入产生非有限logits；该失败保留于[历史报告](reports/m7_cuda_calibration_failure_20261116.md)。当前没有运行SDPA新测试，因此不声称已验证修复。backend单独记录在运行配置与checkpoint身份中，不悄改已冻结论文配置或历史审计哈希。

正式运行仍保留输入身份、有限值、冻结权重、LoRA白名单、hook异常清理与不覆盖结果的保护；它们不是额外测试流程。遇到NaN/Inf或OOM立即失败，不清洗概率、不减少batch、不替换精度，也不伪造Top8。**N250是Adaptive-LoRA算法本身，不能因取消测试而省略。**

## 2. 已有环境和数据

重用既有资源，不重新安装、下载或升级：

- 主机：`autodl`，单RTX4090 24GB。
- Python：`/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python`。
- 已有依赖：Python3.12.3、torch2.6.0+cu124、Transformers5.22.2、PEFT0.21.2。
- 数据根：`/root/autodl-tmp/adaptive-lora-m7-data-20261115-04`，九个JSON在其`data`子目录，固定抽样清单在`audit/audit.json`。
- 数据revision：`816657208af4db747803f87ba40a4c71383fed7a`；训练总体170,420行，已固定抽10,000，校准为其内250条。
- 模型快照：运行配置中的固定1.5B snapshot路径；3,087,467,144 bytes safetensors已经下载并核对过LFS哈希。
- 正式准备目录：`/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01/formal-ready-01`，不覆盖旧仓库或失败目录。

来源仍保留镜像未独立官方认证、固定LICENSE原文未取得及语义/预训练污染未排除的局限，详见历史报告；不要求再做测试作为本轮前置。原始数据、私有ID/生成明细与权重不公开上传。

## 3. 直接使用

在正式源码仓库根目录：

```bash
# 只做数据装配，不加载模型、不使用GPU；只需执行一次。
bash scripts/m7_formal.sh prepare

# 无参数只显示命令，不启动GPU。
bash scripts/m7_formal.sh

# 用户明确启动正式实验后执行：N250 → 三方法完整训练/评估 → JSON/CSV汇总。
bash scripts/m7_formal.sh start
```

无需运行协议预检器、pytest/unittest、N1或pilot。`prepare`直接沿用已有审计索引和本地tokenizer，生成可复用正式输入，不重抽、不丢弃坏行。

如需逐阶段启动：

```bash
PY=/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python
ENTRY=scripts/run_m7_experiment.py
PROFILE=configs/m7_formal_autodl.json

"$PY" -I "$ENTRY" calibrate --execute --profile "$PROFILE"
"$PY" -I "$ENTRY" train --execute --method full --profile "$PROFILE"
"$PY" -I "$ENTRY" evaluate --execute --method full --profile "$PROFILE"
# 另两方法将full改为random_k、adaptive_k；最后汇总：
"$PY" -I "$ENTRY" summarize --profile "$PROFILE"
```

`--execute`只防止误启动实际GPU作业，不运行测试。默认没有旧pilot的1800秒截止；完整实验是新的正式作业，不借用此前失败尝试的窗口，也不在本轮自动启动。

## 4. 数据、结果和续跑

运行配置的`work_dir`下：

```text
prepared/        manifest.json、训练/校准/评估输入与单独参考答案（private）
calibration/     原始概率轨迹、250条校准结果、selection.json
train-full/      3epochs日志、每epoch完整组checkpoint、参数/显存统计、result.json
train-random_k/
train-adaptive_k/
eval-full/       私有生成/逐样本评分、公开metrics.json
eval-random_k/
eval-adaptive_k/
comparison/      summary.json、comparison.csv
```

配置与文件指纹、来源SHA、实际环境版本随结果记录，不需要额外手工报告链。每阶段仅创建新目录；已有失败结果不删除、不覆盖，也不自动重试。评估函数只把前缀传入模型，全部预测生成后才加载独立参考文件评分。

每156个完整累积组及最终边界保存一次训练checkpoint，包含adapter/AdamW/GradScaler/RNG和游标；不是额外续跑测试。如中断，明确指定源断点和新目录：

```bash
"$PY" -I "$ENTRY" train --execute --method full --profile "$PROFILE" \
  --resume /绝对路径/checkpoint-step-000156 --output /新的绝对路径/train-full-resumed
# 对该恢复结果评估时加 --training-dir /新的绝对路径/train-full-resumed。
```

新目录的会话时间只代表本次续跑，不冒充累计训练成本。完整自动汇总默认读取标准目录；使用自定义恢复目录时需明确调整汇总输入，不能把旧未完成目录当成最终结果。

## 5. 当前准备状态与启动边界

环境和原始数据已在服务器；正式代码和运行配置按本协议准备，不再追加测试。**新代码测试状态为NOT RUN（用户要求），不是PASS。** 旧非有限值失败没有被删除或改成成功；SDPA正式计算结果仍未产生。

本轮仅部署和装配数据。只有用户明确要求开始正式实验，才运行`start`或GPU阶段；不以“准备”自动启动训练、大规模评估或新的校准。
