# M7 optimizer/RNG 训练断点机制（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。状态 **`TRAINING_CHECKPOINT_CPU_PASS_CUDA_NOT_RUN`**。

新增训练状态 checkpoint，与旧 M5 adapter-only 推理恢复明确分离。在极小 CPU 线性模型及真实 PEFT 微型随机 Qwen 上验证：完整累积组边界保存，从全新对象恢复后，下一组的参数、AdamW 状态和 loss 与连续运行逐项相同。Python、NumPy legacy 与 Torch 默认 CPU RNG 的后续抽样也相同，含 Gaussian 缓存。

本轮执行了合成小模型 CPU 反向与优化器测试，**没有进行真实 1.5B 训练、远端操作、CUDA、下载、安装或 push**。不把 CPU 单元测试写成正式训练。

## 文件变更

- 新增 [m7_checkpoint.py](../src/adaptive_lora/m7_checkpoint.py)：safetensors/JSON 保存与恢复 API、边界校验、RNG 与 AdamW 检查。
- 新增 [test_m7_checkpoint.py](../tests/test_m7_checkpoint.py)：29 项测试，含微型真实 PEFT 续跑对照。
- 新增本报告与 [机器索引](m7_training_checkpoint_20261116.json)，更新 README 与复现计划。
- 旧 [checkpoint.py](../src/adaptive_lora/checkpoint.py) 与原测试未改；训练引擎、生产配置及哈希链未改。

### 存储内容与信任边界

每次保存必须使用不存在的新目录：

| 文件 | 内容 |
|---|---|
| `state.safetensors` | 可训练参数、AdamW 的 step/一阶二阶矩、Torch CPU RNG 及可选单 CUDA 设备 RNG |
| `checkpoint.json` | 参数有序名称、实验身份、epoch/下一批/global_step、依赖版本、GradScaler、Python/NumPy RNG、张量文件 SHA-256；最后写入作为完成标记 |

不用 pickle，不保存完整基础模型。只有最后的 JSON 与张量文件均完整且校验通过才允许恢复；磁盘失败保留部分目录，不覆盖或自动重试。**不提供断电后磁盘持久性保证或任意底层设备错误的事务回滚**。

保存前需要：

1. 完整 optimizer group 已完成，`scaler.update()` 已执行，所有梯度均为 `None`；
2. 单设备、单参数组、标准 AdamW，FP32 可训练参数，固定 `none/drop` 计划；
3. `named_trainable` 名称和参数对象与 optimizer 的顺序逐一相同，不能只比较集合；
4. global_step 同时匹配 epoch/下一批游标与每个 AdamW 槽位的 step；
5. 参数/优化器状态均有限，形状、dtype 和二阶矩符合约束。

恢复前需要：

1. 调用者从正确 revision 重建并实证基础模型身份、重新注入 LoRA、审计可训练白名单；本模块不自动证明传入参数就是正确的 Q/K/V；
2. 新对象与空优化器，参数顺序、超参数、设备、依赖版本和实验身份一致；
3. 全部文件/状态校验完成后才复制参数并恢复 optimizer/scaler，**RNG 最后恢复**，之后不得重新 seed 或进行会消耗训练 RNG 的初始化。

实验身份至少包括模型 revision、基础参数 SHA-256、协议/抽样/访问顺序 SHA-256、epoch 数、每 epoch micro-batch 数、accumulation 和 `none/drop`。上层还应加入方法、选层及其他运行约束；保存器严格比较完整 identity，但不能认证调用者填写的来源声明。内部哈希也不能认证同时重写了 JSON 与张量文件的外部来源。

### 与现有累积引擎的集成边界

当前 `run_accumulation.on_step` 在 optimizer/scaler 更新后调用，但**仍在清梯度之前**。因此不能把 `save_training_checkpoint` 直接挂在该回调上；本模块会拒绝仍有梯度的保存。调用者必须在组边界清梯度后保存，并维护下一批游标。本轮没有修改引擎的回调时序，也没有创建正式训练入口。

对 10,000/8/8 的 `none/drop` 计划，epoch 内最大已用游标为 1,248，而不是 1,250；末尾两批不形成优化步。epoch 完成可表示为当前 epoch 的游标 1,248，也可表示为下一 epoch 的游标 0，相应 global_step 必须一致。这保证恢复游标不会落在累积组中间，但**调用者仍须确保保存时确实处于所声明的位置**；手工清空未完成组的梯度不能使其成为合法训练断点。

## 论文依据与工程假设

- **[论文] Appendix C.2：** Python/NumPy/PyTorch seed42、AdamW、FP16 AMP、FP32 LoRA、3 epochs、batch8/accumulation8。
- **[工程机制，非论文实验设定]** safetensors/JSON 格式、完整组边界恢复、严格参数顺序与版本门禁是本项目审计设计；论文未给出该存档格式或完整 optimizer/RNG 续跑方案。
- 本轮查阅既有 PDF 提取文本 `.cache/m7-20261115-preflight-01/paper-implementation-utf8.txt` 的 C.2 段落；未新渲染 PDF 页面，不安装缺失的渲染工具。
- 不改变已确认超参数、数据、训练顺序或末组策略，也不修改配置文件。
- 同版本、同 RNG 不足以保证跨设备/硬件/CUDA 内核逐位确定性。调用者仍须保持 train/eval、autocast、后端确定性开关一致；不保存模型 buffers 或模块 training 标志。
- 仅保存 Python 全局、NumPy legacy 全局、Torch 默认 CPU 和当前单 CUDA 设备 RNG；不覆盖独立 Generator、data-loader worker、外部采样器或其他 CUDA 设备。
- 本 API 不支持 DDP、scheduler、组中间断点、跨设备迁移、AMSGrad/fused/capturable 等额外模式。

## 执行命令

```bash
# 定向测试：实际29项通过，4.073秒
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p 'test_m7_checkpoint.py'

# 全套回归：实际321项通过，16.905秒，退出码0
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests \
  > .cache/m7-20261115-assets-04/checkpoint-resume-regression-01.log 2>&1
```

实际本地版本：Python3.12、torch `2.15.0+cpu`、NumPy `2.5.4`、safetensors `0.8.0`、Transformers `5.22.2`、PEFT `0.21.2`。这不是远端 torch `2.6.0+cu124` 的 CUDA 验收。存档与恢复要求保存的 torch/NumPy/safetensors/Python 版本相同；PEFT/Transformers 运行兼容性还需上层环境审计。

## 测试结果（PASS / FAIL / NOT RUN）

| 项 | 状态 | 证据 |
|---|---|---|
| 连续运行 vs 全新对象恢复后续跑 | **PASS** | 参数、AdamW、loss 逐项相同 |
| Python/NumPy/Torch 后续随机序列 | **PASS** | 含 Gaussian 缓存 |
| 已启用 CPU GradScaler 恢复后续跑 | **PASS** | scale/growth 状态与下一组更新一致，不等于 CUDA FP16 验收 |
| 微型随机 Qwen + 真实 PEFT A/B 续跑 | **PASS** | 全新基础模型与 adapter；12 个 A/B 张量、优化器及 loss 相同；基础字节未变 |
| 无基础参数或 pickle 文件 | **PASS** | 只保存白名单与训练状态，两种固定文件名 |
| 保存不推进全局 RNG | **PASS** | 保存前后随机状态一致 |
| 参数对象/名称顺序、重复与冻结参数拒绝 | **PASS** | 逐对象校验，不凭 shape 或参数集合配对 |
| 未清梯度、累积中间游标及错误 global_step 拒绝 | **PASS** | 在创建目录前拒绝 |
| 参数或优化器非有限状态拒绝 | **PASS** | NaN/Inf 负例 |
| 初始零步、epoch 边界与结束位置 | **PASS** | 返回游标与实际 AdamW 步数一致 |
| 身份/超参数/依赖/scaler 模式不符拒绝 | **PASS** | 格式失败不修改活参数、optimizer 或 RNG |
| 文件篡改、缺完成标记、额外文件拒绝 | **PASS** | 哈希/结构门禁 |
| 即使更新哈希，错误张量形状仍拒绝 | **PASS** | 验证不止依赖文件哈希 |
| RNG 格式错误在修改活对象前拒绝 | **PASS** | 参数、optimizer、Torch RNG 保持不变 |
| 写盘失败与已有目录保护 | **PASS** | 不覆盖；部分目录没有有效完成标记 |
| CPU 路径不调用 CUDA RNG 接口 | **PASS** | 用禁止该调用的 patch 验证 |
| 定向测试 | **PASS** | **29 项** |
| 全套离线回归 | **PASS** | **321 项，16.905s，无跳过，退出码0** |
| 真正 CUDA RNG/FP16 GradScaler/设备恢复 | **NOT RUN** | 仅实现单设备分支，无 GPU 授权或运行 |
| 固定 1.5B 模型训练断点续跑 | **NOT RUN** | 使用的是微型随机模型 |
| 正式训练入口的中途停止/恢复集成 | **NOT RUN** | 本轮只验证存档 API 与组间续跑 |

全套日志仍出现此前 Windows 子进程 reader 的 `UnicodeDecodeError`，以及负向夹具的 ERROR 文本；最终 unittest `OK`。本轮不修复该独立日志路径，不称日志完全无异常。

## 运行日志和输出路径

- [源模块](../src/adaptive_lora/m7_checkpoint.py)、[测试](../tests/test_m7_checkpoint.py)。
- `.cache/m7-20261115-assets-04/checkpoint-resume-regression-01.log`：完整回归日志。
- 本轮测试 checkpoint 均在 unittest 的临时新目录，退出时由夹具清理；没有修改既有实验 checkpoint。
- 源码、测试和日志的原始字节 SHA-256 见 [机器索引](m7_training_checkpoint_20261116.json)。

## 未解决问题

1. 单 CUDA 设备的状态捕获/恢复、FP16 AMP 和真实 1.5B 数值尚未验收，不能勾选整个 GPU checkpoint/pilot 门禁。
2. 训练入口仍需连通数据装配、累积层丢弃清单、日志、组后清梯度 checkpoint 和精确游标恢复。
3. 模型基础权重/LoRA 的真实身份审计由调用者执行；本模块是低层存档机制，不是自动可信的模型加载器。
4. 远端仍是旧部署与旧配置；本轮没有同步或远程运行新代码。
5. Windows 回归日志解码异常、语义去重和预训练污染检查均未在本轮处理。

## 下一步建议（不自动执行）

下一单项宜先实现训练入口的纯 CPU 有界集成测试，验证停止后从游标继续不重训、不漏训、不重复步进，且接入累积层私有丢弃清单。之后另行申请真实 CUDA/AMP 有界 pilot 授权；不自动执行 GPU、正式训练、依赖变更或 push。M7 整体仍未完成。
