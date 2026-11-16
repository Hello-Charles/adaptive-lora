# M7 训练执行层：设备/精度解析与梯度累积引擎（2026-11-16）

## 当前任务与完成内容

面向复现实验实现者与执行者。**M7 的 CUDA/AMP/梯度累积执行机制已实现并通过 28 项离线测试**（全套 256 项通过）。状态为 `ENGINE_IMPLEMENTED_GPU_NOT_RUN`：机制可测试，但**真实 CUDA、真实 Qwen 前向、显存与吞吐均未运行**。

本轮**没有**占用 GPU、没有下载、没有训练、没有 push。

### 为什么先做这一层

M7 的三项决定（FP16 基础权重 + FP32 adapter + FP16 AMP、micro-batch 8 × accumulation 8、单卡有效 batch 64）此前只写在配置里，**没有任何可执行代码**。既有 `trainer.py` 的 `train_smoke` 明确写着「不做 accumulation/AMP」，且硬限制 1–3 步、batch=1、CPU。所以这一层是缺口，且不需要 GPU 就能实现和验证。

### 实际实现的三件事

**1. 设备解析不许静默回退**

`resolve_device("cuda")` 在 CUDA 不可用时**报错**，不回退 CPU。理由：静默回退会让一次自称的「GPU 训练」实际在 CPU 上完成，产出的耗时、显存、吞吐证据全部失真。

**2. AMP 在 CPU 上显式禁用并如实上报**

`resolve_amp` 返回 `(autocast_kwargs, scaler)`：

| 环境 | autocast | GradScaler |
|---|---|---|
| CPU + FP16 | 禁用 | 禁用 |
| CUDA + FP16 | 启用 | 启用（需防下溢） |
| CUDA + BF16 | 启用 | 禁用（动态范围足够） |

即使禁用也保留 `device_type`/`dtype`，报告里能如实写「请求了什么、实际是否生效」，而不是含糊说「用了 AMP」。

**3. 梯度累积的数学必须正确**

核心约束：**组内 loss 先除以该组实际 micro-batch 数，再反向**，使累积梯度等于「整组平均 loss」的梯度，而不是组内求和。这是最容易写错、也最容易被误当作「梯度爆炸」的地方，因此用真实梯度做了逐元素验证。

同时只有组末才 `optimizer.step()`；AMP 路径走 `scale → backward → unscale_ → 检查有限性 → step → update`，**unscale 之后**才判断梯度量级（否则看到的是放大后的数值）。

### 一个必须报告的算术事实：10,000 条训练样本无法被 64 整除

`describe_batch_plan` 把这件事在**计划阶段**就暴露出来，而不是等跑到最后一个 epoch 才失败：

| 项目 | 实际值 |
|---|---|
| 有效 batch | 8 × 8 = 64 |
| 每 epoch micro-batch | 1,250 |
| 完整累积组 | 156 |
| 每组余留 micro-batch | **2**（即 16 条样本） |
| 是否整除 | **否** |

10,000 = 64×156 + 16。因此每个 epoch 的最后一组只有 2 个 micro-batch，不是 8 个。

三种处置策略，**必须在任何前向之前确定**（`error` / `step` / `drop`）：

- `error`（默认）：不整除直接报错，不静默改变有效 batch。
- `step`：最后一组按**实际大小 2** 求均值后步进，有效 batch 变小但梯度仍是均值。
- `drop`：丢弃该组，该 epoch 少用 16 条样本——**必须由调用者显式选择**，不能默认。

我默认选 `error`，因为「静默少用样本」或「静默改变有效 batch」都会污染复现记录。

## 文件变更

- **新增** `src/adaptive_lora/m7_training.py`：设备/精度解析、累积组切分、批计划汇总、累积训练引擎
- **新增** `tests/test_m7_training.py`：28 项测试
- 未修改任何既有模块、配置、数据或报告

## 论文依据与工程假设

- **[论文] Appendix C.1–C.2：** FP16 基础权重、FP32 adapter 初始化、FP16 AMP；每卡 batch 8、accumulation 8；3 epochs、lr 1e-4、wd 0.01、betas(0.9, 0.999)、eps 1e-8、无 warmup。本轮实现只覆盖**精度与累积机制**，学习率调度与 epochs 由调用者负责。
- **[复现决策，已确认]** 单卡 8×8=64，明确不同于论文 4 卡 DDP 的 256。
- **[待核验，本轮未解决]** CUDA 上的真实数值、显存占用、吞吐、AMP 溢出行为仍需 GPU 实测。CPU 测试只能证明累积语义正确，不能证明 CUDA 路径等价。

## 执行命令

```bash
# 新增模块的定向测试
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -p "test_m7_training.py"

# 全套回归
.venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests
```

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 状态 | 依据 |
|---|---|---|
| 设备解析：CUDA 不可用时报错不回退 | **PASS** | 3 项（含非法设备值） |
| AMP：CPU 上显式禁用并如实上报 | **PASS** | 4 项（FP16/BF16/显式关闭/非法 dtype） |
| 累积组切分与三种余量策略 | **PASS** | 5 项（含计划阶段就拒绝） |
| 批计划：10,000 与 64 不整除被显式暴露 | **PASS** | 4 项 |
| **累积梯度 = 组内平均 loss 的梯度** | **PASS** | 逐元素与参考梯度比较，误差 0.0 |
| 报告的 loss 是未缩放均值 | **PASS** | 与手算均值一致 |
| 只有组末步进 | **PASS** | 4 micro-batch / 累积 2 → 2 步 |
| 末组按实际大小求均值 | **PASS** | 3 micro-batch → [2,1]；末组梯度等于单样本梯度 |
| 全零梯度被拒绝 | **PASS** | 走「全部为零」分支而非范数分支 |
| 守卫：白名单/空参数/非标量/非有限/推理模式 | **PASS** | 8 项 |
| 正常与异常路径均清梯度 | **PASS** | 2 项 |
| 项目离线回归 | PASS | **256 项，18.092s，无跳过** |
| 真实 CUDA/AMP 数值 | **NOT RUN** | 无 GPU，未进入 |
| 真实 Qwen 1.5B 前向与显存 | **NOT RUN** | 未进入 |
| 正式训练 / pilot / 评估 | **NOT RUN** | 未进入 |

### 测试过程中我自己犯的两个错误（已修正，如实保留）

1. **测试辅助函数签名写错**：`forward_loss` 需要 `micro → 标量`，我却写成了返回闭包的工厂，导致 6 项测试报「must return a scalar tensor」。修的是测试，不是实现。
2. **两项断言本身不成立**：
   - 全零梯度那项：范数检查（`norm <= 0`）先命中，使更具体的「全部为零」分支不可达。**这是实现的诊断顺序问题，已调整**——先判全零，再算范数。
   - 末组均值那项：参考梯度算在初始权重上，而末组实际跑在第一步更新后的权重上；AdamW 第一步就改了参数，两者天然不等。改用 `lr=0` 固定权重后比较才成立。**这是测试写法问题**。

第 2 条的第一个是真实缺陷（诊断信息不可达），第二个不是。两者都不影响数学正确性。

## 运行日志和输出路径

- `tests/test_m7_training.py`：28 项
- `.cache/m7-20261115-assets-04/training-regression-01.log`：全套回归日志

## 未解决问题

1. **CUDA 路径完全未实测**。CPU 测试只能证明累积语义，`resolve_amp` 的 CUDA 分支、GradScaler 的真实溢出处理都需要 GPU 验证。当前环境 `torch.cuda.is_available() == False`。
2. **余量策略需要你决定**：10,000 条与 64 不整除是既定事实，`error`/`step`/`drop` 三选一必须明确，否则正式训练无法启动。
3. 本层不含 checkpoint 的 optimizer/RNG 续跑（现有 `checkpoint.py` 明确只做 adapter-only 推理恢复）。
4. 不变量、去重、内嵌答案、预训练污染四项未检查。
5. 许可原文缺口未解；远端配置仍是修订前字节。

## 下一步建议（不自动执行）

1. **先定余量策略**（`error` / `step` / `drop`）——这是正式训练的前置阻塞项，且只影响 3 epochs 共 48 条样本的处置方式。
2. 有了 GPU 授权后，用本层跑一次**有界 pilot**：真实 1.5B、单卡、少量 step，验证显存峰值、AMP 是否溢出、吞吐是否符合预期，再决定是否开正式训练。

本轮未使用 CUDA、未训练、未 push，未改动配置、数据、审计产物或远端部署。
