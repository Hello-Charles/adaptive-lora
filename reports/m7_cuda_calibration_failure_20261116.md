# M7 离线部署通过、首条 CUDA 校准基线失败 — 2026-11-16

## 当前任务与完成内容

状态：**DEPLOY_CPU_PASS_CALIBRATION_FAIL_NONFINITE_BASELINE**。M7未完成。

经用户另行批准，通过SSH离线交付固定SHA `397df807152a19d530fc6d1a9c5c829a0f561ebb` 的Git对象，避开此前远端GitHub TLS故障。新仓库已成功检出该完整SHA且clean，配置哈希正确。既有venv未替换或安装新包，旧仓库/数据/模型原位保留。

远端适用CPU验收通过；随后实际启动**一次**1800秒上限的N1/N250 CUDA阶段，在首条N1输入的**未消融基线预测**处报错：`ValueError: Prediction logits must be finite`。外层wall-clock **13.633774秒**，非超时。N1未完成、未开始N250、未产生Top8；未进入消融或训练。失败后没有重跑校准、改变dtype/backend或延长时限。依赖校准结果的三方法pilot没有启动。

## 文件变更

- 新增本报告及[机器证据索引](m7_cuda_calibration_failure_20261116.json)。
- 更新[阶段计划](../REPRO_PLAN.md)和[README](../README.md)记录最新实际状态；此前权限/TLS故障保留为历史。
- 已批准执行代码、配置JSON未修改；本轮不提交算法补丁，不推送新执行版本。
- 本地缓存中保存交付/CPU回归/CUDA执行/只读诊断/有限日志取回工具；没有将缓存、模型、原始数据或私有样本身份加入Git。

## 论文依据与工程假设

按需重新读取既有PDF提取文本：Algorithm1、Section4.3公式(4)/(5)、Section4.4选层；训练精度依据AppendixC.2。方法提取文本有部分编码损失，未重新渲染PDF或安装工具，未声称获得新的原文来源认证。

- **[论文]** 冻结未消融基础模型先确定原始`t*`与`q`，Attention/MLP输出独立消融，测同一原token，再逐样本绝对差取均值和Top-K。本次连有效原始`q`都未得到，因此不存在可用重要性或Top8。
- **[论文]** AppendixC.2规定训练冻结FP16基础权重、FP32 adapter及FP16 AMP。
- **[复现决策]** 本固定入口也以FP16/eager进行校准，并严格检查非有限logits、恢复及基础字节；不能把训练精度表直接描述成论文明确规定了校准的每种内部算子精度。
- **[边界]** 不使用`nan_to_num`或过滤坏样本来伪造可用概率，不把失败后的Top8设为默认层，不借用0.5B的Top7，也不把剩余秒数解释成允许另一次校准。

## 执行命令

### 1. 固定Git对象交付与检出（实际通过）

本地创建增量bundle：

```bash
git bundle create .cache/m7-20261115-assets-04/source-397df80-01.bundle \
  2df16eb07bff7990cf9daf93bb189c9d91d06deb..refs/remotes/origin/repro/m7-pretraining-gates
git bundle verify .cache/m7-20261115-assets-04/source-397df80-01.bundle
```

bundle **213,418 bytes**，SHA256 `5f645e24beaeb1e9c6f41212ab721e3d4a780d077c681cb73f3da4424eef48de`；只包含已批准固定SHA相对旧部署所需的Git对象，不是工作树打包。SSH传输长度/哈希及远端bundle prerequisite核验通过后，从旧仓库只读复制独立Git对象，再在新repo fetch bundle并detached checkout固定SHA。

交付器首次使用不存在的 `/usr/bin/python3`，exit127，Python载荷未执行、未写远端目录。修正为已知既有隔离venv后执行成功；原错误日志保留。没有再次连接GitHub克隆，亦没有重跑CUDA。

### 2. 远端CPU回归（实际通过适用部分）

使用既有venv，以`-I`启动受控runner，显式加入固定新repo的src路径；`CUDA_VISIBLE_DEVICES=''`、HF离线、300秒上限，未修改测试源码或假冒0.5B缓存。

结果：总发现335项，明确排除2项缺缓存的旧用例；运行333项，其中30项被原测试skip装饰跳过，**实际303项通过**、0失败、0错误，耗时 **10.498秒**，CUDA未初始化。不能写成“333项全部执行通过”或“335项Linux全通过”。stdout中的两条ERROR是负向夹具输出，不是测试失败。

### 3. 单次N1/N250 CUDA阶段（实际失败）

```bash
PY=/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python
BASE=/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01
DATA=/root/autodl-tmp/adaptive-lora-m7-data-20261115-04
SNAPSHOT=$DATA/models/qwen25_1_5b_8faed761/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323

timeout --signal=TERM --kill-after=20s 1800s "$PY" -I "$BASE/repo/scripts/run_m7_pretraining.py" \
  --phase calibration --repo-sha 397df807152a19d530fc6d1a9c5c829a0f561ebb \
  --data-root "$DATA" --snapshot "$SNAPSHOT" --output "$BASE/calibration-01"
```

实际UTC `06:39:08.775902`–`06:39:22.409633`，exit1。内部失败元数据耗时12.721683秒；外层13.633774秒包含启动/退出开销，不是纯GPU计算时间。

关键调用链：

```text
run_calibration -> calibrate_model(samples[:1])
-> reference_prediction(baseline[0, position]) -> _fp32_logits
ValueError: Prediction logits must be finite
```

源代码及traceback说明首个完整baseline已返回并进入原始预测检查，尚未注册zero-ablation hook；没有完成N1/恢复检查或运行N250。这是核对固定源码后得到的控制流结论，不是失败进程保存的独立forward计数。

### 4. 只读/纯CPU诊断（实际完成）

不构造真实1.5B模型、不做真实模型前向、不重启CUDA：逐张量检查固定safetensors的原始有限性及CPU转FP16有限性，并读取已安装Transformers的eager Attention源码。

- 338个张量、1,543,714,304个元素，原始dtype均BF16；**原始非有限张量0，CPU转FP16后非有限张量0**，最大存储权重绝对值316.0。
- 所安装eager实现先 `torch.matmul(query, key.transpose(...)) * scaling`，随后才FP32 softmax。
- CPU合成Q/K（head_dim128，各值128）验证：FP16点积和缩放后结果非有限，先转FP32点积则有限。
- **该合成探针仅证明一种可能机制，未定位真实N1中首个非有限运算。** 不据此声称已确认真实Q/K溢出，更不能声称换backend或精度已修复。
- 预测位置的FP32 softmax后处理无法修复上游已出现的Inf/NaN。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 结果 | 范围 |
|---|---|---|
| SSH bundle交付/完整SHA/clean | PASS | 长度+SHA256+Git prerequisite/checkout |
| 远端CPU适用测试 | PASS | 303实际通过；30跳过、2明确未运行；0失败/错误 |
| 原始权重及CPU FP16转换有限性 | PASS | 所有338个存储张量；不覆盖真实GPU中间激活 |
| 一次N1/N250 CUDA阶段 | FAIL | 首条N1未消融预测logits非有限；非超时 |
| N1恢复检查 | NOT RUN | 首个baseline检查失败，未到恢复前向 |
| N250/Top8/独立评分复算 | NOT RUN | 没有成功概率轨迹或选层产物 |
| 三方法AMP/最大批宽/GPU续跑 | NOT RUN | 依赖有效选层，pilot尝试0 |
| 正式训练/统一评估 | NOT RUN | optimizer steps0，未启动正式训练 |
| 有限证据取回/一致性核验 | PASS | 13份日志/索引原始字节核验，无原始数据取回 |

## 运行日志和输出路径

远端根：`/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01`。

- `repo`：固定SHA `397df807152a19d530fc6d1a9c5c829a0f561ebb`，clean。
- `delivery-01/source.bundle`：已核验Git对象。
- `cpu-regression-01.log/.json`：完整测试日志与skip/NOT RUN清单。
- `calibration-01.log`、`calibration-01-execution.json`：失败日志和外层计时。
- `calibration-01/failure.json`：该输出目录**唯一文件**，无成功标记、N1结果或概率轨迹。
- `pilot-01`：不存在。

已取回本地的限定日志位于 `.cache/m7-20261115-assets-04/pretraining-remote-evidence-01/`；交付和纯CPU诊断日志在其父目录。所有SHA256见[机器索引](m7_cuda_calibration_failure_20261116.json)，关键CUDA日志为2,142 bytes / SHA256 `86f70bfedd54651adc9fd20db012d1daa2df1b0bd2b6eb7487ba9cc8a66b7827`。

06:44:34 UTC最终只读核验：新旧仓库均clean，旧SHA仍 `2df16eb…`；新配置SHA256仍 `e5e8dad3…e3c87f`，旧审计文件SHA256仍 `840fba0c…2f9`；4090为0MiB/0%。原始题面、答案、私有身份清单、模型及checkpoint均未取回。保留失败目录，不覆盖/删除。

## 未解决问题

1. **真实首个非有限运算尚未定位**。目前排除存储权重非有限/CPU转FP16本身溢出，发现eager FP16 QK中间溢出的可行机制，但MLP/其他路径等尚未排除。
2. 本次获批的校准单次尝试已启动并失败；不能自动再跑或换精度。pilot未启动，不能绕过选层门禁填入默认Adaptive层。
3. 校准换BF16不仅是“同一次重跑”：还会影响分数及当前校准/训练基础字节身份绑定，而且不验证FP16 AMP训练本身；须明确工程决定与审计关系，不能悄改。
4. 32项远端用例仍因缺固定0.5B缓存而未执行；不扩称Linux全测试通过。真实1.5B先前分词验收是独立历史证据。

## 下一步建议（不自动执行）

推荐先另行批准 **`autodl` 最多120秒、只对首个原始输入定位首个非有限算子的FP16数值诊断**：在新 `numeric-probe-01` 目录写仅数值统计/来源指纹，不改权重、依赖、旧输出；不做zero-ablation、不训练、不重跑N250。若需比较backend或高精度子算子，先明确记录该对照，不把它当作成功校准。

定位后再决定稳定实现方案，补CPU回归并固定新执行版本，获批后才另开校准尝试。保持论文zero-ablation/固定原token/Top-K与训练FP16基础配置，不以静默清洗NaN或减batch解阻；正式训练继续禁止启动。
