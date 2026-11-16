# M7 训练前集成与执行门禁 — 2026-11-16

## 当前任务与完成内容

状态：**CPU_SESSION_PASS_REMOTE_DEPLOY_BLOCKED**。M7 未完成。

用户已授权连续推进正式训练前准备，减少逐项确认；本轮不启动正式三轮训练。已完成有界会话集成、CPU精确续跑测试、训练前CUDA入口与固定版本公开交付。但远端部署被工具权限分类器拒绝，命令未执行；未换通道绕过。N=250校准、Top-8、CUDA/AMP pilot和GPU断点续跑均 **NOT RUN**。

固定可执行源码提交：`397df807152a19d530fc6d1a9c5c829a0f561ebb`，已发布至 `repro/m7-pretraining-gates`。本报告和后续文档不包含在该执行SHA内，不能据此漂移远端运行版本。

## 文件变更

- `src/adaptive_lora/m7_session.py`：接通已准备样本→micro-batch→完整累积组→清梯度→checkpoint；显式限定本次新增步数，维护跨epoch游标，记录实际消费样本-次数与两层drop私有清单。绑定实际有序监督张量指纹和micro-batch大小，首次更新前验证身份及优化器参数顺序。
- `src/adaptive_lora/m7_checkpoint.py`：新增只读边界校验接口，不消费RNG、不写盘，不改变既有checkpoint格式。支持会话在开始更新前拒绝非法状态。
- `src/adaptive_lora/calibration.py`：显式device/CUDA选项，输入与模型设备必须一致；不可用时报错，不回退CPU。公式、原始token、独立消融、完整logits恢复和参数字节检查不变。
- `src/adaptive_lora/trainer.py`：默认CPU门禁保留；M7可显式允许迁移后的CUDA参数审计和流式CPU回拷指纹。LoRA仍先在全新CPU基础模型上注入，不放宽旧M5训练器。
- `src/adaptive_lora/m7_training.py`：仅修正注释中“48条不同样本”的错误措辞，运行逻辑未变。
- `scripts/run_m7_pretraining.py`：Linux离线入口，只允许 `calibration` / `pilot`，没有正式训练模式。每阶段一次1800秒上限，新目录、失败不重试、不降低batch/精度、不下载、不安装。显式使用固定仓库src，不依赖venv中旧editable源码。
- 新增10项会话测试和4项CLI门禁测试；README/REPRO_PLAN更新，机器证据见同名JSON。

会话回调仍按只读约定工作；参数指纹不覆盖buffers。checkpoint不保存模块training标志或后端确定性设置，调用者仍须保持相同配置。

## 论文依据与工程假设

- **[论文]** Section 4.5 / 5.1的Q/K/V主对照，Appendix C.2：AdamW `lr=1e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8`；3 epochs、无warmup；micro8、accumulation8；冻结FP16基础模型、FP32 adapter与FP16 AMP；最大512、pad至8倍数。
- 本轮重新读取既有PDF提取文本的C.2第50–70行。未重新渲染PDF，未安装PDF工具；提取文本不是新增的原文来源认证。
- **[复现决策]** none/drop已由用户确认：每epoch计划用9,984条，丢同一批16条；三epoch468步、29,952个使用样本-次、48个丢弃样本-次。计划不等于已执行。
- **[复现决策]** 会话每次只提交完整累积组；相同sample_id但不同监督张量拒绝续跑；组后清梯度才存档。CPU测试的逐位一致不外推为CUDA已验证。
- **[复现决策]** 拟运行CUDA入口采用eager、确定性算法、关闭TF32，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；不声称这是论文完全披露的后端配置。
- 配置原始字节SHA256仍为 `e5e8dad366dc26198d41c9a0ffad3715a30a0c020a4146b02d3e77cb17e3c87f`；协议SHA256仍为 `230a07b8314fc87faa4b59fa1b7025751ecc41ecd0d57f0837e0a8b9f72d528f`。没有修改配置中的历史proposal措辞以免破坏哈希链。
- 原始题面、答案、私有身份列表与权重没有取回本地或公开发布。

## 执行命令

实际执行：

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_m7_session.py' -v
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -I .cache/m7-20261115-assets-04/scan_pretraining_publication.py
git push origin 397df807152a19d530fc6d1a9c5c829a0f561ebb:refs/heads/repro/m7-pretraining-gates
git ls-remote origin refs/heads/repro/m7-pretraining-gates
.venv/Scripts/python.exe -I .cache/m7-20261115-assets-04/build_pretraining_report.py
```

另实际执行一次严格SSH只读核验：新目录是否存在、旧repo HEAD/dirty、GPU名称和占用。未运行Python远端测试或CUDA。

被工具拒绝、**未执行**的部署范围：在 `autodl` 新建 `/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01`，从公开origin克隆到其 `repo` 子目录，detached检出上述完整SHA，核对clean/GPU资源并调用CLI `--help`。没有创建远端部署日志或成功标记；不能把本地日志编号当作执行次数。

后续拟运行，不是本轮已执行命令：

```bash
# 在autodl上，使用既有隔离venv，不重装或替换旧repo。
PY=/root/autodl-tmp/adaptive-lora-m7-20261115-02/venv/bin/python
BASE=/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01
DATA=/root/autodl-tmp/adaptive-lora-m7-data-20261115-04
SNAPSHOT=$DATA/models/qwen25_1_5b_8faed761/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
SHA=397df807152a19d530fc6d1a9c5c829a0f561ebb

timeout --signal=TERM --kill-after=20s 1830s "$PY" -I "$BASE/repo/scripts/run_m7_pretraining.py" \
  --phase calibration --repo-sha "$SHA" --data-root "$DATA" --snapshot "$SNAPSHOT" \
  --output "$BASE/calibration-01"

# 只在上一步成功且结果复核后执行；各阶段不自动重试。
timeout --signal=TERM --kill-after=20s 1830s "$PY" -I "$BASE/repo/scripts/run_m7_pretraining.py" \
  --phase pilot --repo-sha "$SHA" --data-root "$DATA" --snapshot "$SNAPSHOT" \
  --calibration "$BASE/calibration-01" --output "$BASE/pilot-01"
```

校准计划：先N1门禁，再固定250条，每条58次前向，共14,558次（其中250正式评分流程14,500次，含恢复检查）。pilot计划：三方法各连续3个完整组，再从第2组断点在全新基础对象上恢复并执行1组；共12个pilot optimizer steps，另做每方法最大实际批宽的前反向探针，不step。上述所有计划步数本轮实际为0。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 结果 | 证据范围 |
|---|---|---|
| 会话定向测试 | PASS | 10项，2.407s；真实PEFT微型Qwen、CPU |
| 全套回归 | PASS | 335项，20.920s，0跳过；包含4项CLI门禁 |
| 跨epoch fresh-base恢复 | PASS | 参数、AdamW、RNG、下一组loss/日志逐项精确一致；CPU |
| 两层drop/游标/异常清梯度 | PASS | 临时合成样本，非真实1.5B模型 |
| 发布前扫描 | PASS | 旧部署之后18个提交、181个唯一blob；有限敏感模式与路径规则，不是完整DLP |
| GitHub固定引用 | PASS | 已确认远端引用等于完整执行SHA |
| 远端新目录部署 | BLOCKED | 工具权限分类器拒绝，0次实际执行 |
| 只读状态复核 | PASS | 新目录不存在，旧SHA保持且clean，4090为0MiB/0% |
| N1/N250 CUDA校准、Top8 | NOT RUN | 没有真实重要性结果 |
| 三方法CUDA AMP/GPU checkpoint | NOT RUN | 没有显存、吞吐或续跑证据 |
| 正式三epoch训练/统一评估 | NOT RUN | 本轮明确不启动正式训练 |

本轮回归原始日志中没有此前Windows reader线程的 `UnicodeDecodeError`，但部分中文显示仍有编码乱码，不宣称已修复该历史问题。报告生成器首次因Windows CRLF换行断言失败，改为只规范比较用换行后通过；日志原始字节和SHA256未改变。

## 运行日志和输出路径

实际本地证据（机器索引保存原始字节SHA256）：

- `.cache/m7-20261115-assets-04/pretraining-regression-01.log`
- `.cache/m7-20261115-assets-04/pretraining-publication-scan-01.json`
- `.cache/m7-20261115-assets-04/pretraining-readonly-01.log`
- `reports/m7_pretraining_20261116.json`

实际远端新产物：**无**。拟部署目录经只读核验不存在。旧repo仍在 `/root/autodl-tmp/adaptive-lora-m7-data-20261115-01/repo`；旧数据、模型和审计继续留原位。

## 未解决问题

1. 工具要求对 `autodl` 主机及远端写入范围作明确批准，未接受“所有授权均同意”的总体表述。不得自行改权限设置、替换传输方式或以内嵌执行绕过拒绝。
2. 真实CUDA数值、N250选层、三方法AMP/最大批宽显存及fresh-base续跑尚未验证，不能称为“训练前全部就绪”。
3. 该CLI仅用于有界门禁，不是正式三epoch训练命令。完整训练仍需在上述门禁通过后单独启动；后续正式日志/评估尚未产生。
4. 固定模型LICENSE原文字节、镜像来源独立认证、语义去重与预训练污染等已有证据边界不因本轮CPU测试而消失。

## 下一步建议（不自动执行）

先解除远端写入门禁，再按固定SHA部署→远端适用CPU回归→N1/N250→独立复算重要性→三方法AMP pilot/GPU恢复验证的顺序继续；OOM或超时即停，不静默减小batch或延长时限，不自动正式训练。

明确批准范围可表述为：

> 批准在 autodl 新建 /root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01，克隆并检出固定SHA 397df807152a19d530fc6d1a9c5c829a0f561ebb；在该目录写入部署、CPU测试及校准/pilot日志和私有结果。批准用既有隔离venv各执行一次最多1800秒的N1/N250 CUDA校准和三方法有界CUDA/AMP pilot。不安装、不下载模型或数据、不覆盖旧结果、不自动重试、不启动正式三epoch训练。
