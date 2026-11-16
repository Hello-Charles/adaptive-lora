# M7 获批部署执行：GitHub TLS 失败 — 2026-11-16

## 当前任务与完成内容

**状态：BLOCKED_DEPLOY_GITHUB_TLS_FAILURE。此次不再是工具权限拒绝，而是实际执行后的网络错误。**

用户明确批准在 `autodl` 新建 `/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01`，克隆固定SHA `397df807152a19d530fc6d1a9c5c829a0f561ebb`，写入日志/私有结果，使用既有venv各执行一次最多1800秒的N1/N250校准与三方法有界CUDA/AMP pilot。

实际执行1次部署命令：已创建权限700的父目录，随后 `git clone` 因GitHub TLS连接异常退出128。固定SHA checkout、CLI帮助和任何远端Python/CUDA步骤均未执行。按既有不自动重试边界，没有重试克隆、切换传输方式、修改TLS配置、删除空目录或延长执行窗口。

只读核验UTC `2026-11-16T06:27:11Z`：新父目录为空，`repo` 不存在；旧部署仍为 `2df16eb07bff7990cf9daf93bb189c9d91d06deb` 且clean；RTX4090为0MiB/0%。

## 文件变更

- 新增本报告及[机器证据索引](m7_pretraining_deploy_failure_20261116.json)。
- 更新[阶段计划](../REPRO_PLAN.md)记载实际失败，不替换此前权限拒绝的历史。
- 执行代码、配置JSON、模型、数据、旧审计均未修改。未推送新版本。

## 论文依据与工程假设

本轮仅部署已测试的工程入口，没有改变算法、论文配置或任何复现决策。论文依据沿用[训练前集成报告](m7_pretraining_20261116.md)中的Section4.5/5.1及AppendixC.2；本轮不声称重新读取PDF或产生新算法证据。

335项本地CPU测试是此前固定源码验证，不是本轮远端测试。既有tokenizer/模型/数据证据不因GitHub克隆失败而失效，但本轮未重新校验其全部字节和依赖。

## 执行命令

实际部署命令关键部分（经严格SSH执行，顺序为 `set -eu`；clone失败后后续命令未运行）：

```bash
umask 077
base=/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01
test ! -e "$base"
mkdir -m 700 "$base"
timeout 180 git clone --no-checkout --single-branch \
  --branch repro/m7-pretraining-gates \
  https://github.com/Hello-Charles/adaptive-lora.git "$base/repo"
# 以下未执行：git checkout --detach 397df807152a19d530fc6d1a9c5c829a0f561ebb
```

实际stderr：

```text
Cloning into '/root/autodl-tmp/adaptive-lora-m7-pretraining-20261116-01/repo'...
fatal: unable to access 'https://github.com/Hello-Charles/adaptive-lora.git/': GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

随后仅执行 `date -u`、`stat`/`ls`、目标路径存在检查、旧repo `rev-parse HEAD`/`status --porcelain` 和 `nvidia-smi`。本地只计算两个原始日志文件的SHA256。日志编号02区分上轮未执行的权限拒绝，并不表示本次实际执行两次克隆。

## 测试结果（PASS / FAIL / NOT RUN）

| 项目 | 结果 | 实际范围 |
|---|---|---|
| 新目录创建 | PASS | 权限700，失败后保留为空目录 |
| GitHub克隆 | FAIL | 1次实际执行，TLS错误，exit128 |
| 失败后只读核验 | PASS | 新repo不存在，旧SHA/clean/GPU占用如上 |
| 远端CPU回归 | NOT RUN | 源码尚未检出 |
| N1/N250 CUDA校准与Top8 | NOT RUN | 0次校准尝试开始，Top8无结果 |
| 三方法AMP pilot与GPU续跑 | NOT RUN | 0次pilot尝试开始，0个optimizer steps |
| 正式训练与评估 | NOT RUN | 本轮明确不启动正式训练 |

## 运行日志和输出路径

- 本地部署原始日志：`.cache/m7-20261115-assets-04/pretraining-deploy-02.log`，234 bytes，SHA256 `8cdf4aeb3832bf2aeade783a57e6d0d92557fe7a3df0c6f06629256f0c963912`。
- 本地只读核验日志：`.cache/m7-20261115-assets-04/pretraining-deploy-readonly-02.log`，241 bytes，SHA256 `7a7390d1773b9e3cb26038d32bee2df7d42d1d9f2cf219979e10a50dc5b840b3`。
- 远端：仅上述空父目录，无新repo、校准/pilot日志或私有结果，无成功标记。
- 结构化状态：[JSON](m7_pretraining_deploy_failure_20261116.json)。

## 未解决问题

GitHub克隆传输失败是当前阻塞。不能把它表述成源码测试失败、GPU不足、数据重叠或校准超时。校准与pilot各自的一次1800秒执行均尚未开始；并非已使用完CUDA预算。

旧数据、模型及环境仍原位保留；本轮没有下载权重/数据、安装依赖、原始数据回传或正式训练。

## 下一步建议（不自动执行）

需另行批准一次部署续接。建议保持同一完整SHA与既有GPU边界；确认已存在父目录仍为空后，只创建其缺失的 `repo`，不删除/覆盖父目录。可选择一次GitHub克隆重试，或另行批准将本地固定Git对象通过SSH交付，以避免远端GitHub TLS依赖；未经批准本轮未执行任一选项。

源码检出并核对完整SHA/clean后，再按既定顺序执行适用CPU测试、一次N1/N250、结果复算及一次有界pilot。正式训练仍不启动。
