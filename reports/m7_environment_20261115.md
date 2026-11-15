# M7 代码交付与隔离环境准备：SSH入口阻塞（2026-11-15）

## 当前任务与完成内容

**当前阻塞：两次SSH均在TCP阶段返回Connection refused，退出255，尚未进入公钥认证。远端指定SHA检出及环境安装未执行，环境验收状态为BLOCKED，而不是PASS。**

用户在首项协议确认之后，另行选择“批准部署和环境 (Recommended)”，授权范围为：公开推送当前M7分支及未发布祖先、远端数据盘新目录检出批准SHA、新建隔离环境并安装候选依赖/项目、执行框架导入与离线CPU验收。授权不包含模型/数据资源下载、CUDA调用、训练、正式评估、覆盖/删除旧环境或自动重启实例。

已独立完成不受SSH阻塞影响的部分：

- 推送前审查6个此前未发布提交、99个唯一Git文件对象；指定凭据/连接模式及禁止公开的路径未命中。这是有限扫描，不是零泄密或许可证审计保证。
- 本地重新执行全部离线CPU回归：**185 tests，18.466s，OK，无跳过**。
- 已公开推送 [repro/m7-run-preflight](https://github.com/Hello-Charles/adaptive-lora/tree/repro/m7-run-preflight)，Git push退出0；随后用git ls-remote查询分支，确认其实际指向批准的完整SHA：

  `9051fdfe86379aa3502e66bf1a586d4eb00565ba`

- 本地pip check通过；保存当前Windows环境元数据。该快照不是Linux依赖安装/导入验收。
- 保留两次SSH失败输出，没有关闭host key校验、重配公钥、读取私钥、修改known_hosts或尝试其他端口。

没有远端克隆、检出、环境创建、依赖安装、CUDA张量、训练或正式评估。首次失败的inventory-before.json为空文件，**不是有效JSON资源快照**；原样保留，不用同日较早的成功盘点代替本轮结果。

## 文件变更

本轮只修改状态与报告，不修改算法、配置、候选依赖、训练器或测试逻辑：

| 文件 | 内容 |
|---|---|
| [README.md](../README.md) | 区分协议确认与后续操作授权，记录已发布SHA和当前SSH阻塞 |
| [REPRO_PLAN.md](../REPRO_PLAN.md) | 代码发布子项已完成，远端环境验收仍未勾选 |
| [历史重启后报告](autodl_postrestart_20261115.md) | 顶部新增后续状态指引，保留04:29 UTC历史成功及资源快照 |
| 本报告 | 记录操作范围、真实结果、日志和恢复门禁 |

本轮本地归档分支为repro/m7-environment-bootstrap，从批准SHA创建。最终报告提交的完整SHA由Git日志和回复记录；它不是本轮获批公开/部署的代码SHA，不自动推送该后续报告提交。已发布分支保持上述批准版本，远端服务器尚未检出任何版本。

## 论文依据与工程假设

本轮仅交付版本并尝试环境准备，没有实现或修改核心算法，不新增论文算法假设；沿用 [M7_PROTOCOL.md](../M7_PROTOCOL.md) 中此前按原PDF Section 4.5/5.1、Table 1和Appendix C.1–C.3核查的配置。不把本轮环境操作描述成重新核查整篇论文或完成正式实验。

- **[工程候选，未安装/未验收]** [Linux/CUDA直接依赖](../requirements-linux-cu124-m7.in)仍为Torch2.6.0+cu124、Transformers5.22.2、PEFT0.21.2等；不是完整传递依赖锁。
- **[实测，本地]** Python3.12.8、Torch2.15.0+cpu、Transformers5.22.2、PEFT0.21.2；本地185项测试不能证明上述Torch2.6/Linux组合兼容。
- **[历史快照]** 04:29 UTC的远端Python3.12.3、Torch2.5.1+cu124、磁盘与GPU信息仍保留在历史报告；本轮无法再次读取，不能视为当前可用性保证。
- **[待核验]** TCP拒绝可能来自实例关闭、端口映射变化、SSH服务尚未监听或网络入口拒绝。仅凭现有错误无法区分，不能归因为密钥失效，更不能断言服务器已关机。

用户已授权本轮隔离环境安装，不再将“缺安装批准”写成当前阻塞；实际阻塞是SSH入口不可达。协议配置中的execution_authorized=false仍是离线检查器的固定非执行声明，不用于阻止或授予人工批准的环境操作。

## 执行命令

主要实际调用如下，均从本地仓库启动；其中SSH调用未能启动其请求的远端命令：

```bash
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-01/audit_publication.py

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v

git push --porcelain origin \
  9051fdfe86379aa3502e66bf1a586d4eb00565ba:refs/heads/repro/m7-run-preflight
git ls-remote --exit-code --heads origin refs/heads/repro/m7-run-preflight

ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  '/root/miniconda3/bin/python -I -B -' \
  < .cache/autodl-postrestart-20261115-01/inventory_readonly.py

# 仅另做一次有界复查，没有循环重试。
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl 'printf "SSH_LOGIN_OK\n"; id'

.venv/Scripts/python.exe -I -X utf8 scripts/record_environment.py \
  --metadata-only --output .cache/m7-20261115-environment-01/local-environment-metadata.json
.venv/Scripts/python.exe -I -X utf8 -m pip check
git switch -c repro/m7-environment-bootstrap
```

日志重定向使用新文件并启用noclobber；没有执行pip install、远端git clone/checkout、环境创建或任何实例重启命令。没有创建PR或修改默认分支。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 项目 | 实际结果 |
|---|---|---|
| PASS | 批准SHA与待公开历史检查 | 干净9051fdfe…版本；6个未发布提交，99个唯一文件对象；有限模式/路径扫描未命中 |
| PASS | 本地离线CPU回归 | 185 tests，18.466s，OK，无跳过；未新增测试或改变测试逻辑 |
| PASS | 公开推送与发布引用核验 | push退出0；ls-remote退出0并返回精确批准SHA |
| PASS | 本地依赖声明检查 | No broken requirements found；只对应当前Windows环境 |
| PASS | 本地元数据快照 | 记录Python和发行包版本；metadata-only中的not_checked不等于导入成功 |
| FAIL／阻塞 | 本轮SSH盘点及一次有界复查 | 均退出255，TCP Connection refused，未进入公钥认证，远端命令未执行 |
| NOT RUN | 当前远端资源/目录复查 | 未取得新快照，不能确认当前空闲容量或目标目录不存在 |
| NOT RUN | 远端指定SHA检出、环境创建/安装 | SSH不可达，没有修改平台环境，也没有生成Linux安装报告或freeze |
| NOT RUN | Linux导入、pip check和CPU测试 | 尚无本项目新建的候选环境可运行 |
| NOT RUN | 模型/数据下载、CUDA、pilot、正式训练/评估 | 不在本轮授权范围，未执行 |

## 运行日志和输出路径

以下均为Git忽略的本地私有证据；原始SSH错误含连接入口，不加入公开提交：

- [authorization.json](../.cache/m7-20261115-environment-01/authorization.json)：本轮单独操作授权记录，不是未来操作的通用权限凭据。
- [publication-audit.json](../.cache/m7-20261115-environment-01/publication-audit.json)：已检查的提交、Git对象和文件哈希。
- [publication-push.log](../.cache/m7-20261115-environment-01/publication-push.log)、[push stderr](../.cache/m7-20261115-environment-01/publication-push-stderr.log)、[published-ref.log](../.cache/m7-20261115-environment-01/published-ref.log)：实际发布及引用证据。
- [local-regression.log](../.cache/m7-20261115-environment-01/local-regression.log)、[local-pip-check.log](../.cache/m7-20261115-environment-01/local-pip-check.log)、[本地环境元数据](../.cache/m7-20261115-environment-01/local-environment-metadata.json)：本地验证。
- [首次SSH错误](../.cache/m7-20261115-environment-01/inventory-before-ssh.log)、[复查错误](../.cache/m7-20261115-environment-01/ssh-recheck-stderr.log)：两次TCP拒绝；对应stdout为空，不是远端执行成功。
- [verification.json](../.cache/m7-20261115-environment-01/verification.json)：独立存档复核与覆盖边界；[source-snapshot](../.cache/m7-20261115-environment-01/source-snapshot/)保存本轮最终4份文档字节。存档复核通过不等于远端环境验收通过。

未创建占位Linux依赖锁、空训练指标或虚假的环境成功标记；没有下载模型/数据、生成checkpoint或创建远端实验目录。

## 未解决问题

1. SSH入口不可达，无法判断是实例状态、映射、服务还是网络入口问题；不能继续任何远端准备步骤。
2. 此前成功的公钥认证只能证明当时可用，本轮没有到达认证阶段，不能据此要求重新登记公钥或修改authorized_keys。
3. 需要可用的SSH入口后才能重新核查数据盘、目标路径、解释器与候选依赖实际安装/导入；目前没有Linux freeze或兼容性结果。
4. 数据、1.5B模型结构、CUDA/AMP/累积/checkpoint与GPU成本门禁仍未验收，本轮没有放开其运行权限。

## 下一步建议（不自动执行）

需要用户在AutoDL控制台确认实例已开机且SSH入口仍与本地配置一致；若入口已变更，提供更新后的连接信息即可，无需提供私钥或密码。没有控制台访问通道时，助手不能通过被拒绝的SSH替用户确认实例电源状态，也不会自行重启或扩容。

入口恢复后，助手按已获批准的同一范围继续：严格SSH只读复核 → 数据盘不存在的新目录 → 检出完整9051fdfe…批准SHA并检查干净状态 → 独立环境安装及CPU验收。出现已有目录、未知改动、依赖解析冲突或其他超出范围的变化时先停止报告；不自动覆盖、降级、换源或下载模型/数据。

本轮在归档和本地报告提交后停止，不循环探测、不自动进入CUDA/pilot或M8。
