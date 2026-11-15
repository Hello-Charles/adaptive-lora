# AutoDL 重启后 SSH 验证与只读环境盘点（2026-11-15）

## 当前任务与完成内容

用户确认已在AutoDL平台重启当前实例，授权再次验证连接。本轮仅验证SSH及只读盘点；不开始M7、不部署代码、不改变远程Python环境、不下载模型或数据、不启动GPU计算。

- **SSH公钥认证通过，退出码0**；使用本地已有SSH别名，启用BatchMode、StrictHostKeyChecking，禁用密码、键盘交互和agent forwarding。没有修改本地密钥或known_hosts。
- 远端返回 `SSH_LOGIN_OK`，实际身份为 `uid=0(root) gid=0(root)`，内核为Linux 5.15.0-25-generic，架构x86_64。
- 先前的[重启前预检报告](autodl_preflight_20261115.md)保留历史失败证据，不再代表当前认证状态。本次成功与平台文档的重启后生效规则一致；没有读取授权文件，不能据此重建平台具体注入过程或排除其他同时变化。
- 首次盘点未启动Python：非交互SSH的PATH中没有python/python3，命令退出127；这不是SSH认证失败。只读检查发现已有Miniconda解释器，改用其绝对路径后盘点退出0；没有激活、安装、升级或修改环境。
- 远程探针仅通过标准输入执行，使用 `-I -B` 和Python标准库；读取系统信息、包元数据及nvidia-smi，不导入第三方框架、不分配GPU张量、不在远程保存脚本或创建项目目录。

盘点时间：**2026-11-15 04:29:03 UTC**。资源与占用均为该时刻快照，不是性能测试或后续可用性保证。

| 项目 | 实际只读结果 | 边界 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB显存 | nvidia-smi查询；未验证模型实际显存需求 |
| NVIDIA驱动 | 580.76.05 | 本轮实测，不能沿用此前方案中的驱动值 |
| GPU占用 | 显存0 MiB、利用率0% | 单次快照，不保证持续空闲 |
| CPU型号 | Intel Xeon Platinum 8470Q | CPU可见逻辑数/亲和性均为208，不能解释为获分配208 vCPU |
| CPU限额 | cpu.max为 `2000000 100000`，即20个CPU时间配额 | 相当于20 vCPU计算额度，不是20个独占物理核心 |
| 内存限额 | 96,636,764,160 bytes，即90 GiB | cgroup v2 memory.max；不使用宿主/proc总内存冒充容器额度 |
| 系统盘 | 30 GiB，约29.95 GiB可用 | overlay文件系统报告的快照 |
| 数据盘 | 50 GiB，接近50 GiB可用，XFS | 实际挂载已核验，未创建项目或写入数据 |
| 操作系统 | Ubuntu 22.04.4 LTS | 读取当前容器的os-release |
| Python | 3.12.3，现有Miniconda解释器 | 不是本地Windows的3.12.8 |
| PyTorch | 2.5.1+cu124 | 包元数据；未import torch，未测试CUDA张量 |
| NumPy | 2.1.3 | 包元数据 |
| Git | 2.34.1 | 仅查询版本，未clone/fetch/checkout |

数据盘挂载点及本次实际使用的解释器命令：

```text
数据盘：/root/autodl-tmp
解释器：/root/miniconda3/bin/python -I -B -X utf8 -
```

在本次检查的解释器中，未找到Transformers、PEFT、Accelerate、Datasets、Safetensors、Tokenizers、Hugging Face Hub、PyArrow或Pytest的包元数据。该结论不覆盖未检查的其他环境，不把“未找到元数据”扩大为整台主机绝对未安装。

## 文件变更

- 新增本报告，保存本次认证成功、资源限额、软件现状及未执行边界。
- 在[此前预检报告](autodl_preflight_20261115.md)顶部增加后续状态指引，保留原有失败记录和当时测试结果，不改写历史。
- 本地私有盘点脚本、原始JSON和日志保存在Git忽略目录，不加入公开提交。
- 未修改src、scripts、tests、算法规格、依赖文件或里程碑勾选。当前仍为M0–M6已验收、M7未开始。

本次运行前代码基线为 `10a12e9c5ce05875851fcb0d96275be444f8235e`，分支为 `repro/m6-unified-baselines`，工作区干净。报告完成并核验后按用户要求创建本地文档提交；提交SHA由Git日志及最终回复给出，不写入提交自身内容。没有push。

## 论文依据与工程假设

本轮只读接入检查不新增论文算法或正式实验决定；已阅读 [PAPER_SPEC.md](../PAPER_SPEC.md) 与 [REPRO_PLAN.md](../REPRO_PLAN.md)，无需为此重复读取论文PDF。

- CPU以cgroup配额而非宿主可见208逻辑CPU计；内存以90 GiB容器限额而非宿主/proc总量计。
- 先前用户最终说明没有给出数据盘或驱动，本次才通过实际查询确认50 GiB数据盘及580.76.05驱动；不沿用被替换的历史配置。
- PyTorch版本后缀cu124只作为已安装发行包的构建标识记录，不等同于实际框架导入成功、完整CUDA Toolkit已安装或CUDA运算已通过。nvcc不在此次非交互PATH中，也不能据此断言磁盘上不存在Toolkit。
- 本地Transformers5.22.2要求PyTorch>=2.6，不能直接与当前2.5.1组合。远程缺少项目所需包，不能把“SSH通过、GPU可见”当作“实验环境已就绪”。
- [待核验] 正式GPU训练的CUDA/AMP/梯度累积、数据与模型revision、任务协议及显存/吞吐仍需后续获批工作；本轮不保证论文micro-batch可运行，也不引入QLoRA等算法替代。

## 执行命令

实际通过现有SSH别名执行，连接地址、公钥指纹和本机SSH配置不写入公开报告：

```bash
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  'printf "SSH_LOGIN_OK\n"; id; uname -srm'
```

首次环境盘点使用command -v查找python/python3，因非交互PATH没有解释器而退出127；保留其输出文件，没有用后续成功覆盖。随后只读定位常见解释器路径，确认已有Miniconda解释器，再发送本地标准库探针：

```bash
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  'exec /root/miniconda3/bin/python -I -B -X utf8 -' \
  < .cache/autodl-postrestart-20261115-01/inventory_readonly.py \
  > .cache/autodl-postrestart-20261115-01/inventory-miniconda.json \
  2> .cache/autodl-postrestart-20261115-01/inventory-miniconda-ssh.log
```

探针读取os-release、/proc和指定cgroup文件，以importlib.metadata查包版本；调用 `id`、`df -hT`、`git --version` 及 `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits`。没有pip安装、conda环境激活、Git远程获取、模型加载或训练命令。

本地另执行报告章节/链接/尾空白、JSON与资源换算、探针语法及Git变更范围检查；核验索引保存在本机忽略目录。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查 | 结果 |
|---|---|---|
| PASS | 重启后SSH公钥认证 | exit0，返回SSH_LOGIN_OK及root身份；严格host key校验开启 |
| FAIL（已定位） | 首次环境盘点 | exit127，非交互PATH没有python/python3；非认证错误，未更改环境 |
| PASS | 使用现有解释器盘点 | 绝对路径执行exit0，JSON可读取 |
| PASS | GPU可见性查询 | nvidia-smi退出0，RTX4090、24,564 MiB、驱动580.76.05 |
| PASS | 资源与存储查询 | cgroup限额20 CPU时间配额/90 GiB，系统盘30 GiB、数据盘50 GiB |
| PASS | 包元数据检查 | Python3.12.3、torch2.5.1+cu124、numpy2.1.3；其余所查项目包未找到元数据 |
| PASS | 本地文档与证据核验 | 规定章节/本地链接/字节哈希/数值换算及仅文档变更检查，见verification.json |
| NOT RUN | 本轮完整训练单元测试 | 仅新增/维护文档，未修改项目代码；不把既往162项PASS冒充本轮重跑 |
| NOT RUN | 远程项目测试与框架导入 | 未部署仓库，未import torch/transformers/peft |
| NOT RUN | CUDA张量与模型pilot | 未分配GPU张量，未加载模型 |
| NOT RUN | 推送、远程部署、安装、下载与训练 | 没有执行，分别受相应授权约束 |

## 运行日志和输出路径

所有原始日志位于本地Git忽略目录 [.cache/autodl-postrestart-20261115-01/](../.cache/autodl-postrestart-20261115-01/)：

- [ssh-login.log](../.cache/autodl-postrestart-20261115-01/ssh-login.log)：认证成功与远端身份。
- [inventory_readonly.py](../.cache/autodl-postrestart-20261115-01/inventory_readonly.py)：本地探针源码；仅经标准输入发送，未保存到服务器。
- [inventory.json](../.cache/autodl-postrestart-20261115-01/inventory.json)及[inventory-ssh.log](../.cache/autodl-postrestart-20261115-01/inventory-ssh.log)：首次exit127的输出文件，可能为空，不能当作完成的盘点。
- [python-paths.log](../.cache/autodl-postrestart-20261115-01/python-paths.log)：非交互PATH和解释器只读定位结果。
- [inventory-miniconda.json](../.cache/autodl-postrestart-20261115-01/inventory-miniconda.json)及[inventory-miniconda-ssh.log](../.cache/autodl-postrestart-20261115-01/inventory-miniconda-ssh.log)：实际完成的盘点与SSH诊断。
- [verification.json](../.cache/autodl-postrestart-20261115-01/verification.json)：本地一致性核验与原始证据哈希，不含密钥或连接地址。

本轮未创建远程项目、实验输出或checkpoint目录。原始盘点含基础设施路径信息，保持私有；公开提交仅包括经检查的文档摘要，不提交SSH配置、凭据或原始日志。

## 未解决问题

1. **SSH认证阻塞已解除**，但实验软件环境尚未准备；需要经审批选择兼容的Linux/CUDA依赖方案，不能直接安装Windows CPU锁文件。
2. 非交互SSH不自动暴露Miniconda命令；后续应继续使用核验过的绝对解释器或获批实验环境路径，不假定交互式环境激活会继承。
3. 尚未做框架导入、CUDA运算、模型结构、显存或吞吐验收；包元数据和nvidia-smi查询不能替代这些门禁。
4. 本地代码尚未推送，远程尚未克隆或检出任何本项目commit；正式模型、数据、实验协议与费用仍待后续授权。

## 下一步建议（不自动执行）

无需为当前已通过的认证再次重启。建议后续使用已核验的数据盘承载项目、环境、缓存和结果，在获得明确公开推送与远程部署批准后，按 [REMOTE_WORKFLOW.md](../REMOTE_WORKFLOW.md) 检出完整40位SHA。

兼容依赖安装、正式模型/数据下载、GPU pilot和M7分别按用户批准范围执行；本轮在连接验证、只读盘点和本地记录完成后停止。
