# M7 环境恢复：指定版本已部署，隔离安装尚未完成（2026-11-15）

## 当前任务与完成内容

**环境验收仍为 BLOCKED：SSH已恢复，批准SHA检出与独立venv创建通过；依赖和项目未安装，Linux导入/CPU验收未执行。** 本报告续接 [首轮SSH阻塞记录](m7_environment_20261115.md)，不改写此前失败事实。机器可读状态见 [环境索引](m7_environment_resume_20261115.json)。

用户“已重新打开，继续”后，助手沿用此前“批准部署和环境”的范围。官方源超时后，用户另行批准：候选直接版本不变、清华TUNA单一索引、每个wheel先与官方精确版本元数据核对SHA256再离线安装，复用已匹配固定官方哈希的Torch；仅一次新的最多30分钟尝试。不包括模型/数据、CUDA、GPU作业、训练、正式评估、删除/覆盖旧环境或重启实例。

实际完成及停止位置：

1. **08:17 UTC严格SSH恢复通过**，只读重新盘点资源；没有改公钥、known_hosts、端口或认证策略。
2. 官方GitHub/PyPI/Torch端点预检通过；确认目标目录不存在、数据盘余量满足门禁后，在新目录克隆公开仓库并detached checkout完整批准SHA：

   `9051fdfe86379aa3502e66bf1a586d4eb00565ba`

3. 创建独立venv，Python3.12.3，pip24.0；user site关闭，sys.path不包含平台site-packages。检出后工作区干净，保存全部tracked文件字节SHA256，平台包元数据集合未变。
4. 官方源依赖安装在540秒上限处停止，实际540.032856s；Torch完整下载，但后续PyPI wheel约60–90kB/s，停止于NumPy下载。没有安装完成，不能归因为已证实的版本冲突。
5. 单独获批TUNA方案后，重验缓存Torch并完成67个wheel的缓存下载，共**3,179,040,252 bytes**，包括11个固定直接依赖及实际选定的传递/构建依赖。
6. 安装前核验于226.000239s失败。自编核验器把setuptools-84.2.1的1个顶层及12个vendor内嵌METADATA一并计数，误报 `invalid or oversized wheel metadata`；不是已发现hash不符，也不是文件超过2MiB。已保留最初脚本字节，先补vendor回归得到红灯，再修正为“唯一顶层METADATA”，本地11项通过；唯一性、大小、名称/版本与官方整wheel哈希门禁不降低。
7. 准备只续验同批缓存的脚本时，工具权限按 **Remote Shell Writes** 拦截本地组装调用，命令未执行。没有绕过、发送续接脚本、重跑下载或重置时限；仅保存了未执行的续接片段。
8. **09:21 UTC只读复查通过**：新venv仍只有pip24.0；Git HEAD精确匹配批准SHA、工作区干净、tracked源码字节与平台包元数据集合未变。初始核验目录有53个wheel（含单独固定官方哈希的Torch）、52份官方PyPI元数据；全67个wheel尚未完成官方核验，没有完整manifest、Linux freeze或安装成功标记。

原镜像尝试始于 **08:58:32.230328 UTC**，30分钟截止为 **09:28:32.230328 UTC**；失败后的诊断和本地修正不重置此墙钟预算。09:21复查时尚有445.199952秒，不是新的30分钟授权；超出原截止后不得凭本次授权启动续接安装。

### 本轮实时资源快照

| 项目 | 08:17 UTC实测 | 边界 |
|---|---|---|
| 系统 | Ubuntu22.04.4 LTS，Linux5.15.0-25-generic，x86_64 | 当前容器，不是本机Windows |
| GPU | RTX4090，24,564 MiB，驱动580.76.05 | nvidia-smi查询；当时显存0、利用率0，不是CUDA或持续空闲保证 |
| CPU | Xeon Platinum8470Q，可见逻辑CPU208 | cgroup限额20个CPU时间配额，不是获配208 vCPU |
| 内存 | cgroup上限96,636,764,160 bytes，即90 GiB | 不把宿主/proc总量当容器额度 |
| 数据盘 | 50 GiB XFS | 09:21复查可用43,558,199,296 bytes；仅快照 |
| 平台Python/包 | Python3.12.3，Torch2.5.1+cu124、NumPy2.1.3 | 包元数据，未导入框架；平台环境未安装或升级 |

## 文件变更

| 文件 | 内容 |
|---|---|
| [README.md](../README.md) | 最新状态改为SSH恢复、已部署但依赖未安装；链接本报告 |
| [REPRO_PLAN.md](../REPRO_PLAN.md) | 仅勾选指定SHA检出/独立venv子项，安装/导入/CPU及CUDA/pilot父项保持未完成 |
| [首轮报告](m7_environment_20261115.md)、[较早盘点](autodl_postrestart_20261115.md) | 顶部增加最新状态指引，原失败/快照保留 |
| 本报告与[机器索引](m7_environment_resume_20261115.json) | 记录本次操作、证据、测试与未执行边界 |
| Git忽略的本地操作脚本/测试/原始证据 | 核验器规则修正、回归、只读收集及独立归档；不随公开源码提交 |

没有修改src、scripts、tests、算法规格、确认配置或候选依赖文件。获批远端代码仍为9051fdfe…；本地报告基线为 `a2f0ee563e6f4b92ffecd499ccd3664144d39652`，分支repro/m7-environment-bootstrap。完成核验后按既有Git约定创建本地报告提交，完整SHA由Git及最终回复给出；不将后续报告提交冒充远端运行代码，不自动push。

## 论文依据与工程假设

- 本轮是环境准备，不是核心算法、数据预处理或评估实现；已重读PAPER_SPEC/REPRO_PLAN，未重新读取PDF。沿用 [M7协议](../M7_PROTOCOL.md) 此前按原PDF Section4.5/5.1、Table1与AppendixC.1–C.3核查的1.5B/N250/Top8/rank16/alpha32/QKV及单卡batch差异；没有新增论文事实。
- **[复现决策]** 新独立venv使用候选Torch2.6.0+cu124、Transformers5.22.2、PEFT0.21.2等，11项直接版本均未改。镜像派生输入只写在repo之外，变化限于已批准索引、已重验Torch本地URI及wheel构建工具；原[候选文件](../requirements-linux-cu124-m7.in)字节不变。
- **[工程边界]** TUNA索引的哈希不能代替官方核验；要求PyPI精确版本JSON中的filename、size、SHA256一致，Torch使用固定官方URL哈希。缓存存在和pip download成功均不等于全部核验或安装通过；哈希匹配也不保证软件无漏洞。
- **[待核验]** 候选Linux框架导入与实际兼容性仍未验收。Windows本地185项PASS不能替代Torch2.6/Linux验证；包元数据集合未变不等于完整平台文件审计。独立venv不继承平台site-packages，不声称整个操作系统已隔离。
- 未下载Qwen1.5B权重或正式数据；Torch/NVIDIA依赖wheel是软件包，不是模型权重。未测试CUDA张量、真实1.5B模块返回值、AMP、梯度累积或GPU checkpoint。

## 执行命令

SSH均使用已有别名、BatchMode、StrictHostKeyChecking，禁用密码/键盘交互/agent forwarding；不公开实际入口或密钥路径。远端非交互PATH没有python命令，使用已核验的绝对解释器。主要实际命令/脚本入口如下；脚本中的具体参数和stdout保存在私有证据中：

```bash
# 平台解释器：只读盘点和预检，随后执行获批的独立目录bootstrap。
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  '/root/miniconda3/bin/python -I -B -' \
  < .cache/m7-20261115-environment-02/bootstrap_remote.py

# bootstrap内部的关键Git/venv动作；目标均为首次创建的新目录。
git clone --no-checkout --single-branch --branch repro/m7-run-preflight --no-tags \
  https://github.com/Hello-Charles/adaptive-lora.git <新目录>/repo
git checkout --detach 9051fdfe86379aa3502e66bf1a586d4eb00565ba
/root/miniconda3/bin/python -I -B -m venv <新目录>/venv

# 已实际执行的两轮安装准备；同一严格SSH前缀下以stdin传入，不从下载目录启动。
# 官方轮：pip install --index-url https://pypi.org/simple --only-binary=:all: ...，540秒停止。
# 镜像轮：pip download --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
#   --only-binary=:all: --dest <本轮downloads> ...，下载后核验失败，未进入安装。
# 脚本：install_remote_dependencies.py、install_verified_mirror.py。

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/Scripts/python.exe -I -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python.exe -I -B -X utf8 \
  .cache/m7-20261115-environment-02/test_install_verified_mirror.py

# 结束前仅只读收集：固定日志/元数据、wheel字节摘要、包元数据和Git状态。
# 在同一严格SSH前缀下以stdin运行collect_environment_state_readonly.py，exit0。
.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-environment-02/decode_remote_state.py
```

只读Git查询设置GIT_OPTIONAL_LOCKS=0，避免status刷新index。未执行缓存续接、离线依赖/项目安装、远端pip check、CPU导入或CPU测试；被权限拒绝的组装调用不能列作执行成功。没有重新下载整批wheel、创建PR、推送新报告或改变平台原解释器。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查 | 实际结果 |
|---|---|---|
| PASS | SSH恢复/08:17实时盘点 | exit0；严格host key校验，GPU仅可见性查询 |
| PASS | 来源/空间/不存在目标预检 | 官方端点可达，目标初始不存在，数据盘满足20GiB门禁 |
| PASS | 批准SHA检出与独立venv | detached完整SHA、clean；不继承平台site-packages |
| FAIL | 官方源安装尝试 | 540秒TimeoutExpired，停止于下载，未安装 |
| PASS | 已批准镜像下载阶段 | 67个完整wheel；复用官方哈希重验后的Torch |
| FAIL（已定位） | 初次安装前METADATA核验 | 自编规则误计setuptools vendor字段；226.000239s停止，未进入安装 |
| PASS | 原核验工具10项测试 | 0.072s；未覆盖vendor结构，不能当作完备性证明 |
| FAIL（预期红灯） | 新增vendor回归后的11项 | 0.088s，1个error，实际复现旧规则缺陷 |
| PASS | 修正后本地核验工具11项 | 0.111s，OK；合成ZIP/伪响应不证明真实Linux安装兼容 |
| FAIL／阻塞 | 续接准备工具许可 | 本地组装调用被Remote Shell Writes拒绝，未执行，不绕过 |
| PASS | 09:21远端只读复查与证据传回 | exit0；pip24.0 only、HEAD/clean/源码字节/平台包集合一致；63份文件传输哈希通过 |
| PASS | 本地项目离线CPU回归 | **185 tests，19.699s，OK，无跳过**；不包含额外11项私有工具测试 |
| NOT RUN | 全批官方核验与离线安装 | 原核验仅部分完成；无完整manifest、安装报告或freeze |
| NOT RUN | Linux框架导入/pip check/CPU单测 | venv无项目依赖；已准备的CPU探针未执行 |
| NOT RUN | 模型/数据、CUDA/pilot、训练/正式评估 | 不在本轮授权范围，未执行 |

本地文档/证据核验另保存 [verification.json](../.cache/m7-20261115-environment-02/verification.json)，以实际status与覆盖范围为准；这不改变环境BLOCKED结论。

## 运行日志和输出路径

远端新工作区为数据盘下 `/root/autodl-tmp/adaptive-lora-m7-20261115-02/`，子目录为repo、venv、artifacts、cache、tmp和mirror-attempt-01。保留所有完整/部分下载和原失败结果；不能删除或续写成“原失败已成功”。

本地原始证据均位于Git忽略的 [.cache/m7-20261115-environment-02/](../.cache/m7-20261115-environment-02/)：

- [恢复记录](../.cache/m7-20261115-environment-02/resume.json)、[镜像授权](../.cache/m7-20261115-environment-02/mirror-authorization.json)：有限操作范围，不是未来通用权限。
- [实时盘点](../.cache/m7-20261115-environment-02/inventory-before.json)、[来源预检](../.cache/m7-20261115-environment-02/sources/remote-precheck-01/metadata.json)、[bootstrap结果](../.cache/m7-20261115-environment-02/bootstrap-result.json)。
- [官方源失败结果](../.cache/m7-20261115-environment-02/install-dependencies-result.json)、[镜像失败结果](../.cache/m7-20261115-environment-02/mirror-install-result.json)；完整日志/官方JSON另传回至 [retrieved/remote-state-final-01/](../.cache/m7-20261115-environment-02/retrieved/remote-state-final-01/)，每文件独立目录，不执行内容。
- [元数据诊断](../.cache/m7-20261115-environment-02/wheel-metadata-diagnosis.json)、[原执行脚本快照](../.cache/m7-20261115-environment-02/ops-first-attempt-snapshot/)、[红灯日志](../.cache/m7-20261115-environment-02/mirror-vendored-metadata-red.log)、[修正后日志](../.cache/m7-20261115-environment-02/mirror-integrity-tests-corrected.log)。原脚本SHA256为 `9ea3c7d6bf5eab1dfb81ec1dd3113de6a4910cbc63c4c777e2aba476005d86b1`，原10项测试源码SHA256为 `31113d5bec37f44072b2f2ca48a924e67adae99ec40c50f958dd90846f8a1626`；不能用修正版冒充初次执行字节。
- [09:21只读状态响应](../.cache/m7-20261115-environment-02/remote-state-readonly.json)、[解码索引](../.cache/m7-20261115-environment-02/retrieved/remote-state-final-01/index.json)、[续接被拦截记录](../.cache/m7-20261115-environment-02/continuation-blocked.json)。
- [本轮185项日志](../.cache/m7-20261115-environment-02/local-regression.log)、[归档核验](../.cache/m7-20261115-environment-02/verification.json)及其文档字节快照。SSH stderr含基础设施入口，不加入Git。

本轮没有合法的Linux成功freeze/完整wheel锁、cpu-imports.json、CUDA结果、checkpoint或正式指标；不创建占位文件补齐。

## 未解决问题

1. **当前不是SSH或公钥问题**。安装前核验未完成，续接准备另受工具权限限制；不能通过改换工具隐蔽执行被拒绝的写入，也不放宽全局权限。
2. 元数据规则已本地修正，但只有辅助函数测试通过，续接片段没有组装、没有远端执行。后续执行前需在适用许可和有限时段内完成续接工具的本地检查，而非直接重复运行原main（原目录已存在且它会触发新下载）。
3. 原30分钟上限有固定截止，不能因摘要、诊断、报错或等待而重新计算；届满后需要新的明确有限范围许可及工具执行许可。不会自动换版本/源、再下载整批依赖或复用旧授权为无限重试。
4. 53个初始已核验wheel及52份官方JSON不能冒充67个全批完成。后续应复核原缓存、补齐剩余官方小元数据、生成完整manifest与哈希锁后才离线安装；如hash/版本不符即停止。
5. Linux导入/CPU门禁、真实tokenizer缓存与1.5B数据/模型、CUDA/AMP/累积/checkpoint、GPU成本/显存/吞吐均未验收。本地项目PASS不升级这些结论。

## 下一步建议（不自动执行）

在取得工具许可和适用的有限续接授权后，优先复用原67个缓存wheel，不重新下载或改版本：修正工具本地检查 → 剩余逐文件官方核验 → 新完整manifest/require-hashes锁 → 当前隔离venv离线安装依赖与项目 → pip check/freeze → 离线CPU导入及适用单测。所有新产物另建路径，保留旧失败记录，前后复查批准SHA、源码字节和平台包元数据。

若真实tokenizer缓存不存在，相关用例逐项标记NOT RUN，不为凑齐测试数下载资源；只有实际完成依赖/导入/CPU验收才勾选对应子项。本轮在证据归档与本地报告提交后停止，不自动push、下载模型/数据、调用CUDA、运行pilot/训练或进入M8。
