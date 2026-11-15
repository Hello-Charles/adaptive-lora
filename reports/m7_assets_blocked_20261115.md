# M7 资源门禁：固定模型许可交付返回 HTTP403（2026-11-15）

## 当前任务与完成内容

**M7尚未完成；本轮资源门禁为BLOCKED_LICENSE_HTTP403。此前独立环境的PASS_APPLICABLE_CPU结论保留，不退回“依赖未安装”。** 机器状态见[本轮索引](m7_assets_blocked_20261115.json)，前次已通过的安装/适用CPU范围见[环境报告](m7_environment_cpu_20261115.md)。

用户“继续完成M7”后，本轮只推进下一项固定资源与结构门禁。用户明确选择“批准资源与CPU门禁”：允许在数据盘新目录获取已确认的Qwen2.5-1.5B固定safetensors、对应tokenizer/配置和LLM-Adapters固定提交的9个JSON及许可，先查许可，再核对固定字节/字段/行数，最后仅CPU加载一次并对短输入观察28层输出。单次最多1800秒、总下载上限3.3GB，不含新依赖、CUDA、校准、训练、大规模评估、覆盖或push。

实际停止位置：

1. 严格SSH进入远端Python；资源目录初始不存在、15GiB余量、批准SHA、Git clean和tracked源码字节的前置检查均通过。
2. 第一个模型LICENSE的固定revision `resolve` 请求，经重定向后返回 **HTTP403 Forbidden**。远端脚本exit1；未取得可核验许可字节，stdout没有完整JSON，不创建虚假预检PASS。数据许可请求与权重HEAD尚未执行。
3. 按失败即停止，没有启动资源获取、创建资源目录或下载权重。WebFetch的文档读取工具此前也因无法验证域名安全性而失败，不把该工具错误误写成许可内容或同一HTTP403。
4. 用户另行选择“批准有界路径诊断”：最多120秒、最多4个初始小请求，只检查原镜像raw/API路径；只有原固定LICENSE字节匹配后才继续尚未启动的资源任务。实际只发送2个初始请求，禁止自动重定向：**raw LICENSE仍403，固定revision API为200**。
5. API响应8,052 bytes已保存并校验传输SHA256，revision及权重LFS哈希与已确认配置一致，cardData标注apache-2.0。但API标签不能替代这次授权要求的固定许可原文字节，继续条件未满足；没有追加来源请求、更换版本/来源或用API成功掩盖LICENSE失败。
6. **12:34 UTC最终只读确认通过**：资源目录仍不存在；远端HEAD/clean/源码字节、平台包元数据、venv的69个发行包及前次CPU结果哈希均未变。只读确认没有重试下载、导入第三方框架或加载权重。

原获批资源操作的30分钟获取预算尚未开始，不把诊断时间写成已完成下载或默认无限重试。原环境安装尝试和此前失败记录均保留。

## 文件变更

- [README.md](../README.md)：增加最新资源阻塞指引，保留已有CPU通过状态。
- [REPRO_PLAN.md](../REPRO_PLAN.md)：记录本次批准范围、403、诊断和停止位置；资源/结构与GPU/正式实验项仍未勾选。
- [M7_PROTOCOL.md](../M7_PROTOCOL.md)：仅增加当前工程状态说明，纠正读者可能从早期“尚未安装”快照得到的误解，不更改已确认协议。
- 本报告与[机器索引](m7_assets_blocked_20261115.json)：保存真实执行与NOT RUN边界。
- Git忽略目录内新增本地自编预检/诊断/只读确认脚本和8项回归，保存API小响应、日志与授权记录。不加入公开Git。

未改src、scripts、tests、正式配置或候选依赖文件；用户打开的[train.py](../scripts/train.py)仍是原CPU smoke入口，没有为继续M7直接移除它的限制。

## 论文依据与工程假设

- 本轮重读PAPER_SPEC、REPRO_PLAN和已确认M7协议，**未重读PDF**；只做来源/许可预检，不实现核心校准、正式预处理或评估方法，没有新增论文事实。
- **[复现决策，已确认]** 模型固定为Qwen/Qwen2.5-1.5B、revision `8faed761d45a263340a0528343f099c05c9a4323`；原定model.safetensors为3,087,467,144 bytes、SHA256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`。来源仍限已批准镜像，未独立认证官方来源。
- **[工程门禁]** 模型LICENSE预期11,343 bytes、Git blob SHA1 `6634c8cc3133b3848ec74b9f275acaaa1ea618ab`。HTTP403不是hash mismatch：本次没有取得可对比的许可原文，不声称其内容被篡改或许可证禁止使用。
- **[待核验]** API的apache-2.0标签不替代完整许可及附加声明审查。LLM-Adapters代码许可证也不自动授予所有派生数据再发布权；本轮没有取得DATA_LICENSE或实际数据，不能声称数据许可、行数或拆分已验收。
- 保留Qwen非Instruct、28层/N250/Top8、Q/K/V rank16/alpha32、单卡8×8、8任务协议不变。未用另一模型、revision、数据源或量化方案替代。

## 执行命令

SSH使用既有别名、严格host key校验和无交互公钥认证，不公开实际入口或密钥。主要实际入口：

```bash
# 只读预检，资源获取尚未开始。
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ForwardAgent=no \
  -o PreferredAuthentications=publickey -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no autodl \
  '/root/miniconda3/bin/python -I -B -' \
  < .cache/m7-20261115-assets-01/precheck_remote.py

# 另获批准后，同一SSH前缀运行diagnose_mirror_readonly.py；2次初始小请求，
# 不跟随重定向，raw/LICENSE=403、revision API=200；不下载权重。
# 随后仅运行confirm_remote_readonly.py做最终状态确认。

.venv/Scripts/python.exe -I -S -B -X utf8 \
  .cache/m7-20261115-assets-01/test_precheck.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/Scripts/python.exe -I -B -X utf8 -m unittest discover -s tests -v
```

许可、API等下载响应仅作为不可信数据读取；原始小响应置于独立新目录，不执行响应、从下载目录启动解释器或启用trust_remote_code。没有放宽工具/SSH权限、重启实例或安装依赖。

## 测试结果（PASS / FAIL / NOT RUN）

| 状态 | 检查 | 实际结果与边界 |
|---|---|---|
| PASS | SSH及资源前置状态 | 到达远端Python，目标不存在，源码/clean/余量前置检查通过 |
| FAIL | 固定模型许可resolve路径 | HTTP403，exit1；不是完整预检PASS或hash mismatch |
| FAIL | 获批raw许可诊断 | HTTP403；未取得原文，继续条件未满足 |
| PASS（限元数据） | 获批固定revision API诊断 | 200，8,052 bytes，revision/LFS哈希一致；不替代许可原文核验 |
| PASS | 本地预检/诊断回归 | 8项，0.006s；覆盖精确blob、大小/哈希拒绝、403不重试、HTTPS/不跟随重定向、120秒参数与两次请求。mock不证明真实网络/信号可用 |
| PASS | 本地Windows项目离线CPU回归 | 185项，18.765s，无skip，不冒充新的Linux/真实1.5B测试 |
| PASS | 12:34最终只读状态 | 资源目录不存在，远端源码/平台包/venv集合/前次CPU结果未变 |
| NOT RUN | 模型/正式数据获取及内容验收 | 未下载；许可、数据行数/字段/抽样/真实分词/拆分仍未验收 |
| NOT RUN | 真实1.5B CPU结构前向 | 没有加载权重，没有观察28层实际输出 |
| NOT RUN | CUDA/pilot、正式校准/训练/评估 | 没有执行；不在本轮授权范围 |

此前Linux183项通过、2项真实tokenizer用例NOT RUN的结论保持原范围，本轮未重跑或升级它们。

## 运行日志和输出路径

本地证据在Git忽略的 [.cache/m7-20261115-assets-01/](../.cache/m7-20261115-assets-01/)：

- [资源授权](../.cache/m7-20261115-assets-01/authorization.json)、[诊断授权及停止条件](../.cache/m7-20261115-assets-01/diagnostic-authorization.json)。
- [初次预检stderr](../.cache/m7-20261115-assets-01/precheck-ssh.log)及空stdout；空文件不能作为有效JSON或成功报告。
- [诊断响应](../.cache/m7-20261115-assets-01/mirror-diagnosis.json)、[独立保存的API响应](../.cache/m7-20261115-assets-01/sources/model_api/response.txt)，SHA256为 `81cc293858646a07d85ec671172b5eb7f1db6b67a86bd92c88b880331098558a`。
- [最终只读状态](../.cache/m7-20261115-assets-01/final-state.json)、[8项测试日志](../.cache/m7-20261115-assets-01/precheck-tests.log)、[本地185项日志](../.cache/m7-20261115-assets-01/local-regression.log)。

原独立环境及资源工作计划位于数据盘；本轮计划的新资源目录尚未创建。最终可用34,269,646,848 bytes，仅12:34 UTC快照。没有权重、JSON数据、CPU结构结果、checkpoint或正式指标产物，不创建占位文件。远端代码仍为 `9051fdfe86379aa3502e66bf1a586d4eb00565ba`；本地报告基线为 `0d6062db37bee4b2d23dbf7f82ee2221619ac58c`，后续本地报告提交不自动部署或push。

## 未解决问题

1. 当前阻塞是固定模型LICENSE的交付HTTP403，不是SSH、公钥或依赖安装失败。两个已批准许可路径均未提供原文；API可达不说明权重路径可用，也不说明数据源不可用。
2. 本次按明确停止条件没有试其他来源、读取更多许可路径或启动大资源获取。需要明确可行且获准的许可/资源交付方式，不能自行降低“固定许可字节先通过”的门禁。
3. 除资源问题外，M7真实数据与分词/拆分、1.5B模块结构、CUDA/AMP/累积/GPU checkpoint实现、pilot及正式对照训练/评估仍有独立待完成任务。不能把解决403等同完成M7。

## 下一步建议（不自动执行）

先明确可接受的固定许可与模型交付路径，例如另行批准并验证官方来源，或由用户提供与固定blob匹配的许可原文及可达资源路径；仍须保留来源局限，不接收浮动revision或bin/pickle替代。资源路径通过后，再在明确的有限获取范围内核验模型/数据并做真实CPU结构门禁。GPU实现与pilot按计划另行测试、提交固定SHA和批准成本；不自动重试下载、改协议、运行CUDA/训练或push。
