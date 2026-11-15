# M7 首轮候选实验协议与执行门禁

面向复现实验实现者与执行者。本文件区分论文事实、候选复现决定和运行授权，不代表GPU作业已经开始。

**协议状态：2026-11-15用户已确认7项复现决定，协议记录已冻结；确认协议不构成下载、付费GPU或完整训练授权。M7整体未验收。** 后续独立环境已安装并通过适用Linux CPU验收（183项通过、2项真实tokenizer用例NOT RUN），见[CPU环境报告](reports/m7_environment_cpu_20261115.md)。模型LICENSE的镜像resolve/raw仍403，官方HF本轮小请求本地超时、远端不可达，见[历史资源报告](reports/m7_assets_blocked_20261115.md)。单独批准的数据门禁已取得固定DATA_LICENSE/代码LICENSE并部署审计代码，但辅助README超时后原窗口续接到期拒绝；九个JSON与实际审计仍NOT RUN，见[最新数据门禁报告](reports/m7_data_gate_20261115.md)。未下载1.5B权重/真实tokenizer，未运行真实1.5B前向、CUDA或正式训练。

机器可读配置：[qwen2_5_1_5b_commonsense_r16.json](configs/qwen2_5_1_5b_commonsense_r16.json)。预检入口：[check_m7_protocol.py](scripts/check_m7_protocol.py)。

## 1. 原文核查与目标

本轮从本地论文PDF按需重新提取并核查：

- **Section 5.1、Table 1（PDF第7–8页）**：Qwen2.5-1.5B、28层；commonsense_170k抽10,000条，math_10k抽5,000条；主表Adaptive使用N=250、Top-8；Full的Q/K/V覆盖全部28层。
- **Section 4.5、Appendix C.1–C.2（第7、18–19页）**：rank16、alpha32、AdamW lr1e-4、weight_decay0.01、betas(0.9,0.999)、eps1e-8；3 epochs，无warmup；每卡batch8、accumulation8；冻结FP16基础权重，FP32 adapter初始化与FP16 AMP；训练最大512、超长截断、短样本pad到8倍数。
- **Table 1、Appendix C.3（第8、19页）**：8项commonsense各1,000评估样本，OpenBookQA为500；无in-context examples，greedy temperature0，按任务答案抽取后算accuracy。数学任务使用CoT/任务专用解析。
- **Section 5.4（第9页）**：Random比较出现在Alpaca-ZH PPL的独立选层实验，不是主表commonsense同配置的已报告Random准确率。
- **参考文献[14]（第11页）**：训练数据引用LLM-Adapters，论文未给出本项目可直接采用的仓库commit、具体抽样索引、完整模板、loss mask或解析regex。

本轮选择**Table 1主比较的N=250**，而非Table 2跨模型配置的N=30。第一轮候选范围只做commonsense的Full / Random8 / Adaptive8，暂不做数学、LoRA+、DoRA或全11任务；这是项目范围决定，不冒充完整主表复现。

## 2. 固定来源候选

### 2.1 模型

- 仓库：Qwen/Qwen2.5-1.5B，**非Instruct**。
- 候选revision：`8faed761d45a263340a0528343f099c05c9a4323`。
- 权重：model.safetensors，3,087,467,144 bytes。
- LFS SHA256：`a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`。
- config Git blob SHA1：`169d606d5cf502a5628a52381586679a6d0caaab`，684 bytes；已实际读取小配置并核对Git blob头+字节哈希。
- 配置声明28层、hidden1536、12个Q heads、2个KV heads，EOS151643。配置原dtype为BF16，正式加载必须按论文明确转为FP16；不能沿用旧0.5B/BF16参数指纹。

**来源局限：**本地官方HF API连接超时；经服务器查询官方端点也失败，报Network is unreachable。以上候选元数据和小配置来自hf-mirror.com；匹配同镜像提供的Git/LFS标识不构成独立官方来源认证。实际权重尚未下载，必须获准后核对文件大小与LFS SHA256，不启用trust_remote_code，不接收pickle/bin替代。

候选元数据：[镜像API](https://hf-mirror.com/api/models/Qwen/Qwen2.5-1.5B?blobs=true)。这次查询没有下载3GB权重。

### 2.2 数据

候选公开来源：[AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/816657208af4db747803f87ba40a4c71383fed7a)，固定commit：

`816657208af4db747803f87ba40a4c71383fed7a`

此commit是本项目在2026-11-15解析的复现版本，**不是论文已披露的原始版本**。已读取GitHub树元数据，未下载训练或评估JSON。

| 来源文件 | 字节数 | 候选使用数量 |
|---|---:|---:|
| ft-training_set/commonsense_170k.json | 96,529,722 | 抽10,000训练，其中抽250校准 |
| dataset/boolq/test.json | 899,151 | 1,000 |
| dataset/piqa/test.json | 913,157 | 1,000 |
| dataset/social_i_qa/test.json | 843,748 | 1,000 |
| dataset/hellaswag/test.json | 11,099,046 | 1,000 |
| dataset/winogrande/test.json | 497,409 | 1,000 |
| dataset/ARC-Easy/test.json | 1,157,216 | 1,000 |
| dataset/ARC-Challenge/test.json | 622,510 | 1,000 |
| dataset/openbookqa/test.json | 202,545 | 500 |

每个文件的Git blob SHA1已固定在JSON草案。后续必须按 `SHA1("blob " + 十进制字节长度 + NUL + 原始字节)` 核对，并另存普通文件SHA256；二者不能混用。名称170k不是本轮实读行数，当前不声称已核验总体数量。

来源库的test.json路径不自动证明它与原始benchmark官方test split完全同一。实际下载后仍需审查来源/许可、记录行数和ID，并在训练、校准、评估间按ID、规范prompt字节、实际prefix IDs检查直接重叠；不能把哈希检查扩大成语义去重或预训练污染保证。代码许可证不自动授予所有派生数据再发布权，本项目不公开原始数据。

**后续工程状态（2026-11-15）：** 固定DATA_LICENSE已在本地/服务器从上述GitHub commit获取，19,790 bytes、Git blob `fc5b567da08fbcf8f2e2c3532b7a50a31d3de16a`一致；内容为Open Data Commons Attribution License（ODC-By），须遵守署名/通知条件。Section2.4不覆盖每条Contents的独立版权、隐私/人格等权利，Section4.2–4.3涉及数据库/Produced Work公开传播的通知；本项目未逐条清理原benchmark内容权利。代码LICENSE的11,396 bytes及固定blob也匹配，二者不混同。辅助README网络超时，九个JSON未请求；原窗口续接因到期拒绝，不能声称实际行数/字段/样本隔离通过。

[离线数据审计CLI](scripts/audit_m7_dataset.py)及[模块](src/adaptive_lora/m7_dataset.py)已实现并在完整SHA `2df16eb07bff7990cf9daf93bb189c9d91d06deb`经批准发布/部署。它仅核对固定JSON、字段、seed42逐样本索引与user-only prompt哈希交集，不分词或运行模型；即使以后返回TEXT_GATE_PASS_TOKENIZER_PENDING，真实tokenizer/截断/token身份仍须独立验收。当前仅合成夹具22项通过，**真实数据运行NOT RUN**；未来若发现重叠，保留原抽样并BLOCKED，不自动过滤或重抽。

## 3. 七项已由用户确认的复现决定

| ID | 候选决定 | 与论文/现有smoke的边界 |
|---|---|---|
| scope | 先做8项commonsense、rank16，Full/Random8/Adaptive8 | Random是项目扩展对照，不声称复现论文Random主表数字 |
| prompt | 规范化Alpaca纯文本前缀；instruction/input进入前缀，不用Qwen chat/system模板 | 与M2–M6原生chat模板不同，论文未给出完整模板 |
| supervision | prompt labels=-100，答案+显式EOS监督；完整序列右截断512，无可预测监督token时报错 | 论文未规定loss mask/截断方向；不静默丢样本或重抽 |
| sampling | train/cal/eval分别用局部seed42；先抽10,000训练，再从该子集抽250校准；test各抽1,000/500 | 论文给训练seed42和数量，未公开本次这些具体索引 |
| answer_parser | 续文标签大小写不敏感、有词边界；不同合法标签冲突判无效；8任务等权宏平均 | 不复制引用库的无边界首匹配regex；论文未公开其确切regex |
| single_gpu_batch | 单卡8×8=64；不自动降batch或换量化 | 明确不同于4卡DDP的256，显存不足先停止并申请变更 |
| mirror_source | 经镜像获取固定revision/LFS字节并保留来源局限 | 不把镜像完整性当作官方独立认证 |

2026-11-15，用户通过对话选择“确认此协议（推荐）”，确认上述7项决定；机器可读decision.status均已更新为confirmed。被审查草案的规范配置SHA256为 `e9ab92043b66674a78aa6b00461f022dfcf7f01290fd05cec98e0071d49d8625`；确认只改变这7个状态字段，其他配置不变，新的配置哈希由本轮确认后预检输出记录。

该确认不包含公开推送、依赖安装、模型/数据下载、付费GPU或完整训练授权。手工编辑成confirmed本身既不能证明用户授权，也不能开启任何执行入口。

### 3.1 Prompt与预测边界

候选规范文本由 [build_commonsense_prompt](src/adaptive_lora/m7_protocol.py) 生成，与LLM-Adapters原脚本的空白/缩进不保证逐字节相同。该函数没有answer/reference参数，固定尾部为Response标记和换行；训练、校准、生成共享前缀字节。

后续真实tokenizer路径须使用add_special_tokens=False、验证prefix与prefix+answer的BPE边界，答案末尾显式加EOS151643。`labels[i+1]`由`logits[i]`预测；padding位置labels=-100、attention_mask=0，不能因为pad与EOS同ID而误屏蔽有效EOS。对截断后没有可预测监督目标的样本应让数据准备失败，不静默改变10,000条清单。

校准在前缀最后有效位置读取next-token logits，不读teacher-forced答案；先确定未消融的t*，每层Attention/MLP分别置零并测同一t*概率，临时hook须包括异常路径清理。不能沿用0.5B/Top7选择、把整层置零或改为path patching。

当前模块**只实现纯文本模板与协议检查，不实现真实tokenization/截断、CUDA校准或训练**。超长校准/评估前缀的实际分布、保留Response边界与长度处置须在真实tokenizer数据门禁中核验；如既定规则不能处理实际样本，先报告并申请协议修订，不自行丢样本。

### 3.2 候选评分

仅把**生成token续文**交给 [extract_commonsense_answer](src/adaptive_lora/m7_protocol.py)，不能对含题面选项的整个解码输入做答案抽取。合法标签为：

- BoolQ：true/false。
- PIQA：solution1/solution2。
- Social IQa：answer1–answer3。
- HellaSwag：ending1–ending4。
- WinoGrande：option1/option2。
- ARC-Easy/Challenge：answer1–answer5。
- OpenBookQA：answer1–answer4。

只识别当前任务合法标签；重复同一标签允许，不同合法标签同时出现判ambiguous，无标签判no_label。逐样本保存状态，失败仍计入分母；不猜测yes/no映射、不跳过困难样本。参考标签须是来源规范标签，预测完成后按唯一ID连接。

完整8任务各先算accuracy，再等权取宏平均，不按7,500条整体微平均。小型合成评分测试只验证这一数学/边界，不是真实benchmark，也不替代数据拆分审计。

greedy固定do_sample=False、num_beams=1、temperature语义为0；**不要把logits除以0**。候选生成上限32 tokens是复现决定；引用库默认4 beams/temperature0.1不满足论文要求，不能照搬其生成器、旧8bit训练或trust_remote_code=True配置。

## 4. 单卡预算与尚未实现部分

- 校准评分forward预算：`250 × (1 + 2 × 28) = 14,250`；另加250次未消融恢复检查，总预算14,500次。**尚未执行，也不是耗时估计。**
- 从配置推导Q1536→1536、K/V1536→256，rank16每层106,496个LoRA元素；Full28估计2,981,888，K8估计851,968，相差3.5倍。**这些不是实际参数审计结果，也不表示总训练显存减少3.5倍。**
- 权重与所选数据文件合计元数据约3,200,231,648 bytes，未含tokenizer、依赖wheel、环境、缓存及结果。50 GiB数据盘已在[只读盘点](reports/autodl_postrestart_20261115.md)核验，后续仍须检查实际剩余空间。
- 当前M0–M6加载、校准、LoRA训练、checkpoint及生成路径有明确CPU smoke边界。不能给旧CLI加一个cuda参数或移除门禁就当成正式GPU路径。
- 后续CUDA/AMP/GradScaler、梯度累积归一化、批量padding、fresh-base三方法隔离、基础权重/评估不变、GPU checkpoint及显存/吞吐必须另作实现与验收。

## 5. 候选Linux/CUDA环境，不是安装授权

> 本节以下“当前”“未安装/未导入”等措辞保留协议冻结时的候选来源快照，不代表后续安装状态；实际独立venv的固定版本安装、导入、哈希锁和适用CPU验收见[环境CPU报告](reports/m7_environment_cpu_20261115.md)。本节仍不提供CUDA/训练授权。

当前远端仅元数据确认Python3.12.3、torch2.5.1+cu124、numpy2.1.3；本轮未import torch。Transformers/PEFT等所查项目包未在该解释器找到元数据。

候选直接依赖：[requirements-linux-cu124-m7.in](requirements-linux-cu124-m7.in)。保留平台原Miniconda环境，在获批后使用数据盘独立新环境：

- torch2.6.0+cu124：为满足Transformers5.22.2要求PyTorch>=2.6，同时保留CUDA12.4构建的候选最小升级，非论文原版本。
- Transformers5.22.2、PEFT0.21.2等沿用本地直接版本；但本地测试用torch2.15.0+cpu，不等于此Linux组合已验证。
- 已读取PyPI的Transformers/PEFT版本约束及[官方cu124 wheel索引](https://download.pytorch.org/whl/cu124/torch/)，固定CPython3.12/Linux x86_64 GPU wheel URL和SHA256。**没有下载wheel或执行安装。**
- 此文件不是完整传递依赖锁；获批后仍要记录pip解析/安装报告、实际索引、freeze、pip check、框架导入和全部适用测试，最后才产生Linux锁定快照。
- 不自动升级驱动、不覆盖系统Python或原环境、不启用量化。若依赖解析或实际导入失败，保留日志并修订候选方案，不能说已准备就绪。

## 6. 离线预检及返回码

即使没有安装PyTorch等第三方包也可运行；检查器从自己的已审查仓库加载标准库协议模块，不将配置/下载目录加入搜索路径：

```bash
.venv/Scripts/python.exe -I -S -B -X utf8 scripts/check_m7_protocol.py \
  --config configs/qwen2_5_1_5b_commonsense_r16.json \
  --output-dir results/m7-protocol-review-new
```

- 输出目录必须不存在；只写配置快照和preflight.json，不覆盖历史结果。
- pending决定存在：返回2、PENDING_DECISIONS，这是预期的审查阻塞，不是训练失败。
- 配置记录全部confirmed：返回0、REVIEWED_NOT_EXECUTABLE，仍然execution_authorized=false、runtime_validated=false、training_started=false。
- 非法配置/旧目录/写入失败：返回2并明确ERROR；不能只凭返回2认定已产生有效待审查报告。
- 严格拒绝重复JSON键、NaN、浮动revision、错误哈希形状、额外字段、旧模型/Top7、量化、改batch/精度或非greedy配置。
- config_sha256覆盖规范JSON含决策状态，不含外部源码和依赖字节；后者单独记录。Git是运行前基线+dirty，不伪装为尚未生成的提交。哈希都不是来源签名。

## 7. 后续门禁与审批边界

完成当前协议审查后停止，不自动进入下列步骤：

1. **代码交付**：经明确公开推送/部署批准，再发布分支并远端检出完整40位SHA；不传GitHub写凭据到服务器。
2. **依赖准备**：批准新隔离环境安装范围后执行；验证候选组合并归档Linux快照，保留平台原环境。
3. **资源下载**：模型/训练和评估数据须单独批准，使用固定版本、独立新目录及字节核验；检查许可，不覆盖已有缓存。
4. **真实数据与结构门禁**：固定训练/校准/评估IDs和tokenizer协议，确认28层Attention/MLP实际返回值、Q/K/V投影和LoRA白名单。
5. **CUDA实现及最短pilot**：先本地测试核心数学/异常路径，再提交新SHA；明确付费GPU时长/步骤上限后运行，记录峰值显存/吞吐，失败不自动降配或重试。
6. **正式运行**：根据pilot成本估计再次获准，执行N250校准、Full/Random8/Adaptive8统一3epochs及正式评估。单卡与论文四卡差异、解析/模板差异和所有失败如实保留。
