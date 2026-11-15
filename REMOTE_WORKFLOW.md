# 本地开发与远程固定 Commit 实验流程

本流程用于 AutoDL 单卡实验的代码交付，不是 GPU 训练授权。模型/数据下载、关键依赖安装、付费 GPU 作业、扩容、覆盖/删除产物及公开推送仍按 [CLAUDE.md](CLAUDE.md) 审批；已完成的 M6 CPU smoke 不能直接当作正式 GPU 训练入口。

## 1. 职责与版本边界

- 本地完成实现、中文注释、测试和结果核对后创建 Git commit；不得把 dirty 工作区描述成某个干净 commit。
- 远程服务器只获取已发布且明确批准的**完整40位 commit SHA**。分支名仅用于获取对象，不能代替实验版本。
- 用户于2026-11-15明确要求后续采用“本地开发完成后提交 → 远程获取指定commit → 执行实验”的流程。执行由助手负责，访问、费用与关键实验选择由用户批准。
- 不在远程临时修改算法代码后直接训练。远程发现问题时，回到本地修复、测试、提交，再部署新SHA。
- 正式运行前检查工作区干净；运行期间不切换其代码目录的版本。不同并行运行如有需要，使用彼此隔离的检出目录。

## 2. 本地提交门禁

1. 阅读论文规格、当前里程碑及实际依赖；仅完成当前获批任务。
2. 执行适用测试、依赖检查、`git diff --check`，明确 PASS/FAIL/NOT RUN。
3. 检查待提交文件，不包含密钥、SSH配置、连接凭据、完整数据/校准输入、模型、checkpoint、缓存或原始结果。
4. 用明确的源码/文档路径暂存；检查 `git diff --cached --stat` 与暂存区内容，再创建提交。
5. 保存 `git rev-parse HEAD` 的完整输出和提交摘要。公开远端推送另行审批，不把本地提交当作已经推送。

模型/数据/revision清单与完整结果保存在被Git忽略的目录；公开报告只保留经检查的摘要、统计和哈希。哈希不是脱敏、来源签名或公开数据的授权。

## 3. 远程首次准备

先经SSH执行只读巡检，确认实际GPU/驱动、Python/PyTorch/CUDA、依赖、CPU配额、可用内存、系统盘和数据盘挂载/剩余空间。SSH使用本机配置的身份认证并校验已知host key；不把私钥、密码、SSH配置或连接地址放进公开仓库，不启用不必要的agent forwarding。

服务器配置以实际巡检为准；用户提供的硬件说明仅是待核验计划。优先将模型、数据、环境和结果放在确认的数据盘，不能把30GB系统盘误当成空闲30GB。首次创建远程项目目录、安装依赖、下载和运行作业应在相应审批后进行，不覆盖已有目录。

当前本地验证环境为Windows CPU，锁定清单不能不经核对直接安装到Linux/CUDA。特别是本地Transformers5.22.2要求PyTorch>=2.6，平台提供的2.5.1不能直接与其配套；实验环境需单独规划、验证并记录，不自动修改平台原环境。

## 4. 获取并检出批准的版本

以下仅为**审批完成后的操作模板，本文件不代表已执行部署**。`COMMIT`必须替换为实际批准的完整SHA；`REPO`应为事先确认的数据盘目录。

首次部署可以在不存在的新目录克隆公开代码，无需向服务器传入GitHub写凭据：

```bash
REPO=/path/on/data-disk/adaptive-lora
# 必须先确认目录不存在；不要覆盖已有工作区。
test ! -e "$REPO" || exit 1
git clone --no-checkout https://github.com/Hello-Charles/adaptive-lora.git "$REPO"
```

在新克隆或已确认可复用且干净的仓库中检出指定版本：

```bash
cd "$REPO" || exit 1
COMMIT=REPLACE_WITH_APPROVED_40_HEX_SHA
# 先检查身份与工作区，不能用reset/clean丢弃远端改动。
test "$(git remote get-url origin)" = https://github.com/Hello-Charles/adaptive-lora.git || exit 1
test -z "$(git status --porcelain)" || exit 1
case "$COMMIT" in *[!0-9a-f]*|"") exit 1 ;; esac
test "${#COMMIT}" -eq 40 || exit 1
git fetch origin || exit 1
# 确认是已获取的origin分支历史中的commit，不自动切换到最新分支。
git cat-file -e "$COMMIT^{commit}" || exit 1
test -n "$(git for-each-ref --format='%(refname)' --contains="$COMMIT" refs/remotes/origin)" || exit 1
git checkout --detach "$COMMIT" || exit 1
test "$(git rev-parse HEAD)" = "$COMMIT" || exit 1
test -z "$(git status --porcelain)" || exit 1
```

不单独执行不受约束的 `git pull` 后直接训练；不能以“分支大概没变”代替SHA核对。若远程有未知改动或目标commit尚未发布，停止并报告，不自动 `reset --hard`、`clean`、覆盖或从另一来源复制代码。

## 5. 实验门禁与记录

- 先通过依赖导入、模型实际结构、数据来源/拆分与最短GPU pilot门禁，再申请完整运行。
- M7正式配置的模型/data revision、训练与校准样本、prompt/loss mask、有效batch、精度和任务评估须冻结。0.5B/Top7的旧校准不能当作1.5B/Top8结果。
- 记录完整代码commit、dirty状态、运行前后源码字节哈希、依赖版本、模型/data revision、配置、sample IDs、选层seed/IDs、训练轨迹、实际参数量、峰值显存、吞吐/耗时和结果。
- 单卡/四卡的有效batch不同，Full/K-layer参数预算也不同；不以更换训练设置来人为匹配论文数值。
- 每次用新实验目录，成功标记最后写入，失败保留部分证据但不冒充完成。训练前后再次检查代码SHA和工作区状态。
- Git commit校验代码版本；运行时哈希保存实际字节。不同平台换行和不同依赖的差异须如实记录，不把历史Windows结果的字节指纹强行当作Linux结果。

## 6. 当前状态

项目M0–M6已完成既定门禁，详见 [REPRO_PLAN.md](REPRO_PLAN.md)。远程访问/部署状态以每次预检报告和真实命令为准；本流程本身不声称远程GPU、依赖或正式训练已通过。
