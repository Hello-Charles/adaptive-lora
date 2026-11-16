"""M7 数据到 micro-batch 的装配层：按 epoch 定序、切 micro-batch、保留样本身份。

职责边界：

- 只做**确定性装配**：把已核验的抽样行经真实 tokenizer 变成监督样本，再按固定策略
  切成 micro-batch 序列。不加载模型、不训练、不做前向、不写 checkpoint。
- 不替论文未规定的细节另设隐式默认：训练顺序与末组余量策略都是**已确认的复现决策**，
  以 `CONFIRMED_ORDER_POLICY` / `CONFIRMED_FINAL_PARTIAL_GROUP` 两个常量固定，并已写入
  配置 `training.training_order` / `training.final_partial_group`。二者都可在调用时覆盖以
  便对照实验，但覆盖值必须显式给出并记录，不能让"换个默认值"悄悄改变实际参与更新的样本。

关键前提（代码本身无法表达，必须由调用者保证）：

- 传入的 `rows` 与 `records` 必须来自同一次已通过文本门禁的抽样，且 `records[i]`
  对应 `indices[i]` 所指的那一行。本模块会按 `records` 的顺序装配，不做隐式重排。
- `records` 必须携带 `sample_id`；`build_supervised_example` 本身不产出该字段，因此
  身份由本层补上。缺身份会让后续任何失败都无法定位到具体样本。
- 训练顺序是论文**未规定**的实现细节。已确认策略 `none`：每个 epoch 都沿用抽样顺序
  （该顺序本身来自 seed 42 的不放回抽样）。另一个可选值 `per_epoch_shuffle` 会用
  `train_seed` 派生的独立局部 RNG 逐 epoch 打乱。两者都不是"论文原样"，只是可复现的
  工程选择；`none` 的含义是三个 epoch 的访问顺序**完全相同**，这直接影响 `drop` 策略
  下被丢弃的是哪批样本（见 `iter_epoch_micro_batches`）。
"""

import random

from adaptive_lora.m7_tokenization import (
    M7_EOS_TOKEN_ID,
    M7_MAX_LENGTH,
    M7_PAD_MULTIPLE,
    build_supervised_example,
)
from adaptive_lora.m7_training import CONFIRMED_FINAL_PARTIAL_GROUP, plan_groups

# 训练顺序策略。论文只写"3 epochs"，未公开样本顺序，故这是显式的复现决策。
ORDER_POLICIES = ("none", "per_epoch_shuffle")

# 已确认的复现决策（2026-11-16，用户选择默认）：每个 epoch 沿用抽样顺序，不打乱。
CONFIRMED_ORDER_POLICY = "none"


def epoch_order(count, *, epoch, policy=CONFIRMED_ORDER_POLICY, seed=42):
    """返回该 epoch 内样本的访问顺序（0..count-1 的一个排列）。

    `none`：每个 epoch 都按传入顺序，顺序完全来自抽样本身，最易审计。
    `per_epoch_shuffle`：每个 epoch 用 `seed` 与 `epoch` 派生的**独立局部 RNG** 打乱。
    使用局部 RNG 而不是全局 `random`，是为了不受调用方此前是否调用过 `random.seed`
    影响；否则同一次实验在不同调用路径下会得到不同顺序。

    `epoch` 从 1 开始计数；两个 epoch 之间必须能区分（否则打乱等于没打乱）。
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer starting at 1")
    if policy not in ORDER_POLICIES:
        raise ValueError(f"Unknown order policy: {policy!r}")
    positions = list(range(count))
    if policy == "none":
        return positions
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    # 派生 per-epoch 种子：Python 3.12 的 random.Random 只接受 int/float/str/bytes/bytearray，
    # 不接受 tuple，因此用整数混合而不是直接把 (seed, epoch) 传进去。
    # 取黄金比例奇数常量做混合，使相邻 epoch 的种子在整个 32 位空间内分散，
    # 并掩到 [0, 2**32) 以保持可跨平台复现的整数范围。
    derived_seed = ((seed * 0x9E3779B1) ^ (epoch * 0x85EBCA6B)) & 0xFFFFFFFF
    random.Random(derived_seed).shuffle(positions)
    return positions


def prepare_supervised_examples(rows, records, tokenizer, *, eos_token_id=M7_EOS_TOKEN_ID,
                                max_length=M7_MAX_LENGTH, pad_multiple=M7_PAD_MULTIPLE):
    """把已核验的行批量转成监督样本，并为每条补上 `sample_id`。

    输入 `rows[i]` 与 `records[i]` 必须一一对应。训练目标是 `output` 字段（评估用
    `answer`），这里只取 `output`，不会把评估参考答案当成训练目标。

    任一行的 BPE 边界无法判定、或截断后没有可预测监督 token 时，`build_supervised_example`
    会抛错。本函数**不捕获、不跳过、不重抽**：一条坏样本应让整次数据准备失败，而不是
    静默得到 9,999 条 —— 后者会破坏"10,000 条"这一已确认的抽样数量。

    返回值是样本列表，每项是 `build_supervised_example` 的结果加上 `sample_id`、
    `epoch_index` 之外的静态字段；不写文件、不改输入。
    """
    rows, records = list(rows), list(records)
    if not rows or len(rows) != len(records):
        raise ValueError("rows and records must be non-empty and of equal length")
    prepared, seen = [], set()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    for position, (row, record) in enumerate(zip(rows, records)):
        identifier = record.get("sample_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Record {position} lacks a non-empty sample_id")
        if identifier in seen:
            raise ValueError(f"Duplicate sample_id: {identifier}")
        seen.add(identifier)
        answer = row.get("output")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Row {position} requires a non-empty training output")

        prefix = _prompt_for(row)
        example = build_supervised_example(
            tokenizer, prefix, answer, eos_token_id=eos_token_id,
            max_length=max_length, pad_multiple=pad_multiple, pad_token_id=pad_token_id,
        )
        example["sample_id"] = identifier
        example["source_index"] = record.get("source_index")
        example["prompt_sha256"] = record.get("prompt_sha256")
        prepared.append(example)
    return prepared


def _prompt_for(row):
    """按协议渲染 user-only 前缀；import 放在函数内避免模块级循环依赖。"""
    from adaptive_lora.m7_protocol import build_commonsense_prompt

    return build_commonsense_prompt(row["instruction"], row.get("input", ""))


def iter_epoch_micro_batches(examples, *, epoch, micro_batch_size,
                             order_policy=CONFIRMED_ORDER_POLICY, seed=42,
                             final_partial_group=CONFIRMED_FINAL_PARTIAL_GROUP):
    """按该 epoch 的顺序产出 micro-batch（每个是样本列表），保持累积组的边界信息。

    返回 `(micro_batches, plan)`：`micro_batches` 是列表的列表，`plan` 汇总本 epoch 的
    分组情况。**不在这里切累积组**——累积组由 `m7_training.run_accumulation` 依据同一
    `final_partial_group` 策略处理，避免两处各自解释余量。

    末组余量策略（已确认 `drop`）在此的作用是**真正丢弃**末尾不足一批的样本，并把被丢弃
    的 `sample_id` 全部写进 `plan["dropped_sample_ids"]`。丢弃必须有清单：否则"用了
    10,000 条"是一句无法核对的话，事后也没法判断某个 epoch 少训了哪几条。
    """
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError("micro_batch_size must be a positive integer")
    if final_partial_group not in ("error", "step", "drop"):
        raise ValueError(f"Unknown final partial group policy: {final_partial_group!r}")
    examples = list(examples)
    if not examples:
        raise ValueError("iter_epoch_micro_batches requires at least one example")
    order = epoch_order(len(examples), epoch=epoch, policy=order_policy, seed=seed)

    batches, current = [], []
    for position in order:
        current.append(examples[position])
        if len(current) == micro_batch_size:
            batches.append(current)
            current = []
    # 记下落在末尾不完整批里的样本，无论最终丢弃还是保留，都要能追溯。
    trailing = list(current)
    if trailing:
        if final_partial_group == "error":
            raise ValueError(
                f"last micro-batch of epoch {epoch} would hold {len(trailing)} of "
                f"{micro_batch_size} samples; refusing to run a short micro-batch under the "
                f"'error' policy"
            )
        if final_partial_group == "step":
            batches.append(trailing)

    used_ids = [example["sample_id"] for batch in batches for example in batch]
    dropped_ids = [example["sample_id"] for example in trailing] if final_partial_group == "drop" else []
    return batches, {
        "epoch": epoch,
        "order_policy": order_policy,
        "final_partial_group": final_partial_group,
        "samples": len(examples),
        "samples_used": len(used_ids),
        "samples_dropped": len(dropped_ids),
        "dropped_sample_ids": dropped_ids,
        "micro_batch_size": micro_batch_size,
        "micro_batches": len(batches),
        # 逐批的实际大小。累积层据此求和得到真实参与训练的样本数；若改用
        # `批数 × micro_batch_size` 估算，`step` 策略下的末尾短批会被多算。
        "micro_batch_sizes": [len(batch) for batch in batches],
        "full_micro_batches": sum(len(batch) == micro_batch_size for batch in batches),
        "short_micro_batch_sizes": [len(batch) for batch in batches if len(batch) != micro_batch_size],
        "sample_ids_sha256": _ids_sha256([example["sample_id"] for example in examples]),
        "order_sha256": _ids_sha256([examples[position]["sample_id"] for position in order]),
        "used_order_sha256": _ids_sha256(used_ids),
        "dropped_sample_ids_sha256": _ids_sha256(dropped_ids),
    }


def plan_epoch_accumulation(epoch_plan, *, accumulation,
                            final_partial_group=CONFIRMED_FINAL_PARTIAL_GROUP):
    """在 epoch 计划之上叠加累积组划分，返回各组的 micro-batch 数与样本数。

    这是把"数据装配"与"梯度累积"两层的余量事实统一到一处核对的地方：
    micro-batch 层是否整除、累积组层是否整除，必须同时成立才算计划完整。

    已确认 `drop`：末尾不足一组的 micro-batch 不参与任何 optimizer step。返回值把
    **本 epoch 实际用于训练的样本数**与**被丢弃的数量**分开列出，禁止只看
    `samples` 字段就宣称"全部样本都已训练"。
    """
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("accumulation must be a positive integer")
    micro = epoch_plan["micro_batches"]
    groups = plan_groups(micro, accumulation, final_partial_group)
    used_micro = sum(groups)
    dropped_micro = micro - used_micro
    batch_size = epoch_plan["micro_batch_size"]
    # 按**实际**每批大小求和，而不是"批数 × 名义批大小"。`step` 策略下末尾那个短批
    # 只有 2 条，若用名义大小 8 估算会把参与训练的样本数多算 6 条。
    sizes = epoch_plan["micro_batch_sizes"]
    if len(sizes) != micro:
        raise ValueError("epoch_plan micro_batch_sizes length does not match micro_batches")
    used_samples = sum(sizes[:used_micro])
    dropped_samples = sum(sizes[used_micro:])
    return {
        "epoch": epoch_plan["epoch"],
        "order_policy": epoch_plan["order_policy"],
        "final_partial_group": final_partial_group,
        "micro_batches": micro,
        "micro_batches_used": used_micro,
        "micro_batches_dropped": dropped_micro,
        "accumulation": accumulation,
        "accumulation_groups": len(groups),
        "micro_batches_per_group": groups,
        "optimizer_steps": len(groups),
        "samples": epoch_plan["samples"],
        "samples_used_by_micro_batches": epoch_plan["samples_used"],
        "samples_dropped_by_micro_batches": epoch_plan["samples_dropped"],
        "samples_used_by_accumulation": used_samples,
        "samples_dropped_by_accumulation": dropped_samples,
        "samples_unaccounted": (epoch_plan["samples"] - used_samples - dropped_samples),
        "micro_batch_size": batch_size,
        "samples_effective_batch_nominal": batch_size * accumulation,
        "last_group_micro_batches": groups[-1] if groups else None,
        "divides_evenly": micro % accumulation == 0,
    }


def _ids_sha256(values):
    """对标识序列做规范 JSON SHA-256；用于固定顺序证据，不是脱敏或去重手段。"""
    import hashlib
    import json

    raw = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
