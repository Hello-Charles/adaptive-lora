"""M7 训练执行层：设备/精度解析与梯度累积引擎；不加载模型、不下载、不启动正式训练。

职责边界：

- 只提供**可测试的执行机制**——设备选择、AMP 解析、累积组切分与优化步进。
- 不负责构造模型、tokenizer、数据或 checkpoint；这些由调用者注入。
- 不改变论文规定的超参数（micro-batch 8、accumulation 8、FP16 AMP 等），只校验。

关键前提（代码本身无法表达，必须由调用者保证）：

- `trainable` 必须与 `optimizer.param_groups` 完全同一批参数对象；冻结的基础权重不得入优化器。
- `forward_loss` 返回**该 micro-batch 的标量均值 loss**；累积时的除法由本模块统一负责，
  调用者不得再自行除以累积步数，否则梯度会被缩小两次。
- 传入的 micro-batch 顺序即训练顺序；本模块不重排、不重抽、不跨 epoch 混洗。

关于 pad 与 EOS 同 ID：本模块不检查 labels 语义，`forward_loss` 必须用 labels=-100 忽略
prompt 与 padding，不能靠 token id 判断有效性。
"""

import math

import torch

# 论文 Appendix C.1–C.2 的训练精度设定（[论文]）：FP16 基础权重、FP32 adapter、FP16 AMP。
SUPPORTED_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# 累积组末尾不足一个完整组时的处置策略：
# - "drop"：丢弃末尾不足一组的 micro-batch，不为其步进。
# - "step"：按**该组实际大小**求均值后步进，有效 batch 变小但梯度仍是均值。
# - "error"：直接报错。
FINAL_GROUP_POLICIES = ("error", "step", "drop")

# 已确认的复现决策（2026-11-16，用户选择 drop）：末组余量丢弃。
# 事实依据：10,000 条 / micro-batch 8 = 1,250 个 micro-batch；1,250 / 累积 8 = 156 组余 2 个
# micro-batch（16 条样本）。drop 意味着**每个 epoch 都固定少用末尾 16 条**，3 epochs 共 48 条
# 样本从未参与训练。这不是四舍五入，必须在报告中如实记录，不能写成"已使用全部 10,000 条"。
CONFIRMED_FINAL_PARTIAL_GROUP = "drop"


def resolve_device(requested):
    """解析训练设备；请求 CUDA 但不可用时**报错而不静默回退 CPU**。

    静默回退会让一次自称的"GPU 训练"实际在 CPU 上完成，产出的耗时、显存与吞吐证据
    全部失真，因此必须显式失败。返回 torch.device，不改变全局默认设备。
    """
    if not isinstance(requested, str) or requested not in ("cpu", "cuda"):
        raise ValueError("Device must be 'cpu' or 'cuda'")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; refusing to silently fall back to CPU")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp(device, amp_dtype, *, enabled=True):
    """返回 `(autocast_kwargs, scaler)`；CPU 上显式禁用 AMP 而不是假装启用。

    - CPU：FP16 autocast 没有对应的加速路径，与论文的 CUDA AMP 语义不同，故 `enabled=False`。
    - CUDA + FP16：需要 GradScaler 防止小梯度下溢。
    - BF16：动态范围足够，不需要梯度缩放，`scaler` 关闭以免引入多余状态。

    `autocast_kwargs` 可直接作为 `torch.amp.autocast(**kwargs)` 的参数；即使 disabled 也保留
    device_type/dtype，便于在报告中如实记录"请求了什么、实际是否生效"。
    """
    if not isinstance(amp_dtype, str) or amp_dtype not in SUPPORTED_AMP_DTYPES:
        raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a bool")
    dtype = SUPPORTED_AMP_DTYPES[amp_dtype]
    active = enabled and device.type == "cuda"
    autocast = {"device_type": device.type, "dtype": dtype, "enabled": active}
    # 只有 CUDA+FP16 需要缩放；此处即使 enabled=False 也构造 GradScaler，
    # 以便调用方走同一条 scale/unscale 代码路径而不分支。
    scaler = torch.amp.GradScaler(device.type, enabled=active and dtype is torch.float16)
    return autocast, scaler


def plan_groups(count, accumulation, policy=CONFIRMED_FINAL_PARTIAL_GROUP):
    """把 `count` 个 micro-batch 切成累积组，返回每组大小列表。

    末尾不足一组的处置由 `policy` 决定，且必须在任何前向发生前确定，避免"跑到最后才发现
    策略冲突"：

    - `drop`（已确认决策）：只返回完整组，末尾余量被丢弃。**丢弃量不在本函数的返回值里**，
      调用方必须用 `count - sum(返回值)` 自行记账并写入报告，否则会把"丢了 2 个 micro-batch"
      记成"跑完了 1,250 个"。
    - `step`：末组按实际大小步进，有效 batch 变小但梯度仍是均值。
    - `error`：不整除直接报错。
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("accumulation must be a positive integer")
    if policy not in FINAL_GROUP_POLICIES:
        raise ValueError(f"Unknown final partial group policy: {policy!r}")
    full, remainder = divmod(count, accumulation)
    groups = [accumulation] * full
    if remainder:
        if policy == "error":
            raise ValueError(
                f"{count} micro-batches do not divide into groups of {accumulation} "
                f"({full} full groups plus {remainder} left over); "
                f"choose final_partial_group='step' to step on the smaller group, "
                f"or 'drop' to discard the leftover — refusing to silently change the effective batch"
            )
        if policy == "step":
            groups.append(remainder)
        # drop：不追加末尾余量。返回的 groups 只覆盖前 sum(groups) 个 micro-batch。
    return groups


def describe_batch_plan(total_samples, micro_batch_size, accumulation, *, epochs=None,
                        policy=CONFIRMED_FINAL_PARTIAL_GROUP):
    """汇总有效 batch、每 epoch 的 micro-batch 数与不整除余量，供报告如实记录。

    只做算术，不触碰模型或数据。论文的 10,000 条训练样本不能被 8×8=64 整除
    （10,000 = 64×156 + 16），因此每个 epoch 都会出现一个不完整的累积组；本函数把这一
    事实显式暴露出来，而不是让执行期才暴露。

    已确认策略为 `drop`，故 `samples_used_*` 与 `samples_dropped_*` 按"末尾不足一组的
    micro-batch 被丢弃"计算。3 epochs 下 10,000 条中有 9,984 条/epoch 参与训练，
    共 **48 条样本从未被用于更新权重**。这个数字必须写进实验记录。
    """
    for name, value, minimum in (("total_samples", total_samples, 1),
                                 ("micro_batch_size", micro_batch_size, 1),
                                 ("accumulation", accumulation, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be a positive integer")
    if policy not in FINAL_GROUP_POLICIES:
        raise ValueError(f"Unknown final partial group policy: {policy!r}")
    batch = micro_batch_size * accumulation
    per_epoch_micro = total_samples // micro_batch_size
    remainder_micro = total_samples % micro_batch_size
    full_groups, leftover = divmod(per_epoch_micro, accumulation)
    group_sizes = [accumulation] * full_groups
    if leftover and policy == "step":
        group_sizes.append(leftover)
    used_micro = sum(group_sizes)
    # micro-batch 层与累积组层各自可能丢样本：前者是末尾不完整批，后者是末尾不完整组。
    dropped_micro_by_micro = remainder_micro if policy == "drop" else 0
    dropped_micro_by_accumulation = per_epoch_micro - used_micro
    samples_dropped_by_micro = dropped_micro_by_micro
    samples_dropped_by_accumulation = dropped_micro_by_accumulation * micro_batch_size
    result = {
        "total_samples": total_samples,
        "micro_batch_size": micro_batch_size,
        "accumulation": accumulation,
        "final_partial_group_policy": policy,
        "effective_batch_size": batch,
        "samples_per_optimizer_step": batch,
        "micro_batches_per_epoch": per_epoch_micro,
        "full_groups_per_epoch": full_groups,
        "leftover_micro_batches_per_epoch": leftover,
        "samples_left_over_per_epoch": leftover * micro_batch_size + remainder_micro,
        "samples_never_used_per_epoch": samples_dropped_by_micro,
        "optimizer_steps_per_epoch": len(group_sizes),
        "samples_used_per_epoch": used_micro * micro_batch_size,
        "samples_dropped_per_epoch": samples_dropped_by_accumulation + samples_dropped_by_micro,
        "divides_evenly": leftover == 0 and remainder_micro == 0,
    }
    if epochs is not None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        steps = len(group_sizes) * epochs
        dropped = result["samples_dropped_per_epoch"] * epochs
        result.update({
            "epochs": epochs,
            "optimizer_steps_total": steps,
            "samples_used_total_visits": result["samples_used_per_epoch"] * epochs,
            # 计数单位是"样本-次"而不是"不同样本"：每 epoch 丢哪 16 条取决于训练顺序策略。
            # 在已确认的 training_order='none' 下每个 epoch 的访问顺序完全相同，所以被丢弃的
            # 永远是**同一批 16 条**样本，它们从未参与任何权重更新；若改用 per_epoch_shuffle，
            # 则 48 个"样本-次"可能来自最多 48 条不同样本。两种情形都不等于"48 条不同样本，
            # 都不能写成"已使用全部 10,000 条"。
            "samples_dropped_total_visits": dropped,
            "samples_dropped_fraction_total": dropped / (total_samples * epochs),
            "dropped_distinctness_note": ("sample-visits, not distinct samples; under "
                                          "training_order='none' the same samples are dropped every epoch"),
        })
    return result


def _gradient_norm(parameters):
    """以 float64 汇总 adapter 梯度 L2 范数；不修改梯度，不把 NaN 当有限值。"""
    total = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            raise ValueError("Missing gradient on a trainable parameter")
        if not torch.isfinite(gradient).all().item():
            raise ValueError("Nonfinite gradient on a trainable parameter")
        total += gradient.detach().double().pow(2).sum().item()
    return math.sqrt(total)


def run_accumulation(*, optimizer, trainable, micro_batches, forward_loss, accumulation,
                     device, autocast, scaler, final_partial_group=CONFIRMED_FINAL_PARTIAL_GROUP,
                     on_step=None, on_micro_batch=None):
    """执行梯度累积训练，返回逐步证据；不创建 checkpoint、不改 dataloader、不调学习率。

    数学约定（必须与论文的有效 batch 一致）：

    - 每个累积组内的 loss 先除以**该组实际 micro-batch 数**再反向，使累积梯度等于
      "整组平均 loss" 的梯度，而不是组内求和。
    - 只有组末才 `optimizer.step()`；组内 `zero_grad` 不平账，否则累积失效。
    - AMP 走 `scaler.scale(...).backward()` → `unscale_` → 检查有限 → `step` → `update`。
      unscale 之后再检查梯度，才能看到真实量级，而不是放大后的数值。

    异常路径一律清梯度；不回滚已完成的 optimizer 步，调用者应弃用部分训练状态。
    `on_step` 在步进与校验后收到纯标量快照，供上层写日志。
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("Accumulation requires enabled autograd outside inference_mode")
    trainable = tuple(trainable)
    if not trainable or any(not isinstance(parameter, torch.nn.Parameter) for parameter in trainable):
        raise ValueError("trainable must be a non-empty sequence of Parameters")
    if any(parameter.device.type != device.type for parameter in trainable):
        raise ValueError("All trainable parameters must live on the resolved device")
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise ValueError("Optimizer parameters differ from the audited trainable whitelist")
    if not callable(forward_loss):
        raise ValueError("forward_loss must be callable")

    micro_batches = list(micro_batches)
    groups = plan_groups(len(micro_batches), accumulation, final_partial_group)
    # drop 策略下只训练完整组覆盖的前缀；被丢弃的 micro-batch 数与其样本身份必须
    # 如实计入返回值并写进训练日志，否则"用了 10,000 条"是错误陈述。
    used = sum(groups)
    dropped = len(micro_batches) - used

    steps, position = [], 0
    try:
        for number, size in enumerate(groups, start=1):
            group = micro_batches[position:position + size]
            position += size
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset, micro in enumerate(group):
                with torch.amp.autocast(**autocast):
                    loss = forward_loss(micro)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise ValueError("forward_loss must return a scalar tensor")
                if not torch.isfinite(loss).all().item():
                    raise ValueError("Nonfinite training loss")
                # 除以该组实际大小：保证累积梯度是组内均值而非求和。
                scaler.scale(loss / size).backward()
                losses.append(loss.detach().float().item())
                if on_micro_batch is not None:
                    on_micro_batch({"group": number, "offset": offset, "size": size,
                                    "loss": float(losses[-1])})
            # unscale 之后梯度才回到真实量级，此时才能判断有限性与范数。
            scaler.unscale_(optimizer)
            # 先确认每个可训练参数都真的拿到了梯度：`.grad is None` 说明前向没有连到该参数
            # （例如 loss 与模型无关），此时任何范数/非零统计都会以 TypeError 崩掉，
            # 掩盖真正的原因。这一条必须排在其他梯度检查之前。
            missing = [index for index, parameter in enumerate(trainable) if parameter.grad is None]
            if missing:
                raise ValueError(
                    f"Missing gradient on {len(missing)} of {len(trainable)} trainable parameters "
                    f"(first at index {missing[0]}); the accumulated loss is not connected to them"
                )
            # 再判"是否全为零"：否则范数检查会先命中，使更具体的诊断信息不可达。
            nonzero = sum(torch.count_nonzero(parameter.grad).item() > 0 for parameter in trainable)
            if nonzero == 0:
                raise ValueError("Every accumulated gradient is exactly zero")
            norm = _gradient_norm(trainable)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Accumulated gradient must be finite and nonzero")
            scaler.step(optimizer)
            scaler.update()
            if any(not torch.isfinite(parameter).all().item() for parameter in trainable):
                raise ValueError("Nonfinite trainable weights after optimizer step")
            record = {
                "step": number,
                "micro_batches_in_group": size,
                "mean_loss": math.fsum(losses) / len(losses),
                "gradient_l2_norm": norm,
                "gradient_tensors": len(trainable),
                "nonzero_gradient_tensors": nonzero,
                "amp_enabled": bool(autocast.get("enabled")),
                "scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
            }
            steps.append(record)
            if on_step is not None:
                on_step(dict(record))
    finally:
        optimizer.zero_grad(set_to_none=True)

    return {
        "steps": steps,
        "optimizer_steps": len(steps),
        "micro_batches_supplied": len(micro_batches),
        "micro_batches_used": used,
        "micro_batches_dropped": dropped,
        "accumulation": accumulation,
        "final_partial_group_policy": final_partial_group,
        "last_group_size": groups[-1] if groups else None,
        "effective_batch_size_nominal": accumulation,
        "effective_batch_size_actual_last": groups[-1] if groups else None,
        "amp": {"device_type": autocast.get("device_type"), "dtype": str(autocast.get("dtype")),
                "enabled": bool(autocast.get("enabled")), "scaler_enabled": scaler.is_enabled()},
        "policy_note": ("loss divided by the actual group size; gradients equal the mean over the group, "
                        "not the sum"),
    }
