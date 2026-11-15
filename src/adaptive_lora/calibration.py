"""冻结基础模型上的逐样本校准与论文公式 (5)；不选层、不注入 LoRA、不训练。

仅支持本项目已核验的串行 CPU eager 路径。输入是已固定的 IDs/mask，而非答案文本；
每个输入先测未消融预测，再独立消融每层两个模块，始终读取同一个原始 token。
"""

import hashlib
import json
import math
from numbers import Real

import torch

from adaptive_lora.ablation import zero_ablation
from adaptive_lora.data import position_from_mask
from adaptive_lora.model_loader import discover_layer_modules


def _fp32_logits(logits):
    """接收一维有限浮点 logits，返回 FP32；不把 batch/序列维误当成词表维。

    [复现决策] 概率计算转 FP32，不把整模型转 FP32。FP16/BF16 先精确扩展再 softmax，
    减少低精度概率舍入；极端输入在转换后变成非有限值也拒绝，不假装得到有效概率。
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("Expected non-empty one-dimensional floating logits")
    values = logits.float()
    if not torch.isfinite(values).all().item():
        raise ValueError("Prediction logits must be finite")
    return values


def reference_prediction(logits):
    """返回未消融模型的一次 (t*, q)，t* 是零基词表 ID，不是输入序列位置。

    argmax 只在此处执行；并列按 PyTorch 的最小词表 ID。调用者须确保该 logits 来自
    未修改模型的有效末尾；函数本身无法判断是否存在外部干预。返回 Python 数值而非图。
    """
    values = _fp32_logits(logits)
    # 先保留原 logits 的最大项；FP64 调用者的近邻值不能因转 FP32 后并列而换 token。
    token_id = int(logits.argmax().item())
    return token_id, float(torch.softmax(values, dim=-1)[token_id].item())


def token_probability(logits, token_id):
    """读取显式给定 token 的 FP32 softmax 概率；绝不重新选择 argmax。

    在校准中 token_id 必须是原始 t*，即使消融使另一个 token 成为最大概率项也不改变。
    本函数不负责模型模式，calibrate_model 的 inference_mode 覆盖全部实际前向和概率计算。
    """
    values = _fp32_logits(logits)
    if type(token_id) is not int or not 0 <= token_id < values.numel():
        raise ValueError("token_id must be a valid zero-based vocabulary index")
    return float(torch.softmax(values, dim=-1)[token_id].item())


def _probability(value):
    """验证已保存概率，拒绝 NaN、越界和布尔伪数值，避免静默污染均值/排名。"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probabilities must be finite real numbers in [0, 1]")
    return float(value)


def aggregate_importance(records, *, layer_count):
    """按公式 (5) 对样本贡献取算术平均，返回按零基层号排列的分数和两项均值。

    records 的每项含原始 token_id/q 及每层 attention/mlp 的显式 token_id/probability。
    先逐样本计算 |q-qa|+|q-qm| 再除 N，不能先平均有符号差再取绝对值，也不除层数。
    [复现决策] 用 math.fsum（Python 双精度）累积已测得的 FP32 概率，避免累加舍入；
    检查所有测量 token 等于原始 token，层号完整唯一。该函数不执行任何模型前向。
    """
    records = list(records)
    if not records or type(layer_count) is not int or layer_count < 1:
        raise ValueError("Require non-empty calibration records and a positive layer_count")
    attention_deltas = [[] for _ in range(layer_count)]
    mlp_deltas = [[] for _ in range(layer_count)]
    for record in records:
        q = _probability(record["q"])
        token = record["token_id"]
        if type(token) is not int or token < 0:
            raise ValueError("Original token_id must be a non-negative integer")
        rows = record["layers"]
        indices = [row["layer_index"] for row in rows]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(range(layer_count)):
            raise ValueError("Layer indices must cover each zero-based layer exactly once")
        for row in rows:
            index = row["layer_index"]
            for name, target in (("attention", attention_deltas), ("mlp", mlp_deltas)):
                measurement = row[name]
                if type(measurement["token_id"]) is not int or measurement["token_id"] != token:
                    raise ValueError("Every ablation must measure the original token_id")
                target[index].append(abs(q - _probability(measurement["probability"])))
    n = len(records)
    return [
        {
            "layer_index": index,
            "attention_mean_abs_delta": math.fsum(attention_deltas[index]) / n,
            "mlp_mean_abs_delta": math.fsum(mlp_deltas[index]) / n,
            "importance": math.fsum(a + m for a, m in zip(attention_deltas[index], mlp_deltas[index])) / n,
        }
        for index in range(layer_count)
    ]


def parameter_fingerprint(model):
    """流式读取 CPU 参数名/dtype/shape/原始字节，不修改参数；不覆盖 buffers。

    uint8 view 能读取 BF16 的原始字节，memoryview 避免 embedding 再复制成巨大 bytes。
    仅当参数非连续时 contiguous 会复制；只在整次校准前后使用，不放入逐层热循环。
    """
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu":
            raise ValueError("Current calibration supports CPU parameters only")
        header = json.dumps([name, str(parameter.dtype), list(parameter.shape)], ensure_ascii=False).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = parameter.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
        count += parameter.numel()
    return {"sha256": digest.hexdigest(), "parameter_elements": count}


def _has_local_hooks(model):
    """诊断本地 forward/pre-hook；不检查全局 hook、编译图或外部线程。

    M4 要求全新无外部 hook 的基础模型，防止在已消融模型上误选“原始” token；
    诊断只读 PyTorch 内部表，实际注册/清理仍由 M3 的公开 handle API 完成。
    """
    return any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def _assert_base(model):
    """只检查，不偷偷冻结、切换模式或清空已有梯度/hook；调用者须准备全新基础模型。"""
    if any(module.training for module in model.modules()):
        raise ValueError("All model modules must be in eval mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("All parameters must be frozen with no pre-existing gradients")
    if _has_local_hooks(model):
        raise ValueError("Calibration requires a fresh base model without local forward hooks")


def _prepare_samples(samples, vocab_size):
    """在任何前向前验证全部 JSON 样本；固定 mask/预测位置，不重新分词或重新抽样。

    IDs/mask 是一维列表，转换为 CPU long [1, sequence]。预测位置必须是连续二值 mask
    的最后一个有效 token，支持左右 padding；position 不是待插入 next token 的下标。
    source index 可缺省，sample_id 必须唯一；vocab_size 可用时还检查 token 上界。
    """
    samples = list(samples)
    if not samples:
        raise ValueError("Calibration samples must be non-empty")
    prepared, identifiers = [], set()
    for sample in samples:
        identifier = sample.get("sample_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("sample_id must be a unique non-empty string")
        identifiers.add(identifier)
        ids, mask = sample["input_ids"], sample["attention_mask"]
        if not isinstance(ids, list) or not ids or any(
            type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in ids
        ):
            raise ValueError("input_ids must be a non-empty list of legal token IDs")
        if not isinstance(mask, list) or len(mask) != len(ids):
            raise ValueError("attention_mask must match input_ids")
        position = position_from_mask(mask)
        if type(sample["prediction_position"]) is not int or sample["prediction_position"] != position:
            raise ValueError("prediction_position must equal the last active input-token position")
        prepared.append((sample, position, {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }))
    return prepared


def _forward_logits(model, inputs):
    """整段新输入前向，不传历史 KV cache，保留完整序列维以检查位置及恢复。

    本函数只返回当前 logits；调用者不得跨样本保存它们。仅预测位置的词表向量参与
    softmax，不能把整张 [batch, sequence, vocab] 张量误归一化或读取 pad 末列。
    """
    result = model(**inputs, use_cache=False, return_dict=True)
    logits = result.logits
    if logits.ndim != 3 or logits.shape[:2] != inputs["input_ids"].shape or logits.shape[2] < 1:
        raise ValueError("Expected logits shape [1, input_sequence_length, vocabulary_size]")
    if torch.is_grad_enabled() or logits.requires_grad:
        raise RuntimeError("Calibration forward must not record gradients")
    return logits


def calibrate_model(model, samples, *, layers=None, on_sample=None):
    """逐样本冻结校准，返回原 token 概率轨迹、全层均值、调用统计及参数审计。

    调用者须提前 eval/freeze 并保持串行；本函数覆盖 inference_mode，但不改 requires_grad
    或训练模式。layers 缺省时用 M1 发现器；传入时须来自该模型且零基连续、完整有序。
    每个输入执行 1 次未消融、2L 次独立消融和 1 次无消融恢复检查，共 2L+2 次前向。
    额外恢复前向属于工程验证，不计入 q 或分数。完整基线 logits 仅保留当前样本，用于
    精确恢复比较；概率轨迹只保存 Python 数值，不保留激活或计算图。

    on_sample 在已恢复且无 hook 时调用，record 按只读约定传入以便落盘日志；回调异常
    直接传播，不会留下活跃消融。任何前向/概率异常均由 M3 scope 清理；失败不返回成功结果。
    """
    _assert_base(model)
    if layers is None:
        layers = list(discover_layer_modules(model))
    else:
        layers = list(layers)
        # 外部描述符不能只提供前几层，也不能引用另一模型；先重新发现再比对，无前向副作用。
        expected = discover_layer_modules(model, expected_layers=model.config.num_hidden_layers)
        if layers != expected:
            raise ValueError("layers must describe all layers of this model in zero-based order")
    if not layers or [layer.index for layer in layers] != list(range(len(layers))):
        raise ValueError("layers must cover contiguous zero-based layer indices")
    prepared = _prepare_samples(samples, getattr(model.config, "vocab_size", None))
    before = parameter_fingerprint(model)
    records = []
    evaluations = 0
    logits_dtype = None
    with torch.inference_mode():
        for sample, position, inputs in prepared:
            _assert_base(model)
            baseline = _forward_logits(model, inputs).clone()
            evaluations += 1
            logits_dtype = str(baseline.dtype)
            token, q = reference_prediction(baseline[0, position])
            record = {
                "sample_id": sample["sample_id"], "index": sample.get("index"),
                "prediction_position": position, "token_id": token, "q": q, "layers": [],
            }
            for layer in layers:
                row = {"layer_index": layer.index}
                for component in ("attention", "mlp"):
                    with zero_ablation(model, layer, component) as state:
                        logits = _forward_logits(model, inputs)
                        evaluations += 1
                        if logits.shape != baseline.shape or logits.dtype != baseline.dtype:
                            raise ValueError("Ablation logits metadata must match the baseline")
                        probability = token_probability(logits[0, position], token)
                    if state.calls != 1:
                        raise RuntimeError("Each independent target must execute exactly once")
                    row[component] = {"token_id": token, "probability": probability}
                    del logits
                record["layers"].append(row)
            if _has_local_hooks(model):
                raise RuntimeError("Ablation hooks leaked beyond their scopes")
            restored = _forward_logits(model, inputs)
            evaluations += 1
            if not torch.equal(baseline, restored):
                raise RuntimeError("Unablated logits were not exactly restored")
            record["restored_logits_equal"] = True
            record["forward_evaluations"] = 2 * len(layers) + 2
            records.append(record)
            del baseline, restored
            if on_sample is not None:
                on_sample(record)
    _assert_base(model)
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("Base model parameters changed during calibration")
    return {
        "n": len(records), "layer_count": len(layers), "samples": records,
        "layer_importance": aggregate_importance(records, layer_count=len(layers)),
        "forward_evaluations": evaluations, "restoration_evaluations": len(records),
        "parameter_fingerprint_before": before, "parameter_fingerprint_after": after,
        "all_parameters_frozen": True, "all_gradients_absent": True, "hooks_restored": True,
        "model_logits_dtype": logits_dtype, "probability_dtype": "torch.float32",
        "accumulation": "math.fsum over Python floats, divided by N", "use_cache": False, "batch_size": 1,
    }
