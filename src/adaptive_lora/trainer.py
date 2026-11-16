"""M5 的选择性标准 LoRA 与短训练工具，不负责正式数据实验或自动下载。

注入与M5短训练限于经发现器核验的CPU Qwen层栈；M7可显式允许迁移后的CUDA审计。
基础参数冻结与启用autograd是不同约束：adapter梯度仍需穿过冻结运算，不能沿用校准的inference_mode。
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

from adaptive_lora.data import build_calibration_prompt, hash_text
from adaptive_lora.model_loader import EXPECTED_LAYERS, discover_layer_modules


@dataclass(frozen=True)
class LoraSetup:
    """保留注入前基础参数的规范名和对象引用，抵消 PEFT 包装导致的参数路径重命名。

    frozen 仅限制字段重绑，不使模型不可变。base_fingerprint 不覆盖 buffers；调用者不能
    merge、替换基础参数或复用该对象做并发训练。targets 使用注入前模型根节点的完整路径。
    """

    model: object
    selected_layers: tuple[int, ...]
    targets: tuple[str, ...]
    rank: int
    alpha: int
    base_parameters: tuple
    base_fingerprint: dict


def fingerprint_parameters(named_parameters, *, allow_cuda=False):
    """按名称/顺序流式计算参数原始字节SHA-256，不修改权重或读取buffers。

    默认仍限CPU（M5门禁）；M7显式allow_cuda时逐参数拷回CPU散列，避免一次复制整个模型。
    名称/dtype/shape编码与旧CPU算法相同，因此迁移设备不会改变指纹；GPU拷贝会同步，
    必须放在阶段边界而非逐micro-batch热循环。非连续参数的contiguous可能额外分配。
    """
    digest = hashlib.sha256()
    count = 0
    for name, parameter in named_parameters:
        if parameter.device.type != "cpu" and not (allow_cuda and parameter.device.type == "cuda"):
            raise ValueError("M5 smoke supports CPU parameters only; CUDA requires explicit allow_cuda")
        header = json.dumps([name, str(parameter.dtype), list(parameter.shape)], ensure_ascii=False).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = parameter.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
        count += parameter.numel()
    return {"sha256": digest.hexdigest(), "parameter_elements": count}


def _target_paths(model, selected_layers, expected_layers):
    """先验证零基层号和实际线性投影，返回升序层号和完整 Q/K/V 路径；不修改模型。"""
    selected = tuple(selected_layers)
    if (not selected or any(type(index) is not int or not 0 <= index < expected_layers for index in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Selected layer indices must be unique, non-empty, zero-based integers")
    layers = discover_layer_modules(model, expected_layers=expected_layers)
    selected = tuple(sorted(selected))
    targets = tuple(layers[index].projections[projection] for index in selected for projection in ("q", "k", "v"))
    for path in targets:
        module = model.get_submodule(path)
        if not isinstance(module, torch.nn.Linear) or module.in_features < 1 or module.out_features < 1:
            raise ValueError(f"Expected an ordinary Linear Q/K/V projection: {path}")
    return selected, targets


def attach_selective_lora(model, selected_layers, *, rank=16, alpha=32, expected_layers=EXPECTED_LAYERS, revision=None):
    """就地冻结基础模型、注入所选层标准 LoRA，返回含审计依据的 LoraSetup。

    输入必须是无 adapter、无既有梯度/本地 forward hooks 的全新 CPU 基础模型。层结构和
    配置先校验，错误输入不先冻结模型；PEFT 执行异常时不保证撤销第三方已做的结构变更，
    调用者应弃用失败实例，而不是在同一模型上重试。此函数不下载或执行训练。

    完整路径先 re.escape，再组成锚定正则，防止层 1 误匹配层 10/11。标准 A 随机、B=0
    初始化不改基础权重；不使用 PiSSA/OLoRA、bias、modules_to_save 等额外可训练部分。
    PEFT 将 BF16/FP16 adapter 升 FP32，基础参数不转 dtype；这是 CPU smoke 约定而非论文 AMP。
    """
    if type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1:
        raise ValueError("rank and alpha must be positive integers")
    if torch.is_inference_mode_enabled():
        raise ValueError("Do not create trainable LoRA tensors inside inference_mode")
    if getattr(model, "peft_config", None) or any(isinstance(module, LoraLayer) for module in model.modules()):
        raise ValueError("Expected a fresh base model without adapters")
    parameters = tuple(model.named_parameters())
    if not parameters or any(parameter.device.type != "cpu" or parameter.grad is not None for _, parameter in parameters):
        raise ValueError("Expected CPU base parameters without pre-existing gradients")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Expected a base model without local forward hooks")
    selected, targets = _target_paths(model, selected_layers, expected_layers)
    before = fingerprint_parameters(parameters)
    target_pattern = "^(?:" + "|".join(re.escape(path) for path in targets) + ")$"
    config = LoraConfig(
        task_type="CAUSAL_LM", r=rank, lora_alpha=alpha, target_modules=target_pattern,
        lora_dropout=0.0, bias="none", init_lora_weights=True, use_rslora=False,
        use_dora=False, modules_to_save=None, revision=revision,
    )
    model.requires_grad_(False)
    wrapped = get_peft_model(model, config, autocast_adapter_dtype=True)
    setup = LoraSetup(wrapped, selected, targets, rank, alpha, parameters, before)
    audit_lora(setup)
    assert_base_unchanged(setup)
    return setup


def audit_lora(setup, *, trainable=True, allow_cuda=False):
    """核对实际 adapter 模块、A/B 形状、基础参数身份及唯一可训练白名单，返回 JSON 统计。

    不用名称含 lora 的启发式放行参数；任何多余参数、丢失基础参数、merge、禁用 adapter 或
    requires_grad 不符均报错。trainable=False 用于只读恢复检查，此时包括 adapter 在内全部冻结。
    不计算基础字节指纹，避免每步扫描基础模型；字节检查由assert_base_unchanged负责。
    默认保留M5的CPU限定；M7显式allow_cuda时允许同一设备的CUDA参数，不自动迁移设备。
    """
    model = setup.model
    if set(model.peft_config) != {"default"}:
        raise ValueError("Exactly the default adapter must be present")
    root = model.get_base_model()
    actual = {path: module for path, module in root.named_modules() if isinstance(module, LoraLayer)}
    if set(actual) != set(setup.targets):
        raise ValueError("Actual LoRA modules differ from the exact selected Q/K/V paths")
    adapter_ids, dimensions = set(), []
    for path in setup.targets:
        module = actual[path]
        if (module.merged or module.disable_adapters or module.active_adapters != ["default"]
                or set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}):
            raise ValueError(f"Unexpected adapter state: {path}")
        base, a, b = module.get_base_layer(), module.lora_A["default"], module.lora_B["default"]
        if (not isinstance(base, torch.nn.Linear) or a.bias is not None or b.bias is not None
                or tuple(a.weight.shape) != (setup.rank, base.in_features)
                or tuple(b.weight.shape) != (base.out_features, setup.rank)
                or module.lora_alpha["default"] != setup.alpha
                or module.scaling["default"] != setup.alpha / setup.rank):
            raise ValueError(f"Nonstandard LoRA dimensions or scaling: {path}")
        adapter_ids.update((id(a.weight), id(b.weight)))
        dimensions.append({
            "path": path, "in_features": base.in_features, "out_features": base.out_features,
            "rank": setup.rank, "adapter_elements": setup.rank * (base.in_features + base.out_features),
        })
    base_ids = {id(parameter) for _, parameter in setup.base_parameters}
    named = tuple(model.named_parameters())
    if base_ids & adapter_ids or {id(parameter) for _, parameter in named} != base_ids | adapter_ids:
        raise ValueError("Backbone parameter identity changed or unexpected parameters were added")
    for name, parameter in named:
        expected_trainable = trainable and id(parameter) in adapter_ids
        if parameter.requires_grad != expected_trainable:
            raise ValueError(f"Unexpected requires_grad for {name}")
        if id(parameter) in base_ids and parameter.grad is not None:
            raise ValueError(f"Frozen base parameter has a gradient: {name}")
        if parameter.device.type != "cpu" and not (allow_cuda and parameter.device.type == "cuda"):
            raise ValueError("M5 smoke requires CPU parameters; CUDA requires explicit allow_cuda")
    if len({parameter.device for _, parameter in named}) != 1:
        raise ValueError("LoRA audit requires all parameters on one device")
    adapters = [(name, parameter) for name, parameter in named if id(parameter) in adapter_ids]
    expected_elements = sum(item["adapter_elements"] for item in dimensions)
    if len(adapters) != 2 * len(setup.targets) or sum(parameter.numel() for _, parameter in adapters) != expected_elements:
        raise ValueError("Adapter parameter count differs from r*(d_in+d_out)")
    return {
        "selected_layer_indices_ascending": list(setup.selected_layers), "target_modules": list(setup.targets),
        "projection_dimensions": dimensions, "base_frozen": True, "base_gradients_absent": True,
        "base_parameter_elements": sum(parameter.numel() for _, parameter in setup.base_parameters),
        "adapter_parameter_elements": expected_elements,
        "total_parameter_elements": sum(parameter.numel() for _, parameter in named),
        "trainable_parameter_elements": sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
        "trainable_parameter_tensors": sum(parameter.requires_grad for _, parameter in named),
        "trainable_parameter_names": [name for name, parameter in named if parameter.requires_grad],
        "adapter_dtypes": sorted({str(parameter.dtype) for _, parameter in adapters}),
    }


def assert_base_unchanged(setup, *, allow_cuda=False):
    """重扫规范基础参数名，要求字节未变；M7须显式允许CUDA同步拷回CPU散列。"""
    current = fingerprint_parameters(setup.base_parameters, allow_cuda=allow_cuda)
    if current != setup.base_fingerprint:
        raise ValueError("Frozen base parameter fingerprint changed")
    return current


def build_supervised_example(tokenizer, instruction, answer, *, sample_id, max_length=96):
    """构建单条无 padding 的 smoke 监督样本，返回可保存的 IDs、labels、边界与文本哈希。

    [复现决策] 使用与 M2 一致的 user-only generation prefix，再接训练回答和一个 tokenizer
    EOS ID。EOS 不等同于 Qwen 的 im_end，也不声称这是作者的训练模板。回答只在 teacher-
    forcing 后缀出现，不写回 calibration prompt/manifest；这不是推理时把 gold answer 当输入。

    prefix 和 prefix+answer 分别完整分词，验证前者 IDs 确为后者前缀，不能假定 BPE 在拼接处
    不合并。已渲染模板不再自动加 special tokens；长度含显式 EOS，超过上限拒绝而不截断。
    prompt labels 全为 -100；labels[P] 是第一回答 token，loss 应读取 logits[P-1]。
    """
    if not isinstance(sample_id, str) or not sample_id or not isinstance(answer, str) or not answer.strip():
        raise ValueError("Expected a non-empty sample ID and answer")
    if type(max_length) is not int or not 2 <= max_length <= 128:
        raise ValueError("M5 smoke max_length must be between 2 and 128")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("M5 smoke requires an explicit integer tokenizer EOS ID")
    prompt = build_calibration_prompt(tokenizer, instruction, "", add_generation_prompt=True)

    def encode(text):
        """请求无截断/无填充的扁平单样本输出，拒绝缺失或不一致的有效位 mask。"""
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, padding=False, return_attention_mask=True)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(mask, list) or mask != [1] * len(ids)):
            raise ValueError("Expected one unpadded tokenized sequence")
        return ids

    prefix_ids, combined_ids = encode(prompt), encode(prompt + answer)
    p = len(prefix_ids)
    if combined_ids[:p] != prefix_ids or len(combined_ids) <= p:
        raise ValueError("Tokenized prompt must be an unchanged prefix followed by answer tokens")
    ids = combined_ids + [eos]
    if len(ids) > max_length:
        raise ValueError("Supervised example exceeds smoke max_length; truncation is forbidden")
    return {
        "sample_id": sample_id, "input_ids": ids, "attention_mask": [1] * len(ids),
        "labels": [-100] * p + ids[p:], "prompt_token_count": p,
        "answer_token_count": len(combined_ids) - p, "supervised_token_count": len(ids) - p,
        "eos_token_id": eos, "prompt_sha256": hash_text(prompt), "answer_sha256": hash_text(answer),
        "protocol": "user_prefix_plus_answer_then_tokenizer_eos_v1 [复现决策]",
    }


def causal_loss(logits, labels):
    """显式计算 FP32 causal cross entropy，logits 为 [B,S,V]、labels 为 [B,S] 的 long。

    logits[:, i] 预测 labels[:, i+1]；先 shift 再忽略 -100。没有有效监督、越界 token、形状/
    dtype 不符或非有限 logits 均报错，避免全 mask 得到 NaN。FP32 转换保留 autograd 链路。
    """
    if (logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 2
            or labels.shape != logits.shape[:2] or labels.dtype != torch.long):
        raise ValueError("Expected logits [B,S,V] and long labels [B,S], with S,V >= 2")
    shifted = labels[:, 1:]
    active = shifted != -100
    if (not active.any().item() or (shifted[active] < 0).any().item()
            or (shifted[active] >= logits.shape[-1]).any().item() or not torch.isfinite(logits).all().item()):
        raise ValueError("Invalid causal targets or nonfinite logits")
    return F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100)


def _training_batches(examples, vocab_size):
    """在第一次模型前向前验证全部 1–3 条样本，并快照为 CPU long [1,S]。

    仅接受每 batch 一条、全有效 mask、1–128 tokens、非空回答后接明确 EOS 的协议；不做
    自动 padding/truncation、样本拼接或重新抽样。先验证最后一条，防止中途才发现坏 labels。
    """
    examples = list(examples)
    if not 1 <= len(examples) <= 3:
        raise ValueError("M5 smoke accepts exactly 1 to 3 single-example batches")
    prepared, identifiers = [], set()
    for sample in examples:
        identifier = sample["sample_id"]
        ids, mask, labels = (sample[key] for key in ("input_ids", "attention_mask", "labels"))
        p, eos = sample["prompt_token_count"], sample["eos_token_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Training sample IDs must be unique, non-empty strings")
        identifiers.add(identifier)
        if (not isinstance(ids, list) or not 3 <= len(ids) <= 128
                or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
                or not isinstance(mask, list) or any(type(value) is not int for value in mask) or mask != [1] * len(ids)
                or type(p) is not int or not 1 <= p <= len(ids) - 2
                or type(eos) is not int or ids[-1] != eos
                or not isinstance(labels, list) or any(type(value) is not int for value in labels)
                or labels != [-100] * p + ids[p:]):
            raise ValueError("Invalid smoke IDs, prompt mask, answer boundary or EOS")
        prepared.append({
            "sample_id": identifier, "supervised_token_count": len(ids) - p,
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        })
    return prepared


def train_smoke(setup, examples, *, learning_rate=1e-4, weight_decay=0.01, on_step=None):
    """执行恰好 1–3 个 batch/optimizer steps，返回有限 loss、梯度、更新和前后指纹证据。

    调用者负责先固定 RNG/线程及确定性设置。此函数不创建 checkpoint、不调整学习率、不
    做 accumulation/AMP/梯度裁剪；batch_size=1。基础 requires_grad=False，但整体 train()
    且 autograd 必须开启，冻结权重不等于禁用梯度。标准 B=0 导致首步 A 梯度可为零，验收
    要求所有 adapter 梯度存在且有限、整体梯度非零，而非强迫每个张量都非零。

    正常/异常路径均清理梯度并回到 eval；不回滚已完成的 optimizer steps。发生异常须弃用
    部分训练状态，不把它保存为成功 checkpoint。on_step 在更新和审计后收到纯标量快照。
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("Training requires enabled autograd outside inference_mode")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("Invalid AdamW learning_rate or weight_decay")
    audit = audit_lora(setup)
    batches = _training_batches(examples, setup.model.config.vocab_size)
    base_before = assert_base_unchanged(setup)
    adapters = [(name, parameter) for name, parameter in setup.model.named_parameters() if parameter.requires_grad]
    before = fingerprint_parameters(adapters)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in adapters], lr=learning_rate, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False,
    )
    # 优化器不能含冻结基础权重，即使那些参数碰巧不会产生 grad 也不予放行。
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {id(parameter) for _, parameter in adapters}:
        raise ValueError("Optimizer parameters differ from the audited adapter whitelist")
    steps = []
    setup.model.train()
    try:
        for number, batch in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            previous = [parameter.detach().clone() for _, parameter in adapters]
            output = setup.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = causal_loss(output.logits, batch["labels"])
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [parameter.grad for _, parameter in adapters]
            if any(gradient is None or not torch.isfinite(gradient).all().item() for gradient in gradients):
                raise ValueError("Missing or nonfinite adapter gradient")
            nonzero = sum(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
            norm = math.sqrt(math.fsum(gradient.double().square().sum().item() for gradient in gradients))
            if nonzero == 0 or not math.isfinite(norm) or norm <= 0:
                raise ValueError("Adapter gradients must be finite and collectively nonzero")
            audit_lora(setup)
            optimizer.step()
            if any(not torch.isfinite(parameter).all().item() for _, parameter in adapters):
                raise ValueError("Nonfinite adapter weights after optimizer step")
            updated = sum(not torch.equal(old, parameter) for old, (_, parameter) in zip(previous, adapters))
            if updated == 0:
                raise ValueError("Optimizer did not update any adapter tensor")
            record = {
                "step": number, "sample_id": batch["sample_id"], "loss": float(loss.detach().item()),
                "token_count": batch["input_ids"].shape[1], "supervised_token_count": batch["supervised_token_count"],
                "gradient_l2_norm": norm, "gradient_tensors": len(gradients),
                "nonzero_gradient_tensors": nonzero, "updated_adapter_tensors": updated,
                "base_gradients_absent": True,
            }
            steps.append(record)
            if on_step is not None:
                on_step(dict(record))
            # 不跨 step 保留包含完整词表 logits 的计算图；日志只保存上面的纯标量。
            del output, loss
    finally:
        optimizer.zero_grad(set_to_none=True)
        setup.model.eval()
    audit_lora(setup)
    return {
        "protocol": "M5 causal-LM engineering smoke only", "batch_size": 1, "gradient_accumulation": 1,
        "steps": steps, "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
        "weight_decay": weight_decay, "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False},
        "optimizer_parameter_elements": audit["adapter_parameter_elements"],
        "adapter_before": before, "adapter_after": fingerprint_parameters(adapters),
        "base_before": base_before, "base_after": assert_base_unchanged(setup),
    }
