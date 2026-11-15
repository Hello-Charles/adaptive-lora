"""M6 的统一单方法执行与三方法公平性汇总；复用 M5 注入/短训练，不实现正式训练器。

调用者为每种方法提供全新、同字节基础模型。所有方法共享协议与超参数，只有层集合改变；
Full 的实际可训练参数更多，比较结果不能表述为同参数预算或正式下游效果。
"""

import random

import numpy as np
import torch

from adaptive_lora.data import hash_text
from adaptive_lora.evaluation import predict_greedy, score_exact_match
from adaptive_lora.protocol import stable_hash, validate_protocol
from adaptive_lora.trainer import assert_base_unchanged, attach_selective_lora, audit_lora, fingerprint_parameters, train_smoke


def seed_training(seed):
    """显式重置训练侧 Python/NumPy/PyTorch RNG；Random-K 的局部选层 RNG 不受影响。

    每个方法注入前和训练前分别调用，避免目标数量不同导致初始化消耗不同 RNG 后影响
    训练随机流。并不保证不同层集合中同名投影的初始 A 完全相同；该限制必须随报告说明。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_variant(plan, model, tokenizer, protocol, config, *, revision, expected_base_fingerprint, on_step=None):
    """在传入的全新 CPU 基础模型上统一训练/评估，返回 (LoraSetup, 纯 JSON 报告)。

    plan 应由 plan_baselines 生成；protocol 应一次性构建后供所有方法复用。第一次修改模型
    前校验协议、步数、模板与基础字节；训练后再次检查同一协议指纹，回调不得暗改数据。
    此函数不下载、不写 checkpoint、不重试，失败后不能把部分训练实例作为下一个方法基础。
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("Unified training requires enabled autograd outside inference_mode")
    split_audit = validate_protocol(protocol)
    protocol_sha = protocol["content_sha256"]
    settings = config.to_dict()
    if (len(protocol["training_samples"]) != config.steps
            or hash_text(tokenizer.get_chat_template()) != protocol["template_sha256"]
            or tokenizer.eos_token_id != protocol["eos_token_id"]):
        raise ValueError("Shared step count or tokenizer differs from the fixed protocol")
    if plan.method not in {"full", "random_k", "adaptive_k"}:
        raise ValueError("Unknown baseline method")
    if plan.method == "full" and (len(plan.indices) != plan.layer_count or set(plan.indices) != set(range(plan.layer_count))):
        raise ValueError("Full LoRA must target all layers")
    if plan.method == "random_k" and (type(plan.selection_seed) is not int
            or tuple(random.Random(plan.selection_seed).sample(range(plan.layer_count), len(plan.indices))) != plan.indices):
        raise ValueError("Random-K indices do not match the recorded layer seed")
    before = fingerprint_parameters(model.named_parameters())
    if before != expected_base_fingerprint:
        raise ValueError("Each method must start from the same verified base parameter bytes")
    seed_training(config.train_seed)
    setup = attach_selective_lora(model, plan.indices, rank=config.rank, alpha=config.alpha,
                                  expected_layers=plan.layer_count, revision=revision)
    parameters = audit_lora(setup)
    seed_training(config.train_seed)
    training = train_smoke(setup, protocol["training_samples"], learning_rate=config.learning_rate,
                           weight_decay=config.weight_decay, on_step=on_step)
    if training["optimizer"] != settings["optimizer"]:
        raise ValueError("Actual optimizer differs from the shared settings")
    validate_protocol(protocol)
    if protocol["content_sha256"] != protocol_sha:
        raise ValueError("Protocol changed during training")
    # 覆盖基础与 adapter，而不只检查 frozen backbone；评估不得对任何权重做优化或 merge。
    eval_before = fingerprint_parameters(setup.model.named_parameters())
    predictions = predict_greedy(setup.model, tokenizer, protocol["evaluation_prompts"], max_new_tokens=config.max_new_tokens)
    scores = score_exact_match(predictions, protocol["references"])
    eval_after = fingerprint_parameters(setup.model.named_parameters())
    if eval_before != eval_after:
        raise ValueError("Evaluation changed model parameter bytes")
    audit_lora(setup)
    assert_base_unchanged(setup)
    validate_protocol(protocol)
    if protocol["content_sha256"] != protocol_sha:
        raise ValueError("Protocol changed during evaluation")
    return setup, {
        "method": plan.method, "selection": plan.to_dict(), "shared_config": settings,
        "shared_config_sha256": stable_hash(settings), "protocol_sha256": protocol_sha,
        "initial_base_fingerprint": before, "split_audit": split_audit, "parameter_audit": parameters,
        "training": training,
        "evaluation": {"predictions": predictions, "scores": scores, "reference_sha256": stable_hash(protocol["references"]),
                       "parameters_before": eval_before, "parameters_after": eval_after, "parameters_unchanged": True},
        "rng_protocol": "reset training seed before injection and before optimization; separate local layer-selection RNG",
    }


def summarize_comparison(reports):
    """只汇总共享数据/配置/模型/评估身份一致的三个方法，参数量取实际注入审计而非层数估计。

    不要求预测或效果相同，要求训练样本顺序和评估 prefix/参考答案完全相同。Full/K 不同
    参数预算如实标明；若 K=L，允许相同参数量。此函数不从差异推导因果收益或统计显著性。
    """
    reports = list(reports)
    names = [row["method"] for row in reports]
    if len(reports) != 3 or set(names) != {"full", "random_k", "adaptive_k"}:
        raise ValueError("Exactly one Full, Random-K and Adaptive-K report is required")
    by_name = {row["method"]: row for row in reports}
    ordered = [by_name[name] for name in ("full", "random_k", "adaptive_k")]
    first = ordered[0]
    train_ids = [step["sample_id"] for step in first["training"]["steps"]]
    eval_inputs = [(row["sample_id"], row["prefix_token_sha256"]) for row in first["evaluation"]["predictions"]]
    full_count = first["parameter_audit"]["trainable_parameter_elements"]
    rows = []
    for report in ordered:
        if (report["shared_config_sha256"] != stable_hash(report["shared_config"])
                or report["shared_config"] != first["shared_config"] or report["protocol_sha256"] != first["protocol_sha256"]
                or report["initial_base_fingerprint"] != first["initial_base_fingerprint"]
                or report["training"]["base_before"] != report["initial_base_fingerprint"]
                or report["training"]["base_after"] != report["initial_base_fingerprint"]
                or report["training"]["optimizer"] != report["shared_config"]["optimizer"]
                or [step["sample_id"] for step in report["training"]["steps"]] != train_ids
                or [(row["sample_id"], row["prefix_token_sha256"]) for row in report["evaluation"]["predictions"]] != eval_inputs
                or report["evaluation"]["reference_sha256"] != first["evaluation"]["reference_sha256"]
                or not report["evaluation"]["parameters_unchanged"]):
            raise ValueError("Methods differ in data, hyperparameters, base weights or evaluation protocol")
        audit, scores = report["parameter_audit"], report["evaluation"]["scores"]
        count = audit["trainable_parameter_elements"]
        if (type(count) is not int or not 0 < count <= full_count or scores["total"] != len(eval_inputs)
                or scores["metric"] != first["evaluation"]["scores"]["metric"]):
            raise ValueError("Invalid actual parameter count or scoring denominator")
        rows.append({"method": report["method"], "selected_layer_count": report["selection"]["selected_layer_count"],
                     "selected_layer_indices": report["selection"]["selected_layer_indices"],
                     "selection_seed": report["selection"]["selection_seed"], "trainable_parameter_elements": count,
                     "base_parameter_elements": audit["base_parameter_elements"], "trainable_fraction_of_full": count / full_count,
                     "training_steps": len(report["training"]["steps"]), "eval_correct": scores["correct"],
                     "eval_count": scores["total"], "smoke_exact_match": scores["accuracy"]})
    return {"scope": "M6 synthetic interface comparison; not paper benchmark results",
            "shared_config_sha256": first["shared_config_sha256"], "protocol_sha256": first["protocol_sha256"],
            "same_parameter_budget": len({row["trainable_parameter_elements"] for row in rows}) == 1,
            "methods": rows, "metric": first["evaluation"]["scores"]["metric"]}
