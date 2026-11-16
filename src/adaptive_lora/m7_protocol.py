"""M7候选协议、纯文本prompt和commonsense评分，不加载模型或执行实验。

输入配置只表示待审查的实验意图；confirmed字段不是费用、下载、安装或公开推送授权。
本模块仅使用标准库，不能以配置有效、SHA形状正确或评分夹具通过，替代真实数据字节、
模型模块返回值、CUDA/AMP/梯度累积和评估输入隔离的运行验收。论文未定义的模板、
loss mask、抽样及解析细节均是需单独确认的复现决策，见M7_PROTOCOL.md。
"""

import hashlib
import json
import math
from pathlib import Path
import re

TASK_LABELS = {
    "boolq": ("true", "false"),
    "piqa": ("solution1", "solution2"),
    "social_i_qa": ("answer1", "answer2", "answer3"),
    "hellaswag": ("ending1", "ending2", "ending3", "ending4"),
    "winogrande": ("option1", "option2"),
    "ARC-Easy": ("answer1", "answer2", "answer3", "answer4", "answer5"),
    "ARC-Challenge": ("answer1", "answer2", "answer3", "answer4", "answer5"),
    "openbookqa": ("answer1", "answer2", "answer3", "answer4"),
}
DECISION_IDS = {"scope", "prompt", "supervision", "sampling", "answer_parser", "single_gpu_batch",
                "mirror_source", "evaluation_truncation", "training_order", "final_partial_group"}
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
)
_FIXED_TRAINING = {
    "rank": 16, "alpha": 32, "top_k": 8, "target_projections": ["q", "k", "v"],
    "epochs": 3, "train_seed": 42, "learning_rate": 1e-4, "weight_decay": 0.01,
    "betas": [0.9, 0.999], "eps": 1e-8, "warmup_steps": 0, "lora_dropout": 0.0,
    "lora_bias": "none", "micro_batch_size": 8, "gradient_accumulation": 8, "world_size": 1,
    "base_dtype": "float16", "adapter_master_dtype": "float32", "amp_dtype": "float16",
    "max_length": 512, "pad_to_multiple_of": 8, "quantization": None, "use_cache": False,
    # 2026-11-16 用户确认的两项训练循环决策。论文对这两点均未作规定，因此必须固定取值，
    # 而不是留给运行时默认：不同的 shuffle 与末组策略会改变每个 epoch 实际参与更新的样本，
    # 进而让同一份配置产出不同的模型。
    "training_order": "none", "final_partial_group": "drop",
}
_FIXED_PROTOCOL = {
    "methods": ["full", "random_k", "adaptive_k"], "scope": "commonsense_only",
    "prompt_format": "alpaca_canonical_v1", "loss_mask": "answer_and_eos_only",
    "truncation": "right_full_sequence_fail_if_no_supervised_token", "padding_side": "right",
    "evaluation_truncation": "none_keep_full_prefix",
    "calibration_mode": "frozen_eval_zero_ablation_fixed_original_token",
    "metric": "eight_task_macro_accuracy", "answer_parser": "bounded_unique_casefold_v1",
    "max_new_tokens": 32, "do_sample": False, "temperature": 0.0, "num_beams": 1, "eos_token_id": 151643,
}


def _require(condition, message):
    """统一将协议前提不满足报告为ValueError，不自动补默认值或修正输入。"""
    if not condition:
        raise ValueError(message)


def _keys(value, keys, label):
    """严格键集合避免拼写错误或额外执行标志被静默忽略；不作为权限验证机制。"""
    _require(isinstance(value, dict) and set(value) == set(keys), f"Invalid keys for {label}")


def _same(value, expected, label):
    """比较固定值，整数不接受bool；JSON浮点可用数值相等的普通int表示。"""
    if isinstance(expected, list):
        _require(isinstance(value, list) and len(value) == len(expected), f"Invalid {label}")
        for actual, target in zip(value, expected):
            _same(actual, target, label)
    elif type(expected) is float:
        _require(type(value) in (int, float) and math.isfinite(value) and value == expected, f"Invalid {label}")
    else:
        _require(type(value) is type(expected) and value == expected, f"Invalid {label}")


def _integer(value, minimum, maximum, label):
    """数量、seed和索引使用普通整数，避免bool在Python中被当作0/1通过门禁。"""
    _require(type(value) is int and minimum <= value <= maximum, f"Invalid {label}")


def _digest(value, length, label):
    """只检查十六进制格式；不能证明Git对象存在、LFS下载完成或来源已认证。"""
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None, f"Invalid {label}")


def protocol_sha256(config):
    """对配置全部字段做规范UTF-8 JSON SHA256，含决策状态；不覆盖源码/依赖字节。

    字典键顺序不影响结果，数组顺序有意义。禁止NaN/Infinity；外部文件的实际字节
    由CLI另存哈希，不能把此配置哈希称为模型来源签名或完整运行身份。
    """
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_m7_config(config):
    """验证本轮单卡commonsense/r16候选协议，失败抛ValueError，不修改输入。

    固定论文目标与本次明确标注的候选复现决定；不自动接受0.5B/Top7旧结果、量化、
    改rank或修改batch。将来改变实验范围应重新审查配置，而不是让此门禁悄悄兼容。
    文件路径限定为已选公开数据布局；尚未读取数据，样本数和SHA仍只是待验收声明。
    """
    _keys(config, ("schema_version", "profile", "model", "dataset", "training", "protocol", "environment_candidate", "decisions"), "root")
    protocol_sha256(config)
    _same(config["schema_version"], 1, "schema_version")
    _same(config["profile"], "m7-commonsense-r16-single-gpu-v1", "profile")
    model = config["model"]
    _keys(model, ("id", "revision", "layer_count", "hidden_size", "attention_heads", "kv_heads", "config_git_blob_sha1", "weights_file", "weights_bytes", "weights_lfs_sha256", "metadata_source"), "model")
    for key, expected in {"id": "Qwen/Qwen2.5-1.5B", "layer_count": 28, "hidden_size": 1536, "attention_heads": 12, "kv_heads": 2, "weights_file": "model.safetensors", "metadata_source": "mirror_only_not_independently_authenticated"}.items():
        _same(model[key], expected, "model." + key)
    _digest(model["revision"], 40, "model.revision")
    _digest(model["config_git_blob_sha1"], 40, "model.config_git_blob_sha1")
    _digest(model["weights_lfs_sha256"], 64, "model.weights_lfs_sha256")
    _integer(model["weights_bytes"], 1, 10**11, "weights_bytes")

    dataset = config["dataset"]
    _keys(dataset, ("repository", "revision", "train_file", "evaluation_files", "train_samples", "calibration_samples", "train_sampling_seed", "calibration_sampling_seed", "evaluation_sampling_seed"), "dataset")
    _same(dataset["repository"], "AGI-Edgerunners/LLM-Adapters", "dataset.repository")
    _digest(dataset["revision"], 40, "dataset.revision")
    _same(dataset["train_samples"], 10000, "train_samples")
    _same(dataset["calibration_samples"], 250, "calibration_samples")
    for key in ("train_sampling_seed", "calibration_sampling_seed", "evaluation_sampling_seed"):
        _integer(dataset[key], 0, 2**32 - 1, key)
    train = dataset["train_file"]
    _keys(train, ("path", "git_blob_sha1", "size_bytes"), "train_file")
    _same(train["path"], "ft-training_set/commonsense_170k.json", "train_file.path")
    _digest(train["git_blob_sha1"], 40, "train_file.git_blob_sha1")
    _integer(train["size_bytes"], 1, 10**10, "train_file.size_bytes")
    evaluations = dataset["evaluation_files"]
    _require(isinstance(evaluations, list) and len(evaluations) == len(TASK_LABELS), "Expected eight evaluation files")
    tasks = []
    for row in evaluations:
        _keys(row, ("task", "path", "git_blob_sha1", "size_bytes", "sample_count"), "evaluation file")
        task = row["task"]
        _require(isinstance(task, str) and task in TASK_LABELS and task not in tasks, "Invalid or duplicate task")
        tasks.append(task)
        _same(row["path"], f"dataset/{task}/test.json", "evaluation path")
        _digest(row["git_blob_sha1"], 40, "evaluation Git blob")
        _integer(row["size_bytes"], 1, 10**10, "evaluation size")
        _same(row["sample_count"], 500 if task == "openbookqa" else 1000, "evaluation sample_count")

    training = config["training"]
    _keys(training, (*_FIXED_TRAINING, "random_layer_seed"), "training")
    for key, expected in _FIXED_TRAINING.items():
        _same(training[key], expected, "training." + key)
    _integer(training["random_layer_seed"], 0, 2**32 - 1, "random_layer_seed")
    _keys(config["protocol"], _FIXED_PROTOCOL, "protocol")
    for key, expected in _FIXED_PROTOCOL.items():
        _same(config["protocol"][key], expected, "protocol." + key)
    environment = config["environment_candidate"]
    candidate = {"python_series": "3.12", "torch": "2.6.0+cu124", "transformers": "5.22.2", "peft": "0.21.2", "requirements_file": "requirements-linux-cu124-m7.in", "installed_or_verified_by_this_checker": False}
    _keys(environment, candidate, "environment_candidate")
    for key, expected in candidate.items():
        _same(environment[key], expected, "environment_candidate." + key)
    decisions = config["decisions"]
    _require(isinstance(decisions, list) and len(decisions) == len(DECISION_IDS),
             f"Expected {len(DECISION_IDS)} review decisions")
    seen = set()
    for row in decisions:
        _keys(row, ("id", "status", "proposal"), "decision")
        identifier = row["id"]
        _require(isinstance(identifier, str) and identifier in DECISION_IDS and identifier not in seen, "Invalid decision id")
        seen.add(identifier)
        _require(row["status"] in ("pending", "confirmed"), "Invalid decision status")
        _require(isinstance(row["proposal"], str) and bool(row["proposal"].strip()), "Empty decision proposal")


def _unique_object(pairs):
    """JSON重复键必须报错，包括嵌套对象，避免阅读值与实际采用值不一致。"""
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    """拒绝JSON扩展NaN/Infinity，而不是等到数学比较时才发现非有限值。"""
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_m7_config(path):
    """只读最多1MiB的UTF-8 JSON草案，严格解析并验证；不执行配置内任何内容。"""
    with Path(path).open("rb") as handle:
        raw = handle.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, "Configuration exceeds 1 MiB")
    config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_m7_config(config)
    return config


def effective_batch_size(micro_batch_size, gradient_accumulation, world_size):
    """返回三个正普通整数的乘积；world_size是实际并行卡数，不把累计当额外GPU。"""
    for value in (micro_batch_size, gradient_accumulation, world_size):
        _require(type(value) is int and value > 0, "Batch factors must be positive plain integers")
    return micro_batch_size * gradient_accumulation * world_size


def build_preflight_report(config):
    """返回候选协议审查摘要，不触发下载、安装、模型加载或远程命令。

    参数数量只从配置尺寸估算，未审计真实投影或可训练张量。forward预算含每样本
    1次baseline和2L次独立消融，另加1次恢复检查；不能当成已执行次数或GPU耗时。
    所有决策confirmed也仅表示配置记录的审查状态，永远不产生执行授权或运行验收。
    """
    validate_m7_config(config)
    model, data, training = (config[key] for key in ("model", "dataset", "training"))
    pending = [row["id"] for row in config["decisions"] if row["status"] == "pending"]
    n, layers = data["calibration_samples"], model["layer_count"]
    hidden = model["hidden_size"]
    kv_width = hidden // model["attention_heads"] * model["kv_heads"]
    per_layer = training["rank"] * ((hidden + hidden) + 2 * (hidden + kv_width))
    evaluation_bytes = sum(row["size_bytes"] for row in data["evaluation_files"])
    return {
        "review_status": "PENDING_DECISIONS" if pending else "REVIEWED_NOT_EXECUTABLE",
        "config_sha256": protocol_sha256(config), "pending_decisions": pending,
        "execution_authorized": False, "runtime_validated": False, "training_started": False,
        "effective_batch_size": effective_batch_size(training["micro_batch_size"], training["gradient_accumulation"], training["world_size"]),
        "paper_four_gpu_effective_batch_size": effective_batch_size(8, 8, 4),
        "calibration_forward_budget": {"scoring": n * (1 + 2 * layers), "restore_checks": n, "total": n * (2 + 2 * layers)},
        "parameter_estimate": {"full": per_layer * layers, "k_layer": per_layer * training["top_k"], "scope": "from_config_not_measured; no memory/speed claim"},
        "file_bytes_estimate": {"weights": model["weights_bytes"], "train": data["train_file"]["size_bytes"], "evaluation": evaluation_bytes, "weights_and_data": model["weights_bytes"] + data["train_file"]["size_bytes"] + evaluation_bytes, "scope": "metadata only; excludes tokenizer, dependencies, environment and results"},
        "adaptive_selection_computed": False, "adaptive_layer_indices": None,
        "prompt_template_sha256": protocol_sha256({"without_input": PROMPT_WITHOUT_INPUT, "with_input": PROMPT_WITH_INPUT}),
        "unverified_execution_gates": ["Real user authorization for push/install/download/paid compute", "Pinned source bytes and actual dataset manifests/split isolation", "Remote dependency imports and actual 1.5B module paths/returns", "CUDA/AMP/accumulation/checkpoint/metric runtime acceptance"],
    }


def build_commonsense_prompt(instruction, input_text=""):
    """渲染候选规范化Alpaca前缀，不接受答案或reference，也不添加special tokens。

    训练、校准和生成共享该前缀字节；末尾换行属于协议。去掉外围空白且不保留来源
    脚本的缩进，不使用Qwen chat/system模板，均为需单独确认的复现决定。调用者仍须审查
    instruction/input本身是否内嵌答案；参数隔离不能证明语义上无答案泄漏。
    """
    _require(isinstance(instruction, str) and bool(instruction.strip()), "instruction must be non-empty text")
    if input_text is None:
        input_text = ""
    _require(isinstance(input_text, str), "input must be text or None")
    instruction, input_text = instruction.strip(), input_text.strip()
    template = PROMPT_WITH_INPUT if input_text else PROMPT_WITHOUT_INPUT
    return template.format(instruction=instruction, input_text=input_text)


def extract_commonsense_answer(task, generated_text):
    """只解析新生成续文，返回label/status/matched_labels；未知任务或非文本报错。

    调用者必须先裁掉输入token，不能把题面选项交给解析器。本函数无法识别错误混入
    的prompt。大小写归一化及Unicode词边界、多个不同标签判冲突，是复现决定：不
    复制来源脚本的无边界首匹配规则；重复同一标签允许，无标签不猜测答案。
    """
    _require(isinstance(task, str) and task in TASK_LABELS, "Unknown commonsense task")
    _require(isinstance(generated_text, str), "generated_text must be text")
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(label) for label in TASK_LABELS[task]) + r")(?!\w)"
    matches = list(dict.fromkeys(re.findall(pattern, generated_text.casefold())))
    status = "valid" if len(matches) == 1 else "no_label" if not matches else "ambiguous"
    return {"label": matches[0] if status == "valid" else None, "status": status, "matched_labels": matches}


def _indexed_rows(rows, fields, label):
    """按唯一非空字符串ID建索引；拒绝空集和额外字段，不做模糊或位置匹配。"""
    _require(isinstance(rows, list) and bool(rows), f"{label} must be a non-empty list")
    indexed = {}
    for row in rows:
        _keys(row, fields, label)
        identifier = row["sample_id"]
        _require(isinstance(identifier, str) and bool(identifier) and identifier not in indexed, f"Invalid {label} sample_id")
        indexed[identifier] = row
    return indexed


def score_commonsense(task, predictions, references):
    """在生成结束后按ID连接参考答案，返回任务准确率及逐样本解析状态。

    predictions为sample_id/generated_text，references为sample_id/answer；参考标签须已
    是任务规范小写标签。所有预测保留在分母，包括空串和歧义；无效gold或ID不完整
    直接报错，不删除困难样本。不执行生成，也不代替训练/校准/评估输入隔离审计。
    """
    _require(isinstance(task, str) and task in TASK_LABELS, "Unknown commonsense task")
    predicted = _indexed_rows(predictions, ("sample_id", "generated_text"), "predictions")
    gold = _indexed_rows(references, ("sample_id", "answer"), "references")
    _require(set(predicted) == set(gold), "Prediction/reference IDs must match exactly")
    _require(all(isinstance(row["answer"], str) and row["answer"] in TASK_LABELS[task] for row in gold.values()), "Invalid canonical reference label")
    rows = []
    for identifier, row in predicted.items():
        parsed = extract_commonsense_answer(task, row["generated_text"])
        answer = gold[identifier]["answer"]
        rows.append({"sample_id": identifier, "answer": answer, "predicted_label": parsed["label"], "parse_status": parsed["status"], "correct": parsed["label"] == answer})
    correct = sum(row["correct"] for row in rows)
    return {"task": task, "metric": "commonsense_task_accuracy_v1", "correct": correct, "total": len(rows), "accuracy": correct / len(rows), "rows": rows}


def aggregate_commonsense(scores):
    """仅对完整8任务做等权宏平均，不因OBQA样本较少而降低其权重。

    输入为任务评分摘要；检查accuracy与correct/total一致。这里只验证汇总数学，
    不证明这些计数来自实际预测，也不执行正式协议所需的1000/500样本数量门禁。
    """
    _require(isinstance(scores, list) and len(scores) == len(TASK_LABELS), "Expected eight task scores")
    accuracies, total_examples = {}, 0
    for row in scores:
        _require(isinstance(row, dict), "Task score must be an object")
        task = row.get("task")
        _require(isinstance(task, str) and task in TASK_LABELS and task not in accuracies, "Invalid or duplicate scored task")
        correct, total, accuracy = (row.get(key) for key in ("correct", "total", "accuracy"))
        _require(type(total) is int and total > 0 and type(correct) is int and 0 <= correct <= total, "Invalid score counts")
        _require(type(accuracy) in (int, float) and math.isfinite(accuracy) and math.isclose(accuracy, correct / total, rel_tol=0, abs_tol=1e-12), "Inconsistent task accuracy")
        accuracies[task] = correct / total
        total_examples += total
    return {"metric": "eight_task_macro_accuracy", "macro_accuracy": math.fsum(accuracies.values()) / len(TASK_LABELS), "task_accuracies": accuracies, "total_examples": total_examples}
