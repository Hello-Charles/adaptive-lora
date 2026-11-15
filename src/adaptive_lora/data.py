"""校准输入、分词边界与可追溯清单；不执行模型前向、消融或答案评估。

调用者须固定数据 revision、train 行顺序及 tokenizer revision。答案字段只能进入
来源行哈希，不能进入 prompt；哈希不代表脱敏，清单默认应保存在 Git 忽略目录。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import random

DATASET_ID = "tatsu-lab/alpaca"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_N = 30
DEFAULT_SEED = 20261112
DEFAULT_MAX_LENGTH = 512
MANIFEST_CONTENT_KEYS = ("schema_version", "dataset", "sampling", "template", "tokenizer", "samples")


def build_user_content(instruction, input_text):
    """仅拼接文本型 instruction/input，空 instruction 或非文本字段明确报错。

    [复现决策] 首尾 strip；非空 input 前加两个换行。不识别自然语言内部的答案，
    因此字段隔离并不保证源 instruction 自身没有泄漏；调用者仍需审查数据协议。
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty text")
    if input_text is None:
        input_text = ""
    if not isinstance(input_text, str):
        raise ValueError("input must be text or None")
    instruction, input_text = instruction.strip(), input_text.strip()
    return f"{instruction}\n\n{input_text}" if input_text else instruction


def build_calibration_prompt(tokenizer, instruction, input_text, *, add_generation_prompt=True):
    """渲染 user-only 消息，不构造 assistant 答案；模板可自行加入默认 system。

    要求 tokenizer 有适用的 chat template。generation prompt 只是回答起始标记，
    返回文本已含模板特殊标记，后续分词必须使用 add_special_tokens=False 避免重复。
    """
    messages = [{"role": "user", "content": build_user_content(instruction, input_text)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


def sample_indices(population_size, n=DEFAULT_N, seed=DEFAULT_SEED):
    """返回零基、不放回、保留抽取顺序的索引；局部 RNG 不改变全局随机状态。

    同 seed 仅在总体大小、行顺序和 Python 随机实现一致时可重现同一内容；下游必须
    同时保存 revision、索引及哈希，不能把 seed 当作完整的数据版本标识。
    """
    if n < 1:
        raise ValueError("n must be positive")
    if population_size < n:
        raise ValueError(f"population of {population_size} is smaller than n={n}")
    return random.Random(seed).sample(range(population_size), n)


def position_from_mask(mask):
    """返回一维二值 mask 中最后有效 token 的零基索引；空区间、非二值、孔洞拒绝。

    因果模型的 logits[batch, i, :] 预测读完输入 token i 后的下一个 token，返回值
    不是新 token 的插入位置。右 padding 不能读最后一列，左 padding 不能用 sum-1。
    """
    if any(value not in (0, 1) for value in mask):
        raise ValueError("attention mask must be binary (0 or 1)")
    active = [index for index, value in enumerate(mask) if value == 1]
    if not active:
        raise ValueError("attention mask has no active tokens")
    if active != list(range(active[0], active[-1] + 1)):
        raise ValueError("attention mask must contain one contiguous span of active tokens")
    return active[-1]


def to_list(value):
    """将已存在的张量转换为可序列化列表；形状和语义由调用路径另外校验。"""
    return value.tolist() if hasattr(value, "tolist") else value


def _encoded_rows(encoded):
    """规范化 [batch, length] 的 IDs/mask 并检查对应形状，返回每行有效末尾。

    必须显式提供 attention_mask，不能按 token ID 推断有效性：pad token 与 eos token
    可以同 ID。特别是批量路径，缺失 mask 时一律拒绝，不能把填充默认视为真实输入。
    """
    if "attention_mask" not in encoded:
        raise ValueError("tokenizer must return attention_mask")
    ids, masks = to_list(encoded["input_ids"]), to_list(encoded["attention_mask"])
    if not ids or len(ids) != len(masks) or any(len(row) != len(mask) for row, mask in zip(ids, masks)):
        raise ValueError("input_ids and attention_mask shapes must match a non-empty batch")
    positions = [position_from_mask(mask) for mask in masks]
    return ids, masks, positions


def tokenize_single(tokenizer, prompt, max_length=DEFAULT_MAX_LENGTH):
    """返回一维 IDs/mask、有效位置、原始长度和精确截断标记；无模型前向。

    输入须为已渲染的模板文本；不再添加特殊 token，不 padding。先测量完整分词长度，
    超长时才按 tokenizer.truncation_side 截断，所以恰好等于上限不会误报截断。
    本通用工具允许截断，但可能丢失 generation prompt；build_manifest 对超长样本
    明确报错，不把截断后的 user 中途悄悄当作回答起点，也不重新抽样掩盖问题。
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    options = dict(padding=False, add_special_tokens=False, return_attention_mask=True, return_tensors="pt")
    full = tokenizer(prompt, truncation=False, **options)
    ids, masks, positions = _encoded_rows(full)
    if len(ids) != 1:
        raise ValueError("Expected one tokenized prompt")
    original_count = len(ids[0])
    if original_count > max_length:
        encoded = tokenizer(prompt, truncation=True, max_length=max_length, **options)
        ids, masks, positions = _encoded_rows(encoded)
        if len(ids) != 1 or len(ids[0]) > max_length:
            raise ValueError("tokenizer did not honor single-prompt max_length")
    return {
        "input_ids": ids[0],
        "attention_mask": masks[0],
        "prediction_position": positions[0],
        "original_token_count": original_count,
        "truncated": original_count > len(ids[0]),
    }


def tokenize_batch(tokenizer, prompts, max_length=DEFAULT_MAX_LENGTH):
    """返回嵌套 IDs/mask 和每行有效位置，支持 tokenizer 的左右 padding 配置。

    prompts 必须非空且已渲染模板；需要 pad token 和显式 mask。该辅助函数允许按
    tokenizer 方向截断，不能单独保证 assistant 标记保留，也不验证批量 logits 与
    逐样本 logits 数值等价；M2 正式清单仍采用逐样本无 padding 的路径。
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    encoded = tokenizer(
        prompts, truncation=True, padding=True, max_length=max_length,
        add_special_tokens=False, return_attention_mask=True, return_tensors="pt",
    )
    ids, masks, positions = _encoded_rows(encoded)
    if len(ids) != len(prompts):
        raise ValueError("tokenizer batch size must match prompts")
    return {"input_ids": ids, "attention_mask": masks, "prediction_positions": positions}


def hash_text(text):
    """按原始 UTF-8 字节计算 SHA-256，不做 Unicode 或空白归一化。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_hash(row):
    """排序 key 的 JSON 指纹；原始整行可含答案，但不会作为 prompt 构造参数。

    非 JSON 值回退 str，调用者需保证其表示稳定；当前 Alpaca 字段为文本。
    哈希不是脱敏或真实性签名，不能据此把敏感样本及完整 prompt 自动公开。
    """
    return hash_text(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str))


def manifest_content_hash(manifest):
    """仅对协议和样本求指纹；排除时间戳、命令、缓存绝对路径及运行环境等非确定项。

    该字段用于重复运行的一致性检查，不替代来源 parquet 的字节哈希或代码版本记录。
    """
    return row_hash({key: manifest[key] for key in MANIFEST_CONTENT_KEYS})


def build_manifest(
    *, rows, indices, tokenizer, dataset_id, dataset_revision, model_id, model_revision,
    max_length, seed, dataset_source=None,
):
    """按合法唯一索引构建训练输入清单，超长或数据字段不合约时整次失败。

    rows 为固定 train 顺序的可索引数据；dataset_source 保存文件名、字节 SHA/大小，
    由已核验来源的调用者提供。整行哈希、输入字段哈希、完整 prompt 哈希含义不同；
    input_ids 才是下游实际读取的模型输入。所有索引和 prediction_position 均为零基。
    [复现决策] 超过 max_length 不静默截断、不剔除重抽，避免改变固定样本集合及语境。
    模板原文及其哈希、运行时 tokenizer 方向均保存，不把本地选择说成论文模板。
    """
    indices = list(indices)
    if not indices or any(type(index) is not int or not 0 <= index < len(rows) for index in indices):
        raise ValueError("indices must be non-empty, zero-based and in bounds")
    if len(indices) != len(set(indices)):
        raise ValueError("indices must be unique")
    template = tokenizer.get_chat_template()
    samples = []
    for index in indices:
        row = rows[index]
        instruction, input_text = row["instruction"], row.get("input", "")
        prompt = build_calibration_prompt(tokenizer, instruction, input_text)
        tokenized = tokenize_single(tokenizer, prompt, max_length)
        if tokenized["truncated"]:
            raise ValueError(
                f"Sample {index} has {tokenized['original_token_count']} tokens, exceeds max_length={max_length}; "
                "refusing to lose the generation prompt or resample"
            )
        samples.append({
            "sample_id": f"{dataset_id}@{dataset_revision}/train/{index}",
            "index": index,
            "row_hash": row_hash(row),
            "input_hash": hash_text(build_user_content(instruction, input_text)),
            "prompt_hash": hash_text(prompt),
            "prompt": prompt,
            **tokenized,
            "token_count": len(tokenized["input_ids"]),
        })
    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": dataset_id, "revision": dataset_revision, "split": "train",
            "population_size": len(rows), "source": dataset_source,
        },
        "sampling": {
            "n": len(indices), "seed": seed, "index_base": 0,
            "method": "random.Random(seed).sample(range(population_size), n)",
        },
        "template": {
            "kind": "tokenizer_chat_template_user_only", "text": template, "sha256": hash_text(template),
            "add_generation_prompt": True, "source_fields": ["instruction", "input"],
            "user_content": "strip(instruction) + '\\n\\n' + strip(input) when input is non-empty [复现决策]",
        },
        "tokenizer": {
            "model_id": model_id, "model_revision": model_revision, "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast, "max_length": max_length, "add_special_tokens": False,
            "padding": False, "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side, "overlength_policy": "reject [复现决策]",
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "samples": samples,
    }
    manifest["content_sha256"] = manifest_content_hash(manifest)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return manifest
