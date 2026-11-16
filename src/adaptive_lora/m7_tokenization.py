"""M7 真实分词、loss mask、右截断与 padding 协议；不加载模型、不执行前向或训练。

本模块只做文本到监督样本的确定性转换，调用者负责提供已核验的数据行、固定 revision
的真实 tokenizer 与新输出目录。它不下载、不分词以外的语义判断、不评估效果，也不因为
单元测试通过就代表 tokenizer 级隔离或 1.5B 运行已验收。

关键前提（代码本身无法表达，必须由调用者保证）：

- prefix 必须是 build_commonsense_prompt 生成的规范 Alpaca 纯文本前缀，不含答案、
  不含 Qwen chat/system 模板；本模块不检查 prefix 的语义来源。
- tokenizer 必须是目标模型的固定 revision，且 add_special_tokens=False 使用；模板
  已自带必要标记时再加特殊 token 会产生重复。
- 本机 Qwen2.5 分词器的 pad_token_id 与 eos_token_id 同为 151643，因此**不能用 token
  ID 判断某个位置是真实结束还是填充**；唯一可靠依据是 attention_mask 与 labels。
"""

from __future__ import annotations

import hashlib
import json

# 已确认的 M7 协议常量；与 configs/qwen2_5_1_5b_commonsense_r16.json 的 protocol 段一致。
M7_EOS_TOKEN_ID = 151643
M7_MAX_LENGTH = 512
M7_PAD_MULTIPLE = 8
PROMPT_LABEL_IGNORE = -100
# 评估侧不截断（2026-11-16 协议修订）：训练仍右截断 512，但生成入参保留完整前缀。
# 起因：1 条 piqa 评估前缀 521 token，右截断会丢掉 " solution1/solution2\n\n### Response:\n"，
# 使模型在 "Answer format:" 之后续写而非规范的 Response 标记之后。
M7_EVALUATION_TRUNCATION = "none_keep_full_prefix"


def _flat_ids(encoded, label):
    """把分词结果规范为一维整数列表；拒绝批维、空串和非整数。

    分词器在单条文本上通常直接返回一维列表，但不同实现可能返回 Tensor 或嵌套维度。
    这里显式拒绝批量形状，避免把多条文本误当一条继续拼装 labels。
    """
    if not hasattr(encoded, "__getitem__") or "input_ids" not in encoded:
        raise ValueError(f"{label}: tokenizer must return input_ids")
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        raise ValueError(f"{label}: expected a single sequence, got a batch")
    if not isinstance(ids, list):
        raise ValueError(f"{label}: input_ids must be a list")
    if any(type(value) is not int for value in ids):
        raise ValueError(f"{label}: input_ids must be plain integers")
    return ids


def tokenize_ids(tokenizer, text, label="text"):
    """对已渲染文本分词，固定不添加特殊 token、不截断、不 padding。"""
    if not isinstance(text, str) or not text:
        raise ValueError(f"{label} must be non-empty text")
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, padding=False,
                        return_attention_mask=False, return_token_type_ids=False)
    return _flat_ids(encoded, label)


def ids_sha256(ids):
    """对 token 序列做规范 JSON SHA-256；用于跨拆分 token 级身份比较。

    这是内容指纹，不是脱敏手段，也不证明两段文本语义不同。
    """
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_supervised_example(tokenizer, prefix, answer, *, eos_token_id=M7_EOS_TOKEN_ID,
                             max_length=M7_MAX_LENGTH, pad_multiple=M7_PAD_MULTIPLE,
                             pad_token_id=None):
    """返回单条训练样本的 input_ids/labels/attention_mask 及其审计字段。

    步骤与约束：

    1. 分别对 prefix 与 prefix+answer 分词，并要求 `tokenize(prefix)` 恰为
       `tokenize(prefix+answer)` 的前缀。这是 BPE 边界检查：若不成立，说明答案首个
       token 与前缀末尾发生了跨边界合并，答案起点无法按前缀长度切分，此时**报错而不
       猜测切分点**。
    2. 答案 token 之后显式追加 eos_token_id，不依赖分词器自动添加。
    3. 整条序列右截断到 max_length（论文 C.2 的超长截断），截断不重抽、不丢样本。
    4. 前缀位置 labels=-100，答案与 EOS 位置 labels 为原 token ID；截断后若没有任何
       可预测监督 token，抛错而不是静默产生空监督样本。
    5. 长度补齐到 pad_multiple 的倍数；pad 位置 attention_mask=0 且 labels=-100。

    因为本机 pad_token_id 可能等于 eos_token_id，第 5 步的填充在 input_ids 上与真实
    EOS 无法区分，只能由 mask 与 labels 判别——调用者不得用 IDs 判断有效性。

    返回值中的 truncated 表示序列是否被截断；prompt_token_count 是进入 labels 的
    前缀长度（截断后可能小于原始前缀长度）。
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be non-empty text")
    if not isinstance(answer, str) or not answer:
        raise ValueError("answer must be non-empty text")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if type(pad_multiple) is not int or pad_multiple < 1:
        raise ValueError("pad_multiple must be a positive integer")
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_multiple > 1 and (type(pad_token_id) is not int or pad_token_id < 0):
        raise ValueError("padding requires a valid pad_token_id")

    prefix_ids = tokenize_ids(tokenizer, prefix, "prefix")
    combined_ids = tokenize_ids(tokenizer, prefix + answer, "prefix+answer")
    if combined_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "tokenize(prefix) is not a prefix of tokenize(prefix+answer); "
            "the answer boundary is ambiguous, refusing to guess the split"
        )
    answer_ids = combined_ids[len(prefix_ids):]
    if not answer_ids:
        raise ValueError("answer produced no tokens after the prefix boundary")

    full_ids = combined_ids + [eos_token_id]
    truncated = len(full_ids) > max_length
    if truncated:
        full_ids = full_ids[:max_length]
    prompt_token_count = min(len(prefix_ids), len(full_ids))
    labels = [PROMPT_LABEL_IGNORE] * prompt_token_count + full_ids[prompt_token_count:]
    supervised = len(full_ids) - prompt_token_count
    if supervised <= 0:
        raise ValueError(
            "no supervised token remains after truncation; "
            f"prefix={len(prefix_ids)} tokens, max_length={max_length}. "
            "Refusing to drop or resample the sample."
        )

    pad_count = (-len(full_ids)) % pad_multiple if pad_multiple > 1 else 0
    input_ids = full_ids + [pad_token_id] * pad_count
    attention_mask = [1] * len(full_ids) + [0] * pad_count
    labels = labels + [PROMPT_LABEL_IGNORE] * pad_count

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_token_count": prompt_token_count,
        "supervised_token_count": supervised,
        "pad_token_count": pad_count,
        "truncated": truncated,
        "original_token_count": len(combined_ids) + 1,
        "prefix_token_count_untruncated": len(prefix_ids),
        "answer_token_count": len(answer_ids),
        "eos_appended": True,
        "prefix_ids_sha256": ids_sha256(prefix_ids),
        "full_ids_sha256": ids_sha256(full_ids),
        "pad_equals_eos": pad_token_id == eos_token_id,
    }


def build_calibration_input(tokenizer, prefix, *, max_length=M7_MAX_LENGTH):
    """返回校准用的无监督输入，只保留前缀，不含答案也不含 EOS。

    校准读取前缀最后一个有效 token 的 next-token logits 来确定原始 t*，因此这里不追加
    EOS、不构造 labels。若前缀被截断，仍返回截断后的序列并把 truncated 置真，由调用方
    决定是否接受；本函数不静默丢弃样本。
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be non-empty text")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    prefix_ids = tokenize_ids(tokenizer, prefix, "prefix")
    truncated = len(prefix_ids) > max_length
    ids = prefix_ids[:max_length] if truncated else prefix_ids
    if not ids:
        raise ValueError("calibration input has no tokens")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "prediction_position": len(ids) - 1,
        "original_token_count": len(prefix_ids),
        "truncated": truncated,
        "prefix_ids_sha256": ids_sha256(prefix_ids[:max_length]),
    }


def build_evaluation_input(tokenizer, prefix, *, reference_max_length=M7_MAX_LENGTH):
    """返回生成用的完整评估前缀，**不截断**；只把是否超过训练长度作为信息报出。

    2026-11-16 协议修订（`evaluation_truncation = none_keep_full_prefix`）：训练仍右截断
    到 reference_max_length，但生成入参保留完整前缀。原因是 1 条 piqa 评估前缀为 521
    token，右截断到 512 会丢掉末尾 " solution1/solution2\\n\\n### Response:\\n"，使模型在
    "Answer format:" 之后续写，而不是在规范的 Response 标记之后续写。

    本函数不追加 EOS、不构造 labels、不 padding；调用者负责把整个前缀交给生成接口，并在
    任何位置超出模型上下文上限时自行报错（不得在此静默截断）。`exceeds_training_length`
    只是如实标注该样本比训练样本长，不是错误条件。
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be non-empty text")
    if type(reference_max_length) is not int or reference_max_length < 1:
        raise ValueError("reference_max_length must be a positive integer")
    ids = tokenize_ids(tokenizer, prefix, "evaluation prefix")
    if not ids:
        raise ValueError("evaluation input has no tokens")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "truncated": False,
        "token_count": len(ids),
        "exceeds_training_length": len(ids) > reference_max_length,
        "reference_max_length": reference_max_length,
        "prefix_ids_sha256": ids_sha256(ids),
    }


def token_identity_overlap(train_rows, evaluation_rows):
    """统计训练与评估之间 token 级直接身份交集，不做去重或语义判断。

    输入为含 `sample_id` 与 `prefix_ids_sha256` 的记录。只比较**实际分词后的前缀
    token 序列指纹**，这是 ID 与 prompt 字节之外第三个身份域；仍不覆盖语义改写、
    题面内嵌答案与预训练污染。
    """
    def identities(rows):
        return {row["prefix_ids_sha256"] for row in rows}

    common = sorted(identities(train_rows) & identities(evaluation_rows))
    return {
        "token_identity_disjoint": not common,
        "token_identity_intersection": common,
        "intersection_size": len(common),
        "scope": "direct exact token-sequence identity only",
    }
