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


def collate_supervised_examples(examples, *, pad_token_id, pad_multiple=M7_PAD_MULTIPLE,
                                eos_token_id=M7_EOS_TOKEN_ID):
    """把多条 `build_supervised_example` 结果拼成一个右 padding 的批，返回二维列表字段。

    为什么重建 padding 而不是在已有 padding 后继续补：单条样本已按 `pad_multiple` 补过
    一次；若直接在其后追加到批宽，两层 padding 会叠加，而中间那层的边界**无法从 IDs
    恢复**（pad 与 EOS 同 ID）。因此这里先按 `prompt_token_count + supervised_token_count`
    求出真实有效长度，把单条样本自带的那层 padding 视为待重建，再从干净的有效区间重新
    补到批宽。

    关键语义（判据是 mask 与 labels，不是 IDs）：

    - `attention_mask`：有效区间为 1，padding 为 0。
    - `labels`：prompt 区间为 -100，监督区间为原 token id，padding 为 -100。
    - 本模型 `pad_token_id == eos_token_id == 151643`，所以**绝不能**用
      `input_ids != pad_token_id` 推断有效位置——那会把每个真实 EOS 都当成 padding，
      恰好屏蔽掉最该学的结束符。

    前置条件（只校验、不修正）：所有样本来自同一 tokenizer，且 `pad_token_id` 与构造
    样本时一致。已截断的样本允许存在，其有效区间即为截断后长度。

    返回 dict：三个二维列表加批级审计字段。不返回张量（由调用者决定设备与 dtype），
    不写文件，不修改输入。
    """
    if type(pad_multiple) is not int or pad_multiple < 1:
        raise ValueError("pad_multiple must be a positive integer")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    examples = list(examples)
    if not examples:
        raise ValueError("collate requires at least one example")

    rows, identifiers, valid_lengths = [], set(), []
    for position, example in enumerate(examples):
        identifier = example.get("sample_id", f"index-{position}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate sample_id in batch: {identifier}")
        identifiers.add(identifier)

        prompt_n = example["prompt_token_count"]
        supervised_n = example["supervised_token_count"]
        ids = list(example["input_ids"])
        labels = list(example["labels"])
        mask = list(example["attention_mask"])
        if not (len(ids) == len(labels) == len(mask)):
            raise ValueError(f"Length mismatch between ids/labels/mask for {identifier}")
        valid = prompt_n + supervised_n
        if not 1 <= prompt_n < valid <= len(ids):
            raise ValueError(f"Invalid prompt/supervised boundary for {identifier}")

        # 单条样本自带的 padding 必须自成一段：mask=0 且 labels=-100。
        if mask[:valid] != [1] * valid or mask[valid:] != [0] * (len(ids) - valid):
            raise ValueError(f"Single-example attention mask is not contiguous for {identifier}")
        if labels[valid:] != [PROMPT_LABEL_IGNORE] * (len(ids) - valid):
            raise ValueError(f"Single-example padding labels are not ignored for {identifier}")
        if labels[:prompt_n] != [PROMPT_LABEL_IGNORE] * prompt_n:
            raise ValueError(f"Prompt labels are not all ignored for {identifier}")
        if any(value < 0 for value in labels[prompt_n:valid]):
            raise ValueError(f"Negative label inside the supervised region for {identifier}")
        # 未截断时监督区间必须以 EOS 收尾；用 labels 判断，不受 pad==eos 影响。
        if not example.get("truncated", False) and labels[valid - 1] != eos_token_id:
            raise ValueError(f"Supervised region does not end with EOS for {identifier}")

        rows.append({"sample_id": identifier, "prompt_token_count": prompt_n,
                     "supervised_token_count": supervised_n,
                     "input_ids": ids[:valid], "labels": labels[:valid], "mask": mask[:valid]})
        valid_lengths.append(valid)

    # 批宽取最长有效长度并向上取整到 pad_multiple，保持论文的形状约束。
    longest = max(valid_lengths)
    batch_length = -(-longest // pad_multiple) * pad_multiple

    batch_ids, batch_labels, batch_mask = [], [], []
    for row in rows:
        pad_n = batch_length - len(row["input_ids"])
        batch_ids.append(row["input_ids"] + [pad_token_id] * pad_n)
        batch_labels.append(row["labels"] + [PROMPT_LABEL_IGNORE] * pad_n)
        batch_mask.append(row["mask"] + [0] * pad_n)

    return {
        "input_ids": batch_ids,
        "labels": batch_labels,
        "attention_mask": batch_mask,
        "sample_ids": [row["sample_id"] for row in rows],
        "batch_size": len(rows),
        "batch_length": batch_length,
        "longest_valid_length": longest,
        "valid_lengths": valid_lengths,
        "padded_positions_total": batch_length * len(rows) - sum(valid_lengths),
        "pad_equals_eos": pad_token_id == eos_token_id,
        "mask_authority_note": ("pad and EOS share one id here, so validity comes from "
                                "attention_mask and labels; deriving it from token ids would "
                                "wrongly mask every real EOS"),
    }
