"""M6 的共享 smoke 配置、固定监督/评估拆分和直接输入重叠审计。

只定义工程验证协议，不替代论文正式数据、答案抽取或评估器。校准允许来自训练侧；评估
必须与训练/校准按 ID、prompt 字节哈希和实际 prefix token 哈希隔离。不能识别语义改写、
原始题面内嵌答案或模型预训练污染，正式实验仍需独立数据审查。
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

from adaptive_lora.data import build_calibration_prompt, hash_text
from adaptive_lora.trainer import build_supervised_example

TRAIN_ROWS = (
    ("synthetic-m6-v1/train/0", "Reply with yes.", "yes"),
    ("synthetic-m6-v1/train/1", "Return the digit 2.", "2"),
    ("synthetic-m6-v1/train/2", "Write the word red.", "red"),
)
EVAL_ROWS = (
    ("synthetic-m6-v1/eval/0", "Reply with no.", "no"),
    ("synthetic-m6-v1/eval/1", "Return the digit 3.", "3"),
    ("synthetic-m6-v1/eval/2", "Write the word blue.", "blue"),
)


def stable_hash(value):
    """对规范有限 JSON 散列；键排序、紧凑分隔符、UTF-8，不包含外部时间戳或机器路径。"""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SharedConfig:
    """三种方法共用的有界 CPU smoke 设置；不提供正式 epochs、GPU 或批量数据开关。

    rank/alpha 和优化器参数对所有方法相同，不降低 Full 的 rank 来伪造相同参数预算。
    train_seed 控制各方法注入与训练 RNG；Random-K 选层 seed 由独立 BaselinePlan 保存。
    """

    steps: int = 3
    rank: int = 16
    alpha: int = 32
    train_seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_new_tokens: int = 4

    def __post_init__(self):
        limits = ((self.steps, 1, 3), (self.rank, 1, 64), (self.alpha, 1, 256),
                  (self.train_seed, 0, 2 ** 32 - 1), (self.max_new_tokens, 1, 8))
        if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
            raise ValueError("Invalid or unbounded M6 smoke integer setting")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float))
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("Invalid shared AdamW settings")

    def to_dict(self):
        """记录真实公共设置，包括 M5 训练器固定的部分，返回独立 JSON 数据。"""
        return {"steps": self.steps, "rank": self.rank, "alpha": self.alpha, "train_seed": self.train_seed,
                "batch_size": 1, "gradient_accumulation": 1, "max_length": 96,
                "optimizer": {"name": "AdamW", "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                              "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False},
                "dropout": 0.0, "bias": "none", "init_lora_weights": True,
                "target_projections": ["q", "k", "v"], "scheduler": None, "warmup": 0, "gradient_clipping": None,
                "device": "cpu", "precision": "unchanged base dtype; FP32 adapters/loss; no AMP",
                "padding": False, "truncation": False, "use_cache": False,
                "loss_mask": "prompt=-100; answer+tokenizer EOS; one-token causal shift",
                "generation": {"do_sample": False, "num_beams": 1, "temperature": 0,
                               "max_new_tokens": self.max_new_tokens, "use_cache": False, "stop": "tokenizer.eos_token_id"},
                "metric": "smoke_exact_match_strip_only_not_paper_accuracy"}


def _valid_ids(ids):
    """这里只校验非空普通整数 ID；词表上界由持有实际模型的训练/评估器核对。"""
    return isinstance(ids, list) and bool(ids) and all(type(token) is int and token >= 0 for token in ids)


def _signatures(records):
    """从真实 token 列表重算哈希，不信任外部 token_sha256；返回各类唯一标识集合。"""
    records = list(records)
    if not records:
        raise ValueError("Split inputs must be non-empty")
    result = {"sample_id": set(), "prompt_sha256": set(), "token_sha256": set()}
    for row in records:
        identifier, ids, prompt_hash = (row.get(key) for key in ("sample_id", "input_ids", "prompt_sha256"))
        if (not isinstance(identifier, str) or not identifier or identifier in result["sample_id"]
                or not _valid_ids(ids) or not isinstance(prompt_hash, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_hash) is None):
            raise ValueError("Invalid split ID/prompt hash/tokens or duplicate ID within split")
        result["sample_id"].add(identifier)
        result["prompt_sha256"].add(prompt_hash)
        result["token_sha256"].add(stable_hash(ids))
    return result


def check_split_isolation(training, evaluation, calibration):
    """拒绝评估与训练/校准的直接输入重叠，允许校准属于训练集，并返回可审计计数。

    training/evaluation/calibration 均为 prefix-only 的 ID/IDs/prompt_sha256 记录。计数表示
    相交的唯一标识/哈希数，不是重复样本对数量；参考答案相同不构成输入泄漏判据。
    """
    train, test, cal = (_signatures(rows) for rows in (training, evaluation, calibration))
    def overlap(left, right):
        """三种身份域分别统计，避免同一输入换一个 sample_id 就绕过检查。"""
        return {key: len(left[key] & right[key]) for key in left}
    eval_train, eval_cal = overlap(test, train), overlap(test, cal)
    if any(eval_train.values()) or any(eval_cal.values()):
        raise ValueError("Evaluation input overlaps training or calibration")
    return {"evaluation_train_overlaps": eval_train, "evaluation_calibration_overlaps": eval_cal,
            "calibration_train_overlaps": overlap(cal, train),
            "scope": "exact ID/prompt bytes/prefix-token equality only; not semantic or pretraining contamination"}


def protocol_hash(protocol):
    """协议全部内容参与身份检查，只有 content_sha256 自身不参与；改参考答案也会改指纹。"""
    return stable_hash({key: value for key, value in protocol.items() if key != "content_sha256"})


def validate_protocol(protocol):
    """运行前/各方法前复核协议内容、监督边界、评估纯输入及拆分，返回直接重叠审计。

    哈希检测意外修改而非认证外部数据；prefix 文本哈希的来源由 builder/固定校准清单承担。
    即使调用者重算外层指纹，坏 labels、eval 内夹 reference 和直接输入重叠仍明确拒绝。
    """
    if protocol.get("schema_version") != 1 or protocol.get("content_sha256") != protocol_hash(protocol):
        raise ValueError("Comparison protocol content fingerprint mismatch")
    training, evaluation, references, calibration = (protocol[key] for key in
        ("training_samples", "evaluation_prompts", "references", "calibration_inputs"))
    if not 1 <= len(training) <= 3 or not 1 <= len(evaluation) <= 3 or len(references) != len(evaluation):
        raise ValueError("M6 smoke requires 1–3 training and evaluation examples")
    train_inputs = []
    for row in training:
        ids, p, labels = row["input_ids"], row["prompt_token_count"], row["labels"]
        if (not _valid_ids(ids) or len(ids) > 96 or type(p) is not int or not 1 <= p <= len(ids) - 2
                or any(type(value) is not int for value in labels) or labels != [-100] * p + ids[p:]
                or row["attention_mask"] != [1] * len(ids) or ids[-1] != protocol["eos_token_id"]):
            raise ValueError("Invalid shared training mask, answer boundary or EOS")
        train_inputs.append({"sample_id": row["sample_id"], "input_ids": ids[:p], "prompt_sha256": row["prompt_sha256"]})
    for row, reference in zip(evaluation, references):
        if (set(row) != {"sample_id", "input_ids", "attention_mask", "prompt_sha256"}
                or not _valid_ids(row["input_ids"]) or len(row["input_ids"]) > 96
                or row["attention_mask"] != [1] * len(row["input_ids"])
                or reference["sample_id"] != row["sample_id"] or not isinstance(reference["text"], str) or not reference["text"].strip()):
            raise ValueError("Evaluation prompts must be reference-free with matching scoring-only references")
    return check_split_isolation(train_inputs, evaluation, calibration)


def build_smoke_protocol(tokenizer, calibration_samples, *, steps=3):
    """一次性构造共享合成训练/评估协议；不抽样、不读取或改变 M4 的重要性/选层。

    calibration_samples 须来自调用者已认证完整性的 M2 清单。本函数只取其输入身份，绝不
    读取校准答案。训练复用 M5 监督 builder；评估渲染/分词函数不接收 reference，随后才另存
    参考答案供离线评分。全部 prefix 使用相同 tokenizer 模板，不 padding/truncate。
    """
    if type(steps) is not int or not 1 <= steps <= 3:
        raise ValueError("M6 smoke accepts 1–3 training steps")
    training = [build_supervised_example(tokenizer, instruction, answer, sample_id=identifier, max_length=96)
                for identifier, instruction, answer in TRAIN_ROWS[:steps]]
    evaluation = []
    for identifier, instruction, _ in EVAL_ROWS:
        prompt = build_calibration_prompt(tokenizer, instruction, "", add_generation_prompt=True)
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False, padding=False, return_attention_mask=True)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if not _valid_ids(ids) or len(ids) > 96 or mask != [1] * len(ids):
            raise ValueError("Invalid or overlength evaluation prefix")
        evaluation.append({"sample_id": identifier, "input_ids": ids, "attention_mask": mask, "prompt_sha256": hash_text(prompt)})
    protocol = {"schema_version": 1, "scope": "M6 synthetic interface smoke only",
                "template_sha256": hash_text(tokenizer.get_chat_template()), "eos_token_id": tokenizer.eos_token_id,
                "training_samples": training, "evaluation_prompts": evaluation,
                "references": [{"sample_id": identifier, "text": answer} for identifier, _, answer in EVAL_ROWS],
                "calibration_inputs": [{"sample_id": row["sample_id"], "input_ids": list(row["input_ids"]),
                                        "prompt_sha256": row["prompt_hash"]} for row in calibration_samples]}
    protocol["content_sha256"] = protocol_hash(protocol)
    validate_protocol(protocol)
    return protocol
