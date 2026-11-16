"""M7 分词管线的离线真实 tokenizer 测试；不加载模型、不下载、不训练。

使用本地已缓存的 Qwen2.5-0.5B 固定 revision tokenizer。该分词器与 1.5B 同族但**不是**
目标 1.5B tokenizer，因此本测试只验证管线逻辑（BPE 边界、EOS、截断、padding、mask），
不能证明 1.5B 分词结果或 token 级隔离已验收。缺缓存时明确跳过并计数，不伪造通过。
"""
import hashlib
import json
import os
from pathlib import Path
import unittest

from adaptive_lora.m7_protocol import build_commonsense_prompt
from adaptive_lora.m7_tokenization import (
    M7_EOS_TOKEN_ID,
    M7_EVALUATION_TRUNCATION,
    M7_MAX_LENGTH,
    PROMPT_LABEL_IGNORE,
    build_calibration_input,
    build_evaluation_input,
    build_supervised_example,
    collate_supervised_examples,
    ids_sha256,
    token_identity_overlap,
    tokenize_ids,
)

CACHE = Path(__file__).resolve().parents[1] / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def load_tokenizer():
    """只在本地固定缓存可用时加载；返回 None 表示环境无法运行真实分词测试。"""
    if not CACHE.is_dir():
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", revision=REVISION, cache_dir=str(CACHE),
        local_files_only=True, use_fast=True,
    )


TOKENIZER = load_tokenizer()
REQUIRES_TOKENIZER = unittest.skipUnless(TOKENIZER is not None, "cached fixed tokenizer unavailable")


class FakeTokenizer:
    """极简按字符切分替身，仅用于验证与真实分词器无关的纯逻辑分支。

    它不能代表 BPE 行为，因此不使用它断言任何跨边界合并语义。
    """
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **options):
        if options.get("add_special_tokens"):
            raise AssertionError("unexpected special tokens")
        return {"input_ids": [ord(character) for character in text]}


class StructureTests(unittest.TestCase):
    """不依赖第三方环境的形状与拒绝路径。"""

    def test_rejects_batch_shaped_ids(self):
        class Batchy:
            pad_token_id = 0

            def __call__(self, text, **options):
                return {"input_ids": [[1, 2, 3]]}
        with self.assertRaisesRegex(ValueError, "single sequence"):
            tokenize_ids(Batchy(), "abc")

    def test_rejects_missing_input_ids_and_empty_text(self):
        with self.assertRaisesRegex(ValueError, "must return input_ids"):
            tokenize_ids(lambda *a, **k: {}, "abc")
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            tokenize_ids(FakeTokenizer(), "")

    def test_sha_of_ids_is_order_sensitive(self):
        self.assertNotEqual(ids_sha256([1, 2, 3]), ids_sha256([3, 2, 1]))
        self.assertEqual(ids_sha256([1, 2, 3]), hashlib.sha256(b"[1,2,3]").hexdigest())

    def test_padding_requires_pad_token(self):
        class NoPad(FakeTokenizer):
            pad_token_id = None
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            build_supervised_example(NoPad(), "Q:", "A", max_length=64, pad_multiple=8)

    def test_empty_answer_rejected(self):
        with self.assertRaisesRegex(ValueError, "answer must be non-empty"):
            build_supervised_example(FakeTokenizer(), "Q:", "", max_length=64, pad_multiple=8)


@REQUIRES_TOKENIZER
class RealTokenizerTests(unittest.TestCase):
    """真实固定 revision 分词器的边界、mask 与截断行为。"""

    def test_prefix_is_token_prefix_of_prefix_plus_answer(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        prefix_ids = tokenize_ids(TOKENIZER, prefix)
        combined = tokenize_ids(TOKENIZER, prefix + "true")
        self.assertEqual(combined[:len(prefix_ids)], prefix_ids)

    def test_labels_mask_prompt_and_supervise_answer_with_eos(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        example = build_supervised_example(TOKENIZER, prefix, "true")
        prompt_count = example["prompt_token_count"]
        self.assertTrue(all(value == PROMPT_LABEL_IGNORE for value in example["labels"][:prompt_count]))
        supervised_ids = example["input_ids"][prompt_count:prompt_count + example["supervised_token_count"]]
        self.assertEqual(example["labels"][prompt_count:prompt_count + len(supervised_ids)], supervised_ids)
        self.assertEqual(supervised_ids[-1], M7_EOS_TOKEN_ID)
        self.assertEqual(example["supervised_token_count"], example["answer_token_count"] + 1)
        self.assertEqual(len(example["input_ids"]), len(example["labels"]))
        self.assertEqual(len(example["input_ids"]), len(example["attention_mask"]))
        self.assertEqual(len(example["input_ids"]) % 8, 0)

    def test_pad_positions_have_zero_mask_and_ignore_label(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        example = build_supervised_example(TOKENIZER, prefix, "true")
        self.assertGreater(example["pad_token_count"], 0)
        pad_start = len(example["input_ids"]) - example["pad_token_count"]
        self.assertTrue(all(value == 0 for value in example["attention_mask"][pad_start:]))
        self.assertTrue(all(value == PROMPT_LABEL_IGNORE for value in example["labels"][pad_start:]))
        self.assertTrue(all(value == 1 for value in example["attention_mask"][:pad_start]))

    def test_pad_and_eos_share_id_so_mask_is_the_only_authority(self):
        """本机 Qwen2.5 分词器 pad 与 eos 同 ID；必须由 mask/labels 区分，不能用 IDs。"""
        self.assertEqual(TOKENIZER.pad_token_id, TOKENIZER.eos_token_id)
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        example = build_supervised_example(TOKENIZER, prefix, "true")
        self.assertTrue(example["pad_equals_eos"])
        pad_start = len(example["input_ids"]) - example["pad_token_count"]
        self.assertTrue(all(value == M7_EOS_TOKEN_ID for value in example["input_ids"][pad_start:]))

    def test_truncation_keeps_at_most_max_length(self):
        """前缀+答案超出上限时截断，但答案区域仍须有至少一个可预测 token。

        先量前缀 token 数，设 max_length = 前缀长度 + 1（保留 EOS），确保截断确实
        发生、且截断后序列长度符合上限、且仍有监督 token 存在。
        """
        prefix = build_commonsense_prompt("Summarize.", "word " * 50)
        prefix_ids = tokenize_ids(TOKENIZER, prefix)
        # max_length 恰好在前缀末尾之后一个位置：截断命中 EOS，supervised_count >= 1。
        max_length = len(prefix_ids) + 1
        example = build_supervised_example(TOKENIZER, prefix, "true", max_length=max_length)
        self.assertTrue(example["truncated"])
        unpadded = len(example["input_ids"]) - example["pad_token_count"]
        # 截断后长度必须 <= max_length（padding 不计入截断限制）。
        self.assertLessEqual(unpadded, max_length)
        self.assertGreater(example["supervised_token_count"], 0)

    def test_truncation_removing_all_supervision_raises(self):
        """前缀占满预算时不得静默产生空监督样本。"""
        prefix = build_commonsense_prompt("Summarize.", "word " * 400)
        with self.assertRaisesRegex(ValueError, "no supervised token remains"):
            build_supervised_example(TOKENIZER, prefix, "true", max_length=8)

    def test_calibration_input_has_no_answer_or_eos(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        record = build_calibration_input(TOKENIZER, prefix)
        self.assertEqual(record["prediction_position"], len(record["input_ids"]) - 1)
        self.assertNotIn(M7_EOS_TOKEN_ID, record["input_ids"])
        self.assertFalse(record["truncated"])
        expected = tokenize_ids(TOKENIZER, prefix)
        self.assertEqual(record["input_ids"], expected)

    def test_calibration_truncation_reports_flag(self):
        prefix = build_commonsense_prompt("Summarize.", "word " * 400)
        record = build_calibration_input(TOKENIZER, prefix, max_length=32)
        self.assertTrue(record["truncated"])
        self.assertEqual(len(record["input_ids"]), 32)
        self.assertEqual(record["original_token_count"], len(tokenize_ids(TOKENIZER, prefix)))

    def test_deterministic_across_calls(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        first = build_supervised_example(TOKENIZER, prefix, "true")
        second = build_supervised_example(TOKENIZER, prefix, "true")
        self.assertEqual(first, second)

    def test_answer_change_keeps_prefix_hash_but_changes_full_hash(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        left = build_supervised_example(TOKENIZER, prefix, "true")
        right = build_supervised_example(TOKENIZER, prefix, "false")
        self.assertEqual(left["prefix_ids_sha256"], right["prefix_ids_sha256"])
        self.assertNotEqual(left["full_ids_sha256"], right["full_ids_sha256"])

    def test_token_identity_overlap_detects_shared_prefix(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        other = build_commonsense_prompt("Answer the question.", "Is the sky green?")
        train = [{"sample_id": "t", "prefix_ids_sha256": build_calibration_input(TOKENIZER, prefix)["prefix_ids_sha256"]}]
        disjoint = token_identity_overlap(
            train, [{"sample_id": "e", "prefix_ids_sha256": build_calibration_input(TOKENIZER, other)["prefix_ids_sha256"]}])
        self.assertTrue(disjoint["token_identity_disjoint"])
        overlap = token_identity_overlap(
            train, [{"sample_id": "e", "prefix_ids_sha256": build_calibration_input(TOKENIZER, prefix)["prefix_ids_sha256"]}])
        self.assertFalse(overlap["token_identity_disjoint"])
        self.assertEqual(overlap["intersection_size"], 1)

    def test_evaluation_input_is_not_truncated_and_reports_overshoot(self):
        """评估侧不截断（2026-11-16 修订）：即使超出参考训练长度也保留完整前缀。

        用超长前缀构造，确认 ids 与未截断分词完全一致、truncated 恒为 False，
        且超出参考长度只是被标注而不是被裁掉。前缀长度由实测 token 数决定，
        不假设固定词数对应的 token 数。
        """
        prefix = build_commonsense_prompt("Summarize.", "word " * 700)
        full_ids = tokenize_ids(TOKENIZER, prefix)
        self.assertGreater(len(full_ids), M7_MAX_LENGTH)
        record = build_evaluation_input(TOKENIZER, prefix)
        self.assertFalse(record["truncated"])
        self.assertEqual(record["input_ids"], full_ids)
        self.assertEqual(record["token_count"], len(full_ids))
        self.assertTrue(record["exceeds_training_length"])
        self.assertEqual(record["reference_max_length"], M7_MAX_LENGTH)
        self.assertEqual(M7_EVALUATION_TRUNCATION, "none_keep_full_prefix")
        # 不追加 EOS、不构造 labels：生成入参与校准输入同样只含前缀
        self.assertNotIn(M7_EOS_TOKEN_ID, record["input_ids"])
        self.assertNotIn("labels", record)

    def test_evaluation_input_shorter_than_reference_is_not_flagged(self):
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        record = build_evaluation_input(TOKENIZER, prefix)
        self.assertFalse(record["exceeds_training_length"])
        self.assertFalse(record["truncated"])
        self.assertEqual(record["input_ids"], tokenize_ids(TOKENIZER, prefix))

    def test_evaluation_and_calibration_share_prefix_bytes_when_under_limit(self):
        """协议要求训练/校准/生成共享同一前缀字节；未超限时两者必须逐 token 相同。"""
        prefix = build_commonsense_prompt("Answer the question.", "Is the sky blue?")
        evaluation = build_evaluation_input(TOKENIZER, prefix)
        calibration = build_calibration_input(TOKENIZER, prefix)
        self.assertEqual(evaluation["input_ids"], calibration["input_ids"])
        self.assertEqual(evaluation["prefix_ids_sha256"], calibration["prefix_ids_sha256"])

    def test_evaluation_input_rejects_empty_text_and_bad_reference_length(self):
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            build_evaluation_input(TOKENIZER, "")
        for bad in (0, -1, True, 1.0, "512"):
            with self.subTest(reference_max_length=bad), self.assertRaisesRegex(ValueError, "reference_max_length"):
                build_evaluation_input(TOKENIZER, "Q:", reference_max_length=bad)

    def test_real_answer_boundary_strings(self):
        """真实任务答案串在前缀后不得产生空或异常 token。"""
        prefix = build_commonsense_prompt("Answer the question.", "Question?")
        for answer in ("true", "false", "solution1", "answer3", "ending4", "option2"):
            with self.subTest(answer=answer):
                example = build_supervised_example(TOKENIZER, prefix, answer)
                self.assertGreaterEqual(example["supervised_token_count"], 2)
                self.assertEqual(example["input_ids"][example["prompt_token_count"] + example["supervised_token_count"] - 1], M7_EOS_TOKEN_ID)


@REQUIRES_TOKENIZER
class CollateTests(unittest.TestCase):
    """批组装：右 padding、批宽对齐与 pad==eos 下的 mask 权威性。

    使用真实 0.5B 分词器构造长度不同的样本，验证批内每条样本的有效区间都能由
    mask/labels 精确恢复。不涉及模型前向，因此不能证明批训练数值正确。
    """

    def _examples(self, pairs):
        """按 (instruction, input, answer) 构造样本，长度因 input 不同而不同。"""
        return [
            build_supervised_example(
                TOKENIZER,
                build_commonsense_prompt(instruction, input_text),
                answer,
            )
            for instruction, input_text, answer in pairs
        ]

    def test_batch_pads_right_to_multiple_and_preserves_valid_regions(self):
        examples = self._examples([
            ("Answer briefly.", "", "true"),
            ("Answer briefly.", "word " * 40, "false"),
            ("Answer briefly.", "word " * 5, "solution1"),
        ])
        batch = collate_supervised_examples(examples, pad_token_id=TOKENIZER.pad_token_id)

        self.assertEqual(batch["batch_size"], 3)
        self.assertEqual(batch["batch_length"] % 8, 0)
        self.assertGreaterEqual(batch["batch_length"], batch["longest_valid_length"])
        # 三条样本长度不同，所以批内确实发生了 padding。
        self.assertGreater(batch["padded_positions_total"], 0)
        self.assertEqual(len({len(row) for row in batch["input_ids"]}), 1)

        for index, valid in enumerate(batch["valid_lengths"]):
            mask = batch["attention_mask"][index]
            labels = batch["labels"][index]
            ids = batch["input_ids"][index]
            self.assertEqual(len(ids), batch["batch_length"])
            # 有效区间 mask=1，padding 区间 mask=0，且边界单调（右侧补齐）。
            self.assertEqual(mask[:valid], [1] * valid)
            self.assertEqual(mask[valid:], [0] * (batch["batch_length"] - valid))
            # padding 位置的 labels 必须被忽略，否则 loss 会学到填充。
            self.assertEqual(labels[valid:], [PROMPT_LABEL_IGNORE] * (batch["batch_length"] - valid))
            # padding 位置写入 pad id，而此处 pad id 与 EOS 同值。
            self.assertEqual(ids[valid:], [TOKENIZER.pad_token_id] * (batch["batch_length"] - valid))

    def test_real_eos_is_not_masked_despite_sharing_id_with_pad(self):
        """最关键的回归：pad == eos 时，真实 EOS 的 label 必须保留，mask 必须为 1。

        若实现用 `input_ids != pad_token_id` 推断有效位置，凡是 EOS 都会被判成 padding，
        label 被置 -100、mask 置 0——恰好把最该学的结束符屏蔽掉。这里逐条断言每条样本
        监督区间末尾的 EOS 仍然可见。
        """
        self.assertEqual(TOKENIZER.pad_token_id, TOKENIZER.eos_token_id)
        examples = self._examples([
            ("Answer briefly.", "", "true"),
            ("Answer briefly.", "word " * 30, "false"),
        ])
        batch = collate_supervised_examples(examples, pad_token_id=TOKENIZER.pad_token_id)
        self.assertTrue(batch["pad_equals_eos"])

        for index, example in enumerate(examples):
            valid = batch["valid_lengths"][index]
            eos_position = valid - 1
            with self.subTest(sample=index):
                self.assertEqual(batch["input_ids"][index][eos_position], M7_EOS_TOKEN_ID)
                self.assertEqual(batch["labels"][index][eos_position], M7_EOS_TOKEN_ID)
                self.assertEqual(batch["attention_mask"][index][eos_position], 1)
                # 统计每条样本监督区间内有多少个真实 EOS label，至少为 1。
                supervised = batch["labels"][index][example["prompt_token_count"]:valid]
                self.assertGreaterEqual(sum(value == M7_EOS_TOKEN_ID for value in supervised), 1)

    def test_prompt_region_is_ignored_and_supervised_region_is_not(self):
        examples = self._examples([("Answer briefly.", "", "true")])
        batch = collate_supervised_examples(examples, pad_token_id=TOKENIZER.pad_token_id)
        prompt_n = examples[0]["prompt_token_count"]
        labels = batch["labels"][0]
        self.assertEqual(labels[:prompt_n], [PROMPT_LABEL_IGNORE] * prompt_n)
        supervised = labels[prompt_n:batch["valid_lengths"][0]]
        self.assertTrue(all(value != PROMPT_LABEL_IGNORE for value in supervised))

    def test_single_example_batch_has_no_cross_sample_padding(self):
        """单条样本时批宽就等于该样本有效长度的 8 倍数上取整，无额外跨样本 padding。"""
        examples = self._examples([("Answer briefly.", "", "true")])
        batch = collate_supervised_examples(examples, pad_token_id=TOKENIZER.pad_token_id)
        valid = batch["valid_lengths"][0]
        expected = -(-valid // 8) * 8
        self.assertEqual(batch["batch_length"], expected)

    def test_rejects_duplicate_sample_ids_and_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "at least one example"):
            collate_supervised_examples([], pad_token_id=TOKENIZER.pad_token_id)
        example = build_supervised_example(
            TOKENIZER, build_commonsense_prompt("Answer briefly.", ""), "true")
        duplicated = dict(example, sample_id="same")
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            collate_supervised_examples([duplicated, dict(duplicated)], pad_token_id=TOKENIZER.pad_token_id)

    def test_rejects_broken_single_example_regions(self):
        """单条样本内部边界被破坏时必须拒绝，而不是在批组装时静默修正。"""
        example = build_supervised_example(
            TOKENIZER, build_commonsense_prompt("Answer briefly.", ""), "true")

        broken_mask = dict(example)
        broken_mask["attention_mask"] = [1] * (len(example["input_ids"]) - 1) + [1]
        with self.assertRaisesRegex(ValueError, "attention mask is not contiguous"):
            collate_supervised_examples([broken_mask], pad_token_id=TOKENIZER.pad_token_id)

        broken_labels = dict(example)
        valid = example["prompt_token_count"] + example["supervised_token_count"]
        labels = list(example["labels"])
        labels[valid] = 12345  # padding 位置不应带真实 token id
        broken_labels["labels"] = labels
        with self.assertRaisesRegex(ValueError, "padding labels are not ignored"):
            collate_supervised_examples([broken_labels], pad_token_id=TOKENIZER.pad_token_id)

    def test_rejects_invalid_arguments(self):
        example = build_supervised_example(
            TOKENIZER, build_commonsense_prompt("Answer briefly.", ""), "true")
        for kwargs, message in (
            ({"pad_token_id": -1}, "pad_token_id"),
            ({"pad_token_id": TOKENIZER.pad_token_id, "pad_multiple": 0}, "pad_multiple"),
            ({"pad_token_id": TOKENIZER.pad_token_id, "eos_token_id": -5}, "eos_token_id"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                collate_supervised_examples([example], **kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
