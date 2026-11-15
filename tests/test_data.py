"""校准数据契约的离线测试；替身不具备真实 Qwen 分词或模型前向能力。

合成答案专门用于检查字段隔离，不能证明自然语言输入本身不含答案。
真实 tokenizer 测试只读固定本地缓存；真实 Alpaca 清单由独立验收命令验证。
"""

import copy
import importlib.util
from pathlib import Path
import random
import unittest

from adaptive_lora.data import (
    build_calibration_prompt,
    build_manifest,
    build_user_content,
    hash_text,
    manifest_content_hash,
    position_from_mask,
    row_hash,
    sample_indices,
    tokenize_batch,
    tokenize_single,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = ROOT / "data" / "models" / "qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_SHA = "060db6499f32faf8b98477b0a26969ef7d8b9987"

FIXTURE_ROWS = [
    {"instruction": "Write a haiku about code.", "input": "", "output": "Quiet logic flows"},
    {"instruction": "List two cities.", "input": "focus on capitals", "output": "London, Tokyo"},
]


class FakeTokenizer:
    """字符级替身支持左右截断/填充，仅用于长度、选项及索引控制流断言。

    特殊标记不是 Qwen token；记录 add_special_tokens 选项不能替代真实 tokenizer 核验。
    """

    pad_token_id = 0
    eos_token_id = 250
    bos_token_id = None
    padding_side = "right"
    truncation_side = "right"
    is_fast = False

    def __init__(self):
        self.template_calls = []
        self.tokenize_calls = []

    def get_chat_template(self):
        """夹具模板只作清单指纹；不由 Jinja 实际渲染。"""
        return "fixture: user content followed by assistant generation marker"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        """只实现字符串路径；消息字段与模板末尾可观察，不能模拟 Qwen 的默认 system。"""
        self.template_calls.append(
            {"messages": messages, "add_generation_prompt": add_generation_prompt, "tokenize": tokenize}
        )
        text = "\n".join(f"<|{message['role']}|>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\n<|assistant|>"
        return text

    def __call__(self, texts, **kwargs):
        self.tokenize_calls.append(kwargs)
        if isinstance(texts, str):
            texts = [texts]
        rows = [[(ord(char) % 250) + 1 for char in text] for text in texts]
        if kwargs.get("truncation"):
            limit = kwargs["max_length"]
            rows = [row[:limit] if self.truncation_side == "right" else row[-limit:] for row in rows]
        masks = [[1] * len(row) for row in rows]
        if kwargs.get("padding"):
            width = max(len(row) for row in rows)
            for index, (row, mask) in enumerate(zip(rows, masks)):
                pad = width - len(row)
                if self.padding_side == "right":
                    rows[index], masks[index] = row + [0] * pad, mask + [0] * pad
                else:
                    rows[index], masks[index] = [0] * pad + row, [0] * pad + mask
        return {"input_ids": rows, "attention_mask": masks}


class DataPreparationTests(unittest.TestCase):
    """合成样本约束清单结构及输入边界，不把 mock 当作真实来源证据。"""

    def make_manifest(self, **overrides):
        """为边界测试提供同一组小夹具；来源 revision 不是实际数据集 SHA。"""
        kwargs = {
            "rows": FIXTURE_ROWS,
            "indices": [0, 1],
            "tokenizer": FakeTokenizer(),
            "dataset_id": "fixture/alpaca",
            "dataset_revision": "a" * 40,
            "model_id": "Qwen/Qwen2.5-0.5B",
            "model_revision": MODEL_SHA,
            "max_length": 128,
            "seed": 20261112,
        }
        kwargs.update(overrides)
        return build_manifest(**kwargs)

    def test_sample_indices_are_reproducible_distinct_and_in_bounds(self):
        """人口大小只是测试参数；局部 RNG 还必须不改变调用者的全局状态。"""
        state = random.getstate()
        first = sample_indices(52_002, 30, 20261112)
        second = sample_indices(52_002, 30, 20261112)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(len(set(first)), 30)
        self.assertTrue(all(0 <= index < 52_002 for index in first))
        self.assertEqual(random.getstate(), state)

    def test_sample_indices_reject_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            sample_indices(10, 30, 20261112)
        with self.assertRaisesRegex(ValueError, "positive"):
            sample_indices(10, 0, 20261112)

    def test_user_content_uses_only_valid_input_fields(self):
        """strip 和空 input 拼接规则是工程协议；空 instruction/非文本输入明确拒绝。"""
        self.assertEqual(build_user_content(" a ", "  "), "a")
        self.assertEqual(build_user_content("a", " b "), "a\n\nb")
        with self.assertRaisesRegex(ValueError, "instruction"):
            build_user_content(" ", "input")
        with self.assertRaisesRegex(ValueError, "text"):
            build_user_content("a", ["not text"])

    def test_prompt_uses_user_message_and_generation_prompt(self):
        tokenizer = FakeTokenizer()
        prompt = build_calibration_prompt(tokenizer, "do it", "quickly")
        call = tokenizer.template_calls[-1]
        self.assertEqual(call["messages"], [{"role": "user", "content": "do it\n\nquickly"}])
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["tokenize"])
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_position_from_mask_rejects_nonbinary_empty_and_holey_masks(self):
        self.assertEqual(position_from_mask([1, 1, 1]), 2)
        self.assertEqual(position_from_mask([1, 1, 0]), 1)
        self.assertEqual(position_from_mask([0, 1, 1]), 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            position_from_mask([1, 0, 1])
        with self.assertRaisesRegex(ValueError, "no active"):
            position_from_mask([0, 0])
        with self.assertRaisesRegex(ValueError, "binary"):
            position_from_mask([1, 2, 0])

    def test_single_tokenization_reports_actual_not_approximate_truncation(self):
        """长度刚好等于上限不算截断；渲染文本已有特殊标记，不得再次自动添加。"""
        tokenizer = FakeTokenizer()
        exact = tokenize_single(tokenizer, "abcdefgh", max_length=8)
        long = tokenize_single(tokenizer, "abcdefghijk", max_length=8)
        self.assertFalse(exact["truncated"])
        self.assertEqual(exact["original_token_count"], 8)
        self.assertTrue(long["truncated"])
        self.assertEqual(long["original_token_count"], 11)
        self.assertEqual(long["input_ids"], exact["input_ids"])
        self.assertEqual(long["prediction_position"], 7)
        self.assertTrue(all(not call["add_special_tokens"] for call in tokenizer.tokenize_calls))
        with self.assertRaisesRegex(ValueError, "positive"):
            tokenize_single(tokenizer, "hello", max_length=0)

    def test_batch_positions_match_single_with_left_and_right_padding(self):
        """右 pad 下短行末尾不是矩阵末列；左 pad 下索引必须包含左侧偏移。"""
        tokenizer = FakeTokenizer()
        for side in ("right", "left"):
            tokenizer.padding_side = side
            with self.subTest(side=side):
                batch = tokenize_batch(tokenizer, ["short", "a longer prompt"], max_length=32)
                width = len(batch["input_ids"][0])
                expected = 4 if side == "right" else width - 1
                self.assertEqual(batch["prediction_positions"][0], expected)
                self.assertEqual(batch["prediction_positions"][1], width - 1)
                for text, ids, mask in zip(["short", "a longer prompt"], batch["input_ids"], batch["attention_mask"]):
                    single = tokenize_single(tokenizer, text, max_length=32)
                    self.assertEqual([token for token, active in zip(ids, mask) if active], single["input_ids"])

    def test_batch_requires_nonempty_prompts_and_an_explicit_mask(self):
        """缺失 mask 时不能把 pad 当有效 token；替身有意模拟不符合契约的返回。"""
        class NoMaskTokenizer(FakeTokenizer):
            """只破坏 mask 字段，不模拟任何实际第三方 tokenizer。"""

            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                result.pop("attention_mask")
                return result

        with self.assertRaisesRegex(ValueError, "non-empty"):
            tokenize_batch(FakeTokenizer(), [], max_length=32)
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            tokenize_batch(NoMaskTokenizer(), ["a", "longer"], max_length=32)

    def test_manifest_excludes_answers_even_when_reference_fields_change(self):
        """改变答案只能改变原始行指纹，不能改变 prompt、输入 token 或输入来源指纹。"""
        changed = copy.deepcopy(FIXTURE_ROWS)
        for row in changed:
            row.update(output="UNIQUE_SYNTHETIC_ANSWER", reference="UNIQUE_REFERENCE")
        original = self.make_manifest()
        replacement = self.make_manifest(rows=changed)
        for sample, other, row in zip(original["samples"], replacement["samples"], FIXTURE_ROWS):
            self.assertNotIn("output", sample)
            self.assertNotIn("reference", sample)
            self.assertNotIn(row["output"], sample["prompt"])
            self.assertEqual(sample["prompt_hash"], hash_text(sample["prompt"]))
            self.assertEqual(sample["row_hash"], row_hash(row))
            self.assertNotEqual(sample["row_hash"], other["row_hash"])
            for key in ("prompt", "input_hash", "input_ids", "attention_mask", "prediction_position"):
                self.assertEqual(sample[key], other[key])

    def test_manifest_records_actual_template_sides_and_stable_content_hash(self):
        tokenizer = FakeTokenizer()
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        first = self.make_manifest(tokenizer=tokenizer)
        second = self.make_manifest(tokenizer=tokenizer)
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        self.assertEqual(first["content_sha256"], manifest_content_hash(first))
        self.assertEqual(first["tokenizer"]["padding_side"], "left")
        self.assertEqual(first["tokenizer"]["truncation_side"], "left")
        self.assertFalse(first["tokenizer"]["add_special_tokens"])
        self.assertEqual(first["template"]["sha256"], hash_text(tokenizer.get_chat_template()))
        self.assertEqual(first["dataset"]["split"], "train")
        self.assertEqual(first["dataset"]["population_size"], 2)
        first["created_at_utc"] = "different timestamp"
        self.assertEqual(first["content_sha256"], manifest_content_hash(first))
        first["samples"][0]["input_ids"][0] += 1
        self.assertNotEqual(first["content_sha256"], manifest_content_hash(first))

    def test_manifest_rejects_invalid_indices_and_overlength_without_resampling(self):
        """不接受负索引或重复索引；超长时整次失败，避免偷偷改变固定抽样集合。"""
        for indices in ([], [0, 0], [-1], [2]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.make_manifest(indices=indices)
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            self.make_manifest(max_length=8)

    def test_real_tokenizer_template_special_tokens_truncation_and_padding(self):
        """真实 Qwen tokenizer 离线验证，不加载权重，不声称验证批量 logits 等价。

        直接 chat-template 分词必须与渲染后 add_special_tokens=False 一致；长合成输入只
        用于测试截断工具，正式清单拒绝这种超长输入，避免把 user 中途当成回答起点。
        """
        if not MODEL_CACHE.is_dir():
            self.skipTest("pinned model cache is not present on this machine")
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers is not installed in this interpreter")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", revision=MODEL_SHA, cache_dir=str(MODEL_CACHE),
            local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        content = build_user_content("List two cities.", "focus on capitals")
        prompt = build_calibration_prompt(tokenizer, "List two cities.", "focus on capitals")
        direct = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True,
        )
        single = tokenize_single(tokenizer, prompt, max_length=512)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(single["input_ids"], direct["input_ids"])
        self.assertEqual(single["prediction_position"], len(single["input_ids"]) - 1)
        exact = tokenize_single(tokenizer, prompt, max_length=len(single["input_ids"]))
        self.assertFalse(exact["truncated"])
        long = tokenize_single(tokenizer, prompt + " example" * 600, max_length=512)
        self.assertTrue(long["truncated"])
        self.assertEqual(len(long["input_ids"]), 512)
        short_prompt = build_calibration_prompt(tokenizer, "Hi", "")
        batch = tokenize_batch(tokenizer, [short_prompt, prompt], max_length=512)
        for index, text in enumerate([short_prompt, prompt]):
            one = tokenize_single(tokenizer, text, max_length=512)
            active_ids = [token for token, active in zip(batch["input_ids"][index], batch["attention_mask"][index]) if active]
            self.assertEqual(active_ids, one["input_ids"])
            self.assertEqual(batch["prediction_positions"][index], len(one["input_ids"]) - 1)
        self.assertEqual(batch["attention_mask"][0][-1], 0)


if __name__ == "__main__":
    unittest.main()
