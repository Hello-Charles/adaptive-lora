"""M6 统一配置与拆分边界测试；字符级分词替身仅验证控制流，不模拟真实 Qwen 的 BPE。

校准可属于训练集，这是论文允许的来源关系；评估不得与二者按 ID、prompt 或 token 重叠。
这些检查不能识别语义改写重复或预训练污染，不能将哈希隔离表述为自然语言层面的无泄漏。
"""

import copy
import unittest

from adaptive_lora.data import hash_text
from adaptive_lora.protocol import (
    SharedConfig, build_smoke_protocol, check_split_isolation, protocol_hash, stable_hash, validate_protocol,
)
from test_trainer import BoundaryTokenizer


class ProtocolTokenizer(BoundaryTokenizer):
    """复用单字符测试分词，加可追溯模板字段；不声称具备真实模型回答能力。"""

    def get_chat_template(self):
        return "fixture U: instruction then A:"


def calibration_fixture():
    """来源记录只作拆分审计夹具；不代表已有的 30 条真实校准结果。"""
    return [{"sample_id": "calibration/0", "input_ids": [60, 61], "prompt_hash": hash_text("calibration fixture")}]


class SharedProtocolTests(unittest.TestCase):
    """三个方法只能共享同一个不可变配置和内容指纹，不允许按方法偷偷换训练/评估设置。"""

    def test_config_is_explicit_serializable_and_hashes_stably(self):
        config = SharedConfig()
        saved = config.to_dict()
        self.assertEqual(saved["train_seed"], 42)
        self.assertEqual(saved["rank"], 16)
        self.assertEqual(saved["batch_size"], 1)
        self.assertEqual(saved["gradient_accumulation"], 1)
        self.assertEqual(saved["optimizer"]["betas"], [0.9, 0.999])
        self.assertEqual(saved["generation"]["do_sample"], False)
        self.assertEqual(stable_hash(saved), stable_hash(SharedConfig().to_dict()))
        changed = SharedConfig(learning_rate=2e-4)
        self.assertNotEqual(stable_hash(saved), stable_hash(changed.to_dict()))

    def test_invalid_or_unbounded_smoke_settings_are_rejected(self):
        cases = ({"steps": 4}, {"steps": 0}, {"rank": True}, {"rank": 0}, {"alpha": 0},
                 {"train_seed": -1}, {"max_new_tokens": 0}, {"max_new_tokens": 9},
                 {"learning_rate": float("nan")}, {"learning_rate": 0}, {"weight_decay": -1})
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                SharedConfig(**fields)

    def test_builder_keeps_eval_answers_out_of_prompt_and_shares_saved_training_mask(self):
        protocol = build_smoke_protocol(ProtocolTokenizer(), calibration_fixture(), steps=3)
        audit = validate_protocol(protocol)
        self.assertEqual(len(protocol["training_samples"]), 3)
        self.assertEqual(len(protocol["evaluation_prompts"]), 3)
        self.assertEqual(protocol["content_sha256"], protocol_hash(protocol))
        self.assertEqual(audit["evaluation_train_overlaps"], {"sample_id": 0, "prompt_sha256": 0, "token_sha256": 0})
        self.assertEqual(audit["evaluation_calibration_overlaps"], {"sample_id": 0, "prompt_sha256": 0, "token_sha256": 0})
        for sample in protocol["training_samples"]:
            p = sample["prompt_token_count"]
            self.assertEqual(sample["labels"], [-100] * p + sample["input_ids"][p:])
        self.assertTrue(all("reference" not in sample and "answer" not in sample and "labels" not in sample
                            for sample in protocol["evaluation_prompts"]))
        changed = copy.deepcopy(protocol)
        changed["references"][0]["text"] = "different gold answer"
        changed["content_sha256"] = protocol_hash(changed)
        validate_protocol(changed)
        self.assertEqual(changed["evaluation_prompts"], protocol["evaluation_prompts"])
        self.assertNotEqual(changed["content_sha256"], protocol["content_sha256"])

    def test_mutated_protocol_and_rehashed_bad_labels_or_references_are_rejected(self):
        original = build_smoke_protocol(ProtocolTokenizer(), calibration_fixture())
        changed = copy.deepcopy(original)
        changed["training_samples"][0]["labels"][0] = 3
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed["content_sha256"] = protocol_hash(changed)
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = copy.deepcopy(original)
        changed["references"][0]["sample_id"] = "wrong-id"
        changed["content_sha256"] = protocol_hash(changed)
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_builder_steps_are_limited_before_tokenization(self):
        for steps in (0, 4, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                build_smoke_protocol(ProtocolTokenizer(), calibration_fixture(), steps=steps)


class SplitIsolationTests(unittest.TestCase):
    """直接重复检测使用实算 token 指纹，不信任外部宣称的 token hash。"""

    def record(self, sample_id="train/0", ids=None, text="train prompt"):
        return {"sample_id": sample_id, "input_ids": [1, 4] if ids is None else ids, "prompt_sha256": hash_text(text)}

    def test_calibration_training_overlap_is_allowed_but_reported(self):
        train = [self.record()]
        evaluation = [self.record("eval/0", [1, 5], "held out prompt")]
        result = check_split_isolation(train, evaluation, copy.deepcopy(train))
        self.assertEqual(result["calibration_train_overlaps"], {"sample_id": 1, "prompt_sha256": 1, "token_sha256": 1})

    def test_eval_overlap_by_id_prompt_or_token_is_rejected_even_if_other_fields_change(self):
        train = [self.record()]
        calibration = [self.record("cal/0", [6], "calibration prompt")]
        for evaluation in (
            self.record("train/0", [9], "different"),
            self.record("eval/0", [9], "train prompt"),
            self.record("eval/0", [1, 4], "different"),
            self.record("eval/0", [6], "different"),
            self.record("cal/0", [9], "different"),
        ):
            with self.subTest(evaluation=evaluation), self.assertRaises(ValueError):
                check_split_isolation(train, [evaluation], calibration)

    def test_missing_hash_duplicate_ids_and_illegal_tokens_are_rejected(self):
        train = [self.record()]
        evaluation = [self.record("eval/0", [9], "different")]
        for bad in ([self.record(), self.record()], [{"sample_id": "cal/0", "input_ids": [6]}],
                    [self.record("cal/0", [True], "calibration prompt")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_split_isolation(train, evaluation, bad)


if __name__ == "__main__":
    unittest.main()
