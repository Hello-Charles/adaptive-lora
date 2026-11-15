"""M6 共享训练/评估 runner 的微型真实 Qwen2/PEFT 集成测试，不使用下载模型或正式数据。

三个方法均走同一 M5 训练器和同一 prompt-only evaluator；实测小模型参数量，不把规划层数
当作真实权重验证。这里的 strict exact match 只用于接口验收，不评价自然语言能力。
"""

import copy
import unittest
from unittest.mock import patch

import torch

from adaptive_lora.baselines import plan_baselines
from adaptive_lora.comparison import run_variant, summarize_comparison
from adaptive_lora.protocol import SharedConfig, build_smoke_protocol, stable_hash
from adaptive_lora.trainer import fingerprint_parameters
from test_lora_targets import tiny_model
from test_protocol import ProtocolTokenizer, calibration_fixture


class ComparisonTokenizer(ProtocolTokenizer):
    """测试解码只标记 token 数字，不模拟语义回答或真实 tokenizer。"""

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(f"[{token}]" for token in ids if token != self.eos_token_id)


class UnifiedComparisonTests(unittest.TestCase):
    """基础 state_dict 在各实例间复制，但参数对象必须独立，避免前一种方法污染后一种。"""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.state = copy.deepcopy(tiny_model().state_dict())
        cls.tokenizer = ComparisonTokenizer()
        cls.protocol = build_smoke_protocol(cls.tokenizer, calibration_fixture(), steps=1)
        cls.config = SharedConfig(steps=1, rank=2, alpha=4, max_new_tokens=1)
        cls.plans = plan_baselines(3, 1, [0], layer_seed=42)
        cls.reports = []
        for plan in cls.plans:
            model = cls.fresh()
            _, report = run_variant(plan, model, cls.tokenizer, cls.protocol, cls.config,
                                    revision="a" * 40, expected_base_fingerprint=fingerprint_parameters(model.named_parameters()))
            cls.reports.append(report)

    @classmethod
    def fresh(cls):
        """重新构造独立模型再恢复同一随机基础字节，不复用已注入/训练的实例。"""
        model = tiny_model()
        model.load_state_dict(copy.deepcopy(cls.state))
        return model

    def test_actual_full_random_adaptive_parameter_counts_and_common_contract(self):
        self.assertEqual([row["parameter_audit"]["trainable_parameter_elements"] for row in self.reports], [480, 160, 160])
        self.assertEqual(len({row["shared_config_sha256"] for row in self.reports}), 1)
        self.assertEqual(len({row["protocol_sha256"] for row in self.reports}), 1)
        self.assertEqual(len({row["training"]["base_after"]["sha256"] for row in self.reports}), 1)
        for report in self.reports:
            self.assertTrue(report["evaluation"]["parameters_unchanged"])
            self.assertEqual(report["evaluation"]["scores"]["total"], 3)
            self.assertEqual(report["training"]["steps"][0]["sample_id"], self.protocol["training_samples"][0]["sample_id"])
        comparison = summarize_comparison(self.reports)
        self.assertFalse(comparison["same_parameter_budget"])
        self.assertEqual([row["trainable_fraction_of_full"] for row in comparison["methods"]], [1.0, 1 / 3, 1 / 3])

    def test_same_plan_seed_and_data_repeat_exact_tiny_report(self):
        model = self.fresh()
        _, report = run_variant(self.plans[2], model, self.tokenizer, copy.deepcopy(self.protocol), self.config,
                                revision="a" * 40, expected_base_fingerprint=fingerprint_parameters(model.named_parameters()))
        self.assertEqual(report, self.reports[2])

    def test_different_settings_or_protocol_cannot_be_reported_as_fair_comparison(self):
        for change in ("settings", "protocol", "training_ids", "evaluation_ids", "missing", "duplicate"):
            rows = copy.deepcopy(self.reports)
            if change == "settings":
                rows[1]["shared_config"]["generation"]["max_new_tokens"] = 2
                rows[1]["shared_config_sha256"] = stable_hash(rows[1]["shared_config"])
            elif change == "protocol":
                rows[1]["protocol_sha256"] = "0" * 64
            elif change == "training_ids":
                rows[1]["training"]["steps"][0]["sample_id"] = "other-training-sample"
            elif change == "evaluation_ids":
                rows[1]["evaluation"]["predictions"][0]["prefix_token_sha256"] = "0" * 64
            elif change == "missing":
                rows.pop()
            else:
                rows[1] = copy.deepcopy(rows[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_comparison(rows)

    def test_changed_base_is_rejected_before_lora_or_forward(self):
        model = self.fresh()
        before = fingerprint_parameters(model.named_parameters())
        with torch.no_grad():
            next(model.parameters()).reshape(-1)[0].add_(1)
        with patch.object(model, "forward") as forward, self.assertRaises(ValueError):
            run_variant(self.plans[0], model, self.tokenizer, self.protocol, self.config,
                        revision="a" * 40, expected_base_fingerprint=before)
        forward.assert_not_called()
        self.assertFalse(hasattr(model, "peft_config"))

    def test_bad_protocol_step_mismatch_or_disabled_grad_fails_before_injection(self):
        for condition in ("protocol", "steps", "grad"):
            model = self.fresh()
            protocol = copy.deepcopy(self.protocol)
            config = SharedConfig(steps=2) if condition == "steps" else self.config
            if condition == "protocol":
                protocol["references"][0]["text"] = "mutated"
            with patch.object(model, "forward") as forward:
                if condition == "grad":
                    with torch.no_grad(), self.assertRaises(ValueError):
                        run_variant(self.plans[0], model, self.tokenizer, protocol, config, revision="a" * 40,
                                    expected_base_fingerprint=fingerprint_parameters(model.named_parameters()))
                else:
                    with self.assertRaises(ValueError):
                        run_variant(self.plans[0], model, self.tokenizer, protocol, config, revision="a" * 40,
                                    expected_base_fingerprint=fingerprint_parameters(model.named_parameters()))
                forward.assert_not_called()
            self.assertFalse(hasattr(model, "peft_config"))

    def test_protocol_mutation_during_training_is_rejected_before_evaluation(self):
        model = self.fresh()
        protocol = copy.deepcopy(self.protocol)
        def mutate(_):
            """模拟外部日志回调错误地改动参考答案，不能使用混合版本协议继续评估。"""
            protocol["references"][0]["text"] = "changed during training"
        with patch("adaptive_lora.comparison.predict_greedy") as predictor, self.assertRaises(ValueError):
            run_variant(self.plans[2], model, self.tokenizer, protocol, self.config, revision="a" * 40,
                        expected_base_fingerprint=fingerprint_parameters(model.named_parameters()), on_step=mutate)
        predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
