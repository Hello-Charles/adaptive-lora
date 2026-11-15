"""M6 评估边界测试：受控 logits 验证生成输入，另用微型真实 Qwen2/PEFT 检查兼容性。

受控模型不会回答自然语言；它只能证明 argmax、EOS、长度上限、模式恢复和 reference
隔离。严格 strip-only exact match 是 smoke 评分规则，不是论文任务的答案抽取实现。
"""

from types import SimpleNamespace
import unittest

import torch

from adaptive_lora.data import hash_text
from adaptive_lora.evaluation import predict_greedy, score_exact_match
from adaptive_lora.trainer import attach_selective_lora, fingerprint_parameters
from test_lora_targets import tiny_model


class DecodeTokenizer:
    """仅把受控 token 转为测试字符串；不模拟真实 Qwen 的 special-token 解码。"""

    eos_token_id = 2

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join({2: "", 4: " yes ", 5: "no"}.get(token, str(token)) for token in ids)


class ControlledGenerationModel(torch.nn.Module):
    """根据已追加 token 的数量提供可预测 logits，并记录实际输入及梯度/模式。"""

    def __init__(self, outputs=(4, 2), fail_at=None, nan=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.child = torch.nn.Dropout(0.0)
        self.config = SimpleNamespace(vocab_size=8)
        self.outputs, self.fail_at, self.nan = outputs, fail_at, nan
        self.calls = []

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.calls.append({"ids": input_ids.tolist(), "mask": attention_mask.tolist(), "grad": torch.is_grad_enabled(),
                           "training": self.training, "use_cache": use_cache})
        if self.fail_at == len(self.calls):
            raise RuntimeError("controlled generation failure")
        logits = torch.zeros(1, input_ids.shape[1], 8)
        index = min(input_ids.shape[1] - 2, len(self.outputs) - 1)
        logits[0, -1, self.outputs[index]] = float("nan") if self.nan else 10
        return SimpleNamespace(logits=logits)


def request(identifier="eval/0"):
    """两个有效 prefix tokens，不含 gold answer 或 teacher-forced 后缀。"""
    return {"sample_id": identifier, "input_ids": [1, 3], "attention_mask": [1, 1], "prompt_sha256": hash_text(identifier)}


class GreedyEvaluationTests(unittest.TestCase):
    """模型只能接收 prefix 和自身生成 token；不得拼入参考答案、labels 或 padding。"""

    def test_greedy_reads_final_active_logits_appends_only_predictions_and_stops_at_eos(self):
        model = ControlledGenerationModel()
        model.train()
        result = predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        self.assertEqual(result[0]["generated_token_ids"], [4, 2])
        self.assertEqual(result[0]["prediction"], " yes ")
        self.assertEqual(result[0]["finish_reason"], "eos")
        self.assertEqual([call["ids"] for call in model.calls], [[[1, 3]], [[1, 3, 4]]])
        self.assertTrue(all(not call["grad"] and not call["training"] and call["use_cache"] is False for call in model.calls))
        self.assertTrue(model.training)
        self.assertIsNone(model.anchor.grad)

    def test_first_eos_gives_empty_prediction_without_extra_forward(self):
        model = ControlledGenerationModel(outputs=(2,))
        result = predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        self.assertEqual(result[0]["prediction"], "")
        self.assertEqual(result[0]["generated_token_ids"], [2])
        self.assertEqual(len(model.calls), 1)

    def test_length_limit_is_explicit_and_never_runs_unbounded_generation(self):
        model = ControlledGenerationModel(outputs=(4,))
        result = predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=2)
        self.assertEqual(result[0]["generated_token_ids"], [4, 4])
        self.assertEqual(result[0]["finish_reason"], "length_limit")
        self.assertEqual(len(model.calls), 2)

    def test_invalid_all_requests_and_limits_fail_before_first_forward(self):
        cases = ([], [request(), request()], [{**request(), "input_ids": [1, 8]}],
                 [{**request(), "attention_mask": [1, 0]}], [{**request(), "reference": "gold"}],
                 [request(), {**request("eval/1"), "input_ids": [True, 3]}])
        for prompts in cases:
            model = ControlledGenerationModel()
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                predict_greedy(model, DecodeTokenizer(), prompts, max_new_tokens=4)
            self.assertEqual(model.calls, [])
        for limit in (0, 9, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                predict_greedy(ControlledGenerationModel(), DecodeTokenizer(), [request()], max_new_tokens=limit)

    def test_forward_error_restores_each_original_module_mode(self):
        model = ControlledGenerationModel(fail_at=2)
        model.train()
        model.child.eval()
        modes = [module.training for module in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "controlled generation failure"):
            predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertIsNone(model.anchor.grad)

    def test_nonfinite_logits_are_not_silently_decoded(self):
        model = ControlledGenerationModel(nan=True)
        with self.assertRaises(ValueError):
            predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        self.assertTrue(model.training)

    def test_existing_gradients_or_ablation_hooks_are_rejected_without_clearing_them(self):
        model = ControlledGenerationModel()
        model.anchor.grad = torch.ones_like(model.anchor)
        with self.assertRaises(ValueError):
            predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        self.assertTrue(torch.equal(model.anchor.grad, torch.ones(1)))
        model.anchor.grad = None
        handle = model.register_forward_hook(lambda *_: None)
        try:
            with self.assertRaises(ValueError):
                predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        finally:
            handle.remove()

    def test_tiny_real_qwen_peft_evaluation_keeps_all_parameter_bytes_and_flags(self):
        """真实库小张量前向，不是固定 0.5B 权重或自然语言准确率证据。"""
        torch.manual_seed(42)
        setup = attach_selective_lora(tiny_model(), [0], rank=2, alpha=4, expected_layers=3)
        setup.model.train()
        before = fingerprint_parameters(setup.model.named_parameters())
        flags = [parameter.requires_grad for parameter in setup.model.parameters()]
        predict_greedy(setup.model, DecodeTokenizer(), [request()], max_new_tokens=1)
        self.assertEqual(before, fingerprint_parameters(setup.model.named_parameters()))
        self.assertEqual(flags, [parameter.requires_grad for parameter in setup.model.parameters()])
        self.assertTrue(setup.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in setup.model.parameters()))


class ExactMatchScoringTests(unittest.TestCase):
    """评分按 ID 对齐，reference 仅在预测完成后读取，不允许静默丢掉困难样本。"""

    def test_reference_reordering_strip_only_and_case_sensitivity(self):
        predictions = [{"sample_id": "a", "prediction": " yes \n"}, {"sample_id": "b", "prediction": "Blue"}]
        references = [{"sample_id": "b", "text": "blue"}, {"sample_id": "a", "text": "yes"}]
        result = score_exact_match(predictions, references)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual([row["sample_id"] for row in result["items"]], ["a", "b"])

    def test_changed_gold_changes_score_not_generation_inputs_or_predictions(self):
        model = ControlledGenerationModel()
        predictions = predict_greedy(model, DecodeTokenizer(), [request()], max_new_tokens=4)
        calls = list(model.calls)
        yes = score_exact_match(predictions, [{"sample_id": "eval/0", "text": "yes"}])
        no = score_exact_match(predictions, [{"sample_id": "eval/0", "text": "no"}])
        self.assertEqual(yes["accuracy"], 1.0)
        self.assertEqual(no["accuracy"], 0.0)
        self.assertEqual(model.calls, calls)
        self.assertEqual(predictions[0]["generated_token_ids"], [4, 2])

    def test_empty_duplicate_missing_or_extra_ids_are_rejected(self):
        prediction = {"sample_id": "a", "prediction": "yes"}
        reference = {"sample_id": "a", "text": "yes"}
        for predictions, references in (([], []), ([prediction, prediction], [reference]),
                                         ([prediction], []), ([prediction], [reference, reference]),
                                         ([prediction], [{"sample_id": "b", "text": "yes"}])):
            with self.subTest(predictions=predictions, references=references), self.assertRaises(ValueError):
                score_exact_match(predictions, references)


if __name__ == "__main__":
    unittest.main()
