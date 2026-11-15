"""M4 数学与真实 PyTorch hook 集成测试；控制概率的替身不是 Qwen 数值模型。

夹具显式让消融后的 argmax 改变，并让 pad 位置偏好其他 token，检测两种常见测量错误。
实际 Qwen/N=30 的数值、参数字节及样本来源由独立运行命令验收，不由本文件冒充。
"""

import copy
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from adaptive_lora.calibration import (
    aggregate_importance,
    calibrate_model,
    parameter_fingerprint,
    reference_prediction,
    token_probability,
)
from adaptive_lora.model_loader import discover_layer_modules


def probability_record(q, attention, mlp, *, token_id=1, sample_id="fixture"):
    """构造手算用记录；测量值本身就是概率，不使用模型/softmax 计算期望值。"""
    return {
        "sample_id": sample_id, "token_id": token_id, "q": q,
        "layers": [
            {
                "layer_index": index,
                "attention": {"token_id": token_id, "probability": a},
                "mlp": {"token_id": token_id, "probability": m},
            }
            for index, (a, m) in enumerate(zip(attention, mlp))
        ],
    }


class ProbabilityAttention(nn.Module):
    """只提供真实子模块和可置零的三维输出；Q/K/V 名称供发现器核验，不模拟注意力。"""

    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                projection.weight.fill_(0.25)
            setattr(self, name, projection)

    def forward(self, hidden):
        return torch.ones_like(hidden), None


class ProbabilityMLP(nn.Module):
    """输出非零常量以检测消融命中，不用于任何论文效果或训练验证。"""

    def forward(self, hidden):
        return torch.ones_like(hidden)


class ProbabilityModel(nn.Module):
    """两层受控概率表驱动的模型；真实 hook 决定表项，但没有 Transformer 计算。

    每次完整前向均执行四个子模块，只允许最多一个输出为零；pad 的 logits 故意与有效
    末尾不同。requests 仅记录控制流/梯度状态，不影响输出；可注入错误或权重改写以验证门禁。
    """

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2, vocab_size=3)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.self_attn = ProbabilityAttention()
            block.mlp = ProbabilityMLP()
            self.model.layers.append(block)
        self.requests = []
        self.fail_target = None
        self.mutate_parameters = False
        self.bad_logits = False
        self.tables = {
            1: {
                None: [0.1, 0.6, 0.3], (0, "attention"): [0.7, 0.2, 0.1],
                (0, "mlp"): [0.1, 0.1, 0.8], (1, "attention"): [0.1, 0.8, 0.1],
                (1, "mlp"): [0.3, 0.5, 0.2],
            },
            2: {
                None: [0.5, 0.3, 0.2], (0, "attention"): [0.2, 0.7, 0.1],
                (0, "mlp"): [0.4, 0.1, 0.5], (1, "attention"): [0.8, 0.1, 0.1],
                (1, "mlp"): [0.1, 0.6, 0.3],
            },
        }

    def forward(self, input_ids, attention_mask, *, use_cache, return_dict):
        if use_cache or not return_dict:
            raise AssertionError("fixture requires fresh full-input forwards without KV cache")
        hidden = torch.ones((*input_ids.shape, 2), dtype=torch.float32)
        active = []
        for index, block in enumerate(self.model.layers):
            attention = block.self_attn(hidden)[0]
            mlp = block.mlp(hidden)
            if torch.count_nonzero(attention).item() == 0:
                active.append((index, "attention"))
            if torch.count_nonzero(mlp).item() == 0:
                active.append((index, "mlp"))
        if len(active) > 1:
            raise AssertionError("components were jointly ablated")
        target = active[0] if active else None
        self.requests.append({"target": target, "grad_enabled": torch.is_grad_enabled(), "training": self.training})
        if target is not None and target == self.fail_target:
            raise RuntimeError("injected calibration failure")
        if self.mutate_parameters:
            with torch.no_grad():
                next(self.parameters()).add_(0.125)
        key = int(input_ids[attention_mask.bool()][0].item())
        position = int(attention_mask[0].nonzero()[-1].item())
        logits = torch.tensor([20.0, 0.0, 0.0]).repeat(1, input_ids.shape[1], 1)
        logits[0, position] = torch.log(torch.tensor(self.tables[key][target], dtype=torch.float32))
        if self.bad_logits:
            logits[0, position, 0] = float("nan")
        return SimpleNamespace(logits=logits)


class ImportanceMathTests(unittest.TestCase):
    """纯概率记录覆盖公式 (5)，不靠校准器自身的中间输出计算期望值。"""

    def test_n1_is_sum_of_two_absolute_probability_differences(self):
        record = probability_record(0.6, [0.2, 0.8], [0.1, 0.5])
        rows = aggregate_importance([record], layer_count=2)
        self.assertAlmostEqual(rows[0]["importance"], 0.9)
        self.assertAlmostEqual(rows[1]["importance"], 0.3)
        self.assertAlmostEqual(rows[0]["attention_mean_abs_delta"], 0.4)
        self.assertAlmostEqual(rows[0]["mlp_mean_abs_delta"], 0.5)

    def test_n_greater_than_one_averages_per_sample_not_after_cancellation(self):
        records = [
            probability_record(0.5, [0.8], [0.5], sample_id="one"),
            probability_record(0.5, [0.2], [0.5], sample_id="two"),
        ]
        rows = aggregate_importance(records, layer_count=1)
        self.assertAlmostEqual(rows[0]["importance"], 0.3)
        self.assertNotAlmostEqual(rows[0]["importance"], 0.0)

    def test_equal_probabilities_give_zero_and_bounds_allow_zero_one(self):
        zero = aggregate_importance([probability_record(0.4, [0.4], [0.4])], layer_count=1)
        self.assertEqual(zero[0]["importance"], 0.0)
        bound = aggregate_importance([probability_record(1.0, [0.0], [0.0])], layer_count=1)
        self.assertEqual(bound[0]["importance"], 2.0)

    def test_measurements_must_keep_the_original_token(self):
        record = probability_record(0.6, [0.2], [0.1])
        record["layers"][0]["attention"]["token_id"] = 0
        with self.assertRaisesRegex(ValueError, "token"):
            aggregate_importance([record], layer_count=1)

    def test_rejects_empty_missing_duplicate_or_invalid_probability_records(self):
        with self.assertRaises(ValueError):
            aggregate_importance([], layer_count=1)
        for bad in (-0.1, 1.1, float("nan"), float("inf")):
            record = probability_record(bad, [0.2], [0.1])
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                aggregate_importance([record], layer_count=1)
        for layers in ([], [0, 0], [1]):
            record = probability_record(0.6, [0.2], [0.1])
            template = record["layers"][0]
            record["layers"] = [{**template, "layer_index": index} for index in layers]
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                aggregate_importance([record], layer_count=1)

    def test_fp32_probability_reads_requested_token_not_new_argmax(self):
        logits = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32))
        self.assertEqual(reference_prediction(logits)[0], 0)
        self.assertAlmostEqual(token_probability(logits, 1), 0.2, places=6)
        self.assertNotAlmostEqual(token_probability(logits, 1), 0.7, places=6)
        tied = reference_prediction(torch.zeros(3, dtype=torch.bfloat16))
        self.assertEqual(tied[0], 0)
        self.assertAlmostEqual(tied[1], 1 / 3, places=6)

    def test_argmax_precedes_fp32_probability_rounding(self):
        """FP32 是概率计算约定，不应把原始 FP64 最大项因转换并列而替换。"""
        logits = torch.tensor([1.0, 1.0 + 1e-10], dtype=torch.float64)
        token, probability = reference_prediction(logits)
        self.assertEqual(token, 1)
        self.assertEqual(probability, 0.5)

    def test_probability_rejects_bad_shape_nonfinite_and_invalid_token(self):
        for logits in (torch.zeros(1, 3), torch.tensor([float("nan"), 0.0]), torch.tensor([float("inf"), 0.0]), torch.empty(0)):
            with self.subTest(shape=list(logits.shape)), self.assertRaises(ValueError):
                reference_prediction(logits)
        for token in (-1, 3, True):
            with self.subTest(token=token), self.assertRaises(ValueError):
                token_probability(torch.zeros(3), token)


class CalibrationPipelineTests(unittest.TestCase):
    """用真实 nn.Module/hook 验证固定 token、预测位置和生命周期，不代表真实模型精度。"""

    def setUp(self):
        self.model = ProbabilityModel().eval().requires_grad_(False)
        self.layers = discover_layer_modules(self.model, expected_layers=2)
        self.samples = [
            {"sample_id": "one", "index": 0, "input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0], "prediction_position": 1},
            {"sample_id": "two", "index": 1, "input_ids": [2, 1], "attention_mask": [1, 1], "prediction_position": 1},
        ]

    def test_real_hooks_keep_original_token_when_ablation_argmax_changes(self):
        result = calibrate_model(self.model, self.samples[:1], layers=self.layers)
        sample = result["samples"][0]
        self.assertEqual(sample["token_id"], 1)
        self.assertEqual(sample["prediction_position"], 1)
        self.assertAlmostEqual(sample["q"], 0.6, places=6)
        row = sample["layers"][0]
        self.assertEqual(row["attention"]["token_id"], 1)
        self.assertEqual(row["mlp"]["token_id"], 1)
        self.assertAlmostEqual(row["attention"]["probability"], 0.2, places=6)
        self.assertAlmostEqual(row["mlp"]["probability"], 0.1, places=6)
        self.assertAlmostEqual(result["layer_importance"][0]["importance"], 0.9, places=6)
        self.assertEqual([request["target"] for request in self.model.requests], [None, (0, "attention"), (0, "mlp"), (1, "attention"), (1, "mlp"), None])

    def test_n2_means_and_parameter_gradient_hook_invariants(self):
        before = parameter_fingerprint(self.model)
        result = calibrate_model(self.model, self.samples, layers=self.layers)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["forward_evaluations"], 12)
        self.assertAlmostEqual(result["layer_importance"][0]["importance"], 0.65, places=6)
        self.assertAlmostEqual(result["layer_importance"][1]["importance"], 0.5, places=6)
        self.assertEqual(before, result["parameter_fingerprint_before"])
        self.assertEqual(before, result["parameter_fingerprint_after"])
        self.assertTrue(all(not request["grad_enabled"] and not request["training"] for request in self.model.requests))
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in self.model.parameters()))
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))
        self.assertTrue(all(sample["restored_logits_equal"] for sample in result["samples"]))

    def test_left_padding_uses_last_active_position(self):
        sample = {**self.samples[0], "input_ids": [0, 1, 2], "attention_mask": [0, 1, 1], "prediction_position": 2}
        result = calibrate_model(self.model, [sample], layers=self.layers)
        self.assertEqual(result["samples"][0]["token_id"], 1)
        self.assertEqual(result["samples"][0]["prediction_position"], 2)

    def test_same_model_and_inputs_repeat_exact_measurements(self):
        first = calibrate_model(self.model, self.samples, layers=self.layers)
        second = calibrate_model(self.model, self.samples, layers=self.layers)
        self.assertEqual(first, second)

    def test_forward_failure_cleans_hooks_and_allows_a_fresh_run(self):
        self.model.fail_target = (0, "attention")
        with self.assertRaisesRegex(RuntimeError, "injected calibration failure"):
            calibrate_model(self.model, self.samples, layers=self.layers)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))
        self.model.fail_target = None
        result = calibrate_model(self.model, self.samples[:1], layers=self.layers)
        self.assertEqual(result["n"], 1)

    def test_callback_failure_happens_after_restoration_and_without_hooks(self):
        def fail(_record):
            """模拟日志写入失败；此时不能仍有一个消融作用域挂在模型上。"""
            self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))
            raise OSError("callback failure")

        with self.assertRaisesRegex(OSError, "callback failure"):
            calibrate_model(self.model, self.samples, layers=self.layers, on_sample=fail)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def test_parameter_mutation_is_detected_even_when_logits_look_unchanged(self):
        self.model.mutate_parameters = True
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            calibrate_model(self.model, self.samples[:1], layers=self.layers)

    def test_invalid_samples_fail_before_any_forward(self):
        invalid = [
            [], [self.samples[0], self.samples[0]],
            [{**self.samples[0], "prediction_position": 3}],
            [{**self.samples[0], "attention_mask": [1, 2, 0, 0]}],
            [{**self.samples[0], "input_ids": [-1, 2, 0, 0]}],
        ]
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                calibrate_model(self.model, samples, layers=self.layers)
        self.assertEqual(self.model.requests, [])

    def test_requires_frozen_eval_model_with_no_preinstalled_hooks(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            calibrate_model(self.model, self.samples, layers=self.layers)
        self.model.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            calibrate_model(self.model, self.samples, layers=self.layers)
        self.model.requires_grad_(False)
        handle = self.layers[0].attention_module.register_forward_hook(lambda _m, _i, _o: None)
        try:
            with self.assertRaisesRegex(ValueError, "hooks"):
                calibrate_model(self.model, self.samples, layers=self.layers)
        finally:
            handle.remove()
        self.assertEqual(self.model.requests, [])

    def test_partial_or_foreign_layer_descriptors_fail_before_forward(self):
        """传入描述符不能绕过全层/模型归属检查；这不是通用模型结构发现的测试。"""
        foreign = ProbabilityModel().eval().requires_grad_(False)
        foreign_layers = discover_layer_modules(foreign, expected_layers=2)
        for layers in (self.layers[:1], foreign_layers):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "layers"):
                calibrate_model(self.model, self.samples, layers=layers)
        self.assertEqual(self.model.requests, [])

    def test_nonfinite_logits_fail_without_leaking_hooks(self):
        self.model.bad_logits = True
        with self.assertRaisesRegex(ValueError, "finite"):
            calibrate_model(self.model, self.samples, layers=self.layers)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))


if __name__ == "__main__":
    unittest.main()
