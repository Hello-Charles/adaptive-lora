"""M5 短监督训练测试：数学手算、可控分词边界和微型真实 Qwen2/PEFT 反向。

字符级 tokenizer 只能验证 mask/拼接控制流；另有固定本地真实 tokenizer 测试。
微型模型验证梯度与更新机制，不代表 0.5B 收敛、正式数据质量或论文下游表现。
"""

import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from adaptive_lora.trainer import (
    assert_base_unchanged,
    attach_selective_lora,
    build_supervised_example,
    causal_loss,
    train_smoke,
)
from test_lora_targets import tiny_model

ROOT = Path(__file__).resolve().parents[1]
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
CACHE = ROOT / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"


class BoundaryTokenizer:
    """扁平字符级 ID 替身；不模拟 BPE，merge_boundary 仅故意破坏 prefix-ID 前缀契约。"""

    eos_token_id = 2

    def __init__(self, merge_boundary=False):
        self.merge_boundary = merge_boundary
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        return "U:" + messages[0]["content"] + "\nA:"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = [ord(character) % 40 + 3 for character in text]
        if self.merge_boundary and text.endswith("yes"):
            ids[-4] = 63
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def tiny_examples(n=3):
    """已知边界的无 padding 样本；P=2，回答两个 token 后追加 EOS，不读取真实数据。"""
    return [
        {"sample_id": f"fixture/{index}", "input_ids": [1, 4, 5 + index, 9, 2],
         "attention_mask": [1] * 5, "labels": [-100, -100, 5 + index, 9, 2],
         "prompt_token_count": 2, "eos_token_id": 2}
        for index in range(n)
    ]


class SupervisedInputTests(unittest.TestCase):
    """校验训练 teacher forcing 的答案只出现在后缀，不能改变前缀或写回校准清单。"""

    def test_prompt_is_masked_and_answer_plus_eos_is_supervised(self):
        tokenizer = BoundaryTokenizer()
        sample = build_supervised_example(tokenizer, "Reply.", "yes", sample_id="synthetic/0", max_length=64)
        p = sample["prompt_token_count"]
        self.assertEqual(sample["labels"][:p], [-100] * p)
        self.assertEqual(sample["labels"][p:], sample["input_ids"][p:])
        self.assertEqual(sample["input_ids"][-1], 2)
        self.assertEqual(sample["supervised_token_count"], 4)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Reply."}])
        self.assertNotIn("yes", tokenizer.calls[0][0])
        self.assertTrue(all(call[1]["add_special_tokens"] is False and call[1]["truncation"] is False
                            and call[1]["padding"] is False for call in tokenizer.calls))

    def test_boundary_merging_empty_answers_and_overlength_are_rejected(self):
        for tokenizer, answer, length in (
            (BoundaryTokenizer(True), "yes", 64), (BoundaryTokenizer(), " ", 64),
            (BoundaryTokenizer(), "yes", 3),
        ):
            with self.subTest(answer=answer, length=length), self.assertRaises(ValueError):
                build_supervised_example(tokenizer, "Reply.", answer, sample_id="fixture", max_length=length)

    def test_real_qwen_tokenizer_preserves_prefix_and_explicit_eos_protocol(self):
        """只验证真实分词与 mask，不加载权重；EOS 是 smoke 决策，不冒充作者回答模板。"""
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", revision=REVISION, cache_dir=str(CACHE),
            local_files_only=True, trust_remote_code=False,
        )
        sample = build_supervised_example(tokenizer, "Reply with yes.", "yes", sample_id="synthetic/real", max_length=96)
        p = sample["prompt_token_count"]
        self.assertEqual(tokenizer.eos_token_id, 151643)
        self.assertEqual(sample["input_ids"][-1], tokenizer.eos_token_id)
        self.assertEqual(sample["labels"][:p], [-100] * p)
        self.assertEqual(tokenizer.decode(sample["input_ids"][p:-1]), "yes")
        self.assertLessEqual(len(sample["input_ids"]), 96)


class CausalLossTests(unittest.TestCase):
    """logits[i] 预测 labels[i+1]；首回答 label[P] 必须读取 logits[P-1]。"""

    def test_manual_shift_and_prompt_mask_match_hand_selected_logits(self):
        logits = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8).requires_grad_()
        labels = torch.tensor([[-100, -100, 3, 2]])
        loss = causal_loss(logits, labels)
        expected = (F.cross_entropy(logits[0, 1].unsqueeze(0), torch.tensor([3]))
                    + F.cross_entropy(logits[0, 2].unsqueeze(0), torch.tensor([2]))) / 2
        self.assertEqual(loss.item(), expected.item())
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(logits.grad[0, 3], torch.zeros(8)))
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def test_invalid_shapes_targets_all_masked_and_nonfinite_logits_fail(self):
        logits = torch.zeros(1, 4, 8)
        for values, labels in (
            (logits[0], torch.tensor([[-100, -100, 3, 2]])),
            (logits, torch.full((1, 4), -100)),
            (logits, torch.tensor([[-100, -100, 8, 2]])),
            (logits, torch.tensor([[-100, -100, -2, 2]])),
            (logits, torch.zeros(1, 3, dtype=torch.long)),
            (torch.full_like(logits, float("nan")), torch.tensor([[-100, -100, 3, 2]])),
        ):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                causal_loss(values, labels)


class SmokeTrainingTests(unittest.TestCase):
    """真实微型反向/优化器；NaN 注入是受控错误，不代表真实模型曾产生 NaN。"""

    def make_setup(self):
        torch.manual_seed(42)
        return attach_selective_lora(tiny_model(), [0, 2], rank=2, alpha=4, expected_layers=3)

    def test_three_batches_have_finite_gradients_updates_and_unchanged_base(self):
        setup = self.make_setup()
        metrics = train_smoke(setup, tiny_examples())
        self.assertEqual(len(metrics["steps"]), 3)
        self.assertTrue(all(math.isfinite(step["loss"]) and step["gradient_l2_norm"] > 0
                            and step["updated_adapter_tensors"] > 0 for step in metrics["steps"]))
        # 标准 B=0 初始化时第一步 A 梯度可以为零；不能要求所有 A/B 一开始均非零。
        self.assertLess(metrics["steps"][0]["nonzero_gradient_tensors"], 12)
        self.assertGreater(metrics["steps"][0]["nonzero_gradient_tensors"], 0)
        self.assertNotEqual(metrics["adapter_before"]["sha256"], metrics["adapter_after"]["sha256"])
        self.assertEqual(metrics["base_before"], metrics["base_after"])
        self.assertEqual(assert_base_unchanged(setup), setup.base_fingerprint)
        self.assertFalse(setup.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in setup.model.parameters()))

    def test_invalid_batch_count_and_disabled_autograd_fail_before_forward(self):
        setup = self.make_setup()
        with patch.object(setup.model, "forward") as forward:
            for examples in ([], tiny_examples(4)):
                with self.assertRaises(ValueError):
                    train_smoke(setup, examples)
            with torch.no_grad(), self.assertRaises(ValueError):
                train_smoke(setup, tiny_examples(1))
            with torch.inference_mode(), self.assertRaises(ValueError):
                train_smoke(setup, tiny_examples(1))
            forward.assert_not_called()

    def test_all_input_masks_and_labels_are_validated_before_first_forward(self):
        setup = self.make_setup()
        corruptions = (
            {"labels": [-100] * 5}, {"labels": [-100, 4, 5, 9, 2]},
            {"attention_mask": [1, 0, 1, 1, 1]}, {"prompt_token_count": 0},
            {"input_ids": [1, 4, 5, 9, 63]},
        )
        with patch.object(setup.model, "forward") as forward:
            for fields in corruptions:
                samples = tiny_examples(2)
                samples[1].update(fields)
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    train_smoke(setup, samples)
            forward.assert_not_called()

    def test_nonfinite_gradients_abort_before_optimizer_update_and_clear_grads(self):
        setup = self.make_setup()
        before = {name: parameter.detach().clone() for name, parameter in setup.model.named_parameters() if parameter.requires_grad}
        parameter = next(parameter for parameter in setup.model.parameters() if parameter.requires_grad)
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaises(ValueError):
                train_smoke(setup, tiny_examples(1))
        finally:
            handle.remove()
        self.assertTrue(all(torch.equal(before[name], parameter) for name, parameter in setup.model.named_parameters() if name in before))
        self.assertTrue(all(parameter.grad is None for parameter in setup.model.parameters()))
        self.assertFalse(setup.model.training)
        assert_base_unchanged(setup)

    def test_callback_failure_clears_gradients_and_returns_model_to_eval(self):
        setup = self.make_setup()
        def fail(_):
            """模拟日志落盘失败；不得把部分训练说成成功或保留激活图。"""
            raise RuntimeError("callback failure")
        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            train_smoke(setup, tiny_examples(1), on_step=fail)
        self.assertFalse(setup.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in setup.model.parameters()))
        assert_base_unchanged(setup)

    def test_same_seed_and_inputs_repeat_exact_metrics_on_tiny_cpu_model(self):
        first = train_smoke(self.make_setup(), tiny_examples())
        second = train_smoke(self.make_setup(), copy.deepcopy(tiny_examples()))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
