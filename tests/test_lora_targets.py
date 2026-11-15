"""M5 选层契约测试：使用随机初始化的微型 Qwen2 和当前实际 PEFT，不下载权重。

这些测试确实执行 LoRA 注入和小张量前向，但不能代替固定 revision 的 0.5B 模型验证。
维度刻意保留 GQA 的非方形 K/V，避免把所有投影误算成 d×d。
"""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptive_lora.calibration import parameter_fingerprint
from adaptive_lora.trainer import (
    assert_base_unchanged,
    attach_selective_lora,
    audit_lora,
    fingerprint_parameters,
)


def tiny_model(layers=3, dtype=torch.float32):
    """构造本地随机权重的微型模型；种子由调用者控制，不访问 Hub 或真实权重缓存。"""
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).to(dtype=dtype).eval()


class SelectiveLoraTests(unittest.TestCase):
    """检查精确路径、结构白名单和基础参数身份；不通过名称含有 lora 就放行。"""

    def setUp(self):
        torch.manual_seed(42)

    def test_exact_layer_paths_do_not_confuse_one_with_ten_or_eleven(self):
        setup = attach_selective_lora(tiny_model(12), [10, 1], rank=2, alpha=4, expected_layers=12)
        audit = audit_lora(setup)
        expected = {
            f"model.layers.{index}.self_attn.{projection}_proj"
            for index in (1, 10) for projection in ("q", "k", "v")
        }
        self.assertEqual(set(audit["target_modules"]), expected)
        self.assertEqual(audit["selected_layer_indices_ascending"], [1, 10])
        self.assertEqual(audit["trainable_parameter_tensors"], 12)
        # Q 为 16→16；K/V 为 16→8，r=2 时每层共 64+48+48=160 个参数。
        self.assertEqual(audit["adapter_parameter_elements"], 320)
        self.assertEqual(audit["trainable_parameter_elements"], 320)
        self.assertTrue(audit["base_frozen"])
        self.assertTrue(audit["base_gradients_absent"])
        self.assertEqual(assert_base_unchanged(setup), setup.base_fingerprint)
        self.assertTrue(all("lora_A.default.weight" in name or "lora_B.default.weight" in name
                            for name in audit["trainable_parameter_names"]))

    def test_standard_zero_b_initialization_preserves_full_logits(self):
        model = tiny_model()
        ids = torch.tensor([[1, 4, 5, 6]])
        with torch.no_grad():
            baseline = model(input_ids=ids, use_cache=False).logits.clone()
        original = parameter_fingerprint(model)
        setup = attach_selective_lora(model, [0, 2], rank=2, alpha=4, expected_layers=3)
        setup.model.eval()
        with torch.no_grad():
            attached = setup.model(input_ids=ids, use_cache=False).logits
        self.assertTrue(torch.equal(baseline, attached))
        self.assertEqual(original, setup.base_fingerprint)
        self.assertEqual(original, fingerprint_parameters(setup.base_parameters))

    def test_invalid_layer_indices_and_hyperparameters_fail_before_mutation(self):
        invalid = ([], [True], [0, 0], [-1], [3], [1.0])
        for selected in invalid:
            with self.subTest(selected=selected):
                model = tiny_model()
                before = [(id(parameter), parameter.requires_grad) for parameter in model.parameters()]
                with self.assertRaises(ValueError):
                    attach_selective_lora(model, selected, expected_layers=3)
                self.assertEqual(before, [(id(parameter), parameter.requires_grad) for parameter in model.parameters()])
        for keyword in ({"rank": 0}, {"rank": True}, {"alpha": 0}, {"alpha": float("nan")}):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                attach_selective_lora(tiny_model(), [0], expected_layers=3, **keyword)

    def test_existing_adapter_gradients_or_hooks_are_rejected(self):
        setup = attach_selective_lora(tiny_model(), [0], rank=2, expected_layers=3)
        with self.assertRaises(ValueError):
            attach_selective_lora(setup.model, [1], expected_layers=3)
        model = tiny_model()
        parameter = next(model.parameters())
        parameter.grad = torch.zeros_like(parameter)
        with self.assertRaises(ValueError):
            attach_selective_lora(model, [0], expected_layers=3)
        parameter.grad = None
        handle = model.model.layers[0].mlp.register_forward_hook(lambda *_: None)
        try:
            with self.assertRaises(ValueError):
                attach_selective_lora(model, [0], expected_layers=3)
        finally:
            handle.remove()

    def test_audit_rejects_unfrozen_base_and_extra_parameters(self):
        setup = attach_selective_lora(tiny_model(), [0], rank=2, expected_layers=3)
        base = setup.base_parameters[0][1]
        base.requires_grad_(True)
        with self.assertRaises(ValueError):
            audit_lora(setup)
        base.requires_grad_(False)
        setup.model.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1)))
        with self.assertRaises(ValueError):
            audit_lora(setup)

    def test_canonical_base_fingerprint_detects_changes_after_peft_renaming(self):
        setup = attach_selective_lora(tiny_model(), [0], rank=2, expected_layers=3)
        with torch.no_grad():
            setup.base_parameters[0][1].reshape(-1)[0].add_(1)
        with self.assertRaises(ValueError):
            assert_base_unchanged(setup)

    def test_bfloat16_base_stays_bfloat16_and_adapters_are_float32(self):
        setup = attach_selective_lora(tiny_model(dtype=torch.bfloat16), [0], rank=2, expected_layers=3)
        self.assertEqual(audit_lora(setup)["adapter_dtypes"], ["torch.float32"])
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for _, parameter in setup.base_parameters))
        self.assertEqual(assert_base_unchanged(setup), setup.base_fingerprint)


if __name__ == "__main__":
    unittest.main()
