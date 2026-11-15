"""zero-ablation 的真实 PyTorch 小模型测试；不加载 Qwen 权重或计算层重要性。

替身保留残差路径、独立 Attention/MLP 和 Q/K/V 命名，检验 hook/张量语义及异常清理；
它不是 Qwen Attention 的数值实现，真实模型输出结构仍须用独立离线脚本验收。
"""

from collections import namedtuple
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from adaptive_lora.ablation import zero_ablation
from adaptive_lora.model_loader import discover_layer_modules


class ToyAttention(nn.Module):
    """小型线性替身；可切换输出容器/别名/错误结构，辅助项须保留对象身份。

    Q/K/V 仅用线性求和，不含 softmax、KV cache 或位置编码，不能用于模型质量验证。
    """

    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                projection.weight.copy_(torch.eye(4) * 0.1)
            setattr(self, name, projection)
        self.auxiliary = object()
        self.output_kind = "tuple"

    def forward(self, hidden):
        output = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        if self.output_kind == "alias":
            return hidden, self.auxiliary
        if self.output_kind == "tensor":
            return output
        if self.output_kind == "list":
            return [output, self.auxiliary]
        if self.output_kind == "bad":
            return {"hidden_states": output}
        if self.output_kind == "empty":
            return ()
        if self.output_kind == "bad_first":
            return None, output
        if self.output_kind == "namedtuple":
            result_type = namedtuple("AttentionResult", ["hidden", "auxiliary"])
            return result_type(output, self.auxiliary)
        return output, self.auxiliary


class ToyMLP(nn.Module):
    """可注入前向错误的线性 MLP 替身；tuple 输出用于检查不支持的结构须拒绝。"""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4) * 0.2)
        self.fail = False
        self.return_tuple = False

    def forward(self, hidden):
        if self.fail:
            raise RuntimeError("injected forward failure")
        result = self.projection(hidden)
        return (result,) if self.return_tuple else result


class ToyBlock(nn.Module):
    """两个独立子模块加残差；归一化名字用于复用实际模块树发现器。"""

    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, hidden):
        attention = self.self_attn(hidden)
        hidden = hidden + (attention[0] if isinstance(attention, (tuple, list)) else attention)
        return hidden + self.mlp(hidden)


class ToyModel(nn.Module):
    """默认两层以快速验证数学和作用域；24 层只用于合法索引的逐目标遍历测试。"""

    def __init__(self, layer_count=2):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layer_count)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(ToyBlock() for _ in range(layer_count))

    def forward(self, hidden):
        for block in self.model.layers:
            hidden = block(hidden)
        return hidden


class ZeroAblationTests(unittest.TestCase):
    """所有正常前向均显式 eval、冻结参数并禁用梯度；不会替调用者隐藏这些前提。"""

    def setUp(self):
        self.model = ToyModel().eval().requires_grad_(False)
        self.layers = discover_layer_modules(self.model, expected_layers=2)
        self.input = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 4)

    def hooks(self):
        """仅作生命周期诊断使用 PyTorch 内部 hook 表，不作为消融算法实现依赖。"""
        return {name: dict(module._forward_hooks) for name, module in self.model.named_modules()}

    def test_attention_replaces_only_main_tensor_and_preserves_residual(self):
        """原输出不得原地改写；其他组件的输入可变化，但其输出不能被直接强制为零。"""
        layer = self.layers[0]
        original, replaced, mlp_outputs, block_outputs = [], [], [], []
        with ExitStack() as cleanup, torch.inference_mode():
            cleanup.callback(layer.attention_module.register_forward_hook(lambda _m, _i, o: original.append(o)).remove)
            cleanup.callback(layer.mlp_module.register_forward_hook(lambda _m, _i, o: mlp_outputs.append(o)).remove)
            cleanup.callback(self.model.get_submodule(layer.block_path).register_forward_hook(lambda _m, _i, o: block_outputs.append(o)).remove)
            with zero_ablation(self.model, layer, "attention") as state:
                cleanup.callback(layer.attention_module.register_forward_hook(lambda _m, _i, o: replaced.append(o)).remove)
                self.model(self.input)
                self.assertEqual(state.calls, 1)
                self.assertEqual(state.module_path, layer.attention_path)
        self.assertIs(type(replaced[0]), tuple)
        self.assertEqual(len(replaced[0]), 2)
        self.assertIs(replaced[0][1], original[0][1])
        self.assertIsNot(replaced[0][0], original[0][0])
        self.assertEqual(replaced[0][0].shape, original[0][0].shape)
        self.assertEqual(replaced[0][0].dtype, original[0][0].dtype)
        self.assertEqual(replaced[0][0].device, original[0][0].device)
        self.assertEqual(torch.count_nonzero(replaced[0][0]).item(), 0)
        self.assertGreater(torch.count_nonzero(original[0][0]).item(), 0)
        self.assertGreater(torch.count_nonzero(mlp_outputs[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(block_outputs[0]).item(), 0)
        self.assertTrue(all(not hooks for hooks in self.hooks().values()))

    def test_mlp_does_not_zero_attention_or_whole_block(self):
        layer = self.layers[0]
        attention, mlp, block = [], [], []
        with ExitStack() as cleanup, torch.inference_mode(), zero_ablation(self.model, layer, "mlp"):
            for module, values in (
                (layer.attention_module, attention), (layer.mlp_module, mlp),
                (self.model.get_submodule(layer.block_path), block),
            ):
                cleanup.callback(module.register_forward_hook(lambda _m, _i, o, target=values: target.append(o)).remove)
            self.model(self.input)
        self.assertGreater(torch.count_nonzero(attention[0][0]).item(), 0)
        self.assertEqual(torch.count_nonzero(mlp[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(block[0]).item(), 0)

    def test_attention_tensor_list_and_tuple_keep_structure_and_dtype(self):
        """这些结构只在替身上验证；真实 Qwen 本次返回 tuple(Tensor, None)。"""
        module = self.layers[0].attention_module
        for kind in ("tensor", "list", "tuple"):
            for dtype in (torch.float32, torch.bfloat16):
                module.output_kind = kind
                self.model.to(dtype=dtype)
                observed = []
                with self.subTest(kind=kind, dtype=dtype), ExitStack() as cleanup, torch.inference_mode():
                    with zero_ablation(self.model, self.layers[0], "attention"):
                        cleanup.callback(module.register_forward_hook(lambda _m, _i, o: observed.append(o)).remove)
                        module(self.input.to(dtype=dtype))
                result = observed[0]
                main = result if kind == "tensor" else result[0]
                self.assertIs(type(result), {"tensor": torch.Tensor, "list": list, "tuple": tuple}[kind])
                self.assertEqual(main.dtype, dtype)
                self.assertEqual(torch.count_nonzero(main).item(), 0)
                if kind != "tensor":
                    self.assertIs(result[1], module.auxiliary)

    def test_alias_output_does_not_modify_input_tensor(self):
        """Attention 返回输入本身时，zeros_ 会破坏残差；zeros_like 必须另建输出。"""
        self.layers[0].attention_module.output_kind = "alias"
        saved = self.input.clone()
        with torch.inference_mode(), zero_ablation(self.model, self.layers[0], "attention"):
            self.model(self.input)
        self.assertTrue(torch.equal(saved, self.input))

    def test_scope_restores_forward_weights_modes_gradients_and_existing_hook(self):
        """保留调用者原有观察 hook；不靠清空整个 hook 字典完成清理。"""
        observed = []
        handle = self.layers[0].attention_module.register_forward_hook(lambda _m, _i, o: observed.append(o))
        try:
            before_hooks = self.hooks()
            weights = {name: value.detach().clone() for name, value in self.model.named_parameters()}
            with torch.inference_mode():
                baseline = self.model(self.input)
                for component in ("attention", "mlp"):
                    with zero_ablation(self.model, self.layers[0], component) as state:
                        changed = self.model(self.input)
                    self.assertFalse(torch.equal(baseline, changed))
                    self.assertEqual(state.calls, 1)
                    self.assertTrue(torch.equal(baseline, self.model(self.input)))
                    self.assertEqual(before_hooks, self.hooks())
            self.assertTrue(observed)
            self.assertTrue(all(torch.equal(weights[name], value) for name, value in self.model.named_parameters()))
            self.assertTrue(all(not value.requires_grad and value.grad is None for value in self.model.parameters()))
            self.assertTrue(all(not module.training for module in self.model.modules()))
        finally:
            handle.remove()

    def test_same_model_rejects_nested_same_other_component_and_other_layer(self):
        """并列嵌套会变成联合干预；失败的内层不得清除仍在运行的外层 hook。"""
        with torch.inference_mode(), zero_ablation(self.model, self.layers[0], "attention") as outer:
            outer_hooks = self.hooks()
            for layer, component in ((self.layers[0], "attention"), (self.layers[0], "mlp"), (self.layers[1], "mlp")):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with zero_ablation(self.model, layer, component):
                        self.fail("nested ablation was not rejected")
                self.assertEqual(outer_hooks, self.hooks())
            self.model(self.input)
            self.assertEqual(outer.calls, 1)
        with torch.inference_mode(), zero_ablation(self.model, self.layers[1], "mlp"):
            self.model(self.input)

    def test_user_exception_cleans_scope_and_allows_later_ablation(self):
        before = self.hooks()
        with torch.inference_mode():
            baseline = self.model(self.input)
            with self.assertRaisesRegex(RuntimeError, "user failure"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    self.model(self.input)
                    raise RuntimeError("user failure")
            self.assertEqual(before, self.hooks())
            self.assertTrue(torch.equal(baseline, self.model(self.input)))
            with zero_ablation(self.model, self.layers[0], "mlp"):
                self.model(self.input)

    def test_forward_exception_removes_hook_before_original_forward_recovers(self):
        before = self.hooks()
        with torch.inference_mode():
            baseline = self.model(self.input)
            self.layers[0].mlp_module.fail = True
            try:
                with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                    with zero_ablation(self.model, self.layers[0], "attention"):
                        self.model(self.input)
            finally:
                self.layers[0].mlp_module.fail = False
            self.assertEqual(before, self.hooks())
            self.assertTrue(torch.equal(baseline, self.model(self.input)))

    def test_bad_output_shapes_fail_loudly_and_cleanup(self):
        """不把未知 dict/namedtuple 或错误主项猜成可消融输出，避免破坏模型接口。"""
        before = self.hooks()
        module = self.layers[0].attention_module
        with torch.inference_mode():
            for kind in ("bad", "empty", "bad_first", "namedtuple"):
                module.output_kind = kind
                with self.subTest(kind=kind), self.assertRaisesRegex(TypeError, "attention output"):
                    with zero_ablation(self.model, self.layers[0], "attention"):
                        module(self.input)
                self.assertEqual(before, self.hooks())
            module.output_kind = "tuple"
            self.layers[0].mlp_module.return_tuple = True
            try:
                with self.assertRaisesRegex(TypeError, "MLP output"):
                    with zero_ablation(self.model, self.layers[0], "mlp"):
                        self.layers[0].mlp_module(self.input)
            finally:
                self.layers[0].mlp_module.return_tuple = False
            with zero_ablation(self.model, self.layers[0], "attention"):
                self.model(self.input)

    def test_registration_failure_does_not_leave_active_registry_entry(self):
        before = self.hooks()
        with torch.inference_mode():
            with patch.object(self.layers[0].attention_module, "register_forward_hook", side_effect=RuntimeError("registration failure")):
                with self.assertRaisesRegex(RuntimeError, "registration failure"):
                    with zero_ablation(self.model, self.layers[0], "attention"):
                        self.fail("registration should fail")
            self.assertEqual(before, self.hooks())
            with zero_ablation(self.model, self.layers[0], "attention"):
                self.model(self.input)

    def test_unused_target_is_an_error_after_cleanup(self):
        """空 scope 不能被当成成功干预；异常分支则必须保留原始错误而非覆盖为未命中。"""
        before = self.hooks()
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "not executed"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    pass
            self.assertEqual(before, self.hooks())
            with self.assertRaisesRegex(RuntimeError, "original error"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    raise RuntimeError("original error")
            with zero_ablation(self.model, self.layers[0], "attention"):
                self.model(self.input)

    def test_requires_eval_frozen_parameters_and_disabled_gradient_recording(self):
        before = self.hooks()
        with self.assertRaisesRegex(RuntimeError, "no_grad|inference_mode"):
            with zero_ablation(self.model, self.layers[0], "attention"):
                pass
        with torch.no_grad():
            self.model.train()
            with self.assertRaisesRegex(ValueError, "eval"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    pass
            self.model.eval()
            self.model.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "frozen"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    pass
            self.model.requires_grad_(False)
            # root.eval() 后又单独开启子模块 train 也必须被检查出来。
            self.layers[1].mlp_module.train()
            with self.assertRaisesRegex(ValueError, "eval"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    pass
            self.model.eval()
        self.assertEqual(before, self.hooks())

    def test_reenabling_gradients_inside_scope_is_rejected_at_hook(self):
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "no_grad|inference_mode"):
                with zero_ablation(self.model, self.layers[0], "attention"):
                    with torch.enable_grad():
                        self.model(self.input)
        self.assertTrue(all(not hooks for hooks in self.hooks().values()))

    def test_invalid_component_or_layer_from_another_model_is_rejected(self):
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "component"):
                with zero_ablation(self.model, self.layers[0], "block"):
                    pass
            other = ToyModel().eval().requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "belong|match"):
                with zero_ablation(other, self.layers[0], "attention"):
                    pass
            stale = replace(self.layers[0], attention_path="does.not.exist")
            with self.assertRaisesRegex(ValueError, "belong|match"):
                with zero_ablation(self.model, stale, "attention"):
                    pass

    def test_no_grad_allows_multiple_calls_and_mlp_preserves_dtype(self):
        """no_grad 也满足前提；计数是模块命中次数，不把同一 scope 误计成多个独立实验。"""
        self.model.to(dtype=torch.bfloat16)
        module = self.layers[0].mlp_module
        with torch.no_grad(), zero_ablation(self.model, self.layers[0], "mlp") as state:
            for _ in range(2):
                output = module(self.input.to(dtype=torch.bfloat16))
                self.assertIs(type(output), torch.Tensor)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.shape, self.input.shape)
                self.assertEqual(torch.count_nonzero(output).item(), 0)
            self.assertEqual(state.calls, 2)
        self.assertTrue(all(not hooks for hooks in self.hooks().values()))

    def test_invalid_primary_tensor_rank_is_rejected_with_cleanup(self):
        """M1 核验的主输出是三维 hidden states；二维张量不能静默套用该契约。"""
        before = self.hooks()
        with torch.inference_mode():
            for component, module in (("attention", self.layers[0].attention_module), ("mlp", self.layers[0].mlp_module)):
                with self.subTest(component=component), self.assertRaisesRegex(ValueError, "shape"):
                    with zero_ablation(self.model, self.layers[0], component):
                        module(self.input.reshape(2, 4))
                self.assertEqual(before, self.hooks())

    def test_all_24_layer_targets_are_independent_and_restore(self):
        """只验证索引遍历与恢复，不能以这个 24 层线性替身冒充真实 Qwen 全层验证。"""
        model = ToyModel(layer_count=24).eval().requires_grad_(False)
        layers = discover_layer_modules(model)
        with torch.inference_mode():
            baseline = model(self.input)
            for layer in layers:
                for component in ("attention", "mlp"):
                    with self.subTest(layer=layer.index, component=component):
                        with zero_ablation(model, layer, component) as state:
                            model(self.input)
                        self.assertEqual(state.calls, 1)
                        self.assertTrue(torch.equal(baseline, model(self.input)))
                        self.assertTrue(all(not module._forward_hooks for module in model.modules()))


class AblationValidatorTests(unittest.TestCase):
    """验证 CLI 的成本/覆盖门禁及参数指纹工具；不在单元测试里加载真实权重。"""

    @classmethod
    def setUpClass(cls):
        """从项目脚本的绝对路径载入定义，不从数据缓存目录导入任何代码。"""
        path = Path(__file__).resolve().parents[1] / "scripts" / "validate_ablation.py"
        spec = importlib.util.spec_from_file_location("validate_ablation", path)
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def reject(self, args, message):
        """错误参数必须先于模型加载退出，避免验证失败仍读取近 1 GB 权重。"""
        errors = io.StringIO()
        with patch.object(sys, "argv", ["validate_ablation.py", *args]), \
             patch.object(self.cli, "load_model_and_tokenizer") as loader, redirect_stderr(errors):
            with self.assertRaises(SystemExit) as failure:
                self.cli.main()
            self.assertEqual(failure.exception.code, 2)
            self.assertIn(message, errors.getvalue())
            loader.assert_not_called()

    def test_invalid_layers_and_mutable_revision_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ["--output", str(Path(directory) / "validation.json")]
            self.reject([*common, "--layer", "-1"], "0..23")
            self.reject([*common, "--layer", "24"], "0..23")
            self.reject([*common, "--layer", "1", "--layer", "1"], "unique")
            self.reject([*common, "--revision", "main"], "40-character")
            self.reject([*common, "--threads", "0"], "positive")
            self.reject([*common, "--max-input-tokens", "0"], "positive")
            self.reject([*common, "--text", " "], "non-empty")

    def test_existing_validation_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text("historical result", encoding="utf-8")
            self.reject(["--output", str(path)], "Output already exists")
            self.assertEqual(path.read_text(encoding="utf-8"), "historical result")

    def test_parameter_fingerprint_reads_bfloat16_and_detects_changes(self):
        """只证明小参数指纹工具的字节敏感性；真实模型的前后哈希须另行运行。"""
        model = ToyModel().eval().requires_grad_(False).to(dtype=torch.bfloat16)
        first = self.cli.parameter_fingerprint(model)
        self.assertEqual(first, self.cli.parameter_fingerprint(model))
        self.assertEqual(first["parameter_elements"], sum(parameter.numel() for parameter in model.parameters()))
        with torch.no_grad():
            next(model.parameters())[0, 0] += 1
        self.assertNotEqual(first["sha256"], self.cli.parameter_fingerprint(model)["sha256"])


if __name__ == "__main__":
    unittest.main()
