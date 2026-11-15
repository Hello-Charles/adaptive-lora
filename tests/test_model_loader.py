"""模型加载契约的离线测试；模拟模块不包含真实权重或前向计算。

这些测试约束加载选项、路径匹配及预测位置算法，不能替代真实 Qwen 结构实测。
"""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from adaptive_lora.model_loader import (
    DEFAULT_MODEL_ID,
    describe_output,
    discover_layer_modules,
    load_model_and_tokenizer,
    prediction_position,
)


class FakeModule:
    """仅模拟 named_children/named_modules 的树遍历契约，不是 torch.nn.Module。"""

    def __init__(self, children=()):
        self.children = list(children)

    def named_children(self):
        return iter(self.children)

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self.children:
            path = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(prefix=path)


class ModuleList(FakeModule):
    """类名是结构发现器的识别条件，替身须保留 ModuleList 名称。"""

    def __len__(self):
        return len(self.children)


class FakeModel(FakeModule):
    """归一化层也含 attention 字样，模块匹配不能把它当作 Attention 目标。

    missing_projection 指定缺少 K 投影的层号，用于拒绝不完整结构的测试。
    """

    def __init__(self, count=24, missing_projection=None):
        blocks = []
        for index in range(count):
            names = ("q", "k", "v")
            if index == missing_projection:
                names = ("q", "v")
            attention = FakeModule([(f"{name}_proj", FakeModule()) for name in names])
            blocks.append((str(index), FakeModule([
                ("self_attn", attention),
                ("mlp", FakeModule()),
                ("input_layernorm", FakeModule()),
                ("post_attention_layernorm", FakeModule()),
            ])))
        super().__init__([("model", FakeModule([("layers", ModuleList(blocks))]))])
        self.config = SimpleNamespace(num_hidden_layers=count)


class FakeMask:
    """仅模拟一行二值 mask 的 shape 和 tolist，不具备一般张量索引语义。"""

    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, index):
        return SimpleNamespace(tolist=lambda: self.values)


class ModelInspectionTests(unittest.TestCase):
    """真实库加载器由 mock 替代；断言选项并不证明模型或 tokenizer 实际可加载。"""

    def test_loader_is_offline_by_default_and_uses_safe_model_loading(self):
        """默认路径不得联网或启用远端代码；限制只通过调用参数检查。"""
        transformer = ModuleType("transformers")
        model = SimpleNamespace(eval=Mock())
        transformer.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value="tokenizer"))
        transformer.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
        with patch.dict(sys.modules, {"transformers": transformer}):
            loaded, tokenizer = load_model_and_tokenizer()
            self.assertIs(loaded, model)
            self.assertEqual(tokenizer, "tokenizer")
            transformer.AutoTokenizer.from_pretrained.assert_called_once_with(
                DEFAULT_MODEL_ID, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            transformer.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                DEFAULT_MODEL_ID, use_safetensors=True, local_files_only=True, trust_remote_code=False
            )
            model.eval.assert_called_once_with()

    def test_download_requires_explicit_flag(self):
        """显式许可仅改变 mock 调用参数；这里不访问 Hub，也不下载权重。"""
        transformer = ModuleType("transformers")
        transformer.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value="tokenizer"))
        transformer.AutoModelForCausalLM = SimpleNamespace(
            from_pretrained=Mock(return_value=SimpleNamespace(eval=Mock()))
        )
        with patch.dict(sys.modules, {"transformers": transformer}):
            load_model_and_tokenizer(revision="pinned-sha", allow_download=True, cache_dir=Path("data/models/m1"))
            self.assertFalse(transformer.AutoModelForCausalLM.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertEqual(transformer.AutoModelForCausalLM.from_pretrained.call_args.kwargs["revision"], "pinned-sha")
            self.assertEqual(transformer.AutoModelForCausalLM.from_pretrained.call_args.kwargs["cache_dir"], str(Path("data/models/m1")))

    def test_download_cli_requires_pinned_revision_and_new_cache(self):
        """错误参数须在模型加载前退出，不能在已有目录混入新下载或留下成功报告。"""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "structure.json"
            cache = Path(directory) / "cache"
            env = {**os.environ, "PYTHONPATH": str(root / "src")}
            command = [sys.executable, str(root / "scripts" / "inspect_model.py"), "--allow-download", "--output", str(output)]
            missing = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, check=False)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("requires both --revision and --cache-dir", missing.stderr)
            cache.mkdir()
            existing = subprocess.run(
                command + ["--revision", "pinned-sha", "--cache-dir", str(cache)],
                cwd=directory, env=env, capture_output=True, text=True, check=False,
            )
            self.assertEqual(existing.returncode, 2)
            self.assertIn("new, empty directory", existing.stderr)
            self.assertFalse(output.exists())

    def test_discovers_unique_layer_paths_and_qkv_in_fake_model(self):
        layers = discover_layer_modules(FakeModel())
        self.assertEqual(len(layers), 24)
        self.assertEqual([layer.index for layer in layers], list(range(24)))
        self.assertEqual(layers[0].attention_path, "model.layers.0.self_attn")
        self.assertEqual(layers[-1].mlp_path, "model.layers.23.mlp")
        self.assertEqual(layers[0].projections["k"], "model.layers.0.self_attn.k_proj")

    def test_discovers_28_layer_model_for_1_5b(self):
        """M7 目标 1.5B 有 28 层；结构发现器须接受 expected_layers=28 而不报错。"""
        layers = discover_layer_modules(FakeModel(count=28), expected_layers=28)
        self.assertEqual(len(layers), 28)
        self.assertEqual(layers[-1].mlp_path, "model.layers.27.mlp")
        self.assertEqual(layers[-1].projections["v"], "model.layers.27.self_attn.v_proj")
        with self.assertRaisesRegex(ValueError, "Expected 28 layers"):
            discover_layer_modules(FakeModel(count=24), expected_layers=28)


        with self.assertRaisesRegex(ValueError, "Expected 24 layers"):
            discover_layer_modules(FakeModel(count=23))
        with self.assertRaisesRegex(ValueError, "k_proj"):
            discover_layer_modules(FakeModel(missing_projection=5))

    def test_position_uses_mask_instead_of_right_padding(self):
        ids = SimpleNamespace(shape=(1, 4))
        self.assertEqual(prediction_position(ids), 3)
        self.assertEqual(prediction_position(ids, FakeMask([1, 1, 0, 0])), 1)
        self.assertEqual(prediction_position(ids, FakeMask([0, 1, 1, 0])), 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            prediction_position(ids, FakeMask([1, 0, 1, 0]))

    def test_describes_tensor_and_tuple_without_retaining_values(self):
        fake_tensor = SimpleNamespace(shape=(1, 3, 8), dtype="float32")
        self.assertEqual(
            describe_output((fake_tensor, None)),
            {
                "type": "tuple",
                "items": [
                    {"type": "SimpleNamespace", "shape": [1, 3, 8], "dtype": "float32"},
                    {"type": "NoneType"},
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
