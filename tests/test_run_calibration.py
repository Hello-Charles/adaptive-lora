"""M4 CLI 门禁测试；替换模型加载和校准，不能作为真实离线 Qwen/吞吐证据。"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from adaptive_lora.data import build_manifest, manifest_content_hash, sample_indices

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_calibration", ROOT / "scripts" / "run_calibration.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class FixtureTokenizer:
    """恒定的两个 token 只验证协议/调用参数，不模拟 Qwen 模板或真实 tokenizer。"""

    is_fast = True
    padding_side = truncation_side = "right"
    pad_token_id = eos_token_id = 0
    bos_token_id = None

    def get_chat_template(self):
        return "fixture-template"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        return "user:" + messages[0]["content"] + "\nassistant:"

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}


def fixture_manifest():
    """通过已测试的 M2 builder 建立最小 train 清单；不使用任何下载数据。"""
    rows = [{"instruction": "question one", "input": "", "output": "answer one"},
            {"instruction": "question two", "input": "", "output": "answer two"}]
    return build_manifest(
        rows=rows, indices=sample_indices(2, 2, 42), tokenizer=FixtureTokenizer(),
        dataset_id="fixture/data", dataset_revision="b" * 40,
        model_id="fixture/model", model_revision="a" * 40,
        max_length=32, seed=42,
    )


class CalibrationManifestTests(unittest.TestCase):
    """内容哈希只是变化检测，真正输入来源/答案隔离仍依赖 M2 的固定清单验收。"""

    def check(self, manifest):
        return CLI.validate_manifest(manifest, model_id="fixture/model", revision="a" * 40,
                                     expected_content_sha256=manifest["content_sha256"])

    def test_accepts_fixed_train_manifest_and_matching_tokenizer(self):
        manifest = fixture_manifest()
        self.check(manifest)
        CLI.verify_tokenizer(manifest, FixtureTokenizer())

    def test_rejects_changed_content_before_tokenizer_or_model_load(self):
        manifest = fixture_manifest()
        manifest["samples"][0]["input_ids"][0] = 2
        with self.assertRaisesRegex(ValueError, "content_sha256"):
            self.check(manifest)

    def test_rejects_revision_template_answer_fields_and_inconsistent_indices(self):
        for change in ("revision", "answer", "sample_id", "position", "truncation", "prompt_hash"):
            manifest = fixture_manifest()
            if change == "revision":
                manifest["tokenizer"]["model_revision"] = "c" * 40
            elif change == "answer":
                manifest["template"]["source_fields"].append("output")
            elif change == "sample_id":
                manifest["samples"][0]["sample_id"] = "different-row"
            elif change == "position":
                manifest["samples"][0]["prediction_position"] = 0
            elif change == "truncation":
                manifest["samples"][0]["truncated"] = True
            else:
                manifest["samples"][0]["prompt_hash"] = "0" * 64
            # 即使攻击/误改方重算内容哈希，内部协议也应独立拒绝不一致记录。
            manifest["content_sha256"] = manifest_content_hash(manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(manifest)

    def test_retokenization_detects_changed_ids_with_recomputed_hash(self):
        manifest = fixture_manifest()
        manifest["samples"][0]["input_ids"][0] = 2
        manifest["content_sha256"] = manifest_content_hash(manifest)
        self.check(manifest)
        with self.assertRaisesRegex(ValueError, "tokeniz"):
            CLI.verify_tokenizer(manifest, FixtureTokenizer())


class RunCalibrationCLITests(unittest.TestCase):
    """CLI 模型为 mock，只检查执行顺序、独占输出和失败标记，真实校准另行运行。"""

    def rejected(self, args, message):
        stderr = io.StringIO()
        with patch.object(CLI, "load_model_and_tokenizer") as loader, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                CLI.main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            loader.assert_not_called()

    def test_invalid_revision_positive_limits_and_top_k_fail_before_load(self):
        for flag, value, message in (("--revision", "main", "40-character"),
                                     ("--limit", "0", "positive"), ("--threads", "0", "positive"),
                                     ("--top-k", "25", "24")):
            self.rejected(["--output-dir", "unused-m4-test-output", flag, value], message)

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "historical.txt"
            sentinel.write_text("keep", encoding="utf-8")
            self.rejected(["--output-dir", directory], "Output directory already exists")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_missing_cache_or_bad_manifest_fails_without_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.rejected(["--output-dir", str(root / "out"), "--cache-dir", str(root / "missing")], "cache")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            self.rejected(["--output-dir", str(root / "out"), "--cache-dir", directory,
                           "--manifest", str(manifest)], "manifest")
            self.assertFalse((root / "out").exists())

    def run_fixture(self, root, *, failure=False):
        """不执行真实 forward，保留逐样本回调与导出路径，让失败/成功产物契约可测试。"""
        manifest = fixture_manifest()
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        args = ["--output-dir", str(root / "out"), "--manifest", str(path),
                "--cache-dir", str(root), "--model-id", "fixture/model", "--revision", "a" * 40,
                "--expected-manifest-sha256", manifest["content_sha256"], "--limit", "1"]
        model = Mock()
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        model.config = SimpleNamespace(_commit_hash=None, _attn_implementation="fixture")
        layers = [SimpleNamespace(index=index, attention_path=f"layers.{index}.attention", mlp_path=f"layers.{index}.mlp")
                  for index in range(24)]

        def fake_calibrate(_model, samples, *, layers, on_sample):
            """假分数明确不用于验收报告；只模拟 CLI 对已完成样本的保存。"""
            if failure:
                raise RuntimeError("injected CLI failure")
            records = []
            for sample in samples:
                record = {"sample_id": sample["sample_id"], "token_id": 1, "q": 0.6, "layers": [],
                          "restored_logits_equal": True, "forward_evaluations": 50}
                on_sample(record)
                records.append(record)
            return {"n": len(samples), "layer_count": 24, "samples": records, "forward_evaluations": 50,
                    "layer_importance": [{"layer_index": index, "importance": index / 100,
                                          "attention_mean_abs_delta": index / 200,
                                          "mlp_mean_abs_delta": index / 200} for index in range(24)]}

        with patch.object(CLI, "load_model_and_tokenizer", return_value=(model, FixtureTokenizer())) as loader, \
             patch.object(CLI, "discover_layer_modules", return_value=layers), \
             patch.object(CLI, "calibrate_model", side_effect=fake_calibrate), redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "injected CLI failure"):
                    CLI.main(args)
            else:
                CLI.main(args)
            loader.assert_called_once_with("fixture/model", revision="a" * 40, cache_dir=root, allow_download=False)
        return root / "out"

    def test_success_writes_config_sample_subset_raw_trace_and_final_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_fixture(Path(directory))
            summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
            config = json.loads((out / "resolved_config.json").read_text(encoding="utf-8"))
            identifiers = json.loads((out / "sample_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(config["n"], 1)
            self.assertEqual(config["manifest_n"], 2)
            self.assertFalse(config["allow_download"])
            self.assertEqual(len(identifiers), 1)
            self.assertEqual(len((out / "samples.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((out / "environment.json").is_file())
            self.assertTrue((out / "provenance.json").is_file())
            self.assertTrue((out / "importance.html").is_file())
            self.assertFalse((out / "failure.json").exists())

    def test_failed_calibration_never_writes_selection_or_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.run_fixture(Path(directory), failure=True)
            failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "FAIL")
            self.assertFalse((out / "selected_layers.json").exists())
            self.assertFalse((out / "run_summary.json").exists())
            self.assertTrue((out / "run.log").is_file())


if __name__ == "__main__":
    unittest.main()
