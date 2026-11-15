"""M6 离线比较 CLI 的输入门禁与微型真实集成测试，不访问 Hub 或真实大模型。

M4/M2 由结构夹具提供，不能作为真实校准证据。只替换权重加载边界，其余三种方法实际
调用 PEFT、统一训练/评估、safetensors 保存；原始实验必须另外运行固定本地 0.5B CLI。
"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from adaptive_lora.data import build_calibration_prompt, hash_text, manifest_content_hash
from adaptive_lora.trainer import fingerprint_parameters
from test_comparison import ComparisonTokenizer
from test_lora_targets import tiny_model

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m6_compare_baselines", ROOT / "scripts/compare_baselines.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class CLITokenizer(ComparisonTokenizer):
    """补充来源检查所需元数据；仍然是字符级替身，不模拟真实 Qwen 的 BPE。"""

    is_fast = False
    padding_side = truncation_side = "right"
    pad_token_id = 0
    bos_token_id = None


def dump(path, data):
    """仅写新建临时夹具；负例会显式重写这些受控输入，绝不触及真实产物。"""
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def source_fixture(root):
    """建立 N30/24层/Top7 来源结构和与之绑定的输入清单，哈希只代表夹具一致性。"""
    m4 = root / "m4"
    m4.mkdir()
    tokenizer = CLITokenizer()
    samples = []
    for index in range(30):
        prompt = build_calibration_prompt(tokenizer, f"Calibration item {index}.", "")
        encoded = tokenizer(prompt)
        samples.append({"sample_id": f"fixture/calibration/{index}", "prompt": prompt, "prompt_hash": hash_text(prompt),
                        "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                        "prediction_position": len(encoded["input_ids"]) - 1, "truncated": False})
    meta = {"class": type(tokenizer).__name__, "model_id": CLI.DEFAULT_MODEL_ID, "model_revision": CLI.MODEL_REVISION,
            **{name: getattr(tokenizer, name) for name in
               ("is_fast", "padding_side", "truncation_side", "pad_token_id", "eos_token_id", "bos_token_id")}}
    manifest = {"schema_version": 1, "dataset": {"id": "fixture/calibration", "split": "train", "revision": "b" * 40},
                "sampling": {"n": 30}, "template": {"text": tokenizer.get_chat_template(), "sha256": hash_text(tokenizer.get_chat_template())},
                "tokenizer": meta, "samples": samples}
    manifest["content_sha256"] = manifest_content_hash(manifest)
    manifest_path = root / "manifest.json"
    dump(manifest_path, manifest)
    chosen = [0, 1, 2, 5, 3, 8, 10]
    selection = {"index_base": 0, "layer_count": 24, "top_k": 7, "mode": "accuracy: Top-K over all layers",
                 "selected_layer_indices": chosen, "selected_layer_indices_ascending": sorted(chosen),
                 "ranked_layer_indices": chosen + [index for index in range(24) if index not in chosen]}
    config = {"model_id": CLI.DEFAULT_MODEL_ID, "requested_revision": CLI.MODEL_REVISION, "n": 30, "manifest_n": 30,
              "limit": None, "top_k": 7, "layer_count": 24, "index_base": 0, "selection_mode": "accuracy/all layers",
              "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "manifest_content_sha256": manifest["content_sha256"], "tokenizer": meta,
              "template_sha256": hash_text(tokenizer.get_chat_template())}
    dump(m4 / "selected_layers.json", selection)
    dump(m4 / "resolved_config.json", config)
    hashes = {name: hashlib.sha256((m4 / name).read_bytes()).hexdigest() for name in ("selected_layers.json", "resolved_config.json")}
    dump(m4 / "run_summary.json", {"status": "PASS", "n": 30, "completed_samples": 30, "forward_evaluations": 1500,
                                  "selected_layer_indices": chosen, "files_sha256": hashes})
    return manifest_path, hashes["selected_layers.json"]


class ComparisonSourceTests(unittest.TestCase):
    """来源/manifest/tokenizer 必须绑定；评估隔离不能建立在别人手填的 token 哈希上。"""

    def load(self, root, manifest, digest):
        return CLI.load_comparison_sources(root / "m4", manifest, expected_selection_sha256=digest,
                                           model_id=CLI.DEFAULT_MODEL_ID, revision=CLI.MODEL_REVISION)

    def test_matching_m4_and_m2_sources_and_tokenizer_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, digest = source_fixture(root)
            source = self.load(root, manifest, digest)
            CLI.verify_comparison_tokenizer(source, CLITokenizer())
            self.assertEqual(source["selected_layer_indices"], [0, 1, 2, 5, 3, 8, 10])
            self.assertEqual(len(source["calibration_samples"]), 30)

    def test_wrong_pin_manifest_bytes_or_model_revision_are_rejected(self):
        for change in ("pin", "manifest", "revision", "partial"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, digest = source_fixture(root)
                if change == "pin":
                    digest = "0" * 64
                elif change == "manifest":
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    data["samples"][0]["input_ids"][0] = 63
                    dump(manifest, data)
                elif change == "revision":
                    config = json.loads((root / "m4/resolved_config.json").read_text(encoding="utf-8"))
                    config["requested_revision"] = "c" * 40
                    dump(root / "m4/resolved_config.json", config)
                else:
                    summary = json.loads((root / "m4/run_summary.json").read_text(encoding="utf-8"))
                    summary["n"] = 1
                    dump(root / "m4/run_summary.json", summary)
                with self.assertRaises(ValueError):
                    self.load(root, manifest, digest)

    def test_retokenization_checks_real_ids_instead_of_only_tokenizer_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, digest = source_fixture(root)
            source = self.load(root, manifest, digest)
            source["calibration_samples"][0]["input_ids"][0] = 63
            with self.assertRaises(ValueError):
                CLI.verify_comparison_tokenizer(source, CLITokenizer())


class CompareBaselinesCLITests(unittest.TestCase):
    """每次用独立 tiny-model 基础实例；输出只写临时目录，失败保留而不虚构整体 PASS。"""

    def rejected(self, args, message):
        stderr = io.StringIO()
        with patch.object(CLI, "load_model_and_tokenizer") as loader, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                CLI.main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            loader.assert_not_called()

    def test_step_seed_revision_rank_and_generation_bounds_fail_before_load(self):
        for flag, value, message in (("--steps", "4", "3"), ("--layer-seed", "-1", "seed"),
                                     ("--train-seed", "-1", "seed"), ("--revision", "main", "40-character"),
                                     ("--threads", "0", "positive"), ("--rank", "0", "setting"),
                                     ("--max-new-tokens", "9", "setting")):
            self.rejected(["--output-dir", "unused-m6-test-output", flag, value], message)

    def test_full_training_download_and_gpu_flags_are_not_available(self):
        for flag, value in (("--epochs", "3"), ("--device", "cuda"), ("--allow-download", "true")):
            self.rejected(["--output-dir", "unused-m6-test-output", flag, value], "unrecognized")

    def test_existing_output_missing_cache_or_bad_source_fails_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep").write_bytes(b"untouched")
            self.rejected(["--output-dir", str(root)], "already exists")
            self.assertEqual((root / "keep").read_bytes(), b"untouched")
            self.rejected(["--output-dir", str(root / "run"), "--cache-dir", str(root / "missing")], "cache")
            self.rejected(["--output-dir", str(root / "run"), "--cache-dir", str(root), "--m4-dir", str(root / "missing")], "source")

    def run_fixture(self, root, *, fail_method=None):
        """只 mock 权重加载；三方法真实复用注入/优化/生成/保存，参数对象不共享。"""
        manifest, digest = source_fixture(root)
        (root / "cache").mkdir()
        torch.manual_seed(42)
        base = tiny_model(24)
        state = copy.deepcopy(base.state_dict())
        expected = fingerprint_parameters(base.named_parameters())
        models = []
        def load(*args, **kwargs):
            """每次从同一小型基础 state_dict 新建模型，模拟固定缓存加载，不模拟数值前向。"""
            model = tiny_model(24)
            model.load_state_dict(copy.deepcopy(state))
            models.append(model)
            return model, CLITokenizer()
        original = CLI.run_variant
        def run(plan, *args, **kwargs):
            """仅向指定方法注入受控执行异常，前一种成功结果不得冒充三种都完成。"""
            if plan.method == fail_method:
                raise RuntimeError(f"controlled {plan.method} failure")
            return original(plan, *args, **kwargs)
        args = ["--m4-dir", str(root / "m4"), "--manifest", str(manifest), "--expected-selection-sha256", digest,
                "--cache-dir", str(root / "cache"), "--output-dir", str(root / "run"), "--steps", "3", "--threads", "1",
                "--layer-seed", "42", "--max-new-tokens", "1"]
        with patch.object(CLI, "EXPECTED_BASE_FINGERPRINT", expected), patch.object(CLI, "load_model_and_tokenizer", side_effect=load) as loader, \
                patch.object(CLI, "run_variant", side_effect=run), redirect_stdout(io.StringIO()):
            if fail_method:
                with self.assertRaisesRegex(RuntimeError, f"controlled {fail_method} failure"):
                    CLI.main(args)
            else:
                result = CLI.main(args)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(loader.call_count, 3)
                for call in loader.call_args_list:
                    self.assertFalse(call.kwargs["allow_download"])
                    self.assertEqual(call.kwargs["revision"], CLI.MODEL_REVISION)
        self.assertEqual(len({id(model) for model in models}), len(models))
        return models

    def test_all_three_methods_run_real_tiny_pipeline_and_save_actual_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_fixture(root)
            run = root / "run"
            comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
            rows = comparison["methods"]
            self.assertEqual([row["method"] for row in rows], ["full", "random_k", "adaptive_k"])
            self.assertEqual([row["trainable_parameter_elements"] for row in rows], [30720, 8960, 8960])
            self.assertFalse(comparison["same_parameter_budget"])
            self.assertEqual(rows[1]["selection_seed"], 42)
            for method, expected_tensors in (("full", 144), ("random_k", 42), ("adaptive_k", 42)):
                self.assertEqual(len(load_file(str(run / method / "checkpoint/adapter_model.safetensors"))), expected_tensors)
                self.assertEqual(len((run / method / "steps.jsonl").read_text(encoding="utf-8").splitlines()), 3)
                self.assertTrue((run / method / "completion.json").is_file())
            summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
            for name, digest in summary["files_sha256"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)
            self.assertFalse((run / "failure.json").exists())
            self.assertTrue((run / "comparison.csv").is_file())

    def test_second_method_failure_keeps_partial_evidence_without_overall_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = self.run_fixture(root, fail_method="random_k")
            self.assertEqual(len(models), 2)
            self.assertTrue((root / "run/full/completion.json").is_file())
            self.assertFalse((root / "run/random_k/completion.json").exists())
            self.assertFalse((root / "run/adaptive_k").exists())
            self.assertTrue((root / "run/failure.json").is_file())
            self.assertFalse((root / "run/run_summary.json").exists())
            self.assertFalse((root / "run/comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
