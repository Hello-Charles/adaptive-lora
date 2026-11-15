"""M5 CLI 门禁与端到端测试；仅把模型加载替换为微型随机权重，不访问下载或真实大模型。

成功路径实际执行 PEFT 注入、3-step 反向、safetensors 保存及新模型恢复；M4 输入证据是
人工构造的结构夹具，不冒充真实校准。固定 0.5B 的运行证据必须由独立 CLI 命令产生。
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

from adaptive_lora.calibration import parameter_fingerprint
from adaptive_lora.data import hash_text
from test_lora_targets import tiny_model
from test_trainer import BoundaryTokenizer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m5_train", ROOT / "scripts/train.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class FixtureTokenizer(BoundaryTokenizer):
    """只支持当前合成短样本与元数据契约，不模拟真实 Qwen tokenizer。"""

    is_fast = False
    padding_side = truncation_side = "right"
    pad_token_id = 0
    bos_token_id = None

    def get_chat_template(self):
        return "fixture user prefix with assistant marker"


def m4_fixture(directory):
    """创建结构上完整的 N30/24 层选层夹具，哈希仅证明这些夹具字节而非真实校准。"""
    directory.mkdir()
    selected = [0, 1, 2, 5, 3, 8, 10]
    selection = {
        "index_base": 0, "mode": "accuracy: Top-K over all layers", "top_k": 7, "layer_count": 24,
        "selected_layer_indices": selected, "selected_layer_indices_ascending": sorted(selected),
        "ranked_layer_indices": selected + [index for index in range(24) if index not in selected],
        "tie_break": "descending importance, then ascending layer index [复现决策]",
    }
    tokenizer = FixtureTokenizer()
    config = {
        "model_id": CLI.DEFAULT_MODEL_ID, "requested_revision": CLI.MODEL_REVISION,
        "n": 30, "manifest_n": 30, "limit": None, "top_k": 7, "layer_count": 24,
        "index_base": 0, "selection_mode": "accuracy/all layers",
        "template_sha256": hash_text(tokenizer.get_chat_template()),
        "tokenizer": {"class": type(tokenizer).__name__, **{name: getattr(tokenizer, name) for name in
                       ("is_fast", "padding_side", "truncation_side", "pad_token_id", "eos_token_id", "bos_token_id")}},
    }
    for name, value in (("selected_layers.json", selection), ("resolved_config.json", config)):
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ("selected_layers.json", "resolved_config.json")}
    summary = {"status": "PASS", "n": 30, "completed_samples": 30, "forward_evaluations": 1500,
               "selected_layer_indices": selected, "files_sha256": hashes}
    (directory / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return hashes["selected_layers.json"]


class M4SelectionInputTests(unittest.TestCase):
    """确认 CLI 读取 N30 而不是 N1；输入哈希不是独立认证论文作者来源。"""

    def test_fixed_n30_selection_and_zero_based_full_ranking_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m4"
            digest = m4_fixture(path)
            result = CLI.load_m4_selection(path, expected_selection_sha256=digest, model_id=CLI.DEFAULT_MODEL_ID,
                                           revision=CLI.MODEL_REVISION)
            self.assertEqual(result["selection"]["selected_layer_indices"], [0, 1, 2, 5, 3, 8, 10])

    def test_wrong_digest_incomplete_n30_and_rehashed_invalid_indices_fail(self):
        for change in ("digest", "n1", "failed", "indices", "revision"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "m4"
                digest = m4_fixture(path)
                summary_path = path / "run_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if change == "digest":
                    digest = "0" * 64
                elif change == "n1":
                    summary["n"] = summary["completed_samples"] = 1
                elif change == "failed":
                    summary["status"] = "FAIL"
                else:
                    name = "selected_layers.json" if change == "indices" else "resolved_config.json"
                    value = json.loads((path / name).read_text(encoding="utf-8"))
                    if change == "indices":
                        value["selected_layer_indices"][0] = True
                    else:
                        value["requested_revision"] = "b" * 40
                    (path / name).write_text(json.dumps(value), encoding="utf-8")
                    summary["files_sha256"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
                    if change == "indices":
                        digest = summary["files_sha256"][name]
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(ValueError):
                    CLI.load_m4_selection(path, expected_selection_sha256=digest, model_id=CLI.DEFAULT_MODEL_ID,
                                           revision=CLI.MODEL_REVISION)


class TrainCLITests(unittest.TestCase):
    """CLI 的受控小模型集成；真实模型训练由后续新结果目录单独验收。"""

    def rejected(self, args, message):
        stderr = io.StringIO()
        with patch.object(CLI, "load_model_and_tokenizer") as loader, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                CLI.main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            loader.assert_not_called()

    def test_step_limit_revision_seed_and_threads_fail_before_loading(self):
        for flag, value, message in (("--steps", "0", "1"), ("--steps", "4", "3"),
                                     ("--revision", "main", "40-character"), ("--threads", "0", "positive"),
                                     ("--seed", "-1", "seed")):
            self.rejected(["--output-dir", "unused-m5-test-output", flag, value], message)

    def test_existing_output_or_missing_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sentinel").write_bytes(b"keep")
            self.rejected(["--output-dir", str(root)], "already exists")
            self.assertEqual((root / "sentinel").read_bytes(), b"keep")
            self.rejected(["--output-dir", str(root / "new"), "--cache-dir", str(root / "absent")], "cache")

    def run_fixture(self, root, *, fail_training=False, fail_restore=False):
        """只替换离线加载边界；两次提供不同实例、同一基础 state_dict，其他路径真实执行。"""
        digest = m4_fixture(root / "m4")
        (root / "cache").mkdir()
        torch.manual_seed(42)
        first, second = tiny_model(24), tiny_model(24)
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        expected_base = parameter_fingerprint(first)
        args = ["--m4-dir", str(root / "m4"), "--expected-selection-sha256", digest,
                "--cache-dir", str(root / "cache"), "--output-dir", str(root / "run"), "--steps", "3", "--threads", "1"]
        with patch.object(CLI, "EXPECTED_BASE_FINGERPRINT", expected_base), \
                patch.object(CLI, "load_model_and_tokenizer", side_effect=[(first, FixtureTokenizer()), (second, FixtureTokenizer())]) as loader, \
                redirect_stdout(io.StringIO()):
            if fail_training:
                with patch.object(CLI, "train_smoke", side_effect=RuntimeError("controlled training failure")), \
                        patch.object(CLI, "save_adapter_checkpoint") as saver:
                    with self.assertRaisesRegex(RuntimeError, "controlled training failure"):
                        CLI.main(args)
                    saver.assert_not_called()
            elif fail_restore:
                with patch.object(CLI, "restore_adapter_checkpoint", side_effect=RuntimeError("controlled restore failure")):
                    with self.assertRaisesRegex(RuntimeError, "controlled restore failure"):
                        CLI.main(args)
            else:
                result = CLI.main(args)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(loader.call_count, 2)
                for call in loader.call_args_list:
                    self.assertFalse(call.kwargs["allow_download"])
                    self.assertEqual(call.kwargs["revision"], CLI.MODEL_REVISION)

    def test_real_tiny_end_to_end_saves_metrics_and_verifies_fresh_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_fixture(root)
            run = root / "run"
            summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_steps"], 3)
            self.assertTrue(summary["reload_logits_exact"])
            self.assertEqual(summary["selected_layer_indices"], [0, 1, 2, 5, 3, 8, 10])
            self.assertEqual(len((run / "steps.jsonl").read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue((run / "checkpoint/adapter_model.safetensors").is_file())
            for name, digest in summary["files_sha256"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)
            self.assertFalse((run / "failure.json").exists())

    def test_training_failure_has_no_checkpoint_or_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_fixture(root, fail_training=True)
            self.assertTrue((root / "run/failure.json").exists())
            self.assertFalse((root / "run/checkpoint").exists())
            self.assertFalse((root / "run/run_summary.json").exists())

    def test_reload_failure_does_not_mark_saved_checkpoint_as_passed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_fixture(root, fail_restore=True)
            self.assertTrue((root / "run/checkpoint/adapter_model.safetensors").exists())
            self.assertTrue((root / "run/failure.json").exists())
            self.assertFalse((root / "run/run_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
