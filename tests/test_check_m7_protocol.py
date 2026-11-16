"""M7离线协议CLI门禁测试，不连接服务器、访问Hub或申请GPU。

confirmed夹具只覆盖配置审查分支，不代表真实授权；-I -S子进程验证检查器在
完全不加载site-packages时仍可运行，不能把这一结果当作CUDA/训练依赖验收。
"""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_m7_protocol.py"
SPEC = importlib.util.spec_from_file_location("m7_protocol_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def fixture(path, *, confirmed=False):
    """仅在测试临时目录生成配置；不会改变仓库草案的实际确认状态。"""
    config = json.loads((ROOT / "configs/qwen2_5_1_5b_commonsense_r16.json").read_text(encoding="utf-8"))
    for decision in config["decisions"]:
        decision["status"] = "confirmed" if confirmed else "pending"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config


class M7ProtocolCLITests(unittest.TestCase):
    """预检只写新的本地审查目录；不能获得运行训练或覆盖已有结果的能力。"""

    def invoke(self, arguments):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            result = CLI.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_pending_draft_returns_two_and_writes_explicit_nonexecution_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "input.json")
            result, stdout, _ = self.invoke(["--config", str(root / "input.json"), "--output-dir", str(root / "output")])
            self.assertEqual(result, 2)
            self.assertIn("PENDING_DECISIONS", stdout)
            saved = json.loads((root / "output/preflight.json").read_text(encoding="utf-8"))
            # 10 项确认决策：原 7 项 + evaluation_truncation + training_order + final_partial_group
            self.assertEqual(len(saved["pending_decisions"]), 10)
            self.assertFalse(saved["execution_authorized"])
            self.assertFalse(saved["training_started"])
            self.assertEqual(set(path.name for path in (root / "output").iterdir()), {"config.json", "preflight.json"})
            self.assertIn("src/adaptive_lora/m7_protocol.py", saved["source_files_sha256"])

    def test_confirmed_fixture_returns_zero_but_never_marks_runtime_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "input.json", confirmed=True)
            result, _, _ = self.invoke(["--config", str(root / "input.json"), "--output-dir", str(root / "output")])
            self.assertEqual(result, 0)
            saved = json.loads((root / "output/preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["review_status"], "REVIEWED_NOT_EXECUTABLE")
            self.assertFalse(saved["runtime_validated"])
            self.assertFalse(saved["execution_authorized"])

    def test_invalid_protocol_is_rejected_before_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root / "input.json")
            config["model"]["revision"] = "main"
            (root / "input.json").write_text(json.dumps(config), encoding="utf-8")
            result, _, stderr = self.invoke(["--config", str(root / "input.json"), "--output-dir", str(root / "output")])
            self.assertEqual(result, 2)
            self.assertIn("ERROR", stderr)
            self.assertFalse((root / "output").exists())

    def test_existing_output_is_preserved_without_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "input.json")
            output = root / "output"
            output.mkdir()
            (output / "sentinel.txt").write_text("preserve", encoding="utf-8")
            result, _, _ = self.invoke(["--config", str(root / "input.json"), "--output-dir", str(output)])
            self.assertEqual(result, 2)
            self.assertEqual((output / "sentinel.txt").read_text(encoding="utf-8"), "preserve")
            self.assertEqual([path.name for path in output.iterdir()], ["sentinel.txt"])

    def test_cli_exposes_no_download_install_remote_or_training_flags(self):
        for option in ("--allow-download", "--train", "--device", "--epochs", "--ssh", "--install"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                CLI.main(["--output-dir", "unused", option])
            self.assertEqual(raised.exception.code, 2)

    def test_entrypoint_runs_without_site_packages_and_even_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "input.json")
            completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", str(SCRIPT), "--config", str(root / "input.json"), "--output-dir", str(root / "output")], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("PENDING_DECISIONS", completed.stdout)
            saved = json.loads((root / "output/preflight.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["runtime_validated"])
            self.assertFalse(saved["training_started"])


if __name__ == "__main__":
    unittest.main()
