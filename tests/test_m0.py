"""工程骨架测试不执行模型前向，也不下载模型或数据。

测试逻辑使用标准库；环境快照子进程会尝试导入当前解释器中的已安装包。
包缺失或导入失败可以是合法快照内容，测试通过不等于机器学习运行环境可用。
"""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectSkeletonTests(unittest.TestCase):
    """只验证工程入口与证据格式；不把环境状态的记录误作模型可运行保证。"""

    def test_package_imports_from_src(self):
        """必须导入当前源码，不能误用其他目录中同名的已安装包。"""
        package = importlib.import_module("adaptive_lora")
        self.assertEqual(
            Path(package.__file__).resolve(),
            ROOT / "src" / "adaptive_lora" / "__init__.py",
        )

    def test_environment_snapshot_reports_missing_packages_and_git(self):
        """脚本在非项目工作目录也应定位项目，合法的缺包状态不能使记录过程崩溃。"""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "environment.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "scripts" / "record_environment.py"),
                    "--output",
                    str(destination),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            snapshot = json.loads(destination.read_text(encoding="utf-8"))
            self.assertIn("version", snapshot["python"])
            self.assertIn("executable", snapshot["python"])
            for name in ("torch", "transformers", "peft", "datasets", "accelerate", "pytest"):
                self.assertIn(snapshot["packages"][name]["status"], {"ok", "not_installed", "import_failed"})
                self.assertIn("version", snapshot["packages"][name])
            self.assertIn(snapshot["git"]["status"], {"ok", "not_repository", "no_commits", "git_unavailable"})
            if snapshot["git"]["status"] != "ok":
                self.assertIsNone(snapshot["git"]["commit"])

    def test_metadata_only_snapshot_records_versions_without_importing_packages(self):
        """仅查元数据时应标记 not_checked；这里检查报告状态，并非拦截所有导入。"""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "environment.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ROOT / "scripts" / "record_environment.py"),
                    "--metadata-only",
                    "--output",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            snapshot = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["packages"]["pip"]["status"], "not_checked")
            self.assertIsNotNone(snapshot["packages"]["pip"]["version"])

    def test_gitignore_excludes_local_artifacts_but_keeps_sources_and_reports(self):
        """忽略规则只在临时仓库评估，不改项目索引，也不能保护已被跟踪的敏感文件。"""
        git = shutil.which("git")
        if git is None:
            self.skipTest("Git is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(ROOT / ".gitignore", Path(directory) / ".gitignore")
            subprocess.run([git, "-C", directory, "init", "-q"], check=True, capture_output=True)
            for path in (
                "data/train.jsonl",
                "results/trial/run.log",
                "checkpoints/last.ckpt",
                "models/base.safetensors",
                "papers/adaptive_lora.pdf",
                ".venv/Scripts/python.exe",
                ".claude/settings.local.json",
                ".cache/huggingface/hub/model.bin",
                ".env",
                ".env.local",
                "secrets/api_key.txt",
                "private.pem",
                "src/adaptive_lora/__pycache__/__init__.pyc",
            ):
                with self.subTest(ignored=path):
                    result = subprocess.run([git, "-C", directory, "check-ignore", "-q", "--", path])
                    self.assertEqual(result.returncode, 0, path)
            for path in (
                "README.md",
                "pyproject.toml",
                "PAPER_SPEC.md",
                "reports/m0_environment.json",
                "src/adaptive_lora/__init__.py",
            ):
                with self.subTest(tracked=path):
                    result = subprocess.run([git, "-C", directory, "check-ignore", "-q", "--", path])
                    self.assertEqual(result.returncode, 1, path)


if __name__ == "__main__":
    unittest.main()
