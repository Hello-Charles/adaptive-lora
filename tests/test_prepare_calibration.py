"""M2 CLI 的离线门禁测试；Hub 和 parquet reader 均被替换，不产生实际下载。

来源哈希测试使用普通字节夹具，不能替代真实 parquet 的读取与行顺序验收。
"""

from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_calibration", ROOT / "scripts" / "prepare_calibration.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class CalibrationCLITests(unittest.TestCase):
    """检查失败先于下载、旧文件不覆盖以及 tokenizer 始终离线的参数契约。"""

    def rejected(self, args, message):
        """捕获 argparse 错误，并要求在不加载数据/tokenizer 的情况下退出。"""
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["prepare_calibration.py", *args]), \
             patch.object(CLI, "load_rows") as rows, \
             patch.object(CLI, "load_tokenizer") as tokenizer, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                CLI.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            rows.assert_not_called()
            tokenizer.assert_not_called()

    def test_cli_requires_immutable_revisions_and_positive_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ["--dataset-cache-dir", directory, "--output", str(Path(directory) / "out.json")]
            for option, value in (("--dataset-revision", "main"), ("--model-revision", "main")):
                self.rejected([*common, option, value], "40-character")
            for option in ("--n", "--max-length"):
                self.rejected([*common, option, "0"], "positive")
            self.rejected([*common, "--dataset-sha256", "not-a-digest"], "64-character")

    def test_cli_rejects_existing_download_cache_and_missing_offline_cache(self):
        """新下载只能使用尚不存在的目录；离线模式不能悄悄补下载。"""
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "out.json")
            self.rejected(
                ["--allow-download", "--dataset-cache-dir", directory, "--output", output],
                "new, empty directory",
            )
            self.rejected(
                ["--dataset-cache-dir", str(Path(directory) / "missing"), "--output", output],
                "Offline cache does not exist",
            )

    def test_cli_never_overwrites_an_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            output.write_text("historical result", encoding="utf-8")
            self.rejected(["--dataset-cache-dir", directory, "--output", str(output)], "Output already exists")
            self.assertEqual(output.read_text(encoding="utf-8"), "historical result")

    def test_tokenizer_is_always_local_even_when_dataset_download_is_allowed(self):
        """数据下载许可不会传给 tokenizer；这里验证调用参数，不加载真实文件。"""
        transformers = ModuleType("transformers")
        loader = Mock(return_value="tokenizer")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=loader)
        with patch.dict(sys.modules, {"transformers": transformers}):
            self.assertEqual(CLI.load_tokenizer("model", "a" * 40, Path("cache")), "tokenizer")
        loader.assert_called_once_with(
            "model", revision="a" * 40, cache_dir=str(Path("cache")), use_fast=True,
            local_files_only=True, trust_remote_code=False, token=False,
        )

    def test_data_load_checks_file_hash_and_reads_only_local_parquet(self):
        """下载文件身份由固定 SHA 与内容哈希共同限定；parquet reader 不执行仓库代码。"""
        hub = ModuleType("huggingface_hub")
        datasets = ModuleType("datasets")
        parquet = ModuleType("pyarrow.parquet")
        table = object()
        parquet.read_table = Mock(return_value=table)
        rows = [{"instruction": "fixture", "input": "", "output": "answer"}]
        datasets.Dataset = Mock(return_value=rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            path.write_bytes(b"fixture, not a real parquet file")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hub.hf_hub_download = Mock(return_value=str(path))
            kwargs = dict(
                dataset_id="fixture/repo", revision="a" * 40, cache_dir=Path(directory),
                filename="data/train.parquet", expected_sha256=digest,
                allow_download=False, endpoint="https://example.invalid",
            )
            with patch.dict(sys.modules, {"huggingface_hub": hub, "datasets": datasets, "pyarrow.parquet": parquet}):
                loaded, source = CLI.load_rows(**kwargs)
                self.assertIs(loaded, rows)
                self.assertEqual(source["sha256"], digest)
                self.assertEqual(source["file"], "data/train.parquet")
                call = hub.hf_hub_download.call_args.kwargs
                self.assertTrue(call["local_files_only"])
                self.assertFalse(call["token"])
                self.assertEqual(call["revision"], "a" * 40)
                parquet.read_table.assert_called_once_with(str(path), memory_map=True)
                datasets.Dataset.assert_called_once_with(table)
                parquet.read_table.reset_mock()
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    CLI.load_rows(**{**kwargs, "expected_sha256": "0" * 64})
                parquet.read_table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
