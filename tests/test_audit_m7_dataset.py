"""离线数据CLI的合成主流程及真实子进程门禁测试；不联网或运行真实数据审计。

小规格测试只替换配置读取以缩小夹具；正式固定配置仍由原load_m7_config严格验证。
真实-I -S子进程测试证明帮助/旧目录/缺源失败路径不依赖site-packages，不证明数据已下载。
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_m7_dataset import dataset_fixture

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/audit_m7_dataset.py"
SPEC=importlib.util.spec_from_file_location("owned_m7_data_cli",SCRIPT)
CLI=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CLI)
CONFIG=json.loads((ROOT/"configs/qwen2_5_1_5b_commonsense_r16.json").read_text(encoding="utf-8"))


class M7DatasetCliTests(unittest.TestCase):
    """结果目录独占，失败不造summary，重叠结果保留原样而不是静默减样本。"""
    def test_success_scope_keeps_tokenizer_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/"input";data.mkdir();config=copy.deepcopy(CONFIG);config["dataset"]=dataset_fixture(data)
            out=root/"output"
            with patch.object(CLI,"load_m7_config",return_value=config):
                self.assertEqual(CLI.main(["--input-root",str(data),"--output-dir",str(out)]),0)
            summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train_count"],5)
            self.assertFalse(summary["tokenizer_validated"])
            self.assertTrue((out/"audit.json").exists())
            self.assertFalse((out/"error.json").exists())

    def test_overlap_returns_two_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/"input";data.mkdir();config=copy.deepcopy(CONFIG);config["dataset"]=dataset_fixture(data,overlapping=True);out=root/"output"
            with patch.object(CLI,"load_m7_config",return_value=config):
                self.assertEqual(CLI.main(["--input-root",str(data),"--output-dir",str(out)]),2)
            self.assertEqual(json.loads((out/"summary.json").read_text())["status"],"BLOCKED_TEXT_OVERLAP")
            self.assertTrue((out/"audit.json").is_file())

    def test_missing_input_writes_error_not_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/"out"
            self.assertEqual(CLI.main(["--input-root",str(Path(directory)/"absent"),"--output-dir",str(out)]),2)
            self.assertTrue((out/"error.json").is_file())
            self.assertFalse((out/"summary.json").exists())

    def test_unconfirmed_decision_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            config=copy.deepcopy(CONFIG);config["decisions"][0]["status"]="pending";out=Path(directory)/"out"
            with patch.object(CLI,"load_m7_config",return_value=config):
                self.assertEqual(CLI.main(["--input-root",directory,"--output-dir",str(out)]),2)
            self.assertFalse(out.exists())

    def test_real_isolated_help_has_no_site_package_requirement(self):
        result=subprocess.run([sys.executable,"-I","-S","-B",str(SCRIPT),"--help"],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("--input-root",result.stdout)

    def test_real_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/"keep.txt";marker.write_text("keep")
            result=subprocess.run([sys.executable,"-I","-S","-B",str(SCRIPT),"--input-root",directory,"--output-dir",directory],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,2)
            self.assertEqual(marker.read_text(),"keep")
            self.assertFalse((Path(directory)/"summary.json").exists())


if __name__=="__main__":
    unittest.main()
