"""训练前CLI的离线门禁测试：不启动CUDA、不读真实数据，不冒充远端数值验收。"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_m7_pretraining.py'
spec = importlib.util.spec_from_file_location('m7_pretraining_cli_fixture', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class PretrainingCliTests(unittest.TestCase):
    """确认CLI无法选择正式训练，错误与帮助在任何模型/CUDA导入前结束。"""

    def test_help_is_callable_without_gpu(self):
        result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--help'], capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b'{calibration,pilot}', result.stdout)
        self.assertNotIn(b'--epochs', result.stdout)

    def test_formal_training_mode_does_not_exist(self):
        result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--phase', 'train'], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'invalid choice', result.stderr)

    def test_fixed_configuration_fingerprint_matches(self):
        self.assertEqual(cli.file_sha(ROOT / 'configs/qwen2_5_1_5b_commonsense_r16.json'), cli.CONFIG_SHA)

    def test_timeout_is_a_hard_failure(self):
        with self.assertRaises(TimeoutError):
            cli.deadline(None, None)


if __name__ == '__main__':
    unittest.main()
