"""M5 adapter-only checkpoint 的本地真实 PEFT 测试，不下载或保存完整模型。

使用微型随机 Qwen2，把注入前 state_dict 复制给全新实例；不能以重新加载同一已训练对象
代替恢复验证。这里只验证 adapter 推理恢复，不宣称恢复 optimizer/RNG/训练步的断点续训。
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from adaptive_lora.checkpoint import file_sha256, restore_adapter_checkpoint, save_adapter_checkpoint
from adaptive_lora.trainer import attach_selective_lora, audit_lora, train_smoke
from test_lora_targets import tiny_model
from test_trainer import tiny_examples

MODEL_ID = "fixture/tiny-qwen2"
REVISION = "a" * 40


class AdapterCheckpointTests(unittest.TestCase):
    """每项只写临时新目录；篡改仅用于受控负例，不涉及已有实验 checkpoint。"""

    def trained(self):
        torch.manual_seed(42)
        base = tiny_model()
        state = copy.deepcopy(base.state_dict())
        setup = attach_selective_lora(base, [0, 2], rank=2, alpha=4, expected_layers=3)
        train_smoke(setup, tiny_examples())
        return setup, state

    def fresh(self, state):
        model = tiny_model()
        model.load_state_dict(state)
        return model

    def test_adapter_only_safetensors_restores_exact_weights_and_logits_on_fresh_base(self):
        setup, base_state = self.trained()
        ids = torch.tensor([[1, 4, 5, 6]])
        with torch.no_grad():
            expected = setup.model(input_ids=ids, use_cache=False).logits.clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            manifest = save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            keys = load_file(str(checkpoint / "adapter_model.safetensors"))
            self.assertEqual(len(keys), 12)
            self.assertTrue(all("lora_A.weight" in key or "lora_B.weight" in key for key in keys))
            self.assertFalse((checkpoint / "pytorch_model.bin").exists())
            self.assertFalse((checkpoint / "model.safetensors").exists())
            self.assertEqual(manifest["kind"], "adapter_only_no_optimizer_rng_resume")
            restored = restore_adapter_checkpoint(self.fresh(base_state), checkpoint, model_id=MODEL_ID,
                                                  revision=REVISION, expected_layers=3)
            self.assertEqual(audit_lora(restored, trainable=False)["trainable_parameter_elements"], 0)
            with torch.no_grad():
                actual = restored.model(input_ids=ids, use_cache=False).logits
            self.assertTrue(torch.equal(expected, actual))
            self.assertIsNot(setup.model.get_base_model(), restored.model.get_base_model())

    def test_existing_directory_or_checkpoint_is_never_reused(self):
        setup, _ = self.trained()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "keep.txt"
            sentinel.write_bytes(b"do not overwrite")
            with self.assertRaises(FileExistsError):
                save_adapter_checkpoint(setup, path, model_id=MODEL_ID, revision=REVISION)
            self.assertEqual(sentinel.read_bytes(), b"do not overwrite")
            checkpoint = path / "checkpoint"
            save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            before = {file.name: file.read_bytes() for file in checkpoint.iterdir()}
            with self.assertRaises(FileExistsError):
                save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            self.assertEqual(before, {file.name: file.read_bytes() for file in checkpoint.iterdir()})

    def test_wrong_revision_or_base_fingerprint_fails_before_peft_loading(self):
        setup, state = self.trained()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            with patch("adaptive_lora.checkpoint.PeftModel.from_pretrained") as loader:
                for model_id, revision in (("other/model", REVISION), (MODEL_ID, "b" * 40)):
                    with self.assertRaises(ValueError):
                        restore_adapter_checkpoint(self.fresh(state), checkpoint, model_id=model_id,
                                                   revision=revision, expected_layers=3)
                changed = self.fresh(state)
                with torch.no_grad():
                    next(changed.parameters()).reshape(-1)[0].add_(1)
                with self.assertRaises(ValueError):
                    restore_adapter_checkpoint(changed, checkpoint, model_id=MODEL_ID,
                                               revision=REVISION, expected_layers=3)
                loader.assert_not_called()

    def test_changed_weight_bytes_without_digest_are_rejected(self):
        setup, state = self.trained()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            with (checkpoint / "adapter_model.safetensors").open("ab") as target:
                target.write(b"invalid trailing bytes")
            with self.assertRaises(ValueError):
                restore_adapter_checkpoint(self.fresh(state), checkpoint, model_id=MODEL_ID,
                                           revision=REVISION, expected_layers=3)

    def test_rehashed_wrong_targets_are_rejected(self):
        setup, state = self.trained()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            config_path = checkpoint / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["target_modules"] = "q_proj"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            marker = checkpoint / "checkpoint.json"
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            manifest["files_sha256"]["adapter_config.json"] = file_sha256(config_path)
            marker.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                restore_adapter_checkpoint(self.fresh(state), checkpoint, model_id=MODEL_ID,
                                           revision=REVISION, expected_layers=3)

    def test_rehashed_wrong_tensors_or_extra_base_file_are_rejected(self):
        setup, state = self.trained()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            save_adapter_checkpoint(setup, checkpoint, model_id=MODEL_ID, revision=REVISION)
            path = checkpoint / "adapter_model.safetensors"
            weights = load_file(str(path))
            weights[next(iter(weights))].reshape(-1)[0] += 1
            save_file(weights, str(path))
            marker = checkpoint / "checkpoint.json"
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            manifest["files_sha256"][path.name] = file_sha256(path)
            marker.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                restore_adapter_checkpoint(self.fresh(state), checkpoint, model_id=MODEL_ID,
                                           revision=REVISION, expected_layers=3)
            (checkpoint / "model.safetensors").write_bytes(b"unexpected base file")
            with self.assertRaises(ValueError):
                restore_adapter_checkpoint(self.fresh(state), checkpoint, model_id=MODEL_ID,
                                           revision=REVISION, expected_layers=3)


if __name__ == "__main__":
    unittest.main()
