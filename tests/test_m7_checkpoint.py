"""M7 训练断点的纯 CPU 测试：微型线性参数验证续跑机制，不冒充真实 Qwen/CUDA 验收。

所有写入都在临时新目录；破坏文件仅为受控负例。随机化 loss 同时消耗 Python/NumPy/Torch
全局 RNG，以区分只恢复权重、恢复 optimizer、真正恢复三类 RNG 的不同能力。
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from adaptive_lora.m7_checkpoint import restore_training_checkpoint, save_training_checkpoint
from adaptive_lora.m7_training import resolve_amp, run_accumulation


IDENTITY = {
    'model_revision': 'a' * 40, 'base_fingerprint': 'b' * 64,
    'protocol_sha256': 'c' * 64, 'sampling_sha256': 'd' * 64, 'order_sha256': 'e' * 64,
    'epochs': 2, 'micro_batches_per_epoch': 4, 'accumulation': 2,
    'training_order': 'none', 'final_partial_group': 'drop', 'method': 'tiny_cpu_fixture',
}
POSITION = {'epoch': 1, 'next_micro_batch': 2, 'global_step': 1}


def seed(value):
    """测试显式重置三类全局 RNG；不意味着真实 CUDA 数值具有相同确定性。"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def fixture(*, scaler_enabled=False):
    """新建参数与 AdamW；包含一个不参与优化的冻结参数，检测基础权重未入断点。"""
    model = torch.nn.Linear(3, 2)
    model.register_parameter('frozen', torch.nn.Parameter(torch.ones(7), requires_grad=False))
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-3, weight_decay=0.01)
    # CPU GradScaler 的已启用路径只测试状态存储，不能证明 CUDA FP16 的缩放数值。
    scaler = torch.amp.GradScaler('cpu', enabled=scaler_enabled, init_scale=128.0, growth_interval=1)
    return model, named, optimizer, scaler


def train_group(model, optimizer, scaler):
    """走真实累积引擎做一个完整组；每个 loss 消耗三类 RNG，不修改学习率或样本顺序。"""
    def forward(_micro):
        """随机输入及标量同时参与 loss，使随机序列漂移会改变后续更新。"""
        x = torch.rand(1, 3) + random.random() + float(np.random.random())
        return model(x).square().mean()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    autocast, _ = resolve_amp(torch.device('cpu'), 'float16')
    return run_accumulation(optimizer=optimizer, trainable=[p for _, p in named], micro_batches=[0, 1],
                            forward_loss=forward, accumulation=2, device=torch.device('cpu'),
                            autocast=autocast, scaler=scaler)


def assert_tree_equal(case, actual, expected):
    """递归逐张量比较参数/优化器快照，避免只看 loss 接近就宣称续跑一致。"""
    if isinstance(expected, torch.Tensor):
        case.assertEqual(actual.dtype, expected.dtype)
        case.assertTrue(torch.equal(actual, expected))
    elif isinstance(expected, dict):
        case.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_tree_equal(case, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_tree_equal(case, left, right)
    else:
        case.assertEqual(actual, expected)


class TrainingCheckpointTests(unittest.TestCase):
    """只验证局部机制；identity 的真实模型/数据来源仍需上层审计器核对。"""

    def setUp(self):
        self.saved_random = random.getstate()
        self.saved_numpy = np.random.get_state()
        self.saved_torch = torch.get_rng_state()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'checkpoint'
        seed(42)
        self.model, self.named, self.optimizer, self.scaler = fixture()
        train_group(self.model, self.optimizer, self.scaler)
        self.addCleanup(random.setstate, self.saved_random)
        self.addCleanup(np.random.set_state, self.saved_numpy)
        self.addCleanup(torch.set_rng_state, self.saved_torch)

    def save(self, **changes):
        """所有负例由同一已完成组、已清梯度的状态派生，防止夹具前提漂移。"""
        args = dict(named_trainable=self.named, optimizer=self.optimizer, scaler=self.scaler,
                    identity=IDENTITY, position=POSITION)
        args.update(changes)
        return save_training_checkpoint(self.path, **args)

    def restore(self, *, expected_identity=IDENTITY):
        model, named, optimizer, scaler = fixture()
        position = restore_training_checkpoint(self.path, named_trainable=named, optimizer=optimizer,
                                               scaler=scaler, expected_identity=expected_identity)
        return model, optimizer, scaler, position

    def test_continuous_equals_fresh_restore_and_resume(self):
        self.save()
        expected_step = train_group(self.model, self.optimizer, self.scaler)
        expected_weights = copy.deepcopy(self.model.state_dict())
        expected_optimizer = copy.deepcopy(self.optimizer.state_dict())
        expected_draws = (random.random(), float(np.random.random()), torch.rand(5))
        # 构造全新对象会消耗Torch RNG，必须先构造后恢复，而不是在恢复后重新seed。
        seed(999)
        model, optimizer, scaler, position = self.restore()
        self.assertEqual(position, POSITION)
        actual_step = train_group(model, optimizer, scaler)
        assert_tree_equal(self, model.state_dict(), expected_weights)
        assert_tree_equal(self, optimizer.state_dict(), expected_optimizer)
        self.assertEqual(actual_step['steps'], expected_step['steps'])
        actual_draws = (random.random(), float(np.random.random()), torch.rand(5))
        assert_tree_equal(self, actual_draws, expected_draws)

    def test_rng_restoration_preserves_python_gauss_cache_and_numpy_cache(self):
        random.gauss(0, 1)
        np.random.normal()
        self.save()
        expected = (random.gauss(0, 1), float(np.random.normal()), torch.rand(3))
        seed(913)
        self.restore()
        actual = (random.gauss(0, 1), float(np.random.normal()), torch.rand(3))
        assert_tree_equal(self, actual, expected)

    def test_enabled_cpu_scaler_round_trips(self):
        seed(7)
        model, named, optimizer, scaler = fixture(scaler_enabled=True)
        train_group(model, optimizer, scaler)
        self.save(named_trainable=named, optimizer=optimizer, scaler=scaler)
        expected_scaler = copy.deepcopy(scaler.state_dict())
        expected_step = train_group(model, optimizer, scaler)
        expected_weights = copy.deepcopy(model.state_dict())
        fresh, fresh_named, fresh_optimizer, fresh_scaler = fixture(scaler_enabled=True)
        restore_training_checkpoint(self.path, named_trainable=fresh_named, optimizer=fresh_optimizer,
                                    scaler=fresh_scaler, expected_identity=IDENTITY)
        self.assertEqual(fresh_scaler.state_dict(), expected_scaler)
        actual_step = train_group(fresh, fresh_optimizer, fresh_scaler)
        assert_tree_equal(self, fresh.state_dict(), expected_weights)
        self.assertEqual(actual_step['steps'], expected_step['steps'])

    def test_checkpoint_uses_safetensors_and_json_no_base_or_pickle(self):
        manifest = self.save()
        self.assertEqual({p.name for p in self.path.iterdir()}, {'state.safetensors', 'checkpoint.json'})
        tensors = load_file(str(self.path / 'state.safetensors'))
        self.assertEqual(sum(k.startswith('parameter.') for k in tensors), 2)
        self.assertTrue(all('frozen' not in k for k in tensors))
        self.assertEqual(manifest['kind'], 'm7_training_state_no_base')
        self.assertEqual(manifest['position'], POSITION)
        self.assertFalse(manifest['cuda_rng_saved'])

    def test_save_does_not_advance_rng(self):
        py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        self.save()
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(numpy[1], np.random.get_state()[1]))
        self.assertEqual(numpy[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_existing_directory_is_never_overwritten(self):
        self.path.mkdir()
        sentinel = self.path / 'keep'
        sentinel.write_text('existing checkpoint', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'existing checkpoint')

    def test_save_rejects_pending_gradients(self):
        self.named[0][1].grad = torch.ones_like(self.named[0][1])
        with self.assertRaisesRegex(ValueError, 'gradient|boundary'):
            self.save()
        self.assertFalse(self.path.exists())

    def test_save_rejects_middle_of_accumulation(self):
        for cursor in (1, 3):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                self.save(position={**POSITION, 'next_micro_batch': cursor})
        self.assertFalse(self.path.exists())

    def test_save_rejects_inconsistent_global_step(self):
        with self.assertRaises(ValueError):
            self.save(position={**POSITION, 'global_step': 7})

    def test_save_rejects_optimizer_parameter_reordering(self):
        # 同形状参数也不能交换：AdamW状态依赖位置，不是只凭张量尺寸识别。
        with self.assertRaisesRegex(ValueError, 'order|whitelist'):
            self.save(named_trainable=list(reversed(self.named)))

    def test_duplicate_or_frozen_whitelist_rejected(self):
        for named in (self.named + [self.named[0]], [('frozen', self.model.frozen)]):
            with self.subTest(names=[x[0] for x in named]), self.assertRaises(ValueError):
                self.save(named_trainable=named)

    def test_nonfinite_parameter_rejected_before_creating_directory(self):
        with torch.no_grad():
            self.named[0][1].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.save()
        self.assertFalse(self.path.exists())

    def test_nonfinite_optimizer_state_rejected(self):
        self.optimizer.state[self.named[0][1]]['exp_avg'].fill_(float('inf'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.save()

    def test_wrong_experiment_identity_rejected_before_mutation(self):
        self.save()
        model, named, optimizer, scaler = fixture()
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, 'identity'):
            restore_training_checkpoint(self.path, named_trainable=named, optimizer=optimizer, scaler=scaler,
                                        expected_identity={**IDENTITY, 'method': 'another_method'})
        assert_tree_equal(self, model.state_dict(), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(len(optimizer.state), 0)

    def test_optimizer_hyperparameter_mismatch_rejected(self):
        self.save()
        model, named, optimizer, scaler = fixture()
        optimizer.param_groups[0]['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'hyperparameter'):
            restore_training_checkpoint(self.path, named_trainable=named, optimizer=optimizer,
                                        scaler=scaler, expected_identity=IDENTITY)

    def test_restore_requires_fresh_optimizer(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'fresh|empty'):
            restore_training_checkpoint(self.path, named_trainable=self.named, optimizer=self.optimizer,
                                        scaler=self.scaler, expected_identity=IDENTITY)

    def test_tensor_file_corruption_rejected(self):
        self.save()
        path = self.path / 'state.safetensors'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.restore()

    def test_missing_completion_marker_rejected(self):
        self.save()
        (self.path / 'checkpoint.json').unlink()
        with self.assertRaises(ValueError):
            self.restore()

    def test_extra_file_rejected(self):
        self.save()
        (self.path / 'untracked.pkl').write_bytes(b'not loaded')
        with self.assertRaises(ValueError):
            self.restore()

    def test_write_failure_leaves_no_completion_marker(self):
        with patch('adaptive_lora.m7_checkpoint.save_file', side_effect=OSError('injected disk failure')):
            with self.assertRaises(OSError):
                self.save()
        self.assertTrue(self.path.exists())
        self.assertFalse((self.path / 'checkpoint.json').exists())

    def test_cpu_checkpoint_never_initializes_cuda(self):
        # 阻断触发设备初始化的接口，验证CPU状态保存不走CUDA RNG分支。
        with patch('torch.cuda.get_rng_state', side_effect=AssertionError('CUDA not allowed')):
            self.save()
            self.restore()

    def test_restore_rejects_rehashed_parameter_shape_mismatch(self):
        """主动更新文件哈希仍不能使错误shape通过，证明不只检查哈希。"""
        self.save()
        path = self.path / 'state.safetensors'
        tensors = load_file(str(path))
        tensors['parameter.0'] = tensors['parameter.0'].reshape(-1)
        save_file(tensors, str(path))
        marker = self.path / 'checkpoint.json'
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        manifest['state_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        marker.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.restore()

    def test_invalid_rng_rejected_without_modifying_live_objects(self):
        self.save()
        marker = self.path / 'checkpoint.json'
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        # 保存树的显式tag使本测试能只破坏Python RNG，不碰optimizer/参数文件。
        rng = next(value for key, value in manifest['state']['dict'] if key == {'scalar': 'rng'})
        python = next(value for key, value in rng['dict'] if key == {'scalar': 'python'})
        python['tuple'][0] = {'scalar': -1}
        marker.write_text(json.dumps(manifest), encoding='utf-8')
        model, named, optimizer, scaler = fixture()
        before = copy.deepcopy(model.state_dict())
        rng_before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            restore_training_checkpoint(self.path, named_trainable=named, optimizer=optimizer,
                                        scaler=scaler, expected_identity=IDENTITY)
        assert_tree_equal(self, model.state_dict(), before)
        self.assertEqual(len(optimizer.state), 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_scaler_mode_mismatch_rejected(self):
        self.save()
        model, named, optimizer, scaler = fixture(scaler_enabled=True)
        with self.assertRaisesRegex(ValueError, 'Scaler mode'):
            restore_training_checkpoint(self.path, named_trainable=named, optimizer=optimizer,
                                        scaler=scaler, expected_identity=IDENTITY)

    def test_restore_rejects_dependency_version_drift(self):
        self.save()
        marker = self.path / 'checkpoint.json'
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        manifest['environment']['torch'] = 'different-version'
        marker.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'environment'):
            self.restore()

    def test_initial_zero_step_checkpoint_can_resume(self):
        model, named, optimizer, scaler = fixture()
        position = {'epoch': 1, 'next_micro_batch': 0, 'global_step': 0}
        self.save(named_trainable=named, optimizer=optimizer, scaler=scaler, position=position)
        actual, restored_optimizer, _, cursor = self.restore()
        self.assertEqual(cursor, position)
        self.assertEqual(len(restored_optimizer.state), 0)
        assert_tree_equal(self, actual.state_dict(), model.state_dict())

    def test_final_epoch_position_round_trips(self):
        # 2 epochs各两组，共4步；最终位置只能是epoch=3,cursor=0。
        for _ in range(3):
            train_group(self.model, self.optimizer, self.scaler)
        position = {'epoch': 3, 'next_micro_batch': 0, 'global_step': 4}
        self.save(position=position)
        self.assertEqual(self.restore()[3], position)

    def test_real_tiny_qwen_peft_adapter_resumes_exactly_on_fresh_base(self):
        """真实PEFT A/B白名单的微型Qwen续跑；不加载正式权重、不代表1.5B/CUDA验收。"""
        from adaptive_lora.trainer import attach_selective_lora, assert_base_unchanged, audit_lora, causal_loss
        from test_lora_targets import tiny_model
        from test_trainer import tiny_examples
        base = tiny_model()
        base_state = copy.deepcopy(base.state_dict())
        setup = attach_selective_lora(base, [0, 2], rank=2, alpha=4, expected_layers=3)
        named = [(n, p) for n, p in setup.model.named_parameters() if p.requires_grad]
        self.assertEqual(len(named), 12)
        audit_lora(setup)
        identity = {**IDENTITY, 'base_fingerprint': setup.base_fingerprint['sha256'],
                    'method': 'fixture_adaptive2', 'selected_layer_indices': [0, 2]}
        optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-3)
        _, scaler = resolve_amp(torch.device('cpu'), 'float16')
        examples = tiny_examples(2)

        def group(target, optim, scale):
            """用实际Qwen logits与causal shift反向；少量随机项使RNG错位可观测。"""
            target.model.train()
            def loss(index):
                row = examples[index]
                ids = torch.tensor([row['input_ids']])
                labels = torch.tensor([row['labels']])
                logits = target.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                coefficient = random.random() + float(np.random.random()) + torch.rand(()).item()
                return causal_loss(logits, labels) + coefficient * logits.float().square().mean() * 0.01
            autocast, _ = resolve_amp(torch.device('cpu'), 'float16')
            return run_accumulation(optimizer=optim, trainable=[p for _, p in target.model.named_parameters() if p.requires_grad],
                                    micro_batches=[0, 1], forward_loss=loss, accumulation=2,
                                    device=torch.device('cpu'), autocast=autocast, scaler=scale)

        group(setup, optimizer, scaler)
        self.save(named_trainable=named, optimizer=optimizer, scaler=scaler, identity=identity)
        expected_step = group(setup, optimizer, scaler)
        expected_adapter = {n: p.detach().clone() for n, p in named}
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        # 真正重建基础模型与adapter，不复用已训练实例；恢复前单独核对基础字节身份。
        seed(999)
        fresh_base = tiny_model()
        fresh_base.load_state_dict(base_state)
        fresh = attach_selective_lora(fresh_base, [0, 2], rank=2, alpha=4, expected_layers=3)
        self.assertEqual(fresh.base_fingerprint, setup.base_fingerprint)
        fresh_named = [(n, p) for n, p in fresh.model.named_parameters() if p.requires_grad]
        fresh_optimizer = torch.optim.AdamW([p for _, p in fresh_named], lr=1e-3)
        _, fresh_scaler = resolve_amp(torch.device('cpu'), 'float16')
        restore_training_checkpoint(self.path, named_trainable=fresh_named, optimizer=fresh_optimizer,
                                    scaler=fresh_scaler, expected_identity=identity)
        actual_step = group(fresh, fresh_optimizer, fresh_scaler)
        assert_tree_equal(self, {n: p.detach() for n, p in fresh_named}, expected_adapter)
        assert_tree_equal(self, fresh_optimizer.state_dict(), expected_optimizer)
        self.assertEqual(actual_step['steps'], expected_step['steps'])
        assert_base_unchanged(setup)
        assert_base_unchanged(fresh)
        audit_lora(fresh)

    def test_epoch_boundary_position_round_trips(self):
        # 当前微型计划每epoch两组；首个epoch结束时global_step必须为2。
        position = {'epoch': 2, 'next_micro_batch': 0, 'global_step': 2}
        train_group(self.model, self.optimizer, self.scaler)
        self.save(position=position)
        self.assertEqual(self.restore()[3], position)


if __name__ == '__main__':
    unittest.main()
