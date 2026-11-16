"""真实PEFT微型Qwen的CPU会话测试；验证控制流与精确恢复，不替代1.5B/CUDA验收。

使用小词表且EOS=31模拟pad==eos；数据完全合成，临时目录不含任何真实训练题目。
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
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptive_lora.calibration import parameter_fingerprint, calibrate_model
from adaptive_lora.m7_checkpoint import save_training_checkpoint
from adaptive_lora.m7_session import run_training_session, supervised_examples_sha256
from adaptive_lora.m7_training import resolve_amp
from adaptive_lora.trainer import attach_selective_lora, audit_lora, assert_base_unchanged, fingerprint_parameters


def seed(value):
    """三种默认RNG重置仅为夹具复现；会话恢复后不再次seed。"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def fixture():
    """每次同样seed创建全新小模型；真实PEFT适配器、AdamW与CPU禁用AMP路径。"""
    seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.1, tie_word_embeddings=False, eos_token_id=31, pad_token_id=31)
    config._attn_implementation = 'eager'
    setup = attach_selective_lora(Qwen2ForCausalLM(config), [0, 1], expected_layers=2, rank=2, alpha=4)
    optimizer = torch.optim.AdamW([p for p in setup.model.parameters() if p.requires_grad], lr=1e-3)
    autocast, scaler = resolve_amp(torch.device('cpu'), 'float16')
    return setup, optimizer, autocast, scaler


def examples(count=11):
    """11条/micro2/累积2：每epoch两组用8条，累积层丢2条、micro层丢1条。"""
    return [{'sample_id': 'fixture-%d' % i, 'input_ids': [1, 2 + i % 10, 31],
             'attention_mask': [1, 1, 1], 'labels': [-100, 2 + i % 10, 31],
             'prompt_token_count': 1, 'supervised_token_count': 2} for i in range(count)]


def identity(setup, samples):
    """固定伪来源哈希只用于格式测试，不声称是真实模型/数据来源。"""
    ids = [x['sample_id'] for x in samples]
    return {'model_revision': 'a' * 40, 'base_fingerprint': setup.base_fingerprint['sha256'],
            'protocol_sha256': 'b' * 64, 'sampling_sha256': 'c' * 64,
            'order_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            'epochs': 2, 'micro_batches_per_epoch': len(samples) // 2, 'accumulation': 2,
            'training_order': 'none', 'final_partial_group': 'drop', 'micro_batch_size': 2,
            'selected_layer_indices': list(setup.selected_layers),
            'supervised_examples_sha256': supervised_examples_sha256(samples)}


def assert_tree(case, actual, expected):
    """比较完整参数/optimizer而非只比近似loss。"""
    if isinstance(expected, torch.Tensor):
        case.assertTrue(torch.equal(actual, expected))
    elif isinstance(expected, dict):
        case.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_tree(case, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        case.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_tree(case, a, b)
    else:
        case.assertEqual(actual, expected)


class SessionTests(unittest.TestCase):
    """跨epoch的drop与fresh-base恢复；attention dropout让RNG漂移会改变更新。"""

    def setUp(self):
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.addCleanup(random.setstate, self.rng[0])
        self.addCleanup(np.random.set_state, self.rng[1])
        self.addCleanup(torch.set_rng_state, self.rng[2])
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.setup, self.optimizer, self.autocast, self.scaler = fixture()
        self.samples = examples()
        self.identity = identity(self.setup, self.samples)

    def run_session(self, name, **changes):
        """统一默认参数使每个负例只破坏一种前提。"""
        args = dict(setup=self.setup, examples=self.samples, optimizer=self.optimizer, scaler=self.scaler,
                    autocast=self.autocast, device='cpu', identity=self.identity, directory=self.root / name,
                    micro_batch_size=2, max_steps=4, eos_token_id=31)
        args.update(changes)
        return run_training_session(**args)

    def test_continuous_equals_resume_across_epoch_with_exact_visits(self):
        expected_visits = []
        result = self.run_session('continuous', on_micro_batch=lambda x: expected_visits.extend(x['sample_ids']))
        weights = copy.deepcopy(self.setup.model.state_dict())
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        draws = (random.random(), float(np.random.random()), torch.rand(4))
        self.setup, self.optimizer, self.autocast, self.scaler = fixture()
        visits = []
        first = self.run_session('first', max_steps=1, on_micro_batch=lambda x: visits.extend(x['sample_ids']))
        self.setup, self.optimizer, self.autocast, self.scaler = fixture()
        resumed = self.run_session('resumed', max_steps=9, resume_from=first['latest_checkpoint'],
                                   on_micro_batch=lambda x: visits.extend(x['sample_ids']))
        self.assertEqual(visits, expected_visits)
        self.assertEqual(visits, [x['sample_id'] for x in self.samples[:8]] * 2)
        self.assertEqual(resumed['position'], result['position'])
        self.assertEqual(resumed['position'], {'epoch': 3, 'next_micro_batch': 0, 'global_step': 4})
        self.assertEqual(resumed['sample_visits_this_session'], 12)
        self.assertEqual(result['sample_visits_this_session'], 16)
        assert_tree(self, self.setup.model.state_dict(), weights)
        assert_tree(self, self.optimizer.state_dict(), optimizer)
        assert_tree(self, (random.random(), float(np.random.random()), torch.rand(4)), draws)
        expected_log = [json.loads(x) for x in (self.root / 'continuous/steps.jsonl').read_text().splitlines()]
        actual_log = [json.loads(x) for x in (self.root / 'resumed/steps.jsonl').read_text().splitlines()]
        for a, b in zip(actual_log, expected_log[1:]):
            a.pop('optimizer_elapsed_seconds')
            b.pop('optimizer_elapsed_seconds')
            self.assertEqual(a, b)
        self.assertTrue(all(p.grad is None for p in self.setup.model.parameters()))
        assert_base_unchanged(self.setup)

    def test_two_drop_layers_conserve_identities(self):
        result = self.run_session('drop', max_steps=1)
        manifest = json.loads((self.root / 'drop/drop_manifest.private.json').read_text())
        self.assertEqual(manifest['dropped_sample_ids'], ['fixture-8', 'fixture-9', 'fixture-10'])
        self.assertEqual(manifest['samples_planned_used_per_epoch'], 8)
        self.assertEqual(result['planned_dropped_samples_per_epoch'], 3)
        self.assertEqual(result['optimizer_steps_this_session'], 1)

    def test_resume_completed_session_executes_zero_steps(self):
        done = self.run_session('done')
        self.setup, self.optimizer, self.autocast, self.scaler = fixture()
        result = self.run_session('empty', resume_from=done['latest_checkpoint'])
        self.assertEqual(result['optimizer_steps_this_session'], 0)
        self.assertTrue(result['all_epochs_complete'])

    def test_resume_legacy_epoch_end_cursor_normalizes(self):
        first = self.run_session('epoch', max_steps=2)
        named = [(n, p) for n, p in self.setup.model.named_parameters() if p.requires_grad]
        save_training_checkpoint(self.root / 'equivalent', named_trainable=named, optimizer=self.optimizer,
                                 scaler=self.scaler, identity=self.identity,
                                 position={'epoch': 1, 'next_micro_batch': 4, 'global_step': 2})
        self.setup, self.optimizer, self.autocast, self.scaler = fixture()
        result = self.run_session('next', max_steps=1, resume_from=self.root / 'equivalent')
        self.assertEqual(result['start_position'], first['position'])
        self.assertEqual(result['position']['global_step'], 3)

    def test_bad_identity_or_optimizer_order_fails_before_forward(self):
        variants = [dict(identity={**self.identity, 'model_revision': 'invalid'}),
                    dict(identity={**self.identity, 'micro_batch_size': 1}),
                    dict(identity={**self.identity, 'epochs': 0}),
                    dict(identity={**self.identity, 'selected_layer_indices': [1]})]
        with patch.object(self.setup.model, 'forward', side_effect=AssertionError('must not forward')):
            for index, changes in enumerate(variants):
                with self.assertRaises(ValueError):
                    self.run_session('bad-%d' % index, **changes)
            self.optimizer.param_groups[0]['params'].reverse()
            with self.assertRaisesRegex(ValueError, 'order'):
                self.run_session('reversed')

    def test_tensor_mutation_with_same_ids_rejected(self):
        self.samples[0]['input_ids'][1] = 9
        with self.assertRaisesRegex(ValueError, 'tensors'):
            self.run_session('mutated')

    def test_output_collision_or_unbounded_steps_rejected(self):
        (self.root / 'exists').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_session('exists')
        for value in [None, 0, -1, True]:
            with self.assertRaises(ValueError):
                self.run_session('invalid', max_steps=value)

    def test_forward_failure_clears_gradients_without_success_marker(self):
        with patch.object(self.setup.model, 'forward', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):
                self.run_session('failed')
        self.assertTrue(all(p.grad is None for p in self.setup.model.parameters()))
        self.assertTrue((self.root / 'failed/failure.json').is_file())
        self.assertFalse((self.root / 'failed/summary.json').exists())

    def test_explicit_cuda_flags_preserve_cpu_fingerprints_and_audit(self):
        self.assertEqual(audit_lora(self.setup), audit_lora(self.setup, allow_cuda=True))
        self.assertEqual(assert_base_unchanged(self.setup), assert_base_unchanged(self.setup, allow_cuda=True))
        self.assertEqual(parameter_fingerprint(self.setup.model), parameter_fingerprint(self.setup.model, allow_cuda=True))
        self.assertEqual(fingerprint_parameters(self.setup.base_parameters),
                         fingerprint_parameters(self.setup.base_parameters, allow_cuda=True))

    def test_calibration_cuda_unavailable_fails_without_initializing(self):
        model = self.setup.model.get_base_model()
        model.requires_grad_(False).eval()
        with patch('torch.cuda.is_available', return_value=False), patch('torch.cuda.current_device', side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                calibrate_model(model, [], device='cuda')
        with self.assertRaisesRegex(ValueError, 'only CPU'):
            calibrate_model(model, [], device='meta')


if __name__ == '__main__':
    unittest.main()
