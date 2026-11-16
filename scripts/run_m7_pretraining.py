"""正式训练前的离线、单CUDA、有界门禁；没有正式epochs训练模式。

调用者负责授权GPU成本并提供可信固定SHA仓库、已下载快照及历史审计目录。本入口不联网、
不安装、不修改配置/旧审计，不覆盖结果。calibration最多1800秒，pilot最多1800秒，各单次，
失败不自动重试或调整batch/精度。私有概率/样本清单只能留远端；summary仅计数/统计/哈希。
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SAMPLING_SHA = '76a61e01d6e0b2400b4870ed54b6d0a5afa12347389f14d19d58b05164288c3b'
AUDIT_SHA = '840fba0ca65d436587c6941b89e066689682c73bbcff8be6cd179c0eb69202f9'
CONFIG_SHA = 'e5e8dad366dc26198d41c9a0ffad3715a30a0c020a4146b02d3e77cb17e3c87f'
SNAPSHOT_HASHES = {
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': 'c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'config.json': '0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77',
}


def require(value, message):
    """输入门禁失败直接停止；消息不含题面或答案。"""
    if not value:
        raise ValueError(message)


def file_sha(path):
    """流式原始字节SHA256，不改变缓存，也不把哈希匹配当作独立来源认证。"""
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def git_state():
    """包括未跟踪文件的clean门禁；结果目录必须放在repo外。"""
    def call(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=15).strip()
    return {'commit': call('rev-parse', 'HEAD'), 'dirty': bool(call('status', '--porcelain'))}


def deadline(_signal, _frame):
    """单次时限不刷新；外层还应使用timeout覆盖卡在原生内核的情况。"""
    raise TimeoutError('Bounded pretraining gate exceeded 1800 seconds')


def seed_training():
    """对照方法注入前和首次优化前各固定42；恢复checkpoint后不得再调用。"""
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def load_inputs(args, config):
    """重放固定文本审计，不重抽或改旧哈希；只返回训练行和不含答案的校准输入。"""
    from adaptive_lora.m7_dataset import canonical_sha256, read_verified_rows, sample_indices, selected_records
    from adaptive_lora.m7_protocol import build_commonsense_prompt
    from adaptive_lora.m7_batches import prepare_supervised_examples
    from adaptive_lora.m7_tokenization import build_calibration_input
    from transformers import AutoTokenizer
    audit_path = args.data_root / 'audit/audit.json'
    require(file_sha(audit_path) == AUDIT_SHA, 'Historical audit bytes changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    sampling = audit['sampling']
    require(audit['status'] == 'TEXT_GATE_PASS_TOKENIZER_PENDING', 'Historical text gate missing')
    require(audit['sampling_sha256'] == canonical_sha256(sampling) == SAMPLING_SHA, 'Sampling identity mismatch')
    dataset = config['dataset']
    rows, provenance = read_verified_rows(args.data_root / 'data', dataset['train_file'])
    indices = sample_indices(len(rows), dataset['train_samples'], dataset['train_sampling_seed'])
    positions = sample_indices(len(indices), dataset['calibration_samples'], dataset['calibration_sampling_seed'])
    require(indices == sampling['train_indices'] and positions == sampling['calibration_training_positions'],
            'Replayed sampling order changed')
    records = selected_records(rows, indices, provenance['path'], dataset['revision'])
    require(records == audit['train'], 'Training record identities changed')
    require([indices[p] for p in positions] == sampling['calibration_source_indices'], 'Calibration indices changed')
    require([records[p] for p in positions] == audit['calibration'], 'Calibration record identities changed')
    for name, sha in SNAPSHOT_HASHES.items():
        require(file_sha(args.snapshot / name) == sha, 'Snapshot bytes changed: ' + name)
    require(args.snapshot.name == config['model']['revision'], 'Snapshot revision directory mismatch')
    weights = args.snapshot / config['model']['weights_file']
    require(weights.stat().st_size == config['model']['weights_bytes'] and
            file_sha(weights) == config['model']['weights_lfs_sha256'], 'Model weight bytes changed')
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True, trust_remote_code=False)
    require(tokenizer.pad_token_id == tokenizer.eos_token_id == 151643, 'EOS/pad protocol mismatch')
    chosen = [rows[index] for index in indices]
    calibration = []
    for position in positions:
        row, record = chosen[position], records[position]
        prefix = build_commonsense_prompt(row['instruction'], row.get('input', ''))
        require(hashlib.sha256(prefix.encode()).hexdigest() == record['prompt_sha256'], 'Calibration prompt mismatch')
        calibration.append({**build_calibration_input(tokenizer, prefix),
                            'sample_id': record['sample_id'], 'index': record['source_index']})
    examples = prepare_supervised_examples(chosen, records, tokenizer) if args.phase == 'pilot' else None
    return calibration, examples


def fresh_base(snapshot):
    """固定本地safetensors、FP16、eager；先CPU加载供LoRA精确白名单注入，不下载。"""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True, dtype=torch.float16,
                                                attn_implementation='eager')
    require(model.config.num_hidden_layers == 28 and model.config.hidden_size == 1536, 'Unexpected model structure')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float16 for p in model.parameters()), 'Unexpected base dtype/device')
    model.config.use_cache = False
    model.requires_grad_(False).eval()
    return model


def run_calibration(args, config, samples):
    """先N1门禁再原250条；额外恢复前向不进入公式，逐样本私有轨迹及时落盘。"""
    import torch
    from adaptive_lora.artifacts import write_json
    from adaptive_lora.calibration import calibrate_model
    from adaptive_lora.model_loader import discover_layer_modules
    from adaptive_lora.selection import select_top_k
    seed_training()
    model = fresh_base(args.snapshot).to('cuda:0')
    layers = discover_layer_modules(model, expected_layers=28)
    gate = calibrate_model(model, samples[:1], layers=layers, device='cuda:0')
    write_json(args.output / 'n1.private.json', gate)
    with (args.output / 'probabilities.private.jsonl').open('x', encoding='utf-8') as handle:
        def progress(record):
            """只公开完成数与耗时，不公开sample_id、题面或原token轨迹。"""
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            progress.count += 1
            if progress.count % 10 == 0:
                print(json.dumps({'phase': 'calibration', 'samples_complete': progress.count}), flush=True)
        progress.count = 0
        result = calibrate_model(model, samples, layers=layers, device='cuda:0', on_sample=progress)
    require(result['samples'][0] == gate['samples'][0], 'N1/N250 first sample differs')
    require(result['n'] == 250 and result['forward_evaluations'] == 14500, 'Calibration workload mismatch')
    write_json(args.output / 'calibration.private.json', result)
    scores = [row['importance'] for row in result['layer_importance']]
    selection = {'n': 250, 'layer_count': 28, 'selected_ranked': select_top_k(scores, 8),
                 'layer_importance': result['layer_importance'],
                 'base_fingerprint': result['parameter_fingerprint_before']['sha256'],
                 'calibration_file_sha256': file_sha(args.output / 'calibration.private.json')}
    write_json(args.output / 'selection.json', selection)
    del model, layers
    gc.collect()
    torch.cuda.empty_cache()
    return {'status': 'CALIBRATION_N250_CUDA_PASS', **selection, 'forward_evaluations': 14558,
            'n1_first_sample_equal': True, 'base_unchanged': True, 'hooks_restored': True,
            'calibration_samples_sha256': hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def stress_longest_batch(setup, optimizer, examples):
    """最大实际批宽的FP16 AMP前反向显存探针，不step、不保存为训练断点。

    只选原有none顺序中的最大批（不是把最长8行重排成新批）；一次micro反向保留的激活图
    不小于稳态累积的单micro图。另记录连续组峰值，二者均不是完整epoch吞吐基准。
    """
    import torch
    from adaptive_lora.m7_batches import iter_epoch_micro_batches
    from adaptive_lora.m7_tokenization import collate_supervised_examples
    from adaptive_lora.trainer import causal_loss, audit_lora, assert_base_unchanged
    batches, _ = iter_epoch_micro_batches(examples, epoch=1, micro_batch_size=8)
    batches = batches[:1248]
    offset, longest = max(enumerate(batches), key=lambda item: max(
        x['prompt_token_count'] + x['supervised_token_count'] for x in item[1]))
    batch = collate_supervised_examples(longest, pad_token_id=151643)
    tensors = {key: torch.tensor(batch[key], device='cuda:0', dtype=torch.long)
               for key in ('input_ids', 'labels', 'attention_mask')}
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = setup.model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'], use_cache=False).logits
            loss = causal_loss(logits, tensors['labels'])
        require(torch.isfinite(loss).item(), 'Longest-batch loss is nonfinite')
        scaler.scale(loss / 8).backward()
        scaler.unscale_(optimizer)
        parameters = [p for p in setup.model.parameters() if p.requires_grad]
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters),
                'Longest-batch gradient missing or nonfinite')
        torch.cuda.synchronize()
        result = {'original_micro_batch_offset': offset, 'batch_size': 8, 'batch_width': batch['batch_length'],
                  'loss': float(loss.item()), 'optimizer_steps_executed': 0,
                  'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()}
    finally:
        optimizer.zero_grad(set_to_none=True)
    audit_lora(setup, allow_cuda=True)
    assert_base_unchanged(setup, allow_cuda=True)
    return result


def run_pilot(args, config, examples):
    """每方法连续3完整组+全新基础恢复第2组后再1组；不是正式训练、不是效果评估。

    三方法都使用全10,000条计划，但此门禁仅消费前192条，不能据此宣称全序列显存已通过。
    safetensors逐张量精确比较包含adapter/AdamW/RNG；不只检查loss接近。
    """
    import torch
    from safetensors.torch import load_file
    from adaptive_lora.artifacts import write_json
    from adaptive_lora.m7_protocol import protocol_sha256
    from adaptive_lora.m7_session import run_training_session, supervised_examples_sha256
    from adaptive_lora.m7_training import resolve_amp
    from adaptive_lora.trainer import attach_selective_lora, audit_lora
    require(args.calibration is not None, 'pilot requires a completed calibration directory')
    cal_summary = json.loads((args.calibration / 'summary.json').read_text())
    require(cal_summary['status'] == 'CALIBRATION_N250_CUDA_PASS', 'Calibration gate incomplete')
    require(cal_summary['config_sha256'] == CONFIG_SHA and cal_summary['sampling_sha256'] == SAMPLING_SHA,
            'Calibration protocol identity mismatch')
    require(cal_summary['source_commit'] == args.repo_sha, 'Calibration source SHA differs')
    require(file_sha(args.calibration / 'selection.json') == cal_summary['selection_file_sha256'], 'Selection file changed')
    selection = json.loads((args.calibration / 'selection.json').read_text())
    require(file_sha(args.calibration / 'calibration.private.json') == selection['calibration_file_sha256'], 'Calibration trace changed')
    random_layers = random.Random(config['training']['random_layer_seed']).sample(range(28), 8)
    methods = {'full': list(range(28)), 'random_k': random_layers, 'adaptive_k': selection['selected_ranked']}
    results = {}
    for method, selected in methods.items():
        def construct():
            """每次重建base/adapter/optimizer，再由会话恢复RNG；不复用训练后的基础对象。"""
            seed_training()
            base = fresh_base(args.snapshot)
            setup = attach_selective_lora(base, selected, expected_layers=28, rank=16, alpha=32,
                                          revision=config['model']['revision'])
            require(setup.base_fingerprint['sha256'] == selection['base_fingerprint'], 'Base differs from calibration')
            setup.model.to('cuda:0')
            audit = audit_lora(setup, allow_cuda=True)
            require(audit['adapter_dtypes'] == ['torch.float32'], 'Adapter master weights must remain FP32')
            optimizer = torch.optim.AdamW([p for p in setup.model.parameters() if p.requires_grad], lr=1e-4,
                                          weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
            autocast, scaler = resolve_amp(torch.device('cuda:0'), 'float16')
            return setup, optimizer, autocast, scaler, audit
        setup, optimizer, autocast, scaler, audit = construct()
        identity = {'model_revision': config['model']['revision'], 'base_fingerprint': selection['base_fingerprint'],
                    'protocol_sha256': protocol_sha256(config), 'sampling_sha256': SAMPLING_SHA,
                    'order_sha256': hashlib.sha256(json.dumps([x['sample_id'] for x in examples], separators=(',', ':')).encode()).hexdigest(),
                    'epochs': 3, 'micro_batches_per_epoch': 1250, 'accumulation': 8, 'micro_batch_size': 8,
                    'training_order': 'none', 'final_partial_group': 'drop', 'method': method,
                    'selected_layer_indices': list(setup.selected_layers), 'rank': 16, 'alpha': 32,
                    'supervised_examples_sha256': supervised_examples_sha256(examples),
                    'precision': 'base_fp16_adapter_fp32_amp_fp16', 'train_seed': 42}
        kwargs = dict(examples=examples, device='cuda:0', identity=identity, micro_batch_size=8, max_steps=3)
        seed_training()
        torch.cuda.reset_peak_memory_stats()
        continuous = run_training_session(setup=setup, optimizer=optimizer, autocast=autocast, scaler=scaler,
                                          directory=args.output / (method + '-continuous'), **kwargs)
        peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()}
        stress = stress_longest_batch(setup, optimizer, examples)
        del setup, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        setup, optimizer, autocast, scaler, restored_audit = construct()
        resumed = run_training_session(setup=setup, optimizer=optimizer, autocast=autocast, scaler=scaler,
                                       directory=args.output / (method + '-resumed'), **{**kwargs, 'max_steps': 1},
                                       resume_from=args.output / (method + '-continuous/checkpoint-step-000002'))
        expected = load_file(str(Path(continuous['latest_checkpoint']) / 'state.safetensors'))
        actual = load_file(str(Path(resumed['latest_checkpoint']) / 'state.safetensors'))
        require(set(expected) == set(actual) and all(torch.equal(expected[k], actual[k]) for k in expected),
                'Fresh-base CUDA checkpoint continuation differs')
        left = [json.loads(x) for x in (args.output / (method + '-continuous/steps.jsonl')).read_text().splitlines()][-1]
        right = json.loads((args.output / (method + '-resumed/steps.jsonl')).read_text().strip())
        left.pop('optimizer_elapsed_seconds')
        right.pop('optimizer_elapsed_seconds')
        require(left == right, 'Restored next-group loss/gradient/scaler evidence differs')
        result = {'selected_draw_order': selected, 'audit': audit, 'continuous': continuous, 'resumed': resumed,
                  'cuda_peak_memory': peak, 'longest_batch_backward_probe': stress,
                  'fresh_base_resume_tensor_exact': True, 'next_step_log_exact': True,
                  'formal_training_started': False}
        write_json(args.output / (method + '-result.json'), result)
        results[method] = result
        print(json.dumps({'phase': 'pilot', 'method': method, 'status': 'PASS', 'peak_memory': peak}), flush=True)
        del setup, optimizer, scaler, expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    return {'status': 'THREE_METHOD_CUDA_AMP_PILOT_PASS', 'methods': results,
            'pilot_optimizer_steps_executed': 12, 'formal_training_started': False}


def main():
    """只允许两种训练前门禁；1800秒单次上限与新目录先于任何CUDA计算生效。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['calibration', 'pilot'], required=True)
    parser.add_argument('--repo-sha', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    require(os.name == 'posix', 'This bounded CUDA entrypoint requires Linux')
    require(git_state() == {'commit': args.repo_sha, 'dirty': False}, 'Fixed clean source commit required')
    require(not args.output.exists() and not args.output.is_symlink(), 'Output exists; refusing overwrite')
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}.items():
        os.environ[key] = value
    # -I下不依赖venv的旧editable映射，明确使用当前已核对SHA的src，不从数据目录导入。
    sys.path.insert(0, str(ROOT / 'src'))
    from adaptive_lora.artifacts import write_json
    from adaptive_lora.m7_protocol import load_m7_config, protocol_sha256
    try:
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one available CUDA device required')
        config_path = ROOT / 'configs/qwen2_5_1_5b_commonsense_r16.json'
        require(file_sha(config_path) == CONFIG_SHA, 'Protocol configuration bytes changed')
        config = load_m7_config(config_path)
        for package in ('torch', 'transformers', 'peft'):
            require(importlib.metadata.version(package) == config['environment_candidate'][package], 'Dependency changed: ' + package)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        samples, examples = load_inputs(args, config)
        result = run_calibration(args, config, samples) if args.phase == 'calibration' else run_pilot(args, config, examples)
        require(git_state() == {'commit': args.repo_sha, 'dirty': False}, 'Source state changed during gate')
        result.update({'source_commit': args.repo_sha, 'config_sha256': CONFIG_SHA,
                       'protocol_sha256': protocol_sha256(config), 'sampling_sha256': SAMPLING_SHA,
                       'audit_file_sha256': AUDIT_SHA, 'model_revision': config['model']['revision'],
                       'elapsed_seconds': time.monotonic() - started, 'deadline_seconds': 1800,
                       'gpu': torch.cuda.get_device_name(), 'completed_utc': datetime.now(timezone.utc).isoformat(),
                       'formal_training_started': False,
                       'environment': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'numpy', 'safetensors')}})
        if args.phase == 'calibration':
            result['selection_file_sha256'] = file_sha(args.output / 'selection.json')
        write_json(args.output / 'summary.json', result)
        signal.alarm(0)
        print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        return 0
    except BaseException as error:
        write_json(args.output / 'failure.json', {'exception_type': type(error).__name__,
                   'elapsed_seconds': time.monotonic() - started, 'formal_training_started': False})
        print('GATE_FAILED: ' + type(error).__name__, file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
