"""M7正式实验的顺序执行层：数据准备、一次N250、三方法完整训练与统一评估。

没有单测/N1/pilot/数值诊断前置。prepare仅使用既有本地资源，不加载模型或使用CUDA；其他
阶段由CLI显式--execute启动。保留计算所需的输入身份、有限值、冻结/LoRA白名单与hook清理，
这些不是额外测试作业。所有阶段新建目录，失败不重试、不降配，原始样本与生成结果留远端。
"""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from adaptive_lora.artifacts import write_json
from adaptive_lora.m7_protocol import load_m7_config, protocol_sha256

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('full', 'random_k', 'adaptive_k')


def file_sha(path):
    """文件字节指纹用于运行身份，不是来源认证或匿名化；不修改原始资源。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    """有序列表保留顺序，字典排序；用于抽样、协议和恢复身份的一致编码。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def require(condition, message):
    """不自动修复不一致输入；错误文本不得包含题面或答案。"""
    if not condition:
        raise ValueError(message)


def load_context(profile_path):
    """读独立运行配置，不修改已冻结论文配置；本地资源缺失时直接报错，不下载/安装。"""
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    require(profile.get('schema_version') == 1, 'Unsupported execution profile')
    require(profile.get('attention_backend') in ('sdpa', 'eager'), 'Unsupported attention backend')
    require(type(profile.get('checkpoint_every')) is int and profile['checkpoint_every'] > 0,
            'checkpoint_every must be positive')
    protocol_path = ROOT / profile['protocol_config']
    config = load_m7_config(protocol_path)
    return {'profile': profile, 'profile_sha256': file_sha(profile_path), 'config': config,
            'config_sha256': file_sha(protocol_path), 'protocol_sha256': protocol_sha256(config),
            'work': Path(profile['work_dir']), 'snapshot': Path(profile['snapshot']),
            'data_root': Path(profile['data_root'])}


def _environment():
    """只记录实际解释器/依赖，不运行测试或安装器；缺包不谎报环境已就绪。"""
    return {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'numpy', 'safetensors')}}


def _run_identity(ctx):
    """记录源码与运行配置；数据不进入Git，历史报告提交不影响计算源码的内容指纹。"""
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain']).strip())
    sources = {path.name: file_sha(path) for path in sorted((ROOT / 'src/adaptive_lora').glob('*.py'))}
    return {'source_commit': commit, 'source_dirty': dirty, 'implementation_sha256': canonical_sha(sources),
            'config_sha256': ctx['config_sha256'], 'protocol_sha256': ctx['protocol_sha256'],
            'profile_sha256': ctx['profile_sha256'], 'attention_backend': ctx['profile']['attention_backend'],
            'environment': _environment()}


def _new_directory(path):
    """仅创建新目录，保留任何已有/失败结果；调用者不能用此函数覆盖续跑源。"""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Output directory exists; choose a new output path')
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    return path


def prepare_data(ctx, output=None):
    """重用固定审计清单，装配正式输入到私有文件；不做模型前向或单测。

    训练目标只取output，校准只取instruction/input；评估参考答案单独保存，不送入生成。
    固定原抽样，不重抽、不跳过坏行。路径和已下载缓存的生命周期由调用者负责。
    """
    from transformers import AutoTokenizer
    from adaptive_lora.m7_batches import prepare_supervised_examples
    from adaptive_lora.m7_dataset import read_verified_rows, selected_records
    from adaptive_lora.m7_protocol import build_commonsense_prompt, TASK_LABELS
    from adaptive_lora.m7_tokenization import build_calibration_input, build_evaluation_input
    profile, config = ctx['profile'], ctx['config']
    dataset = config['dataset']
    audit_path = ctx['data_root'] / 'audit/audit.json'
    require(file_sha(audit_path) == profile['audit_sha256'], 'Saved audit bytes changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    require(canonical_sha(audit['sampling']) == audit['sampling_sha256'] == profile['sampling_sha256'],
            'Saved sampling identity changed')
    require(ctx['snapshot'].name == config['model']['revision'], 'Snapshot revision differs from protocol')
    require((ctx['snapshot'] / config['model']['weights_file']).stat().st_size == config['model']['weights_bytes'],
            'Existing model file size differs from protocol')
    tokenizer = AutoTokenizer.from_pretrained(str(ctx['snapshot']), local_files_only=True, trust_remote_code=False)
    require(tokenizer.pad_token_id == tokenizer.eos_token_id == config['protocol']['eos_token_id'], 'EOS/pad mismatch')
    rows, provenance = read_verified_rows(ctx['data_root'] / 'data', dataset['train_file'])
    indices = audit['sampling']['train_indices']
    records = selected_records(rows, indices, provenance['path'], dataset['revision'])
    require(records == audit['train'] and len(records) == dataset['train_samples'], 'Training identities differ')
    chosen = [rows[index] for index in indices]
    training = prepare_supervised_examples(chosen, records, tokenizer)
    positions = audit['sampling']['calibration_training_positions']
    require([records[p] for p in positions] == audit['calibration'] and len(positions) == dataset['calibration_samples'],
            'Calibration identities differ')
    calibration = []
    for position in positions:
        row, record = chosen[position], records[position]
        prefix = build_commonsense_prompt(row['instruction'], row.get('input', ''))
        calibration.append({**build_calibration_input(tokenizer, prefix),
                            'sample_id': record['sample_id'], 'index': record['source_index']})
    evaluation, references = {}, {}
    for spec in dataset['evaluation_files']:
        task = spec['task']
        source, provenance = read_verified_rows(ctx['data_root'] / 'data', spec)
        indices = audit['sampling']['evaluation_indices'][task]
        records = selected_records(source, indices, provenance['path'], dataset['revision'])
        require(records == audit['evaluation'][task] and len(records) == spec['sample_count'], 'Evaluation identities differ')
        inputs, gold = [], []
        for index, record in zip(indices, records):
            row = source[index]
            prefix = build_commonsense_prompt(row['instruction'], row.get('input', ''))
            inputs.append({**build_evaluation_input(tokenizer, prefix), 'sample_id': record['sample_id']})
            require(row['answer'] in TASK_LABELS[task], 'Noncanonical evaluation reference')
            gold.append({'sample_id': record['sample_id'], 'answer': row['answer']})
        evaluation[task], references[task] = inputs, gold
    output = _new_directory(output or ctx['work'] / 'prepared')
    payloads = {'train.private.json': training, 'calibration.private.json': calibration,
                'evaluation_inputs.private.json': evaluation, 'evaluation_references.private.json': references}
    files = {}
    for name, value in payloads.items():
        path = output / name
        write_json(path, value)
        path.chmod(0o600)
        files[name] = {'sha256': file_sha(path), 'bytes': path.stat().st_size}
    manifest = {**_run_identity(ctx), 'status': 'DATA_PREPARED_NO_MODEL_EXECUTED', 'files': files,
                'model_revision': config['model']['revision'], 'sampling_sha256': profile['sampling_sha256'],
                'audit_file_sha256': profile['audit_sha256'], 'train_samples': len(training),
                'calibration_samples': len(calibration), 'evaluation_samples': {k: len(v) for k, v in evaluation.items()},
                'samples_planned_used_per_epoch': 9984, 'samples_planned_dropped_per_epoch': 16,
                'optimizer_steps_per_method': 468, 'tests_executed': False, 'formal_training_started': False}
    write_json(output / 'manifest.json', manifest)
    return manifest


def _prepared(ctx, directory=None):
    """使用已生成输入，不要求测试报告；内容哈希避免恢复时误换样本/labels。"""
    directory = Path(directory or ctx['work'] / 'prepared')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'DATA_PREPARED_NO_MODEL_EXECUTED', 'Data preparation incomplete')
    require(manifest['config_sha256'] == ctx['config_sha256'] and manifest['profile_sha256'] == ctx['profile_sha256'],
            'Prepared data configuration differs')
    return directory, manifest


def _artifact(prepared, manifest, name):
    """只解析已绑定文件；不从数据目录导入Python或反序列化pickle。"""
    path = prepared / name
    require(file_sha(path) == manifest['files'][name]['sha256'], 'Prepared input bytes changed')
    return json.loads(path.read_text(encoding='utf-8'))


def setup_gpu():
    """仅由显式正式执行分支调用；prepare不调用。不可用时失败，不回退CPU。"""
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Single available CUDA device required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seed():
    """首次优化前固定三类RNG；checkpoint恢复之后不再调用。"""
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _base(ctx):
    """固定本地safetensors/FP16/显式backend；不因失败切换精度或下载替代权重。

    SDPA是运行配置中的复现实现选择，未声称已验证修复旧eager错误；非有限值仍会终止运行。
    CPU加载后可注入FP32 LoRA，再整体迁至单CUDA。基础eval/freeze与训练autograd分别控制。
    """
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ctx['snapshot']), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
        attn_implementation=ctx['profile']['attention_backend'])
    require(model.config.num_hidden_layers == 28, 'Unexpected model layer count')
    require(all(p.dtype == torch.float16 for p in model.parameters()), 'Base weights must be FP16')
    model.config.use_cache = False
    model.requires_grad_(False).eval()
    return model


def calibrate(ctx, output=None, prepared_dir=None):
    """直接进行正式N250，不运行N1或pilot；核心校准器继续负责固定原token和hook异常清理。"""
    import torch
    from adaptive_lora.calibration import calibrate_model
    from adaptive_lora.model_loader import discover_layer_modules
    from adaptive_lora.selection import select_top_k
    prepared, manifest = _prepared(ctx, prepared_dir)
    samples = _artifact(prepared, manifest, 'calibration.private.json')
    require(len(samples) == 250, 'Expected the fixed 250 calibration samples')
    output = _new_directory(output or ctx['work'] / 'calibration')
    _seed()
    model = _base(ctx).to('cuda:0')
    layers = discover_layer_modules(model, expected_layers=28)
    started = time.perf_counter()
    with (output / 'probabilities.private.jsonl').open('x', encoding='utf-8') as handle:
        def progress(record):
            """轨迹仅留私有目录，stdout只输出完成数量，不输出输入或样本身份。"""
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            progress.count += 1
            if progress.count % 10 == 0:
                print(json.dumps({'phase': 'calibration', 'samples_complete': progress.count}), flush=True)
        progress.count = 0
        result = calibrate_model(model, samples, layers=layers, on_sample=progress, device='cuda:0')
    write_json(output / 'calibration.private.json', result)
    scores = [row['importance'] for row in result['layer_importance']]
    selection = {**_run_identity(ctx), 'n': result['n'], 'layer_count': result['layer_count'],
                 'selected_ranked': select_top_k(scores, 8), 'layer_importance': result['layer_importance'],
                 'model_revision': ctx['config']['model']['revision'],
                 'base_fingerprint': result['parameter_fingerprint_before']['sha256'],
                 'prepared_manifest_sha256': file_sha(prepared / 'manifest.json'),
                 'calibration_file_sha256': file_sha(output / 'calibration.private.json')}
    write_json(output / 'selection.json', selection)
    summary = {'status': 'FORMAL_CALIBRATION_COMPLETE', 'selected_ranked': selection['selected_ranked'],
               'forward_evaluations': result['forward_evaluations'], 'elapsed_seconds': time.perf_counter() - started,
               'selection_sha256': file_sha(output / 'selection.json'), 'formal_training_started': False}
    write_json(output / 'summary.json', summary)
    del model, layers
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _selection(ctx, prepared, path=None):
    """Adaptive层来自本次真实N250，不从旧smoke、默认层或测试输出补造。"""
    path = Path(path or ctx['work'] / 'calibration/selection.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    require(value['config_sha256'] == ctx['config_sha256'] and value['profile_sha256'] == ctx['profile_sha256'],
            'Selection configuration differs')
    require(value['prepared_manifest_sha256'] == file_sha(prepared / 'manifest.json'), 'Selection data identity differs')
    selected = value['selected_ranked']
    require(value['n'] == 250 and value['layer_count'] == 28 and len(selected) == len(set(selected)) == 8
            and all(type(i) is int and 0 <= i < 28 for i in selected), 'Invalid N250 Top8 selection')
    require(file_sha(path.parent / 'calibration.private.json') == value['calibration_file_sha256'],
            'Calibration output bytes changed')
    return value, file_sha(path)


def _training_objects(ctx, method, selection):
    """每方法全新基础模型与标准Q/K/V LoRA；选层RNG独立，不进入基础参数优化。"""
    import torch
    from adaptive_lora.trainer import attach_selective_lora, audit_lora
    from adaptive_lora.m7_training import resolve_amp
    require(method in METHODS, 'Unknown experiment method')
    config = ctx['config']['training']
    selected = list(range(28)) if method == 'full' else (
        random.Random(config['random_layer_seed']).sample(range(28), 8) if method == 'random_k' else selection['selected_ranked'])
    _seed()
    setup = attach_selective_lora(_base(ctx), selected, expected_layers=28, rank=config['rank'], alpha=config['alpha'],
                                  revision=ctx['config']['model']['revision'])
    require(setup.base_fingerprint['sha256'] == selection['base_fingerprint'], 'Base differs from calibration')
    setup.model.to('cuda:0')
    audit = audit_lora(setup, allow_cuda=True)
    require(audit['adapter_dtypes'] == ['torch.float32'], 'Adapter master weights must be FP32')
    optimizer = torch.optim.AdamW([p for p in setup.model.parameters() if p.requires_grad],
        lr=config['learning_rate'], weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['eps'],
        foreach=False, fused=False)
    autocast, scaler = resolve_amp(torch.device('cuda:0'), config['amp_dtype'])
    return setup, optimizer, autocast, scaler, audit


def _training_identity(ctx, method, setup, manifest, examples, selection_sha):
    """把数据、实际监督张量、backend和方法共同绑定到训练断点，避免换输入后误续跑。"""
    from adaptive_lora.m7_session import supervised_examples_sha256
    training = ctx['config']['training']
    return {'model_revision': ctx['config']['model']['revision'], 'base_fingerprint': setup.base_fingerprint['sha256'],
            'protocol_sha256': ctx['protocol_sha256'], 'sampling_sha256': manifest['sampling_sha256'],
            'order_sha256': canonical_sha([x['sample_id'] for x in examples]), 'epochs': training['epochs'],
            'micro_batches_per_epoch': len(examples) // training['micro_batch_size'],
            'micro_batch_size': training['micro_batch_size'], 'accumulation': training['gradient_accumulation'],
            'training_order': training['training_order'], 'final_partial_group': training['final_partial_group'],
            'method': method, 'selected_layer_indices': list(setup.selected_layers), 'rank': setup.rank, 'alpha': setup.alpha,
            'supervised_examples_sha256': supervised_examples_sha256(examples), 'selection_sha256': selection_sha,
            'profile_sha256': ctx['profile_sha256'], 'attention_backend': ctx['profile']['attention_backend'],
            'implementation_sha256': _run_identity(ctx)['implementation_sha256'],
            'precision': 'base_fp16_adapter_fp32_amp_fp16', 'train_seed': training['train_seed']}


def train(ctx, method, output=None, prepared_dir=None, selection_path=None, resume_from=None):
    """正式跑完整3 epochs（none/drop共468步）；按epoch间隔存档，恢复仅接受完整组断点。

    这是正式训练而非pilot。调用者必须显式启动并承担GPU成本；OOM/非有限值直接失败，
    不缩batch、不量化、不重抽。输出与resume源必须不同，原失败目录保留。
    """
    import torch
    from adaptive_lora.m7_session import run_training_session
    prepared, manifest = _prepared(ctx, prepared_dir)
    examples = _artifact(prepared, manifest, 'train.private.json')
    selection, selection_sha = _selection(ctx, prepared, selection_path)
    setup, optimizer, autocast, scaler, audit = _training_objects(ctx, method, selection)
    identity = _training_identity(ctx, method, setup, manifest, examples, selection_sha)
    output = Path(output or ctx['work'] / ('train-' + method))
    _seed()
    torch.cuda.reset_peak_memory_stats()
    def progress(record):
        """正式日志只发布step/loss等标量，完整访问身份清单留会话私有文件。"""
        print(json.dumps({'method': method, **record}, allow_nan=False), flush=True)
    summary = run_training_session(setup=setup, examples=examples, optimizer=optimizer, scaler=scaler,
        autocast=autocast, device='cuda:0', identity=identity, directory=output,
        micro_batch_size=ctx['config']['training']['micro_batch_size'], max_steps=468,
        checkpoint_every=ctx['profile']['checkpoint_every'], resume_from=resume_from, on_step=progress)
    require(summary['all_epochs_complete'] and summary['position']['global_step'] == 468, 'Training did not finish the plan')
    result = {**_run_identity(ctx), 'status': 'FORMAL_TRAINING_COMPLETE', 'method': method, 'identity': identity,
              'audit': audit, 'session': summary, 'checkpoint_directory': summary['latest_checkpoint'],
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'formal_training_started': True}
    write_json(output / 'result.json', result)
    return result


def summarize(ctx, output=None):
    """汇总三方法已完成结果，不运行模型；明确Full与K8不是同参数预算。

    续跑时session时间/显存只代表最后一次会话，不能冒充完整训练累计成本。
    """
    import csv
    rows = []
    for method in METHODS:
        metrics = json.loads((ctx['work'] / ('eval-' + method) / 'metrics.json').read_text(encoding='utf-8'))
        trained = json.loads((ctx['work'] / ('train-' + method) / 'result.json').read_text(encoding='utf-8'))
        require(metrics['status'] == 'FORMAL_EVALUATION_COMPLETE' and trained['status'] == 'FORMAL_TRAINING_COMPLETE',
                'Cannot summarize incomplete experiment')
        require(metrics['profile_sha256'] == trained['profile_sha256'] == ctx['profile_sha256'], 'Result profiles differ')
        row = {'method': method, 'macro_accuracy': metrics['macro_accuracy'],
               'trainable_parameters': trained['audit']['trainable_parameter_elements'],
               'optimizer_steps_total': trained['session']['position']['global_step'],
               'optimizer_steps_last_session': trained['session']['optimizer_steps_this_session'],
               'training_last_session_seconds': trained['session']['elapsed_seconds'],
               'peak_allocated_last_session_bytes': trained['peak_allocated_bytes'],
               **{'accuracy_' + task: value for task, value in metrics['task_accuracies'].items()}}
        rows.append(row)
    output = _new_directory(output or ctx['work'] / 'comparison')
    result = {**_run_identity(ctx), 'status': 'FORMAL_COMPARISON_COMPLETE', 'rows': rows,
              'same_parameter_budget': False, 'cost_scope': 'Last session only if training resumed'}
    write_json(output / 'summary.json', result)
    with (output / 'comparison.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _greedy(model, sample, limit, eos):
    """只接无padding的实际前缀；逐步追加自身预测，参考答案绝不传入模型。

    仅最后位置logits预测下一个token，logits_to_keep=1减少完整词表输出内存，不改变输入。
    不采样、不beam、不把temperature除以0；无KV cache，与校准前缀语义一致。
    """
    import torch
    require(sample['attention_mask'] == [1] * len(sample['input_ids']), 'Evaluation expects unpadded prefixes')
    ids = list(sample['input_ids'])
    generated = []
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(limit):
            inputs = torch.tensor([ids], dtype=torch.long, device='cuda:0')
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1).logits[0, -1]
            require(torch.isfinite(logits).all().item(), 'Nonfinite generation logits')
            token = int(logits.argmax().item())
            generated.append(token)
            ids.append(token)
            if token == eos:
                break
    return generated


def evaluate(ctx, method, output=None, prepared_dir=None, selection_path=None, training_dir=None):
    """从该方法完整训练断点恢复，全7500条greedy评估并按8任务等权宏平均。

    输入与参考文件分开：先生成全部预测，再按ID交给已有评分器。私有生成/评分明细远端保存，
    公共metrics仅任务计数/准确率。此函数不进行恢复对照测试，不更新权重。
    """
    import torch
    from transformers import AutoTokenizer
    from adaptive_lora.m7_checkpoint import restore_training_checkpoint
    from adaptive_lora.trainer import fingerprint_parameters, audit_lora, assert_base_unchanged
    from adaptive_lora.m7_protocol import score_commonsense, aggregate_commonsense
    prepared, manifest = _prepared(ctx, prepared_dir)
    examples = _artifact(prepared, manifest, 'train.private.json')
    inputs = _artifact(prepared, manifest, 'evaluation_inputs.private.json')
    selection, selection_sha = _selection(ctx, prepared, selection_path)
    trained = json.loads((Path(training_dir or ctx['work'] / ('train-' + method)) / 'result.json').read_text(encoding='utf-8'))
    require(trained['status'] == 'FORMAL_TRAINING_COMPLETE' and trained['method'] == method, 'Training result incomplete')
    setup, optimizer, autocast, scaler, _ = _training_objects(ctx, method, selection)
    identity = _training_identity(ctx, method, setup, manifest, examples, selection_sha)
    require(identity == trained['identity'], 'Training/evaluation experiment identity differs')
    named = [(n, p) for n, p in setup.model.named_parameters() if p.requires_grad]
    position = restore_training_checkpoint(trained['checkpoint_directory'], named_trainable=named, optimizer=optimizer,
                                           scaler=scaler, expected_identity=identity)
    require(position == {'epoch': 4, 'next_micro_batch': 0, 'global_step': 468}, 'Expected complete training checkpoint')
    setup.model.requires_grad_(False).eval()
    audit_lora(setup, trainable=False, allow_cuda=True)
    before = fingerprint_parameters(named, allow_cuda=True)
    output = _new_directory(output or ctx['work'] / ('eval-' + method))
    tokenizer = AutoTokenizer.from_pretrained(str(ctx['snapshot']), local_files_only=True, trust_remote_code=False)
    predictions = {}
    with (output / 'predictions.private.jsonl').open('x', encoding='utf-8') as handle:
        for task, samples in inputs.items():
            rows = []
            for sample in samples:
                generated = _greedy(setup.model, sample, ctx['config']['protocol']['max_new_tokens'], tokenizer.eos_token_id)
                text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                rows.append({'sample_id': sample['sample_id'], 'generated_text': text})
                handle.write(json.dumps({'task': task, **rows[-1], 'generated_token_ids': generated}, ensure_ascii=False) + '\n')
                handle.flush()
            predictions[task] = rows
            print(json.dumps({'phase': 'evaluation', 'method': method, 'task': task, 'samples': len(rows)}), flush=True)
    references = _artifact(prepared, manifest, 'evaluation_references.private.json')
    scores = [score_commonsense(task, predictions[task], references[task]) for task in inputs]
    write_json(output / 'scores.private.json', scores)
    audit_lora(setup, trainable=False, allow_cuda=True)
    assert_base_unchanged(setup, allow_cuda=True)
    require(fingerprint_parameters(named, allow_cuda=True) == before, 'Evaluation changed adapter weights')
    metrics = {**_run_identity(ctx), **aggregate_commonsense(scores), 'status': 'FORMAL_EVALUATION_COMPLETE',
               'method': method, 'selection_sha256': selection_sha, 'training_checkpoint': trained['checkpoint_directory'],
               'task_counts': {s['task']: {'correct': s['correct'], 'total': s['total']} for s in scores}}
    write_json(output / 'metrics.json', metrics)
    return metrics
