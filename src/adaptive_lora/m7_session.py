"""M7有界训练会话：连接真实切批、累积组、私有余量清单与完整组断点。

调用者先审计模型/数据、固定RNG与设备；本模块不加载/下载模型，不隐式授权正式训练。
每次调用必须给出max_steps并使用新目录。只支持已确认none/drop，恢复后从下一批继续，
不重置seed。旧引擎每次执行一个完整累积组，返回后梯度已清空，此时才写checkpoint。
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from adaptive_lora.artifacts import write_json
from adaptive_lora.m7_batches import iter_epoch_micro_batches, plan_epoch_accumulation
from adaptive_lora.m7_checkpoint import (
    restore_training_checkpoint, save_training_checkpoint, validate_training_boundary,
)
from adaptive_lora.m7_tokenization import collate_supervised_examples, M7_EOS_TOKEN_ID
from adaptive_lora.m7_training import run_accumulation
from adaptive_lora.trainer import audit_lora, assert_base_unchanged, causal_loss


def _ids_hash(ids):
    """访问序列身份与装配层一致；指纹不能被称为匿名化保证。"""
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def supervised_examples_sha256(examples):
    """绑定有序实际IDs/labels/mask与监督边界，避免相同sample_id却换了训练张量的续跑。

    流式按行编码且加长度前缀；不含答案文本，但指纹仍不是匿名化或来源认证。
    """
    digest = hashlib.sha256()
    for example in examples:
        row = {key: example[key] for key in ('sample_id', 'input_ids', 'attention_mask', 'labels',
                                             'prompt_token_count', 'supervised_token_count')}
        row['truncated'] = example.get('truncated', False)
        raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def run_training_session(*, setup, examples, optimizer, scaler, autocast, device, identity,
                         directory, micro_batch_size, max_steps, resume_from=None,
                         checkpoint_every=1, eos_token_id=M7_EOS_TOKEN_ID, on_step=None,
                         on_micro_batch=None):
    """执行至多max_steps个完整组并写安全边界断点，返回游标与仅本次实际消费的样本统计。

    identity须包括模型/基础字节/协议/抽样/序列/方法/选层与epoch计划。调用者传入的examples
    已经准备完监督区间；本函数在任何更新前逐批校验，坏样本不被静默跳过。私有余量清单按
    累积组实际使用边界生成，不误用micro层空清单。可选回调只读，不得修改模型/游标。
    max_steps是本次调用的新增步数，不是最终global_step；resume时保持跨epoch计数。
    """
    for name, value in [('max_steps', max_steps), ('checkpoint_every', checkpoint_every),
                        ('micro_batch_size', micro_batch_size)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError('Session output directory already exists')
    if identity.get('training_order') != 'none' or identity.get('final_partial_group') != 'drop':
        raise ValueError('Session requires the confirmed none/drop policy')
    examples = list(examples)
    identifiers = [x['sample_id'] for x in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate sample identities')
    if identity['order_sha256'] != _ids_hash(identifiers):
        raise ValueError('Actual sample order differs from checkpoint identity')
    if identity.get('micro_batch_size') != micro_batch_size:
        raise ValueError('Micro-batch size differs from identity')
    if identity.get('supervised_examples_sha256') != supervised_examples_sha256(examples):
        raise ValueError('Actual supervised tensors differ from identity')
    if identity['base_fingerprint'] != setup.base_fingerprint['sha256']:
        raise ValueError('Actual base fingerprint differs from identity')
    if identity.get('selected_layer_indices') != list(setup.selected_layers):
        raise ValueError('Actual selected layers differ from identity')
    device = torch.device(device)
    cuda = device.type == 'cuda'
    if any(p.device != device for p in setup.model.parameters()):
        raise ValueError('Model parameters differ from session device')
    audit_lora(setup, allow_cuda=cuda)
    assert_base_unchanged(setup, allow_cuda=cuda)
    named = [(n, p) for n, p in setup.model.named_parameters() if p.requires_grad]
    if any(p.grad is not None for _, p in named):
        raise ValueError('Session requires cleared gradients')
    batches, plan = iter_epoch_micro_batches(examples, epoch=1, micro_batch_size=micro_batch_size)
    accumulation = identity['accumulation']
    epoch_plan = plan_epoch_accumulation(plan, accumulation=accumulation)
    if plan['micro_batches'] != identity['micro_batches_per_epoch'] or epoch_plan['optimizer_steps'] < 1:
        raise ValueError('Actual batch count differs from identity plan')
    # 预先验证所有批次，不把数据错误延迟到已更新了一半权重之后才发现。
    for batch in batches:
        collate_supervised_examples(batch, pad_token_id=eos_token_id, eos_token_id=eos_token_id)
    used_micro = epoch_plan['micro_batches_used']
    dropped = [x['sample_id'] for batch in batches[used_micro:] for x in batch] + plan['dropped_sample_ids']
    if len(identifiers) != epoch_plan['samples_used_by_accumulation'] + len(dropped):
        raise ValueError('Used/dropped sample accounting does not conserve total samples')
    position = {'epoch': 1, 'next_micro_batch': 0, 'global_step': 0}
    setup.model.train()
    if resume_from is not None:
        # 模型、optimizer、批次计划均已构建；此后恢复RNG，不能再seed或随机初始化模型。
        position = restore_training_checkpoint(resume_from, named_trainable=named, optimizer=optimizer,
                                               scaler=scaler, expected_identity=identity)
    elif optimizer.state:
        raise ValueError('A new session requires an empty optimizer or an explicit checkpoint')
    # checkpoint也接受上一epoch已用末尾的等价游标；统一到下一epoch再切批，防重复/漏训。
    if position['next_micro_batch'] == used_micro:
        position['epoch'] += 1
        position['next_micro_batch'] = 0
    validate_training_boundary(named_trainable=named, optimizer=optimizer, identity=identity, position=position)
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_json(directory / 'identity.json', identity)
    write_json(directory / 'drop_manifest.private.json', {'epochs': identity['epochs'],
               'same_order_every_epoch': True, 'dropped_sample_ids': dropped,
               'dropped_sample_ids_sha256': _ids_hash(dropped), 'samples_per_epoch': len(identifiers),
               'samples_planned_used_per_epoch': epoch_plan['samples_used_by_accumulation']})
    (directory / 'drop_manifest.private.json').chmod(0o600)
    start_position = dict(position)
    steps, visited, latest_checkpoint = [], [], None
    started = perf_counter()
    try:
        with (directory / 'steps.jsonl').open('x', encoding='utf-8') as log:
            while len(steps) < max_steps and position['epoch'] <= identity['epochs']:
                offset = position['next_micro_batch']
                group = batches[offset:offset + accumulation]
                if len(group) != accumulation or offset >= used_micro:
                    raise ValueError('Resume position is outside complete used accumulation groups')
                group_ids = [x['sample_id'] for batch in group for x in batch]

                def loss(batch):
                    """右pad只从labels/mask识别；causal_loss负责shift与FP32交叉熵。"""
                    merged = collate_supervised_examples(batch, pad_token_id=eos_token_id, eos_token_id=eos_token_id)
                    tensors = {key: torch.tensor(merged[key], dtype=torch.long, device=device)
                               for key in ('input_ids', 'attention_mask', 'labels')}
                    logits = setup.model(input_ids=tensors['input_ids'], attention_mask=tensors['attention_mask'],
                                         use_cache=False).logits
                    value = causal_loss(logits, tensors['labels'])
                    if on_micro_batch is not None:
                        on_micro_batch({'sample_ids': merged['sample_ids'], 'epoch': position['epoch'],
                                        'batch_length': merged['batch_length']})
                    return value

                if cuda:
                    torch.cuda.synchronize(device)
                before = perf_counter()
                result = run_accumulation(optimizer=optimizer, trainable=[p for _, p in named], micro_batches=group,
                                           forward_loss=loss, accumulation=accumulation, device=device,
                                           autocast=autocast, scaler=scaler, final_partial_group='error')
                if cuda:
                    torch.cuda.synchronize(device)
                epoch_before = position['epoch']
                position['global_step'] += 1
                position['next_micro_batch'] += accumulation
                if position['next_micro_batch'] == used_micro:
                    position['epoch'] += 1
                    position['next_micro_batch'] = 0
                record = {**result['steps'][0], 'step': position['global_step'], 'epoch': epoch_before,
                          'samples_in_group': len(group_ids), 'sample_ids_sha256': _ids_hash(group_ids),
                          'optimizer_elapsed_seconds': perf_counter() - before}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                steps.append(record)
                visited.extend(group_ids)
                if on_step is not None:
                    on_step(dict(record))
                # run_accumulation已返回并在finally清空梯度；不用其早于zero_grad的on_step保存。
                finished = position['epoch'] > identity['epochs'] or len(steps) == max_steps
                if position['global_step'] % checkpoint_every == 0 or finished:
                    latest_checkpoint = directory / ('checkpoint-step-%06d' % position['global_step'])
                    save_training_checkpoint(latest_checkpoint, named_trainable=named, optimizer=optimizer,
                                             scaler=scaler, identity=identity, position=dict(position))
    except BaseException as error:
        # 不把包含题目/答案的异常文本写进公共状态，不重试，不宣称失败前步骤可回滚。
        write_json(directory / 'failure.json', {'exception_type': type(error).__name__,
                   'completed_global_step': position['global_step'], 'session_complete': False})
        raise
    audit_lora(setup, allow_cuda=cuda)
    assert_base_unchanged(setup, allow_cuda=cuda)
    summary = {'status': 'BOUNDED_SESSION_COMPLETE', 'start_position': start_position,
               'position': position, 'optimizer_steps_this_session': len(steps),
               'sample_visits_this_session': len(visited), 'sample_visit_order_sha256': _ids_hash(visited),
               'planned_dropped_samples_per_epoch': len(dropped), 'elapsed_seconds': perf_counter() - started,
               'latest_checkpoint': str(latest_checkpoint) if latest_checkpoint else None,
               'all_epochs_complete': position['epoch'] > identity['epochs'], 'base_unchanged': True}
    write_json(directory / 'summary.json', summary)
    return summary
