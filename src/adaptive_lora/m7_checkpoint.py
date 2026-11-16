"""M7完整累积组边界的训练状态断点：可训练权重、AdamW、GradScaler与全局RNG。

使用safetensors+严格JSON，不使用pickle、不存基础模型、不加载模型或发起训练。
调用者必须先审计LoRA白名单、基础模型字节和真实数据，并把相应身份纳入identity；
本模块只核对传入身份是否相同，不把调用者的声明当成来源认证。仅支持单设备、单参数组、
FP32可训练参数的标准AdamW，固定none/drop计划；不支持scheduler、DDP或组中途续跑。

保存必须在scaler.update与zero_grad(set_to_none=True)之后；现有run_accumulation.on_step
回调发生在清梯度之前，不能直接在其中保存。恢复应在模型/优化器/数据对象全部重建后调用，
恢复RNG后不得再重置seed。调用者还须保持相同的train/eval标志、autocast精度和后端确定性
开关；本模块不保存模型buffers、模块training标志或后端全局设置。仅覆盖Python、NumPy legacy、
Torch默认CPU/单CUDA设备RNG，不覆盖独立Generator、worker RNG或外部采样器。
真实CUDA路径需另行运行验收。
"""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import safetensors
from safetensors.torch import load_file, save_file
import torch


def _require(condition, message):
    """不自动修复非法状态，避免让损坏断点产生看似成功的续跑。"""
    if not condition:
        raise ValueError(message)


def _canonical(value):
    """JSON拒绝非有限浮点，身份只允许JSON字典而不携带任意可执行对象。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(path):
    """流式文件指纹仅检测一致性，不认证外部来源。"""
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _environment():
    """拒绝依赖版本漂移；相同版本也不承诺跨硬件/CUDA内核逐位确定性。"""
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__),
            'numpy': np.__version__, 'safetensors': safetensors.__version__}


def _identity(identity, position):
    """固定none/drop的完整组游标；global_step与epoch/cursor必须相互一致。

    微型CPU夹具可用较小批数；真实配置哈希和基础字节身份由上层先行实证，不能用格式代替审计。
    epoch从1开始；epoch结束可以表示为当前epoch的已用末尾，或下一epoch的cursor=0。
    """
    _require(isinstance(identity, dict), 'Expected experiment identity dictionary')
    _canonical(identity)
    for key, length in [('model_revision', 40), ('base_fingerprint', 64), ('protocol_sha256', 64),
                        ('sampling_sha256', 64), ('order_sha256', 64)]:
        _require(isinstance(identity.get(key), str) and
                 re.fullmatch(r'[0-9a-f]{%d}' % length, identity[key]) is not None, 'Invalid identity: ' + key)
    for key in ('epochs', 'micro_batches_per_epoch', 'accumulation'):
        _require(type(identity.get(key)) is int and identity[key] > 0, 'Invalid identity plan: ' + key)
    _require(identity.get('training_order') == 'none' and identity.get('final_partial_group') == 'drop',
             'Checkpoint supports the confirmed none/drop identity only')
    _require(isinstance(position, dict) and set(position) == {'epoch', 'next_micro_batch', 'global_step'},
             'Invalid checkpoint position keys')
    _require(all(type(x) is int for x in position.values()), 'Checkpoint position must use plain integers')
    epoch, cursor, step = (position[k] for k in ('epoch', 'next_micro_batch', 'global_step'))
    accumulation = identity['accumulation']
    groups = identity['micro_batches_per_epoch'] // accumulation
    _require(groups > 0, 'Plan has no complete accumulation group')
    _require(1 <= epoch <= identity['epochs'] + 1 and 0 <= cursor <= groups * accumulation
             and cursor % accumulation == 0, 'Position is not a complete accumulation boundary')
    _require(epoch <= identity['epochs'] or cursor == 0, 'Final epoch position must use zero cursor')
    _require(step == (epoch - 1) * groups + cursor // accumulation, 'Position global_step differs from epoch/cursor')


def _parameters(named_trainable, optimizer):
    """顺序必须与AdamW参数列表逐对象一致，不允许凭同shape配对或只比集合。

    本模块无法从参数名证明它们真的是Q/K/V LoRA，因此调用者必须传入已审计的完整白名单。
    未保存冻结基础权重；加载前必须从相同revision重建并验证基础字节。
    """
    named = list(named_trainable)
    _require(named and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in named), 'Empty or invalid whitelist')
    names, parameters = zip(*named)
    _require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names), 'Duplicate or invalid whitelist names')
    _require(all(isinstance(p, torch.nn.Parameter) and p.requires_grad and p.dtype == torch.float32 for p in parameters),
             'Whitelist requires trainable FP32 Parameters')
    _require(len({id(p) for p in parameters}) == len(parameters), 'Duplicate parameter in whitelist')
    _require(all(p.grad is None for p in parameters), 'Checkpoint boundary requires cleared gradients')
    device = parameters[0].device
    _require(device.type in ('cpu', 'cuda') and all(p.device == device for p in parameters), 'Checkpoint requires one device')
    _require(all(torch.isfinite(p).all().item() for p in parameters), 'Nonfinite trainable parameter')
    _require(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1, 'Expected standard single-group AdamW')
    group = optimizer.param_groups[0]
    _require([id(p) for p in group['params']] == [id(p) for p in parameters], 'Optimizer parameter order differs from whitelist')
    _require(not any(group.get(k, False) for k in ('amsgrad', 'capturable', 'differentiable', 'maximize', 'fused')),
             'Unsupported nonstandard AdamW mode')
    if device.type == 'cuda':
        _require(torch.cuda.is_initialized(), 'CUDA must already be initialized by the approved caller')
    return list(names), list(parameters), device


def _options(optimizer_state):
    """排除按位置编码的参数索引，仅比较真实优化器超参数，禁止load_state_dict悄悄改lr。"""
    return [{k: v for k, v in group.items() if k != 'params'} for group in optimizer_state['param_groups']]


def _check_optimizer(state, parameters, step):
    """在加载前验证AdamW槽位形状、dtype、步数与有限性，拒绝部分初始化或错误配对。"""
    _require(set(state) == {'state', 'param_groups'} and len(state['param_groups']) == 1, 'Invalid AdamW state structure')
    _require(state['param_groups'][0]['params'] == list(range(len(parameters))), 'Invalid AdamW parameter indices')
    expected = set(range(len(parameters))) if step else set()
    _require(set(state['state']) == expected, 'Missing or extra AdamW state entries')
    for index, parameter in enumerate(parameters):
        if not step:
            continue
        slots = state['state'][index]
        _require(set(slots) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Invalid AdamW slots')
        counter = slots['step']
        _require(isinstance(counter, torch.Tensor) and counter.numel() == 1 and
                 torch.isfinite(counter).all().item() and counter.item() == step, 'AdamW step differs from checkpoint position')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor = slots[name]
            _require(isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape and tensor.dtype == parameter.dtype,
                     'AdamW state shape/dtype mismatch')
            _require(torch.isfinite(tensor).all().item(), 'Nonfinite AdamW state')
        _require((slots['exp_avg_sq'] >= 0).all().item(), 'Negative AdamW second moment')


def _pack(value, tensors):
    """使用显式tag保存tuple和整数dict键；JSON直接转换会丢掉optimizer索引及RNG元组类型。"""
    if isinstance(value, torch.Tensor):
        key = 'state.' + str(len(tensors))
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {'tensor': key}
    if isinstance(value, dict):
        return {'dict': [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'tuple' if isinstance(value, tuple) else 'list': [_pack(x, tensors) for x in value]}
    _require(value is None or type(value) in (bool, int, float, str), 'Unsupported state value type')
    _require(type(value) is not float or math.isfinite(value), 'Nonfinite JSON state value')
    return {'scalar': value}


def _unpack(node, tensors, used):
    """仅解释有限tag，不反序列化Python对象；张量引用不重复且不得缺少。"""
    _require(isinstance(node, dict) and len(node) == 1, 'Invalid encoded state node')
    kind, value = next(iter(node.items()))
    if kind == 'tensor':
        _require(value in tensors and value not in used, 'Missing or duplicate state tensor')
        used.add(value)
        return tensors[value]
    if kind == 'scalar':
        _require(value is None or type(value) in (bool, int, float, str), 'Invalid scalar state')
        _require(type(value) is not float or math.isfinite(value), 'Nonfinite scalar state')
        return value
    if kind in ('list', 'tuple'):
        _require(isinstance(value, list), 'Invalid encoded sequence')
        result = [_unpack(x, tensors, used) for x in value]
        return tuple(result) if kind == 'tuple' else result
    _require(kind == 'dict' and isinstance(value, list), 'Unknown encoded state tag')
    result = {}
    for pair in value:
        _require(isinstance(pair, list) and len(pair) == 2, 'Invalid encoded dictionary')
        key = _unpack(pair[0], tensors, used)
        _require(type(key) in (str, int) and key not in result, 'Invalid or duplicate state key')
        result[key] = _unpack(pair[1], tensors, used)
    return result


def _rng_state(device):
    """CPU不访问CUDA RNG；单CUDA设备路径只采集该设备，不能作为多卡/DDP续跑实现。"""
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _validate_rng(state, device):
    """在局部生成器上试装RNG，失败不改变进程全局随机序列；返回NumPy可用的原生元组。"""
    _require(set(state) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}, 'Invalid RNG keys')
    random.Random().setstate(state['python'])
    numpy = state['numpy']
    _require(isinstance(numpy, tuple) and len(numpy) == 5 and numpy[0] == 'MT19937'
             and isinstance(numpy[1], list) and len(numpy[1]) == 624
             and all(type(x) is int and 0 <= x < 2**32 for x in numpy[1]), 'Invalid NumPy RNG')
    _require(type(numpy[2]) is int and 0 <= numpy[2] <= 624 and type(numpy[3]) is int
             and numpy[3] in (0, 1) and type(numpy[4]) in (int, float) and math.isfinite(numpy[4]),
             'Invalid NumPy RNG position or Gaussian cache')
    numpy = (numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:])
    np.random.RandomState().set_state(numpy)
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    if device.type == 'cuda':
        _require(isinstance(state['torch_cuda'], torch.Tensor), 'Missing CUDA RNG state')
        torch.Generator(device=device).set_state(state['torch_cuda'])
    else:
        _require(state['torch_cuda'] is None, 'CPU checkpoint unexpectedly has CUDA RNG state')
    return numpy


def _validate_scaler(state, enabled, device):
    """校验GradScaler显式标量并试装局部对象；已禁用的缩放器不得假装恢复FP16 AMP。"""
    if enabled:
        _require(set(state) == {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}, 'Invalid scaler keys')
        _require(all(type(state[k]) in (int, float) and math.isfinite(state[k]) for k in
                     ('scale', 'growth_factor', 'backoff_factor')), 'Invalid scaler values')
        _require(state['scale'] > 0 and state['growth_factor'] > 1 and 0 < state['backoff_factor'] < 1,
                 'Invalid scaler scale/factors')
        _require(type(state['growth_interval']) is int and state['growth_interval'] > 0 and
                 type(state['_growth_tracker']) is int and 0 <= state['_growth_tracker'] < state['growth_interval'],
                 'Invalid scaler growth tracker')
    else:
        _require(state == {}, 'Disabled scaler state must be empty')
    temporary = torch.amp.GradScaler(device.type, enabled=enabled)
    temporary.load_state_dict(state)


def save_training_checkpoint(directory, *, named_trainable, optimizer, scaler, identity, position):
    """在完整optimizer组边界保存新目录，checkpoint.json最后写入作为完成标记。

    不保存未完成累积的梯度；失败保留部分目录、不重试、不删除，也不伪造完成标记。
    全局RNG必须在恢复的最后一步安装；保存本身不得推进全局随机序列。
    """
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError('Checkpoint directory already exists: ' + str(directory))
    _identity(identity, position)
    names, parameters, device = _parameters(named_trainable, optimizer)
    optimizer_state = optimizer.state_dict()
    _check_optimizer(optimizer_state, parameters, position['global_step'])
    scaler_state = scaler.state_dict()
    _validate_scaler(scaler_state, scaler.is_enabled(), device)
    rng = _rng_state(device)
    _validate_rng(rng, device)
    tensors = {'parameter.' + str(i): p.detach().cpu().contiguous().clone() for i, p in enumerate(parameters)}
    state = _pack({'optimizer': optimizer_state, 'scaler': scaler_state, 'rng': rng}, tensors)
    manifest = {
        'schema_version': 1, 'kind': 'm7_training_state_no_base', 'identity': identity, 'position': position,
        'environment': _environment(), 'parameter_names': names,
        'device': str(device), 'scaler_enabled': scaler.is_enabled(), 'cuda_rng_saved': device.type == 'cuda',
        'state': state,
    }
    # 先完成全部内存校验再创建目录。磁盘写失败时目录保留，调用者不能原地覆盖重试。
    directory.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(directory / 'state.safetensors'))
    manifest['state_file_sha256'] = _sha(directory / 'state.safetensors')
    with (directory / 'checkpoint.json').open('x', encoding='utf-8') as handle:
        handle.write(_canonical(manifest) + '\n')
    return manifest


def validate_training_boundary(*, named_trainable, optimizer, identity, position):
    """在会话开始/恢复后只读校验完整边界，不写盘、不消费RNG、不初始化GradScaler。

    用于第一次更新前拒绝非法身份/优化器顺序；scaler的已初始化状态仍在实际保存时校验。
    """
    _identity(identity, position)
    _, parameters, _ = _parameters(named_trainable, optimizer)
    _check_optimizer(optimizer.state_dict(), parameters, position['global_step'])


def _pairs(pairs):
    """严格JSON拒绝重复键，避免完成标记存在多个互相覆盖的身份或路径字段。"""
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(_value):
    """拒绝Python JSON默认接受的NaN/Infinity扩展。"""
    raise ValueError('Nonfinite JSON constant')


def restore_training_checkpoint(directory, *, named_trainable, optimizer, scaler, expected_identity):
    """先完整校验再恢复权重/optimizer/scaler，最后安装全局RNG并返回下一批游标。

    调用者必须提供同设备的全新已审计参数与空AdamW，禁止加载进已有半训练状态。
    格式/身份失败不会修改参数、optimizer或RNG；设备复制/底层库失败不保证回滚，调用者须
    弃用该恢复对象。来源必须可信；文件内部哈希不认证恶意重写了JSON与张量文件的第三方。
    """
    directory = Path(directory)
    _require(directory.is_dir() and not directory.is_symlink(), 'Expected checkpoint directory')
    required = {'state.safetensors', 'checkpoint.json'}
    _require({x.name for x in directory.iterdir()} == required and
             all(not (directory / name).is_symlink() and (directory / name).is_file() for name in required),
             'Incomplete checkpoint or unexpected files')
    with (directory / 'checkpoint.json').open('rb') as handle:
        raw = handle.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, 'Checkpoint metadata exceeds 1 MiB')
    manifest = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    _require(set(manifest) == {'schema_version', 'kind', 'identity', 'position', 'environment', 'parameter_names',
                               'device', 'scaler_enabled', 'cuda_rng_saved', 'state', 'state_file_sha256'}, 'Invalid checkpoint keys')
    _require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and
             manifest['kind'] == 'm7_training_state_no_base', 'Invalid checkpoint schema')
    _require(_canonical(manifest['identity']) == _canonical(expected_identity), 'Checkpoint experiment identity mismatch')
    _identity(expected_identity, manifest['position'])
    names, parameters, device = _parameters(named_trainable, optimizer)
    _require(not optimizer.state, 'Restore requires a fresh empty optimizer')
    _require(names == manifest['parameter_names'], 'Checkpoint parameter names/order mismatch')
    _require(manifest['device'] == str(device), 'Checkpoint device mismatch; cross-device restore is not supported')
    _require(manifest['environment'] == _environment(), 'Checkpoint environment mismatch')
    _require(type(manifest['scaler_enabled']) is bool and manifest['scaler_enabled'] == scaler.is_enabled(), 'Scaler mode mismatch')
    _require(type(manifest['cuda_rng_saved']) is bool and manifest['cuda_rng_saved'] == (device.type == 'cuda'), 'CUDA RNG mode mismatch')
    state_path = directory / 'state.safetensors'
    _require(_sha(state_path) == manifest['state_file_sha256'], 'Checkpoint tensor file hash mismatch')
    tensors = load_file(str(state_path), device='cpu')
    used = set()
    for i, parameter in enumerate(parameters):
        key = 'parameter.' + str(i)
        _require(key in tensors and tensors[key].shape == parameter.shape and tensors[key].dtype == parameter.dtype,
                 'Checkpoint parameter shape/dtype mismatch')
        _require(torch.isfinite(tensors[key]).all().item(), 'Nonfinite checkpoint parameter')
        used.add(key)
    state = _unpack(manifest['state'], tensors, used)
    _require(set(used) == set(tensors) and set(state) == {'optimizer', 'scaler', 'rng'}, 'Extra or missing state tensors')
    _require(_options(state['optimizer']) == _options(optimizer.state_dict()), 'Optimizer hyperparameter mismatch')
    _check_optimizer(state['optimizer'], parameters, manifest['position']['global_step'])
    _validate_scaler(state['scaler'], scaler.is_enabled(), device)
    numpy = _validate_rng(state['rng'], device)
    # 所有格式/身份/形状/RNG校验完成后才修改活对象；RNG最后恢复，避免初始化过程推进序列。
    with torch.no_grad():
        for i, parameter in enumerate(parameters):
            parameter.copy_(tensors['parameter.' + str(i)])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(numpy)
    torch.set_rng_state(state['rng']['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['rng']['torch_cuda'], device)
    return dict(manifest['position'])
