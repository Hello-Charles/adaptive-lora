"""冻结基础模型上的独立子模块 zero-ablation；不负责预测 token、概率或层评分。

仅支持 eager、串行前向及经 discover_layer_modules 确认的层。调用者须先冻结参数、
将所有子模块设为 eval，并在 no_grad/inference_mode 中调用；不能与训练或共享模型的
并发请求混用。这里不是 path patching，也不修改参数、残差路径或整个 block 的输出。
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from weakref import WeakKeyDictionary

import torch

from adaptive_lora.model_loader import LayerModules


@dataclass
class AblationState:
    """单次作用域的观测计数，不保存激活；calls 只累计成功替换的目标模块调用。

    layer_index 为零基，module_path 相对于所传模型根节点；调用者不应改写这些字段。
    一个 scope 可有多次前向；正常退出却从未命中目标时会报错，防止假消融静默成功。
    """

    layer_index: int
    component: str
    module_path: str
    calls: int = 0


# 注册表不往模型对象写私有属性，也不持有模型强引用。锁只防止两个 scope 同时登记，
# 不隔离前向执行；外部代码在 scope 内调用同一模型也会受到 hook 影响，须由调用者串行化。
_ACTIVE = WeakKeyDictionary()
_ACTIVE_LOCK = Lock()


def _zero_output(output, component):
    """只替换 [batch, sequence, hidden] 主输出，保持 dtype/device/shape 及辅助项身份。

    M1 的 Qwen Attention 是 (Tensor, None)，MLP 是 Tensor。Attention 也支持裸 Tensor
    和普通 tuple/list 的首项 Tensor；未核验的 dict/namedtuple 等结构明确拒绝。
    zeros_like 新建张量，绝不原地清零：输出可能与输入/残差共享存储，原地修改会扩大干预。
    """
    if component == "mlp":
        if not isinstance(output, torch.Tensor):
            raise TypeError("Unsupported MLP output: expected a Tensor")
        main = output
    elif isinstance(output, torch.Tensor):
        main = output
    elif type(output) in (tuple, list) and output and isinstance(output[0], torch.Tensor):
        main = output[0]
    else:
        raise TypeError("Unsupported attention output: expected Tensor or plain tuple/list with Tensor first")
    if main.ndim != 3:
        raise ValueError("Expected submodule output shape [batch, sequence, hidden]")
    zero = torch.zeros_like(main)
    if isinstance(output, torch.Tensor):
        return zero
    # 浅重建容器只换第 0 项；None、注意力权重或其他辅助对象不复制、不清零、不重新排序。
    return type(output)([zero, *output[1:]])


def _target(model, layer, component):
    """确认发现器给出的模块引用仍属于该根模型，避免陈旧/跨模型引用挂错 hook。

    仅接收 LayerModules；不接受 block 或 Q/K/V 投影作为直接目标，路径必须是 block
    的直接子模块。发现器的输出是调用前提，不把用户任意伪造的路径当作结构实测。
    """
    if component not in ("attention", "mlp"):
        raise ValueError("component must be 'attention' or 'mlp', not a whole block")
    if not isinstance(layer, LayerModules):
        raise ValueError("layer must be a discovered LayerModules record")
    path, module = (
        (layer.attention_path, layer.attention_module) if component == "attention"
        else (layer.mlp_path, layer.mlp_module)
    )
    try:
        actual = model.get_submodule(path)
    except AttributeError as exc:
        raise ValueError("Layer path does not belong to this model") from exc
    if actual is not module or path.rpartition(".")[0] != layer.block_path:
        raise ValueError("Layer module reference/path does not match this model's block child")
    return path, module


@contextmanager
def zero_ablation(model, layer, component):
    """在 with 作用域内独立消融一个层的 Attention 或 MLP 输出，返回 AblationState。

    model 须为完整、未共享包装的 eager 根模型；layer 由该模型的发现器获得。进入前
    要求参数全部 requires_grad=False、全部子模块 eval、梯度记录已关闭；本接口不代替
    调用者修改这些状态。调用者还应使用新输入、use_cache=False 且不传历史 KV cache。

    同一根模型的任何第二个 scope（即使换层/组件）都被拒绝，避免联合干预。已有观察
    hooks 不会被清空，只移除本 scope 创建的 handle；正常退出、调用方异常、前向异常
    或输出结构校验异常均经过 finally 清理。正常退出零次命中会在清理前报错；原异常
    不会被“未命中”覆盖。锁不使共享前向线程安全，也不支持编译/分布式或别名根模型。
    """
    path, module = _target(model, layer, component)
    if torch.is_grad_enabled():
        raise RuntimeError("zero_ablation requires torch.no_grad() or torch.inference_mode()")
    if any(child.training for child in model.modules()):
        raise ValueError("All model modules must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All base model parameters must be frozen before zero_ablation")
    state = AblationState(layer.index, component, path)
    with _ACTIVE_LOCK:
        if model in _ACTIVE:
            raise RuntimeError("An ablation is already active on this model; independent runs cannot be nested")
        _ACTIVE[model] = state

    def replace_output(_module, _inputs, output):
        """运行时再次检查梯度开关；返回替代输出，不能像观察 hook 一样返回 None。"""
        if torch.is_grad_enabled():
            raise RuntimeError("Keep torch.no_grad() or torch.inference_mode() active during the forward")
        replacement = _zero_output(output, component)
        state.calls += 1
        return replacement

    handle = None
    try:
        # 注册本身也可能抛错；先占用的登记项必须释放，不能让后续合法 scope 永久被拒绝。
        handle = module.register_forward_hook(replace_output)
        yield state
        if state.calls == 0:
            raise RuntimeError(f"Ablation target was not executed: {path}")
    finally:
        try:
            if handle is not None:
                handle.remove()
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.pop(model, None)
