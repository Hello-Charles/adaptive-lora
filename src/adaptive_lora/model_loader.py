"""模型结构探查接口；不执行消融、校准评分或训练。

模块路径必须由实际模型树确认，不能用层数或命名约定代替前向输出验证。
重型依赖仅在加载函数中导入，使纯结构测试不必初始化 PyTorch/Transformers。
"""

from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
EXPECTED_LAYERS = 24


@dataclass(frozen=True)
class LayerModules:
    """层索引为零基，路径相对于模型根节点，模块引用用于注册临时 hook。

    frozen 仅阻止字段重绑；不会冻结引用的模型参数，也不会让 projections 字典只读。
    """

    index: int
    block_path: str
    attention_path: str
    mlp_path: str
    projections: dict[str, str]
    attention_module: object
    mlp_module: object


def load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, *, revision=None, allow_download=False, cache_dir=None):
    """返回匹配同一来源的 (model, tokenizer)，不隐式授权联网下载。

    可重复实验需由调用者传入不可变的提交 SHA；此接口不验证 revision 的格式，
    也不创建隔离目录，下载许可和缓存目录门禁由 CLI 负责。模型只接受 safetensors，
    不启用仓库自定义 Python 代码。设备和 dtype 仍由 Transformers 默认加载规则决定。

    eval() 只切换推理行为，不冻结 requires_grad，也不禁用自动求导；
    前向调用者仍需显式使用 no_grad/inference_mode，校准时还需满足权重冻结约束。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = {"local_files_only": not allow_download, "trust_remote_code": False}
    if revision is not None:
        source["revision"] = revision
    if cache_dir is not None:
        source["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **source)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True, **source)
    model.eval()
    return model, tokenizer


def discover_layer_modules(model, expected_layers=EXPECTED_LAYERS):
    """按零基层序返回 LayerModules；结构不符合约定或候选不唯一时拒绝猜测。

    仅支持一个连续编号的 ModuleList 层栈，以及每层独立的 Attention、MLP 和
    q_proj/k_proj/v_proj 命名；不是任意 Transformer 的通用适配器，也不支持融合 QKV。
    路径来自当前模块树，但输出是 Tensor 还是 tuple 仍需真实前向确认。
    """
    configured_layers = getattr(model.config, "num_hidden_layers", None)
    if configured_layers != expected_layers:
        raise ValueError(f"Expected {expected_layers} layers; model config reports {configured_layers}")

    stacks = []
    for path, module in model.named_modules():
        if type(module).__name__ != "ModuleList" or len(module) != expected_layers:
            continue
        children = list(module.named_children())
        if [name for name, _ in children] == [str(index) for index in range(expected_layers)]:
            stacks.append((path, children))
    if len(stacks) != 1:
        raise ValueError(f"Expected one {expected_layers}-layer ModuleList; found {[path for path, _ in stacks]}")

    stack_path, blocks = stacks[0]
    discovered = []
    for index, (_, block) in enumerate(blocks):
        children = list(block.named_children())
        # 归一化层可能含有 attention 字样（如 post_attention_layernorm），但不是干预目标。
        attention = [(name, module) for name, module in children if name.lower().endswith(("attn", "attention"))]
        mlps = [(name, module) for name, module in children if "mlp" in name.lower()]
        if len(attention) != 1 or len(mlps) != 1:
            raise ValueError(f"Layer {index} has unexpected submodules: {[name for name, _ in children]}")
        attention_name, attention_module = attention[0]
        mlp_name, mlp_module = mlps[0]
        block_path = f"{stack_path}.{index}" if stack_path else str(index)
        attention_path = f"{block_path}.{attention_name}"
        projections = {}
        attention_children = list(attention_module.named_modules())
        for projection in ("q", "k", "v"):
            matches = [
                f"{attention_path}.{name}"
                for name, _ in attention_children
                if name and name.rsplit(".", 1)[-1] == f"{projection}_proj"
            ]
            if len(matches) != 1:
                raise ValueError(f"Layer {index} has {len(matches)} {projection}_proj modules")
            projections[projection] = matches[0]
        discovered.append(
            LayerModules(
                index=index,
                block_path=block_path,
                attention_path=attention_path,
                mlp_path=f"{block_path}.{mlp_name}",
                projections=projections,
                attention_module=attention_module,
                mlp_module=mlp_module,
            )
        )
    return discovered


def prediction_position(input_ids, attention_mask=None):
    """返回 [1, sequence_length] 输入中最后有效 token 的零基索引。

    因果模型的 logits[0, i, :] 表示读完第 i 个输入 token 后对下一个 token 的预测，
    所以返回的不是新 token 的插入下标。mask 须为同形状的二值张量：有效区间连续，
    左/右 padding 不属于有效位置；未传 mask 时，调用者必须保证没有 padding。
    本接口仅支持单样本，不能直接用 sum(mask)-1 处理左 padding 或批量数据。
    """
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected one tokenized input with shape [1, sequence_length]")
    sequence_length = input_ids.shape[1]
    if sequence_length < 1:
        raise ValueError("Input must have at least one token")
    if attention_mask is None:
        return sequence_length - 1
    if attention_mask.shape != input_ids.shape:
        raise ValueError("Attention mask shape must match input IDs")
    active = [index for index, value in enumerate(attention_mask[0].tolist()) if value == 1]
    if not active or active != list(range(active[0], active[-1] + 1)):
        raise ValueError("Attention mask must contain one contiguous span of valid tokens")
    return active[-1]


def describe_output(value):
    """仅保留类型、形状和 dtype，不持有激活张量或复制张量值到报告中。

    tuple/list 的嵌套位置有意义：Attention 的主输出与附加返回项不能合并成一张量。
    第二项可能为 None 或注意力权重，不能仅凭函数名假定结构；其他对象只记录类型。
    """
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [describe_output(item) for item in value]}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}
