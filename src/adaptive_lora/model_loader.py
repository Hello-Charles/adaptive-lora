"""Load the M1 base model and discover its layer modules at runtime."""

from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
EXPECTED_LAYERS = 24


@dataclass(frozen=True)
class LayerModules:
    index: int
    block_path: str
    attention_path: str
    mlp_path: str
    projections: dict[str, str]
    attention_module: object
    mlp_module: object


def load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID, *, revision=None, allow_download=False, cache_dir=None):
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
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [describe_output(item) for item in value]}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}
