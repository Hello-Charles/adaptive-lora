"""M5 的本地 adapter-only safetensors 保存/恢复，不提供 optimizer/RNG 训练断点续跑。

所有 checkpoint 必须写入尚不存在的新目录。恢复需要调用者提供同一固定 revision 的全新
CPU 基础模型；不自动加载基础模型、不联网、不执行下载目录代码、不使用 pickle 权重。
内部哈希检测变化与一致性，不能认证外部来源；调用者仍须信任自己的来源和外层运行索引。
"""

import hashlib
import json
from pathlib import Path
import re

from peft import PeftModel, get_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file
import torch

from adaptive_lora.artifacts import write_json
from adaptive_lora.model_loader import EXPECTED_LAYERS
from adaptive_lora.trainer import (
    LoraSetup, _target_paths, assert_base_unchanged, audit_lora, fingerprint_parameters,
)


def file_sha256(path):
    """流式散列文件原始字节；只读取文件，不解析或执行模型/数据中的代码。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(model_id, revision):
    """要求显式模型身份与完整提交 SHA；格式正确不等于独立认证了模型来源。"""
    if not isinstance(model_id, str) or not model_id or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Checkpoint requires a model ID and immutable 40-character revision")


def _state_fingerprint(setup, state):
    """要求存储的 key/shape/dtype/值恰好对应所选 A/B，不允许 embedding、基础权重或额外 tensor。

    使用实际模块的参数对象身份建立白名单；PEFT 存储 key 去掉 .default. adapter 名。
    再与公开 get_peft_model_state_dict 的结果核对，避免误把参数名包含 lora 当成充分证据。
    """
    root = setup.model.get_base_model()
    adapter_ids = {
        id(weight)
        for path in setup.targets
        for weight in (root.get_submodule(path).lora_A["default"].weight,
                       root.get_submodule(path).lora_B["default"].weight)
    }
    expected = {
        name.replace(".default.", "."): parameter
        for name, parameter in setup.model.named_parameters() if id(parameter) in adapter_ids
    }
    exported = get_peft_model_state_dict(setup.model, save_embedding_layers=False)
    if set(state) != set(expected) or set(exported) != set(expected):
        raise ValueError("Checkpoint tensor keys differ from the exact A/B whitelist")
    for key, value in state.items():
        parameter = expected[key]
        if (value.shape != parameter.shape or value.dtype != parameter.dtype
                or not torch.isfinite(value).all().item() or not torch.equal(value, parameter.detach())
                or not torch.equal(exported[key], parameter.detach())):
            raise ValueError(f"Checkpoint tensor mismatch: {key}")
    return fingerprint_parameters(sorted(state.items()))


def save_adapter_checkpoint(setup, directory, *, model_id, revision):
    """保存可独立恢复推理的标准 LoRA safetensors，最后写 checkpoint.json 作为完成标记。

    写前要求基础字节未变、可训练白名单通过。显式 save_embedding_layers=False，禁止自动
    把基础 embedding 放入 adapter 文件。异常保留部分新目录，不删除/重试/覆盖，亦不宣称
    其已完成。只保存 adapter，不保存优化器、RNG、scheduler 或 tokenizer 全量副本。
    """
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Checkpoint directory already exists: {directory}")
    _identity(model_id, revision)
    audit_lora(setup)
    assert_base_unchanged(setup)
    config = setup.model.peft_config["default"]
    config.base_model_name_or_path, config.revision = model_id, revision
    directory.mkdir(parents=True, exist_ok=False)
    setup.model.save_pretrained(str(directory), safe_serialization=True, save_embedding_layers=False)
    files = {path.name for path in directory.iterdir()}
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required <= files or not files <= required | {"README.md"}:
        raise ValueError("Unexpected files in adapter-only checkpoint")
    state = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    manifest = {
        "schema_version": 1, "kind": "adapter_only_no_optimizer_rng_resume",
        "model_id": model_id, "revision": revision,
        "selected_layer_indices_ascending": list(setup.selected_layers), "target_modules": list(setup.targets),
        "rank": setup.rank, "alpha": setup.alpha, "base_fingerprint": setup.base_fingerprint,
        "adapter_fingerprint": _state_fingerprint(setup, state),
        "tensor_shapes": {key: list(value.shape) for key, value in sorted(state.items())},
        "files_sha256": {name: file_sha256(directory / name) for name in sorted(files)},
    }
    write_json(directory / "checkpoint.json", manifest)
    return manifest


def restore_adapter_checkpoint(base_model, directory, *, model_id, revision, expected_layers=EXPECTED_LAYERS):
    """在传入的全新基础模型上恢复本地 adapter，返回全部冻结、eval 的 LoraSetup。

    先核对身份、完成标记、文件哈希、精确 targets 和基础字节，再调用实际 PEFT 恢复。
    加载后再次核对所有 A/B 与存储值和基础指纹。与保存时同一 dtype/设备/依赖才能要求
    精确 logits；推理输出的逐元素比较由调用者执行，不能仅凭哈希就宣称前向一致。
    """
    directory = Path(directory)
    _identity(model_id, revision)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Expected a local non-symlink checkpoint directory")
    marker = directory / "checkpoint.json"
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["kind"] != "adapter_only_no_optimizer_rng_resume"
            or manifest["model_id"] != model_id or manifest["revision"] != revision):
        raise ValueError("Checkpoint identity or schema mismatch")
    files = manifest["files_sha256"]
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if (not required <= set(files) or not set(files) <= required | {"README.md"}
            or {path.name for path in directory.iterdir()} != set(files) | {"checkpoint.json"}):
        raise ValueError("Unexpected or missing adapter-only checkpoint files")
    for name, digest in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Checkpoint file hash mismatch: {name}")
    if getattr(base_model, "peft_config", None) or any(isinstance(module, LoraLayer) for module in base_model.modules()):
        raise ValueError("Restore requires a fresh base without adapters")
    parameters = tuple(base_model.named_parameters())
    if any(parameter.grad is not None for _, parameter in parameters):
        raise ValueError("Restore base must not have pre-existing gradients")
    before = fingerprint_parameters(parameters)
    if before != manifest["base_fingerprint"]:
        raise ValueError("Fresh base fingerprint differs from the checkpoint base")
    selected, targets = _target_paths(base_model, manifest["selected_layer_indices_ascending"], expected_layers)
    rank, alpha = manifest["rank"], manifest["alpha"]
    if type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1 or list(targets) != manifest["target_modules"]:
        raise ValueError("Invalid checkpoint targets, rank or alpha")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    expected = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha,
        "lora_dropout": 0.0, "bias": "none", "modules_to_save": None,
        "target_modules": "^(?:" + "|".join(re.escape(path) for path in targets) + ")$",
        "base_model_name_or_path": model_id, "revision": revision,
        "use_rslora": False, "use_dora": False, "inference_mode": True,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint PEFT config differs from standard selected Q/K/V LoRA")
    stored = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    if (fingerprint_parameters(sorted(stored.items())) != manifest["adapter_fingerprint"]
            or {key: list(value.shape) for key, value in sorted(stored.items())} != manifest["tensor_shapes"]):
        raise ValueError("Checkpoint adapter tensor fingerprint/shape mismatch")
    base_model.requires_grad_(False)
    model = PeftModel.from_pretrained(
        base_model, str(directory), is_trainable=False, autocast_adapter_dtype=True,
        local_files_only=True, use_safetensors=True,
    )
    model.requires_grad_(False)
    model.eval()
    setup = LoraSetup(model, selected, targets, rank, alpha, parameters, before)
    audit_lora(setup, trainable=False)
    if _state_fingerprint(setup, stored) != manifest["adapter_fingerprint"]:
        raise ValueError("Restored adapter fingerprint mismatch")
    assert_base_unchanged(setup)
    return setup
