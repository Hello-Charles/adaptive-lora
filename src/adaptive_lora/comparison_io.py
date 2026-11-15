"""M6 的本地来源完整性与运行身份记录，不重做校准、不加载模型或读取数据集答案。

选层绑定最终 M4 N30；校准输入绑定 M4 当时使用的 M2 清单，再由实际 tokenizer 复核。
哈希只检查字节/内容一致性，不是作者来源签名。调用者负责信任固定文件、缓存和许可证，
不得把完整 prompt/token 清单或 checkpoint 自动发布；本模块不联网、不覆盖文件。
"""

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

from adaptive_lora.checkpoint import file_sha256
from adaptive_lora.data import hash_text, manifest_content_hash
from adaptive_lora.model_loader import EXPECTED_LAYERS


def load_comparison_sources(directory, manifest_path, *, expected_selection_sha256, model_id, revision):
    """返回已核对的 Top7、30 条校准输入及本地来源审计；任何冲突均在模型加载前拒绝。

    检查 M4 完成标记、全部已列产物、完整零基排名，以及 M4→M2 文件/内容指纹和模板。
    不从概率重新计算重要性，也不读取原 parquet：user-only 构造和数据来源仍依赖 M2/M4
    已有独立验收。显式替换 expected hash 时，新的来源责任归调用者，不能据此认证作者。
    """
    directory, manifest_path = Path(directory), Path(manifest_path)
    if not directory.is_dir() or directory.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Expected non-symlink local M4 directory and M2 manifest")
    selection_path = directory / "selected_layers.json"
    if file_sha256(selection_path) != expected_selection_sha256:
        raise ValueError("M4 selection differs from the fixed final N30 SHA256")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    ranked, selected = selection["ranked_layer_indices"], selection["selected_layer_indices"]
    if (selection["index_base"] != 0 or selection["layer_count"] != EXPECTED_LAYERS or selection["top_k"] != 7
            or selection["mode"] != "accuracy: Top-K over all layers"
            or not isinstance(ranked, list) or any(type(index) is not int for index in ranked)
            or len(ranked) != EXPECTED_LAYERS or sorted(ranked) != list(range(EXPECTED_LAYERS))
            or not isinstance(selected, list) or any(type(index) is not int for index in selected)
            or selected != ranked[:7] or selection["selected_layer_indices_ascending"] != sorted(selected)):
        raise ValueError("Invalid M4 zero-based Top7 or complete layer ranking")
    if (summary["status"] != "PASS" or summary["n"] != 30 or summary["completed_samples"] != 30
            or summary["forward_evaluations"] != 1500 or summary["selected_layer_indices"] != selected):
        raise ValueError("M6 requires completed M4 N30, not N1 or partial calibration")
    expected = {"model_id": model_id, "requested_revision": revision, "n": 30, "manifest_n": 30,
                "limit": None, "top_k": 7, "layer_count": EXPECTED_LAYERS, "index_base": 0,
                "selection_mode": "accuracy/all layers"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("M4 model revision/configuration differs from the comparison")
    hashes = summary["files_sha256"]
    if not {"selected_layers.json", "resolved_config.json"} <= set(hashes):
        raise ValueError("M4 summary lacks required artifact fingerprints")
    source_hashes = {}
    for name, digest in hashes.items():
        if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("M4 artifact names must be local basenames")
        path = directory / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"M4 artifact hash mismatch: {name}")
        source_hashes[str(path.resolve())] = digest
    source_hashes[str((directory / "run_summary.json").resolve())] = file_sha256(directory / "run_summary.json")
    raw_manifest_hash = file_sha256(manifest_path)
    if raw_manifest_hash != config["manifest_file_sha256"]:
        raise ValueError("M2 manifest bytes differ from the manifest used for M4 selection")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_hash = manifest_content_hash(manifest)
    if (manifest["schema_version"] != 1 or manifest["content_sha256"] != content_hash
            or config["manifest_content_sha256"] != content_hash or manifest["sampling"]["n"] != 30
            or manifest["dataset"]["split"] != "train" or not isinstance(manifest["dataset"]["id"], str)
            or re.fullmatch(r"[0-9a-f]{40}", manifest["dataset"]["revision"]) is None
            or manifest["tokenizer"] != config["tokenizer"]
            or manifest["tokenizer"]["model_id"] != model_id or manifest["tokenizer"]["model_revision"] != revision
            or hash_text(manifest["template"]["text"]) != manifest["template"]["sha256"]
            or manifest["template"]["sha256"] != config["template_sha256"]
            or ("dataset" in config and config["dataset"] != manifest["dataset"])):
        raise ValueError("M2/M4 calibration content, split, tokenizer or template mismatch")
    samples, identifiers = manifest["samples"], set()
    if not isinstance(samples, list) or len(samples) != 30:
        raise ValueError("Exactly 30 saved calibration inputs are required")
    for row in samples:
        identifier, ids, mask = (row[key] for key in ("sample_id", "input_ids", "attention_mask"))
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or not isinstance(ids, list) or not 1 <= len(ids) <= 512
                or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(mask, list) or any(type(value) is not int for value in mask) or mask != [1] * len(ids)
                or type(row["prediction_position"]) is not int or row["prediction_position"] != len(ids) - 1
                or row["truncated"] is not False or not isinstance(row["prompt"], str)
                or hash_text(row["prompt"]) != row["prompt_hash"]):
            raise ValueError("Invalid saved calibration identity, prompt hash or unpadded token position")
        identifiers.add(identifier)
    source_hashes[str(manifest_path.resolve())] = raw_manifest_hash
    return {"selected_layer_indices": list(selected), "calibration_samples": samples, "m4_config": config,
            "audit": {"m4_directory": str(directory.resolve()), "manifest_path": str(manifest_path.resolve()),
                      "selection_sha256": expected_selection_sha256, "manifest_file_sha256": raw_manifest_hash,
                      "manifest_content_sha256": content_hash, "files_sha256": source_hashes,
                      "scope": "local integrity and M4-to-M2 binding; not remote source authentication"}}


def verify_comparison_tokenizer(source, tokenizer):
    """逐方法核对实际模板/special token 元数据，并重新分词全部30条既有校准 prompt。

    不构造校准答案或重新抽样。显式无特殊 token 叠加、无 padding/截断；完整 IDs/mask 必须
    与 M2 保存值相同，不能只相信 tokenizer 类名或手填的 token 哈希。
    """
    config = source["m4_config"]
    saved = config["tokenizer"]
    fields = ("is_fast", "padding_side", "truncation_side", "pad_token_id", "eos_token_id", "bos_token_id")
    if (hash_text(tokenizer.get_chat_template()) != config["template_sha256"]
            or type(tokenizer).__name__ != saved["class"] or any(getattr(tokenizer, name) != saved[name] for name in fields)):
        raise ValueError("Actual tokenizer metadata/template differs from M4/M2")
    for row in source["calibration_samples"]:
        encoded = tokenizer(row["prompt"], add_special_tokens=False, truncation=False, padding=False, return_attention_mask=True)
        if encoded["input_ids"] != row["input_ids"] or encoded["attention_mask"] != row["attention_mask"]:
            raise ValueError(f"Calibration input changed under actual tokenizer: {row['sample_id']}")


def code_hashes(root):
    """记录全部项目 Python 源码的真实字节，不包括缓存/虚拟环境/下载目录或敏感输入。"""
    root = Path(root)
    sources = sorted(path for folder in ("src", "scripts", "tests") for path in (root / folder).rglob("*.py"))
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sources}


def provenance(root, command):
    """只读记录 Git 基线、dirty 与实际源码；HEAD 不代表尚未提交的 M6 实现。"""
    def git(*args):
        """不提交、联网或修改索引；Git 不可用时留下 None 而非编造提交。"""
        try:
            process = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
            "git": {"baseline_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                    "dirty": bool(status) if status is not None else None, "status": status.splitlines() if status else []},
            "code_sha256": code_hashes(root)}


def environment():
    """记录当前 CPU/Python/依赖元数据和确定性设置；不联网、不读取凭据或改变安装环境。

    元数据版本不独立证明包可导入；真实前向/训练是另外的执行证据。线程与确定性开关由
    CLI 在捕获前统一设置，跨设备、依赖版本或不同线程数不保证数值逐字节一致。
    """
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "tokenizers", "huggingface-hub"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "python_executable": sys.executable, "platform": platform.platform(), "package_metadata_versions": packages,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
