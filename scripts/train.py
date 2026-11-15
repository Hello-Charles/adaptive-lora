"""M5 离线入口：复用 M4 N30 的 Top-7，只运行 1–3 个短合成 batch 并验证 adapter 恢复。

从项目根目录使用 python -I 启动。只接受已有本地模型缓存和全新结果目录；没有下载、GPU、
正式数据训练或跳过恢复的开关。本入口不重做校准，不把合成答案写入 M2/M4，也不进入 M6。
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from adaptive_lora.artifacts import write_json
from adaptive_lora.checkpoint import file_sha256, restore_adapter_checkpoint, save_adapter_checkpoint
from adaptive_lora.data import hash_text
from adaptive_lora.model_loader import DEFAULT_MODEL_ID, EXPECTED_LAYERS, load_model_and_tokenizer
from adaptive_lora.trainer import (
    attach_selective_lora, audit_lora, build_supervised_example, fingerprint_parameters, train_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_CACHE = ROOT / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
M4_DIRECTORY = ROOT / "results/m4-20261114-n30"
SELECTION_SHA256 = "0f5db68e4451ade4f92bc4353ca1c8c40226f9165f81a009ab129f962c640bdb"
EXPECTED_BASE_FINGERPRINT = {
    "sha256": "3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3",
    "parameter_elements": 494032768,
}
# 版本化的仓库内合成监督数据，固定顺序、不抽样；只验证工程反向，不代表正式训练分布。
SMOKE_ROWS = (
    {"sample_id": "synthetic-m5-v1/0", "instruction": "Reply with yes.", "answer": "yes"},
    {"sample_id": "synthetic-m5-v1/1", "instruction": "Return the digit 2.", "answer": "2"},
    {"sample_id": "synthetic-m5-v1/2", "instruction": "Write the word red.", "answer": "red"},
)


def _hex_sha(value, length):
    """只接受完整 SHA 格式，不把 main/tag 当固定版本；不执行远端来源认证。"""
    if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        raise argparse.ArgumentTypeError(f"Expected a {length}-character hexadecimal SHA")
    return value.lower()


def _positive_int(value):
    """在加载模型前拒绝非正 CPU 线程数。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def load_m4_selection(directory, *, expected_selection_sha256, model_id, revision):
    """加载完整 N30/24 层/Top7 accuracy-mode 结果，并验证固定 selection 字节与产物哈希。

    此处验证已完成运行的结构与完整性，不从概率重新评分；M4 的独立数学复算由既有验收器
    负责。显式替换 expected hash 时来源责任归调用者，哈希不是作者签名。N1 门禁结果拒绝
    作为正式选层来源，层号使用整数零基，完整 ranking 必须是 0..23 的一个排列。
    """
    directory = Path(directory)
    raw = (directory / "selected_layers.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_selection_sha256:
        raise ValueError("M4 selected_layers.json does not match the fixed N30 selection SHA256")
    selection = json.loads(raw)
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    ranked, selected = selection["ranked_layer_indices"], selection["selected_layer_indices"]
    if (selection["index_base"] != 0 or selection["layer_count"] != EXPECTED_LAYERS or selection["top_k"] != 7
            or selection["mode"] != "accuracy: Top-K over all layers"
            or not isinstance(ranked, list) or any(type(index) is not int for index in ranked)
            or len(ranked) != EXPECTED_LAYERS or sorted(ranked) != list(range(EXPECTED_LAYERS))
            or not isinstance(selected, list) or any(type(index) is not int for index in selected)
            or selected != ranked[:7] or selection["selected_layer_indices_ascending"] != sorted(selected)):
        raise ValueError("Invalid zero-based M4 Top7/ranking structure")
    if (summary["status"] != "PASS" or summary["n"] != 30 or summary["completed_samples"] != 30
            or summary["forward_evaluations"] != 1500 or summary["selected_layer_indices"] != selected):
        raise ValueError("M5 requires the completed M4 N30 run, not an N1 or partial run")
    expected = {"model_id": model_id, "requested_revision": revision, "n": 30, "manifest_n": 30,
                "limit": None, "top_k": 7, "layer_count": EXPECTED_LAYERS, "index_base": 0,
                "selection_mode": "accuracy/all layers"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("M4 configuration/model revision does not match M5")
    hashes = summary["files_sha256"]
    if not {"selected_layers.json", "resolved_config.json"} <= set(hashes):
        raise ValueError("M4 summary lacks required artifact fingerprints")
    for name, expected_hash in hashes.items():
        if Path(name).name != name or file_sha256(directory / name) != expected_hash:
            raise ValueError(f"M4 artifact hash mismatch: {name}")
    return {"selection": selection, "config": config, "selection_sha256": digest,
            "summary_sha256": file_sha256(directory / "run_summary.json"),
            "config_sha256": hashes["resolved_config.json"]}


def verify_m4_tokenizer(source, tokenizer):
    """核对本地 tokenizer 模板和 special/padding 元数据与 M4 一致，不重读训练答案。"""
    config = source["config"]
    saved = config["tokenizer"]
    if (hash_text(tokenizer.get_chat_template()) != config["template_sha256"]
            or type(tokenizer).__name__ != saved["class"] or any(getattr(tokenizer, name) != saved[name]
                for name in ("is_fast", "padding_side", "truncation_side", "pad_token_id", "eos_token_id", "bos_token_id"))):
        raise ValueError("Local tokenizer differs from the M4 tokenizer/template")


def provenance(command):
    """记录 Git 基线/dirty 和实际源码字节；不把 M1 HEAD 冒充尚未提交的 M5 版本。"""
    def git(*args):
        """只读 Git，不提交、联网或更改索引；缺失 Git 时明确留下 None。"""
        try:
            process = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None
    status = git("status", "--porcelain")
    sources = sorted(path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py"))
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
            "git": {"baseline_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                    "dirty": bool(status) if status is not None else None, "status": status.splitlines() if status else []},
            "code_sha256": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sources}}


def environment():
    """记录实际 Python/CPU/依赖发行版元数据，不读取凭据、不联网、不改变安装环境。"""
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


def probe_logits(model, samples):
    """在 eval/no_grad 下仅输入各监督样本的 user prefix，返回完整 CPU logits 快照。

    回答不参与探针输入；此处只比较注入/保存恢复的数值一致性，不生成答案、不算准确率。
    整体模型不转 dtype，不复用 KV cache。快照不含计算图，也不把完整 logits 写入 Git。
    """
    model.eval()
    result = {}
    with torch.no_grad():
        for sample in samples:
            ids = torch.tensor([sample["input_ids"][:sample["prompt_token_count"]]], dtype=torch.long)
            values = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            if not torch.isfinite(values).all().item():
                raise ValueError("Nonfinite evaluation probe logits")
            result[sample["sample_id"]] = values.detach().cpu().clone()
    return result


def compare_probes(expected, actual):
    """要求同一 CPU/dtype 的完整 logits torch.equal；记录字节哈希与最大绝对误差而非近似通过。"""
    if set(expected) != set(actual):
        raise ValueError("Probe IDs differ")
    rows = []
    for identifier, left in expected.items():
        right = actual[identifier]
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"Probe logits did not restore exactly: {identifier}")
        rows.append({"sample_id": identifier, "shape": list(left.shape), "dtype": str(left.dtype), "exact": True,
                     "max_abs_difference": (left.float() - right.float()).abs().max().item(),
                     "expected_sha256": fingerprint_parameters([(identifier, left)])["sha256"],
                     "actual_sha256": fingerprint_parameters([(identifier, right)])["sha256"]})
    return {"all_exact": True, "probes": rows, "comparison": "full prefix-only logits; torch.equal; no tolerance"}


def main(argv=None):
    """一次新目录对应一次受限 smoke；失败保留证据但不写成功标记，不重试或覆盖 checkpoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-dir", type=Path, default=M4_DIRECTORY)
    parser.add_argument("--expected-selection-sha256", type=lambda value: _hex_sha(value, 64), default=SELECTION_SHA256)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=lambda value: _hex_sha(value, 40), default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=3, help="Exactly 1–3 batches, batch_size=1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, never reused")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.cache_dir.is_dir():
        parser.error(f"Offline model cache does not exist: {args.cache_dir}")
    try:
        source = load_m4_selection(args.m4_dir, expected_selection_sha256=args.expected_selection_sha256,
                                   model_id=args.model_id, revision=args.revision)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid M4 input: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, completed = perf_counter(), 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as log:
        def emit(message):
            """同步终端和新日志并及时 flush，不输出完整 prompt 或训练答案。"""
            text = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()
        try:
            config = {
                "scope": "M5 synthetic engineering smoke; not a paper training/evaluation run",
                "model_id": args.model_id, "requested_revision": args.revision, "model_cache_dir": str(args.cache_dir.resolve()),
                "m4_directory": str(args.m4_dir.resolve()), "m4_n": 30, "m4_selection_sha256": source["selection_sha256"],
                "m4_summary_sha256": source["summary_sha256"], "m4_config_sha256": source["config_sha256"],
                "selected_layer_indices": source["selection"]["selected_layer_indices"],
                "selected_layer_indices_ascending": source["selection"]["selected_layer_indices_ascending"],
                "index_base": 0, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0,
                "target_projections": ["q", "k", "v"], "init_lora_weights": True, "bias": "none",
                "steps": args.steps, "batch_size": 1, "gradient_accumulation": 1, "max_length": 96,
                "seed": args.seed, "cpu_threads": args.threads, "deterministic_algorithms": True,
                "device": "cpu", "precision": "preserve base dtype; PEFT FP32 adapters; FP32 cross entropy; no AMP",
                "loss_mask": "prompt=-100; answer and explicit tokenizer EOS supervised; causal shift by one",
                "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
                "scheduler": None, "warmup": 0, "gradient_clipping": None, "padding": False, "truncation": False,
                "synthetic_data": "repository SMOKE_ROWS v1, fixed prefix order, no calibration/benchmark data",
                "synthetic_rows_sha256": hash_text(json.dumps(SMOKE_ROWS, ensure_ascii=False, sort_keys=True)),
                "use_cache": False, "allow_download": False, "local_files_only": True, "trust_remote_code": False,
                "checkpoint_scope": "adapter-only inference restore, not optimizer/RNG resume",
            }
            write_json(args.output_dir / "resolved_config.json", config)
            evidence = provenance(["scripts/train.py", *(sys.argv[1:] if argv is None else argv)])
            write_json(args.output_dir / "provenance.json", evidence)
            with (args.output_dir / "git_commit.txt").open("x", encoding="utf-8") as output:
                output.write((evidence["git"]["baseline_commit"] or "unavailable") + "\n")
            write_json(args.output_dir / "environment.json", environment())
            timings = {}
            phase = perf_counter()
            model, tokenizer = load_model_and_tokenizer(args.model_id, revision=args.revision, allow_download=False, cache_dir=args.cache_dir)
            timings["base_model_load_seconds"] = perf_counter() - phase
            verify_m4_tokenizer(source, tokenizer)
            if fingerprint_parameters(model.named_parameters()) != EXPECTED_BASE_FINGERPRINT:
                raise ValueError("Local base weights differ from the verified M4 base fingerprint")
            rows = SMOKE_ROWS[:args.steps]
            samples = [build_supervised_example(tokenizer, row["instruction"], row["answer"],
                       sample_id=row["sample_id"], max_length=96) for row in rows]
            write_json(args.output_dir / "training_samples.json", samples)
            write_json(args.output_dir / "sample_ids.json", {"source": "synthetic-m5-v1", "sample_ids": [row["sample_id"] for row in rows]})
            baseline = probe_logits(model, samples)
            phase = perf_counter()
            setup = attach_selective_lora(model, config["selected_layer_indices"], revision=args.revision)
            audit = audit_lora(setup)
            write_json(args.output_dir / "trainable_parameters.json", audit)
            initialized = probe_logits(setup.model, samples)
            write_json(args.output_dir / "initialization_check.json", compare_probes(baseline, initialized))
            timings["attach_and_initialization_check_seconds"] = perf_counter() - phase
            write_json(args.output_dir / "model_runtime.json", {
                "requested_revision": args.revision, "config_resolved_revision": getattr(model.config, "_commit_hash", None),
                "model_class": type(model).__name__, "attention_implementation": getattr(model.config, "_attn_implementation", None),
                "base_dtypes": sorted({str(parameter.dtype) for _, parameter in setup.base_parameters}),
                "adapter_dtypes": audit["adapter_dtypes"], "base_fingerprint": setup.base_fingerprint,
                "tokenizer_class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id,
                "eos_token": getattr(tokenizer, "eos_token", None), "template_sha256": hash_text(tokenizer.get_chat_template()),
            })
            emit(f"Top7={config['selected_layer_indices']}; projections={len(setup.targets)}; trainable={audit['trainable_parameter_elements']}; initialization exact")
            phase = perf_counter()
            with (args.output_dir / "steps.jsonl").open("x", encoding="utf-8") as trace:
                def on_step(record):
                    """只记录已完成的优化步骤；同步写失败向上传播，不能伪装训练全部完成。"""
                    nonlocal completed
                    trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    trace.flush()
                    completed += 1
                    emit(f"step {completed}/{args.steps}; loss={record['loss']:.9g}; grad_norm={record['gradient_l2_norm']:.9g}; updated_tensors={record['updated_adapter_tensors']}")
                metrics = train_smoke(setup, samples, on_step=on_step)
            timings["training_seconds"] = perf_counter() - phase
            write_json(args.output_dir / "training_metrics.json", metrics)
            write_json(args.output_dir / "post_training_audit.json", audit_lora(setup))
            trained = probe_logits(setup.model, samples)
            probe_changed = any(not torch.equal(initialized[key], trained[key]) for key in trained)
            phase = perf_counter()
            save_adapter_checkpoint(setup, args.output_dir / "checkpoint", model_id=args.model_id, revision=args.revision)
            timings["checkpoint_save_seconds"] = perf_counter() - phase
            # 释放训练对象及其规范参数引用；恢复必须再次从固定本地缓存创建基础模型，不能在原实例上挂回。
            del baseline, initialized, setup, model, tokenizer
            gc.collect()
            phase = perf_counter()
            fresh_base, fresh_tokenizer = load_model_and_tokenizer(args.model_id, revision=args.revision, allow_download=False, cache_dir=args.cache_dir)
            verify_m4_tokenizer(source, fresh_tokenizer)
            restored = restore_adapter_checkpoint(fresh_base, args.output_dir / "checkpoint", model_id=args.model_id, revision=args.revision)
            timings["fresh_base_load_and_adapter_restore_seconds"] = perf_counter() - phase
            reload_check = compare_probes(trained, probe_logits(restored.model, samples))
            reload_check["fresh_base_loaded_from_cache"] = True
            reload_check["optimizer_rng_resume_tested"] = False
            write_json(args.output_dir / "reload_check.json", reload_check)
            write_json(args.output_dir / "reload_audit.json", audit_lora(restored, trainable=False))
            if any(file_sha256(ROOT / name) != digest for name, digest in evidence["code_sha256"].items()):
                raise RuntimeError("Source files changed during smoke; refuse mixed-version evidence")
            files = {path.relative_to(args.output_dir).as_posix(): file_sha256(path)
                     for path in sorted(args.output_dir.rglob("*")) if path.is_file() and path.name != "run.log"}
            summary = {
                "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(), "completed_steps": completed,
                "selected_layer_indices": config["selected_layer_indices"], "trainable_parameter_elements": audit["trainable_parameter_elements"],
                "base_parameter_elements": audit["base_parameter_elements"], "adapter_projections": len(audit["target_modules"]),
                "initialization_logits_exact": True, "reload_logits_exact": reload_check["all_exact"],
                "reload_probe_count": len(reload_check["probes"]), "post_training_probe_changed": probe_changed,
                "base_fingerprint_unchanged": metrics["base_before"] == metrics["base_after"],
                "phase_timings_seconds": timings, "total_elapsed_seconds": perf_counter() - started,
                "timing_scope": "CPU wall-clock including audits/probes/hash/save/reload; not a paper benchmark",
                "files_sha256": files, "formal_training_run": False, "m6_started": False,
            }
            write_json(args.output_dir / "run_summary.json", summary)
            emit(f"PASS steps={completed}; base unchanged; fresh-base reload exact on {len(reload_check['probes'])} probes")
            return summary
        except Exception as error:
            traceback.print_exc(file=log)
            log.flush()
            write_json(args.output_dir / "failure.json", {"status": "FAIL", "error_type": type(error).__name__,
                       "error": str(error), "completed_steps": completed, "elapsed_seconds": perf_counter() - started})
            raise


if __name__ == "__main__":
    main()
