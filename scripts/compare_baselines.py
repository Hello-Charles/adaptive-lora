"""M6 离线比较入口：统一 Full / Random-K / Adaptive-K 的短训练和 prompt-only 评估。

从项目根目录以 python -I 启动；只读固定本地权重、M4 N30 及其 M2 输入清单。每方法重新
加载基础模型，最多3个短CPU batch、每评估样本最多8个生成token。无下载/GPU/正式epoch
开关，不重做校准、不进入M7。输出目录必须新建，失败保留部分证据但不写整体成功标记。
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import traceback

import torch

from adaptive_lora.artifacts import write_json
from adaptive_lora.baselines import plan_baselines
from adaptive_lora.checkpoint import file_sha256, save_adapter_checkpoint
from adaptive_lora.comparison import run_variant, seed_training, summarize_comparison
from adaptive_lora.comparison_io import (
    code_hashes, environment, load_comparison_sources, provenance, verify_comparison_tokenizer,
)
from adaptive_lora.data import hash_text
from adaptive_lora.model_loader import DEFAULT_MODEL_ID, EXPECTED_LAYERS, load_model_and_tokenizer
from adaptive_lora.protocol import SharedConfig, build_smoke_protocol, stable_hash, validate_protocol
from adaptive_lora.trainer import audit_lora

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_CACHE = ROOT / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
M4_DIRECTORY = ROOT / "results/m4-20261114-n30"
MANIFEST_PATH = ROOT / "results/m2-20261114/manifest.json"
SELECTION_SHA256 = "0f5db68e4451ade4f92bc4353ca1c8c40226f9165f81a009ab129f962c640bdb"
EXPECTED_BASE_FINGERPRINT = {
    "sha256": "3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3",
    "parameter_elements": 494032768,
}


def _hex_sha(value, length):
    """只接受完整 SHA 格式，拒绝浮动 main/tag；格式正确不代表独立认证了来源。"""
    if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        raise argparse.ArgumentTypeError(f"Expected a {length}-character hexadecimal SHA")
    return value.lower()


def _positive_int(value):
    """加载前拒绝非正 CPU 线程数，不隐式改成另一个值。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _seed(value):
    """训练和选层 seed 都保存为有界普通整数，但分别使用独立 RNG。"""
    number = int(value)
    if not 0 <= number < 2 ** 32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32)")
    return number


def _file_hashes(directory):
    """完成标记不散列自身；运行日志仍在写入，另在关闭后的验收索引中保存它的最终哈希。"""
    return {path.relative_to(directory).as_posix(): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "run.log"}


def _write_comparison_csv(path, comparison):
    """用实际审计行输出纯表格；层ID列表用JSON单元格，不画未经验证的效果/速度曲线。"""
    rows = comparison["methods"]
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected_layer_indices": json.dumps(row["selected_layer_indices"])})


def main(argv=None):
    """执行一次全新目录的有界比较并返回成功摘要；失败保留轨迹、异常向调用者传播。

    三方法同一 SharedConfig/协议、CPU线程和确定性开关，只有选层改变；完整基础参数字节
    由 runner 每次注入前核对。校准允许来自训练侧，但评估不与训练/校准直接重叠。
    checkpoint 复用 M5 adapter-only 保存器；本次不额外做三方法恢复或训练断点续跑验证。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-dir", type=Path, default=M4_DIRECTORY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-selection-sha256", type=lambda value: _hex_sha(value, 64), default=SELECTION_SHA256)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=lambda value: _hex_sha(value, 40), default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=3, help="1–3 batches per method, batch_size=1")
    parser.add_argument("--train-seed", type=_seed, default=42)
    parser.add_argument("--layer-seed", type=_seed, default=20261115, help="Independent Random-K layer seed")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int, default=4, help="1–8 greedy tokens per evaluation prompt")
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, never reused")
    args = parser.parse_args(argv)
    try:
        shared = SharedConfig(steps=args.steps, rank=args.rank, alpha=args.alpha, train_seed=args.train_seed,
                              learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_new_tokens=args.max_new_tokens)
    except ValueError as error:
        parser.error(f"Invalid shared setting: {error}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.cache_dir.is_dir():
        parser.error(f"Offline model cache does not exist: {args.cache_dir}")
    try:
        source = load_comparison_sources(args.m4_dir, args.manifest, expected_selection_sha256=args.expected_selection_sha256,
                                         model_id=args.model_id, revision=args.revision)
        plans = plan_baselines(EXPECTED_LAYERS, 7, source["selected_layer_indices"], layer_seed=args.layer_seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid comparison source: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, total_steps, active_method = perf_counter(), 0, None
    completed_methods, reports, timings = [], [], {}
    seed_training(shared.train_seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as log:
        def emit(message):
            """同步终端和新日志，及时flush；不打印完整prompt、答案、token清单或权重。"""
            text = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()
        try:
            settings = shared.to_dict()
            config = {"scope": "M6 synthetic interface comparison only; not paper benchmark training/evaluation",
                      "model_id": args.model_id, "requested_revision": args.revision, "model_cache_dir": str(args.cache_dir.resolve()),
                      "m4_selection_sha256": source["audit"]["selection_sha256"], "shared_config_sha256": stable_hash(settings),
                      "shared_config": settings, "layer_seed": args.layer_seed, "index_base": 0, "top_k": 7,
                      "cpu_threads": args.threads, "deterministic_algorithms": True, "allow_download": False,
                      "local_files_only": True, "trust_remote_code": False, "checkpoint_reload_tested": False,
                      "rng_limit": "shared seed does not guarantee identical initial A tensors for common layers across different target sets"}
            write_json(args.output_dir / "resolved_config.json", config)
            write_json(args.output_dir / "shared_config.json", settings)
            write_json(args.output_dir / "source_audit.json", source["audit"])
            write_json(args.output_dir / "baseline_plans.json", [plan.to_dict() for plan in plans])
            evidence = provenance(ROOT, ["scripts/compare_baselines.py", *(sys.argv[1:] if argv is None else argv)])
            write_json(args.output_dir / "provenance.json", evidence)
            with (args.output_dir / "git_commit.txt").open("x", encoding="utf-8") as output:
                output.write((evidence["git"]["baseline_commit"] or "unavailable") + "\n")
            write_json(args.output_dir / "environment.json", environment())
            protocol = None
            for plan in plans:
                active_method, method_steps, method_started = plan.method, 0, perf_counter()
                directory = args.output_dir / plan.method
                directory.mkdir(exist_ok=False)
                emit(f"{plan.method}: loading fresh fixed-revision CPU base; layers={list(plan.indices)}")
                phase = perf_counter()
                model, tokenizer = load_model_and_tokenizer(args.model_id, revision=args.revision, allow_download=False, cache_dir=args.cache_dir)
                method_timings = {"base_model_load_seconds": perf_counter() - phase}
                verify_comparison_tokenizer(source, tokenizer)
                if protocol is None:
                    protocol = build_smoke_protocol(tokenizer, source["calibration_samples"], steps=shared.steps)
                    write_json(args.output_dir / "protocol.json", protocol)
                    write_json(args.output_dir / "split_audit.json", validate_protocol(protocol))
                    write_json(args.output_dir / "sample_ids.json", {"source": "synthetic-m6-v1 plus fixed M2 calibration inputs",
                               "training": [row["sample_id"] for row in protocol["training_samples"]],
                               "evaluation": [row["sample_id"] for row in protocol["evaluation_prompts"]],
                               "calibration": [row["sample_id"] for row in protocol["calibration_inputs"]]})
                write_json(directory / "selection.json", plan.to_dict())
                write_json(directory / "resolved_config.json", {**config, "selection": plan.to_dict(), "protocol_sha256": protocol["content_sha256"]})
                phase = perf_counter()
                with (directory / "steps.jsonl").open("x", encoding="utf-8") as trace:
                    def on_step(record):
                        """只记录完成的optimizer step；回调失败向上传播，不把部分训练记成完整成功。"""
                        nonlocal total_steps, method_steps
                        trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        trace.flush()
                        method_steps += 1
                        total_steps += 1
                        emit(f"{plan.method}: step {method_steps}/{shared.steps}; loss={record['loss']:.9g}; grad_norm={record['gradient_l2_norm']:.9g}")
                    setup, report = run_variant(plan, model, tokenizer, protocol, shared, revision=args.revision,
                                                expected_base_fingerprint=EXPECTED_BASE_FINGERPRINT, on_step=on_step)
                method_timings["unified_train_eval_and_audits_seconds"] = perf_counter() - phase
                audit = audit_lora(setup)
                write_json(directory / "model_runtime.json", {"model_class": type(model).__name__, "requested_revision": args.revision,
                           "config_resolved_revision": getattr(model.config, "_commit_hash", None),
                           "attention_implementation": getattr(model.config, "_attn_implementation", None),
                           "base_dtypes": sorted({str(parameter.dtype) for _, parameter in setup.base_parameters}),
                           "adapter_dtypes": audit["adapter_dtypes"], "tokenizer_class": type(tokenizer).__name__,
                           "template_sha256": hash_text(tokenizer.get_chat_template()), "eos_token_id": tokenizer.eos_token_id,
                           "eos_token": getattr(tokenizer, "eos_token", None), "base_fingerprint": setup.base_fingerprint})
                write_json(directory / "trainable_parameters.json", audit)
                write_json(directory / "training_metrics.json", report["training"])
                write_json(directory / "evaluation.json", report["evaluation"])
                phase = perf_counter()
                save_adapter_checkpoint(setup, directory / "checkpoint", model_id=args.model_id, revision=args.revision)
                method_timings["checkpoint_save_seconds"] = perf_counter() - phase
                method_timings["method_total_elapsed_seconds"] = perf_counter() - method_started
                write_json(directory / "report.json", report)
                write_json(directory / "timings.json", method_timings)
                write_json(directory / "completion.json", {"status": "PASS", "method": plan.method, "completed_steps": method_steps,
                           "protocol_sha256": report["protocol_sha256"], "shared_config_sha256": report["shared_config_sha256"],
                           "files_sha256": _file_hashes(directory), "checkpoint_reload_tested": False})
                completed_methods.append(plan.method)
                reports.append(report)
                timings[plan.method] = method_timings
                emit(f"{plan.method}: saved; trainable={audit['trainable_parameter_elements']}; smoke EM={report['evaluation']['scores']['correct']}/{report['evaluation']['scores']['total']}; base unchanged")
                # 释放 PEFT、基础模型及参数引用；下一方法必须重新加载，不能继承已训练的前一种实例。
                del setup, model, tokenizer
                gc.collect()
            comparison = summarize_comparison(reports)
            if code_hashes(ROOT) != evidence["code_sha256"]:
                raise RuntimeError("Source files changed during comparison; refuse mixed-version evidence")
            if any(file_sha256(path) != digest for path, digest in source["audit"]["files_sha256"].items()):
                raise RuntimeError("M4/M2 source artifacts changed during comparison")
            write_json(args.output_dir / "comparison.json", comparison)
            _write_comparison_csv(args.output_dir / "comparison.csv", comparison)
            summary = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                       "completed_methods": completed_methods, "completed_optimizer_steps": total_steps,
                       "shared_config_sha256": comparison["shared_config_sha256"], "protocol_sha256": comparison["protocol_sha256"],
                       "same_parameter_budget": comparison["same_parameter_budget"], "methods": comparison["methods"],
                       "phase_timings_seconds": timings, "total_elapsed_seconds": perf_counter() - started,
                       "timing_scope": "CPU wall-clock including hashes/audits/IO; not paper training speed or memory benchmark",
                       "files_sha256": _file_hashes(args.output_dir), "formal_training_run": False,
                       "checkpoint_reload_tested": False, "m7_started": False}
            write_json(args.output_dir / "run_summary.json", summary)
            emit(f"PASS methods={completed_methods}; steps={total_steps}; identical data/settings; same_parameter_budget={comparison['same_parameter_budget']}")
            return summary
        except Exception as error:
            traceback.print_exc(file=log)
            log.flush()
            write_json(args.output_dir / "failure.json", {"status": "FAIL", "error_type": type(error).__name__, "error": str(error),
                       "active_method": active_method, "completed_methods": completed_methods, "completed_optimizer_steps": total_steps,
                       "elapsed_seconds": perf_counter() - started})
            raise


if __name__ == "__main__":
    main()
