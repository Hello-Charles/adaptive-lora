"""M4 离线校准入口；复用 M2 固定清单与 M1 本地模型，绝不下载/训练/覆盖结果。

从项目目录以 python -I 运行，不能从模型/数据下载目录启动解释器。默认只接受已验收的
M2 内容哈希；替换清单须显式给出新哈希并自行确认实验设计、数据协议和答案泄漏边界。
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
import traceback

import torch

from adaptive_lora.artifacts import write_calibration_artifacts, write_json
from adaptive_lora.calibration import calibrate_model
from adaptive_lora.data import hash_text, manifest_content_hash, position_from_mask, sample_indices, tokenize_single
from adaptive_lora.model_loader import DEFAULT_MODEL_ID, EXPECTED_LAYERS, discover_layer_modules, load_model_and_tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_CACHE = ROOT / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
MANIFEST = ROOT / "results/m2-20261114/manifest.json"
# 固定的是已验收 M2 的协议+样本指纹，不含时间戳/绝对路径；不是论文作者数据的签名。
MANIFEST_CONTENT_SHA256 = "7c829ccb4c332291ea432c62b3a256bf8459d5a901e4eb9a575b0818aedfedc9"


def file_sha256(path):
    """流式读取文件字节，不解析或执行下载内容；也用于 dirty 工作树源码的身份记录。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value):
    """解析阶段拒绝非正样本数、线程数和 K，不允许无意中启动无界/空实验。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def hex_sha(value, length):
    """固定完整 SHA 而非 main/tag；格式检查不代表已经认证远端来源。"""
    if re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None:
        raise argparse.ArgumentTypeError(f"Expected a {length}-character hexadecimal SHA")
    return value.lower()


def validate_manifest(manifest, *, model_id, revision, expected_content_sha256):
    """在模型加载前核对完整 M2 清单的内部协议、采样顺序、输入形状和内容指纹。

    此处不重新读取原 parquet/答案；输入来源和 user-only 构造依赖已验收 M2 的固定指纹。
    内容哈希是完整性/一致性检查而非真实性签名；自行换 expected 值时由调用者负责来源。
    拒绝重新抽样、截断、padding 或读取不一致的预测位置，不默默修复输入改变实验。
    """
    try:
        content = manifest_content_hash(manifest)
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid M2 manifest structure") from error
    if content != manifest.get("content_sha256") or content != expected_content_sha256:
        raise ValueError("manifest content_sha256 does not match the fixed protocol and samples")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema_version")
    dataset, sampling, template, tokenizer = (manifest[key] for key in ("dataset", "sampling", "template", "tokenizer"))
    if dataset["split"] != "train" or re.fullmatch(r"[0-9a-f]{40}", dataset["revision"]) is None:
        raise ValueError("Calibration requires a fixed train dataset revision")
    if (template["kind"] != "tokenizer_chat_template_user_only" or template["source_fields"] != ["instruction", "input"]
            or template["add_generation_prompt"] is not True or hash_text(template["text"]) != template["sha256"]):
        raise ValueError("Expected the fixed user-only template without answer fields")
    if tokenizer["model_id"] != model_id or tokenizer["model_revision"] != revision:
        raise ValueError("Model/tokenizer revision must match the M2 manifest")
    if tokenizer["add_special_tokens"] is not False or tokenizer["padding"] is not False or not tokenizer["overlength_policy"].startswith("reject"):
        raise ValueError("Expected M2 no-padding, no-added-special-token, overlength-reject protocol")
    if type(tokenizer["max_length"]) is not int or tokenizer["max_length"] < 1:
        raise ValueError("Invalid tokenizer max_length")
    samples = manifest["samples"]
    n, population, seed = sampling["n"], dataset["population_size"], sampling["seed"]
    if type(n) is not int or n < 1 or type(population) is not int or population < n or type(seed) is not int:
        raise ValueError("Invalid fixed sampling configuration")
    if len(samples) != n or sampling["index_base"] != 0 or sampling["method"] != "random.Random(seed).sample(range(population_size), n)":
        raise ValueError("Expected the M2 zero-based, without-replacement sampling protocol")
    expected_indices = sample_indices(population, n, seed)
    if [sample["index"] for sample in samples] != expected_indices:
        raise ValueError("Sample indices/order differ from the fixed M2 draw")
    for sample, index in zip(samples, expected_indices):
        expected_id = f"{dataset['id']}@{dataset['revision']}/train/{index}"
        if sample["sample_id"] != expected_id or type(sample["index"]) is not int:
            raise ValueError("Sample ID does not identify its dataset revision/train row")
        ids, mask = sample["input_ids"], sample["attention_mask"]
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid saved input_ids")
        if not isinstance(mask, list) or any(type(value) is not int for value in mask) or mask != [1] * len(ids):
            raise ValueError("M2 manifest inputs must be single unpadded sequences")
        if (sample["truncated"] is not False or type(sample["token_count"]) is not int
                or sample["token_count"] != len(ids) or sample["original_token_count"] != len(ids)
                or len(ids) > tokenizer["max_length"]):
            raise ValueError("Saved token count/truncation violates M2 length policy")
        if type(sample["prediction_position"]) is not int or sample["prediction_position"] != position_from_mask(mask):
            raise ValueError("Saved prediction_position must be the final active input token")
        if any(not isinstance(sample[key], str) or re.fullmatch(r"[0-9a-f]{64}", sample[key]) is None
               for key in ("row_hash", "input_hash", "prompt_hash")) or hash_text(sample["prompt"]) != sample["prompt_hash"]:
            raise ValueError("Saved prompt or source hashes are inconsistent")


def verify_tokenizer(manifest, tokenizer):
    """真实本地 tokenizer 复核全部样本 IDs/mask/有效位置，不重渲染答案或改变清单。

    get_chat_template 与记录逐字比较，配置方向/special IDs/类也匹配；随后对已保存 prompt
    使用 add_special_tokens=False 重新分词。这里只证明当前 tokenizer 与 M2 相同，不再次
    证明 prompt 与原始 instruction/input 的关系；后者由 M2 原 parquet 独立核验承担。
    """
    config = manifest["tokenizer"]
    if tokenizer.get_chat_template() != manifest["template"]["text"]:
        raise ValueError("Local tokenizer chat template differs from M2")
    if type(tokenizer).__name__ != config["class"] or any(
        getattr(tokenizer, name) != config[name]
        for name in ("is_fast", "padding_side", "truncation_side", "pad_token_id", "eos_token_id", "bos_token_id")
    ):
        raise ValueError("Local tokenizer metadata differs from M2")
    for sample in manifest["samples"]:
        encoded = tokenize_single(tokenizer, sample["prompt"], config["max_length"])
        if any(encoded[key] != sample[key] for key in encoded):
            raise ValueError(f"Local tokenizer produced different saved IDs/mask/position: {sample['sample_id']}")


def provenance(command):
    """记录 Git 基线+dirty+实际源码哈希；不把 HEAD 误写成尚未创建的 M4 提交。

    只扫描项目源码/脚本/测试的 .py，不扫描下载目录和完整输入；不读取凭据或环境变量。
    已安装依赖只查发行版元数据，不把版本存在说成逐包隔离导入成功。
    """
    def git(*args):
        """Git 不存在时显式返回 None，不借用或编造版本证据。"""
        try:
            process = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None

    commit, status, branch = git("rev-parse", "HEAD"), git("status", "--porcelain"), git("branch", "--show-current")
    sources = sorted(path for folder in ("src", "scripts", "tests") for path in (ROOT / folder).rglob("*.py"))
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "git": {"baseline_commit": commit, "branch": branch, "dirty": bool(status) if status is not None else None,
                "status": status.splitlines() if status else []},
        "code_sha256": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sources},
    }


def environment():
    """当前 Python/CPU 和关键包的元数据快照；不查询密钥、不联网、不改安装环境。"""
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "tokenizers", "pyarrow", "huggingface-hub", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable}, "platform": platform.platform(),
        "package_metadata_versions": packages, "device": "cpu", "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
    }


def main(argv=None):
    """一个新目录只对应一次串行运行；--limit 取固定清单前缀做门禁，不重新抽样。

    CLI 显式设置随机种子、CPU 线程和确定性算法，负责 eval/freeze。成功标记最后写入；
    失败保留配置/已完成样本与 failure.json，绝不自动重试、删目录或把部分输出当成功。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--expected-manifest-sha256", type=lambda value: hex_sha(value, 64), default=MANIFEST_CONTENT_SHA256)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", type=lambda value: hex_sha(value, 40), default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=MODEL_CACHE)
    parser.add_argument("--top-k", type=positive_int, default=7)
    parser.add_argument("--limit", type=positive_int, help="Take first N saved samples, without a new random draw")
    parser.add_argument("--seed", type=int, default=20261114, help="Inference RNG seed, not M2 sampling seed")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment directory; never reused or overwritten")
    args = parser.parse_args(argv)
    if args.top_k > EXPECTED_LAYERS:
        parser.error(f"--top-k cannot exceed {EXPECTED_LAYERS}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.cache_dir.is_dir():
        parser.error(f"Offline model cache does not exist: {args.cache_dir}")
    try:
        # 读取原文件一次并同时保存字节指纹，避免分别读取时发生变化造成证据错配。
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw)
        validate_manifest(manifest, model_id=args.model_id, revision=args.revision,
                          expected_content_sha256=args.expected_manifest_sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid calibration manifest: {error}")
    if args.limit is not None and args.limit > len(manifest["samples"]):
        parser.error("--limit cannot exceed the fixed manifest sample count")
    samples = manifest["samples"][:args.limit] if args.limit is not None else manifest["samples"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = ["scripts/run_calibration.py", *(sys.argv[1:] if argv is None else argv)]
    started = perf_counter()
    completed = 0
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    with (args.output_dir / "run.log").open("x", encoding="utf-8") as log:
        def record_log(message):
            """同步终端与独立运行日志并立即 flush；不输出完整 prompt/答案。"""
            text = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        try:
            config = {
                "model_id": args.model_id, "requested_revision": args.revision, "model_cache_dir": str(args.cache_dir.resolve()),
                "manifest_path": str(args.manifest.resolve()), "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
                "manifest_content_sha256": manifest["content_sha256"], "manifest_n": manifest["sampling"]["n"],
                "n": len(samples), "limit": args.limit, "sampling_seed": manifest["sampling"]["seed"], "inference_seed": args.seed,
                "top_k": args.top_k, "layer_count": EXPECTED_LAYERS, "index_base": 0, "selection_mode": "accuracy/all layers",
                "batch_size": 1, "use_cache": False, "allow_download": False, "local_files_only": True,
                "trust_remote_code": False, "device": "cpu", "cpu_threads": args.threads, "deterministic_algorithms": True,
                "probability_dtype": "float32", "accumulation": "Python math.fsum", "tie_break": "ascending layer index",
                "dataset": manifest["dataset"], "template_sha256": manifest["template"]["sha256"], "tokenizer": manifest["tokenizer"],
                "expected_forward_evaluations": len(samples) * (2 * EXPECTED_LAYERS + 2), "restore_check_per_sample": True,
            }
            write_json(args.output_dir / "resolved_config.json", config)
            identities = [{key: sample[key] for key in ("sample_id", "index", "row_hash", "input_hash", "prompt_hash",
                                                       "input_ids", "attention_mask", "prediction_position", "token_count")}
                          for sample in samples]
            write_json(args.output_dir / "sample_ids.json", identities)
            evidence = provenance(command)
            write_json(args.output_dir / "provenance.json", evidence)
            write_json(args.output_dir / "environment.json", environment())
            with (args.output_dir / "git_commit.txt").open("x", encoding="utf-8") as output:
                output.write((evidence["git"]["baseline_commit"] or "unavailable") + "\n")
            record_log(f"START n={len(samples)} / manifest_n={len(manifest['samples'])}; Top-{args.top_k}; CPU threads={args.threads}; no download/training")
            load_started = perf_counter()
            model, tokenizer = load_model_and_tokenizer(args.model_id, revision=args.revision, cache_dir=args.cache_dir, allow_download=False)
            load_seconds = perf_counter() - load_started
            verify_tokenizer(manifest, tokenizer)
            model.eval().requires_grad_(False)
            layers = discover_layer_modules(model)
            write_json(args.output_dir / "model_runtime.json", {
                "requested_revision": args.revision, "config_resolved_revision": getattr(model.config, "_commit_hash", None),
                "attention_implementation": getattr(model.config, "_attn_implementation", None),
                "layers": [{"layer_index": layer.index, "attention_path": layer.attention_path, "mlp_path": layer.mlp_path} for layer in layers],
            })
            calibration_started = perf_counter()
            with (args.output_dir / "samples.jsonl").open("x", encoding="utf-8") as trace:
                def on_sample(record):
                    """恢复检查通过才写入一条轨迹；未完成样本不会冒充完整样本，异常则向上传播。"""
                    nonlocal completed
                    trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    trace.flush()
                    completed += 1
                    record_log(f"sample {completed}/{len(samples)}; id={record['sample_id']}; t*={record['token_id']}; q={record['q']:.9g}; restored=True; elapsed={perf_counter() - calibration_started:.3f}s")
                result = calibrate_model(model, samples, layers=layers, on_sample=on_sample)
            calibration_seconds = perf_counter() - calibration_started
            if any(file_sha256(ROOT / name) != digest for name, digest in evidence["code_sha256"].items()):
                raise RuntimeError("Source files changed during calibration; refuse mixed-version evidence")
            write_json(args.output_dir / "calibration.json", result)
            selection = write_calibration_artifacts(args.output_dir, result, top_k=args.top_k)
            # 最终标记覆盖除动态 run.log 外的所有已闭合产物；哈希本身不是来源认证或签名。
            files = {path.name: file_sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name != "run.log"}
            summary = {
                "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(), "n": result["n"],
                "forward_evaluations": result["forward_evaluations"], "selected_layer_indices": selection["selected_layer_indices"],
                "model_load_elapsed_seconds": load_seconds, "calibration_elapsed_seconds": calibration_seconds,
                "total_elapsed_seconds": perf_counter() - started, "completed_samples": completed, "files_sha256": files,
                "timing_scope": "CPU wall-clock; calibration includes parameter fingerprints, all ablations, restores and flushed sample logs; not a paper benchmark",
            }
            write_json(args.output_dir / "run_summary.json", summary)
            record_log(f"PASS n={result['n']}; forwards={result['forward_evaluations']}; ranked Top-{args.top_k}={selection['selected_layer_indices']}; calibration={calibration_seconds:.3f}s")
        except Exception as error:
            traceback.print_exc(file=log)
            log.flush()
            write_json(args.output_dir / "failure.json", {
                "status": "FAIL", "error_type": type(error).__name__, "error": str(error),
                "completed_samples": completed, "elapsed_seconds": perf_counter() - started,
            })
            raise


if __name__ == "__main__":
    main()
