"""独立核验 M4 已保存产物；只读文件、不加载模型、不导入校准/选层实现。

从原始逐样本概率重新计算公式 (5) 和排名，同时检查清单、文件字节、固定 token、恢复、
参数及 JSONL/CSV/JSON 一致性。它不能重新证明模型实际前向，须结合核心测试和真实运行日志。
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETER_SHA256 = "3e59c59f388044257c042e93b86fc1a0064dd9af6cd106c102cc685469427db3"
MANIFEST_KEYS = ("schema_version", "dataset", "sampling", "template", "tokenizer", "samples")
IDENTITY_KEYS = ("sample_id", "index", "row_hash", "input_hash", "prompt_hash", "input_ids", "attention_mask", "prediction_position", "token_count")


def require(condition, message):
    """验收使用显式异常而非 assert，不能被 python -O 静默禁用。"""
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    """计算原始字节；文本行尾变化也算不同版本，不做掩盖差异的自动归一化。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    """只解析 JSON 数据；不从产物目录加载 Python、模板代码或包。"""
    return json.loads(Path(path).read_bytes())


def close_number(actual, expected, label):
    """重算使用独立数据路径，按 1e-12 容差比较 Python 双精度结果而非字符串格式。"""
    require(type(actual) in (float, int) and math.isfinite(actual), f"{label}: expected finite number")
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: recomputed value differs")


def probability(value):
    """拒绝布尔、NaN/Infinity 和越界概率，不把损坏记录当作零贡献。"""
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, "Invalid probability")
    return float(value)


def check_layer_rows(actual, expected, *, ranks=None, selected=None):
    """比较完整按层序的均值及可选导出排名；不调用被测聚合器或 Top-K helper。"""
    require(len(actual) == len(expected), "Missing layer importance rows")
    for index, (row, wanted) in enumerate(zip(actual, expected)):
        require(type(row["layer_index"]) is int and row["layer_index"] == index, "Layer row order/index mismatch")
        for key in ("importance", "attention_mean_abs_delta", "mlp_mean_abs_delta"):
            close_number(row[key], wanted[key], f"layer {index} {key}")
        if ranks is not None:
            require(row["rank"] == ranks[index] and row["selected"] is (index in selected), "Exported rank/selection mismatch")


def validate_run(directory, manifest_path, *, expected_n, expected_layers=24, top_k=7,
                 expected_parameter_sha256=None, compare_prefix_run=None, check_source=True):
    """独立复算一次完整运行，返回无原始 prompt/答案的验收记录；输入文件须在期间只读。

    文件/源码 SHA 是一致性检测而非来源认证。check_source 比较运行前记录的源码原始字节，
    不要求新增的后处理脚本曾存在于旧快照；跳过时明确返回未核验，不能声称版本相同。
    compare_prefix_run 仅允许另一目录的 N=1 运行，只证明首样本轨迹一致，不冒充 N=30 重跑。
    """
    directory, manifest_path = Path(directory), Path(manifest_path)
    summary = read_json(directory / "run_summary.json")
    require(summary["status"] == "PASS" and not (directory / "failure.json").exists(), "Run did not finish successfully")
    fingerprints = summary["files_sha256"]
    required = {"calibration.json", "sample_ids.json", "resolved_config.json", "provenance.json", "samples.jsonl",
                "selected_layers.json", "layer_importance.csv", "layer_importance.json"}
    require(required <= set(fingerprints), "Missing required artifact fingerprints")
    for name, digest in fingerprints.items():
        require(Path(name).name == name and name not in (".", ".."), "Unsafe artifact filename")
        require(file_sha256(directory / name) == digest, f"Artifact SHA-256 mismatch: {name}")
    config = read_json(directory / "resolved_config.json")
    manifest = read_json(manifest_path)
    payload = {key: manifest[key] for key in MANIFEST_KEYS}
    content = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    require(content == manifest["content_sha256"] == config["manifest_content_sha256"], "Manifest content SHA-256 mismatch")
    require(file_sha256(manifest_path) == config["manifest_file_sha256"], "Manifest file SHA-256 mismatch")
    require(type(expected_n) is int and 1 <= expected_n <= len(manifest["samples"]), "Invalid expected_n")
    require(type(expected_layers) is int and expected_layers > 0 and type(top_k) is int and 1 <= top_k <= expected_layers, "Invalid layer/K expectation")
    require(config["n"] == expected_n and config["manifest_n"] == len(manifest["samples"]), "Config sample count mismatch")
    require(config["top_k"] == top_k and config["layer_count"] == expected_layers, "Config layer count/Top-K mismatch")
    require(config["batch_size"] == 1 and config["use_cache"] is False and config["allow_download"] is False, "Unexpected batching/cache/download configuration")
    identities = read_json(directory / "sample_ids.json")
    expected_ids = [{key: sample[key] for key in IDENTITY_KEYS} for sample in manifest["samples"][:expected_n]]
    require(identities == expected_ids, "Saved sample identities/IDs differ from fixed manifest prefix")
    require(len({sample["sample_id"] for sample in identities}) == expected_n, "Duplicate sample IDs")
    result = read_json(directory / "calibration.json")
    require(result["n"] == expected_n and result["layer_count"] == expected_layers, "Core sample/layer count mismatch")
    records = result["samples"]
    require(len(records) == expected_n, "Missing completed sample records")
    expected_forwards = expected_n * (2 * expected_layers + 2)
    require(result["forward_evaluations"] == config["expected_forward_evaluations"] == summary["forward_evaluations"] == expected_forwards,
            "Forward count does not include each baseline, independent intervention and restore")
    require(summary["n"] == summary["completed_samples"] == expected_n and result["restoration_evaluations"] == expected_n, "Restore/completion count mismatch")
    require(all(result[key] is True for key in ("all_parameters_frozen", "all_gradients_absent", "hooks_restored")), "Frozen/gradient/hook audit failed")
    require(result["batch_size"] == 1 and result["use_cache"] is False and result["probability_dtype"] == "torch.float32", "Unexpected probability/forward contract")
    before, after = result["parameter_fingerprint_before"], result["parameter_fingerprint_after"]
    require(before == after and before["parameter_elements"] > 0, "Parameter fingerprint changed")
    require(re.fullmatch(r"[0-9a-f]{64}", before["sha256"]) is not None, "Invalid parameter fingerprint")
    if expected_parameter_sha256 is not None:
        require(before["sha256"] == expected_parameter_sha256, "Parameter fingerprint differs from expected M3 base")
    attention = [[] for _ in range(expected_layers)]
    mlp = [[] for _ in range(expected_layers)]
    for record, identity in zip(records, identities):
        require(all(record[key] == identity[key] for key in ("sample_id", "index", "prediction_position")), "Sample record identity/position mismatch")
        token = record["token_id"]
        require(type(token) is int and token >= 0, "Original token_id is invalid")
        q = probability(record["q"])
        require(record["restored_logits_equal"] is True and record["forward_evaluations"] == 2 * expected_layers + 2, "Sample restoration/forward audit failed")
        rows = record["layers"]
        require(len(rows) == expected_layers and [row["layer_index"] for row in rows] == list(range(expected_layers)), "Missing/duplicate intervention layers")
        for index, row in enumerate(rows):
            for component, target in (("attention", attention), ("mlp", mlp)):
                measurement = row[component]
                require(type(measurement["token_id"]) is int and measurement["token_id"] == token, "Intervention changed the original reference token")
                target[index].append(abs(q - probability(measurement["probability"])))
    # 直接从原始概率得到逐样本绝对差，再平均；不读取任何保存的 contribution/排名作为期望值。
    recomputed = [{"layer_index": index, "importance": math.fsum(a + m for a, m in zip(attention[index], mlp[index])) / expected_n,
                   "attention_mean_abs_delta": math.fsum(attention[index]) / expected_n,
                   "mlp_mean_abs_delta": math.fsum(mlp[index]) / expected_n} for index in range(expected_layers)]
    check_layer_rows(result["layer_importance"], recomputed)
    order = sorted(range(expected_layers), key=lambda index: (-recomputed[index]["importance"], index))
    selected = order[:top_k]
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    selection = read_json(directory / "selected_layers.json")
    require(selection["index_base"] == 0 and selection["top_k"] == top_k and selection["layer_count"] == expected_layers, "Selection metadata mismatch")
    for key, wanted in (("ranked_layer_indices", order), ("selected_layer_indices", selected), ("selected_layer_indices_ascending", sorted(selected))):
        require(all(type(index) is int for index in selection[key]) and selection[key] == wanted, f"Selection mismatch: {key}")
    require(summary["selected_layer_indices"] == selected, "Summary selection mismatch")
    exported = read_json(directory / "layer_importance.json")
    require(exported["n"] == expected_n and exported["index_base"] == 0, "Importance JSON metadata mismatch")
    check_layer_rows(exported["layers"], recomputed, ranks=ranks, selected=selected)
    with (directory / "layer_importance.csv").open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    require(len(csv_rows) == expected_layers, "CSV layer count mismatch")
    for index, row in enumerate(csv_rows):
        require(int(row["layer_index"]) == index and int(row["rank"]) == ranks[index] and row["selected"] == str(index in selected), "CSV index/rank/selection mismatch")
        for key in ("importance", "attention_mean_abs_delta", "mlp_mean_abs_delta"):
            close_number(float(row[key]), recomputed[index][key], f"CSV layer {index} {key}")
    trace = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    require(trace == records, "JSONL progress trace differs from final raw records")
    evidence = read_json(directory / "provenance.json")
    if check_source:
        require(bool(evidence["code_sha256"]), "No source fingerprint evidence")
        for name, digest in evidence["code_sha256"].items():
            parts = PurePosixPath(name).parts
            require(parts and parts[0] in ("src", "scripts", "tests") and ".." not in parts and (ROOT / name).resolve().is_relative_to(ROOT), "Unsafe source path")
            require(file_sha256(ROOT / name) == digest, f"Current source SHA-256 mismatch: {name}")
    repeatability = {"full_n30_repeated": False, "n1_matches_first_sample": None}
    if compare_prefix_run is not None:
        comparison = Path(compare_prefix_run)
        require(comparison.resolve() != directory.resolve(), "Comparison must be a distinct N=1 run")
        validate_run(comparison, manifest_path, expected_n=1, expected_layers=expected_layers, top_k=top_k,
                     expected_parameter_sha256=expected_parameter_sha256, check_source=check_source)
        prefix = read_json(comparison / "calibration.json")
        require(records[0] == prefix["samples"][0], "Independent N=1 run does not reproduce the first full-run sample")
        other_config = read_json(comparison / "resolved_config.json")
        for key in ("model_id", "requested_revision", "manifest_content_sha256", "inference_seed", "sampling_seed", "cpu_threads", "deterministic_algorithms"):
            require(config[key] == other_config[key], f"N=1 comparison config differs: {key}")
        other_evidence = read_json(comparison / "provenance.json")
        runtime_sources = ("src/adaptive_lora/calibration.py", "src/adaptive_lora/ablation.py", "src/adaptive_lora/model_loader.py",
                           "src/adaptive_lora/data.py", "src/adaptive_lora/selection.py", "src/adaptive_lora/artifacts.py", "scripts/run_calibration.py")
        require(all(evidence["code_sha256"][name] == other_evidence["code_sha256"][name] for name in runtime_sources), "Runtime source versions differ between runs")
        repeatability = {"full_n30_repeated": False, "n1_matches_first_sample": True,
                         "scope": "Independent N=1 vs the first full-run sample; not a repeated N=30 calibration",
                         "comparison_directory": str(comparison)}
    return {
        "status": "PASS", "captured_at_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(directory),
        "manifest_path": str(manifest_path), "manifest_content_sha256": content, "n": expected_n, "layer_count": expected_layers,
        "fixed_token_measurements": expected_n * expected_layers * 2, "forward_evaluations": expected_forwards,
        "full_logits_restorations": expected_n, "parameter_fingerprint": before, "top_k": top_k,
        "selected_layer_indices": selected, "ranked_layer_indices": order, "recomputed_layer_importance": recomputed,
        "verified_artifact_count": len(fingerprints), "source_hashes_match_current": True if check_source else None,
        "recorded_source_count": len(evidence["code_sha256"]), "git": evidence["git"], "repeatability": repeatability,
        "validator_source_sha256": file_sha256(Path(__file__)),
        "limitations": "Audits saved evidence, not a new model forward or source authentication; parameter fingerprints exclude buffers.",
    }


def main():
    """只写新的验收 JSON；校验失败非零退出且不生成成功记录，不改变任何原产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/m2-20261114/manifest.json")
    parser.add_argument("--expected-n", type=int, default=30)
    parser.add_argument("--expected-layers", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--expected-parameter-sha256", default=DEFAULT_PARAMETER_SHA256)
    parser.add_argument("--compare-prefix-run", type=Path)
    parser.add_argument("--skip-source-check", action="store_true", help="Explicitly leave current-source identity unverified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        report = validate_run(args.run_dir, args.manifest, expected_n=args.expected_n, expected_layers=args.expected_layers,
                              top_k=args.top_k, expected_parameter_sha256=args.expected_parameter_sha256,
                              compare_prefix_run=args.compare_prefix_run, check_source=not args.skip_source_check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: calibration artifact verification: {error}\n")
    report["command"] = sys.argv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"PASS: n={report['n']}; {report['fixed_token_measurements']} fixed-token measurements; Top-{args.top_k}={report['selected_layer_indices']}")
    print(f"Independent calibration audit written: {args.output}")


if __name__ == "__main__":
    main()
