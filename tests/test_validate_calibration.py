"""M4 独立验收脚本的反例测试；人工 JSON/CSV 夹具不是模型预测或真实来源认证。"""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_calibration", ROOT / "scripts/validate_calibration.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def save(path, value):
    """只给全新临时夹具写 JSON，不接触工作区真实校准输出。"""
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def rehash_run(directory):
    """重新建立夹具产物指纹，用于验证即使哈希被同步修改，数学/固定 token 门禁仍能拦错。"""
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    summary["files_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in directory.iterdir() if path.name != "run_summary.json"}
    save(directory / "run_summary.json", summary)


def fixture_run(root):
    """两输入、两层可手算夹具，I=[0.65,0.5]；指纹为假值，仅表示内部前后相等。"""
    directory = root / "run"
    directory.mkdir()
    samples = [{"sample_id": f"fixture/train/{index}", "index": index, "row_hash": "d" * 64,
                "input_hash": "e" * 64, "prompt_hash": "f" * 64, "input_ids": [index + 1, 2],
                "attention_mask": [1, 1], "prediction_position": 1, "token_count": 2} for index in range(2)]
    manifest = {"schema_version": 1, "dataset": {"id": "fixture"}, "sampling": {"n": 2},
                "template": {}, "tokenizer": {}, "samples": samples}
    manifest["content_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest_path = root / "manifest.json"
    save(manifest_path, manifest)
    records = []
    for sample, q, token, a, m in zip(samples, [0.6, 0.5], [1, 0], [[0.2, 0.8], [0.2, 0.8]], [[0.1, 0.5], [0.4, 0.1]]):
        records.append({"sample_id": sample["sample_id"], "index": sample["index"], "prediction_position": 1,
                        "token_id": token, "q": q, "restored_logits_equal": True, "forward_evaluations": 6,
                        "layers": [{"layer_index": index, "attention": {"token_id": token, "probability": a[index]},
                                    "mlp": {"token_id": token, "probability": m[index]}} for index in range(2)]})
    rows = [{"layer_index": 0, "importance": 0.65, "attention_mean_abs_delta": 0.35, "mlp_mean_abs_delta": 0.3},
            {"layer_index": 1, "importance": 0.5, "attention_mean_abs_delta": 0.25, "mlp_mean_abs_delta": 0.25}]
    fingerprint = {"sha256": "a" * 64, "parameter_elements": 10}
    result = {"n": 2, "layer_count": 2, "samples": records, "layer_importance": rows, "forward_evaluations": 12,
              "restoration_evaluations": 2, "parameter_fingerprint_before": fingerprint, "parameter_fingerprint_after": fingerprint,
              "all_parameters_frozen": True, "all_gradients_absent": True, "hooks_restored": True,
              "batch_size": 1, "use_cache": False, "probability_dtype": "torch.float32"}
    save(directory / "calibration.json", result)
    save(directory / "sample_ids.json", samples)
    save(directory / "resolved_config.json", {"n": 2, "manifest_n": 2, "top_k": 1, "layer_count": 2,
         "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
         "manifest_content_sha256": manifest["content_sha256"], "expected_forward_evaluations": 12,
         "batch_size": 1, "use_cache": False, "allow_download": False})
    save(directory / "provenance.json", {"code_sha256": {}, "git": {"baseline_commit": "fixture", "dirty": True}})
    enriched = [{**row, "rank": index + 1, "selected": index == 0} for index, row in enumerate(rows)]
    save(directory / "layer_importance.json", {"n": 2, "index_base": 0, "layers": enriched})
    save(directory / "selected_layers.json", {"index_base": 0, "top_k": 1, "layer_count": 2,
         "ranked_layer_indices": [0, 1], "selected_layer_indices": [0], "selected_layer_indices_ascending": [0]})
    with (directory / "layer_importance.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    (directory / "samples.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    save(directory / "run_summary.json", {"status": "PASS", "n": 2, "completed_samples": 2, "forward_evaluations": 12,
                                         "selected_layer_indices": [0], "files_sha256": {}})
    rehash_run(directory)
    return directory, manifest_path


class IndependentCalibrationAuditTests(unittest.TestCase):
    """不调用校准器/排序器来判断预期结果，专门测试保存后证据的二次核验。"""

    def validate(self, directory, manifest):
        return CHECKER.validate_run(directory, manifest, expected_n=2, expected_layers=2, top_k=1, check_source=False)

    def test_hand_calculated_saved_records_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest = fixture_run(Path(temporary))
            report = self.validate(directory, manifest)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["fixed_token_measurements"], 8)
            self.assertEqual(report["selected_layer_indices"], [0])
            self.assertAlmostEqual(report["recomputed_layer_importance"][0]["importance"], 0.65)

    def test_changed_file_without_new_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest = fixture_run(Path(temporary))
            (directory / "calibration.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.validate(directory, manifest)

    def test_rehashed_wrong_scores_token_or_parameter_audit_are_rejected(self):
        for change in ("score", "token", "parameters", "restore", "trace", "source_ids"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                directory, manifest = fixture_run(Path(temporary))
                result = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
                if change == "score":
                    result["layer_importance"][0]["importance"] = 0.1
                elif change == "token":
                    result["samples"][0]["layers"][0]["attention"]["token_id"] = 0
                elif change == "parameters":
                    result["parameter_fingerprint_after"]["sha256"] = "b" * 64
                elif change == "restore":
                    result["samples"][0]["restored_logits_equal"] = False
                elif change == "trace":
                    (directory / "samples.jsonl").write_text("{}\n", encoding="utf-8")
                else:
                    identities = json.loads((directory / "sample_ids.json").read_text(encoding="utf-8"))
                    identities[0]["input_ids"][0] = 2
                    save(directory / "sample_ids.json", identities)
                save(directory / "calibration.json", result)
                rehash_run(directory)
                with self.assertRaises(ValueError):
                    self.validate(directory, manifest)

    def test_rehashed_wrong_rank_or_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, manifest = fixture_run(Path(temporary))
            selected = json.loads((directory / "selected_layers.json").read_text(encoding="utf-8"))
            selected["selected_layer_indices"] = [1]
            save(directory / "selected_layers.json", selected)
            rehash_run(directory)
            with self.assertRaises(ValueError):
                self.validate(directory, manifest)


if __name__ == "__main__":
    unittest.main()
