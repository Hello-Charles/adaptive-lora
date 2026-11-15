"""M7候选协议的纯标准库测试，不下载数据、加载真实模型或执行CUDA。

测试从仓库配置派生pending/confirmed夹具；状态变化只验证控制流，不代表用户授权。
合成instruction和生成续文仅验证协议边界，不能作为真实benchmark结果或数据无泄漏证明。
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from adaptive_lora.m7_protocol import (
    TASK_LABELS, aggregate_commonsense, build_commonsense_prompt, build_preflight_report,
    effective_batch_size, extract_commonsense_answer, load_m7_config, protocol_sha256,
    score_commonsense, validate_m7_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen2_5_1_5b_commonsense_r16.json"


def draft():
    """每次生成独立pending夹具，不改变仓库配置，也不代表其实际确认状态。"""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for decision in config["decisions"]:
        decision["status"] = "pending"
    return config


class M7ConfigTests(unittest.TestCase):
    """只检查候选协议结构与论文固定值，不证明revision对象存在或下载字节已验证。"""

    def test_shipped_config_is_valid_but_never_implies_execution_authority(self):
        config = load_m7_config(CONFIG)
        self.assertIsNone(validate_m7_config(config))
        self.assertFalse(build_preflight_report(config)["execution_authorized"])

    def test_pending_fixture_validates_but_is_not_approved_or_executable(self):
        config = draft()
        before = copy.deepcopy(config)
        self.assertIsNone(validate_m7_config(config))
        report = build_preflight_report(config)
        self.assertEqual(report["review_status"], "PENDING_DECISIONS")
        self.assertEqual(len(report["pending_decisions"]), 7)
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["runtime_validated"])
        self.assertFalse(report["training_started"])
        self.assertEqual(config, before)

    def test_batch_and_forward_budgets_keep_single_gpu_distinct_from_ddp(self):
        self.assertEqual(effective_batch_size(8, 8, 1), 64)
        self.assertEqual(effective_batch_size(8, 8, 4), 256)
        for values in ((True, 8, 1), (8, 0, 1), (8, 8, -1), (8, 8, 1.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                effective_batch_size(*values)
        report = build_preflight_report(draft())
        self.assertEqual(report["effective_batch_size"], 64)
        self.assertEqual(report["paper_four_gpu_effective_batch_size"], 256)
        self.assertEqual(report["calibration_forward_budget"], {"scoring": 14250, "restore_checks": 250, "total": 14500})
        self.assertEqual(report["parameter_estimate"]["full"], 2981888)
        self.assertEqual(report["parameter_estimate"]["k_layer"], 851968)
        self.assertIn("not_measured", report["parameter_estimate"]["scope"])

    def test_hash_ignores_key_order_but_tracks_decisions_and_nested_values(self):
        config = draft()
        reordered = dict(reversed(list(config.items())))
        self.assertEqual(protocol_sha256(config), protocol_sha256(reordered))
        changed = copy.deepcopy(config)
        changed["decisions"][0]["status"] = "confirmed"
        self.assertNotEqual(protocol_sha256(config), protocol_sha256(changed))
        changed["training"]["learning_rate"] = float("nan")
        with self.assertRaises(ValueError):
            protocol_sha256(changed)

    def test_reviewed_fixture_never_grants_execution_or_runtime_readiness(self):
        config = draft()
        for decision in config["decisions"]:
            decision["status"] = "confirmed"
        report = build_preflight_report(config)
        self.assertEqual(report["review_status"], "REVIEWED_NOT_EXECUTABLE")
        self.assertEqual(report["pending_decisions"], [])
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["runtime_validated"])
        self.assertFalse(report["training_started"])
        self.assertTrue(report["unverified_execution_gates"])

    def test_m0_m6_or_algorithm_drift_is_rejected(self):
        cases = (
            ("model", "id", "Qwen/Qwen2.5-0.5B"), ("model", "revision", "main"),
            ("model", "layer_count", 24), ("model", "weights_lfs_sha256", "a" * 40),
            ("dataset", "calibration_samples", 30), ("dataset", "train_samples", 9999),
            ("training", "top_k", 7), ("training", "rank", 8),
            ("training", "target_projections", ["q", "v"]), ("training", "world_size", 4),
            ("training", "base_dtype", "bfloat16"), ("training", "quantization", "nf4"),
            ("training", "micro_batch_size", True), ("training", "gradient_accumulation", 0),
            ("training", "learning_rate", float("nan")), ("training", "use_cache", True),
            ("protocol", "calibration_mode", "path_patching"), ("protocol", "do_sample", True),
            ("protocol", "num_beams", 4), ("protocol", "max_new_tokens", 0),
            ("environment_candidate", "torch", "2.5.1+cu124"),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                config = draft()
                config[section][key] = value
                validate_m7_config(config)

    def test_unknown_fields_unsafe_paths_duplicate_tasks_and_bad_decisions_fail(self):
        for change in ("unknown", "path", "duplicate_task", "missing_task", "decision_id", "duplicate_decision", "status", "bool_seed"):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config = draft()
                if change == "unknown":
                    config["allow_download"] = True
                elif change == "path":
                    config["dataset"]["train_file"]["path"] = "../private.json"
                elif change == "duplicate_task":
                    config["dataset"]["evaluation_files"][1] = copy.deepcopy(config["dataset"]["evaluation_files"][0])
                elif change == "missing_task":
                    config["dataset"]["evaluation_files"].pop()
                elif change == "decision_id":
                    config["decisions"][0]["id"] = "anything"
                elif change == "duplicate_decision":
                    config["decisions"].append(copy.deepcopy(config["decisions"][0]))
                elif change == "status":
                    config["decisions"][0]["status"] = "approved_run_now"
                else:
                    config["dataset"]["train_sampling_seed"] = True
                validate_m7_config(config)

    def test_json_loader_rejects_duplicate_keys_nonfinite_and_nonobject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for text in ('{"schema_version":1,"schema_version":2}', '{"value":NaN}', '[]'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text, encoding="utf-8")
                    load_m7_config(path)
        self.assertEqual(load_m7_config(CONFIG), json.loads(CONFIG.read_text(encoding="utf-8")))


class CommonsensePromptTests(unittest.TestCase):
    """规范化Alpaca字节是复现决定，不宣称等同LLM-Adapters的空白或论文原模板。"""

    def test_prompt_has_no_answer_argument_and_explicit_response_boundary(self):
        prompt = build_commonsense_prompt("  Choose one.  ", "")
        self.assertEqual(prompt, "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nChoose one.\n\n### Response:\n")
        self.assertNotIn("### Input:", prompt)
        self.assertNotIn("<|im_start|>", prompt)
        with self.assertRaises(TypeError):
            build_commonsense_prompt("Question", "", answer="private reference")

    def test_input_branch_is_identical_for_calibration_and_generation(self):
        first = build_commonsense_prompt("Question", "  Context  ")
        self.assertIn("### Input:\nContext\n\n### Response:\n", first)
        self.assertEqual(first, build_commonsense_prompt("Question", "Context"))
        self.assertEqual(build_commonsense_prompt("Question", None), build_commonsense_prompt("Question", ""))

    def test_invalid_instruction_or_input_is_not_silently_stringified(self):
        for instruction, input_text in ((" ", ""), (None, ""), ("Question", 3)):
            with self.subTest(instruction=instruction, input_text=input_text), self.assertRaises(ValueError):
                build_commonsense_prompt(instruction, input_text)


class CommonsenseAnswerTests(unittest.TestCase):
    """只测试新生成文本的候选解析；不模拟实际分词、生成或真实任务分布。"""

    def test_each_task_accepts_its_own_casefolded_label(self):
        for task, labels in TASK_LABELS.items():
            for label in labels:
                with self.subTest(task=task, label=label):
                    result = extract_commonsense_answer(task, "The answer is " + label.upper() + ".")
                    self.assertEqual(result["label"], label)
                    self.assertEqual(result["status"], "valid")

    def test_word_boundaries_prevent_substring_matches(self):
        for task, text in (("boolq", "untrue falsehood"), ("piqa", "solution10"), ("ARC-Easy", "answer10"), ("hellaswag", "myending1")):
            with self.subTest(task=task):
                self.assertEqual(extract_commonsense_answer(task, text)["status"], "no_label")
        self.assertEqual(extract_commonsense_answer("social_i_qa", "answer4")["status"], "no_label")
        self.assertEqual(extract_commonsense_answer("openbookqa", "answer5")["status"], "no_label")

    def test_conflicting_labels_fail_but_repeated_same_label_is_unambiguous(self):
        self.assertEqual(extract_commonsense_answer("boolq", "true then false")["status"], "ambiguous")
        same = extract_commonsense_answer("boolq", "true. TRUE.")
        self.assertEqual(same["label"], "true")
        self.assertEqual(same["matched_labels"], ["true"])
        with self.assertRaises(ValueError):
            extract_commonsense_answer("unknown", "true")
        with self.assertRaises(ValueError):
            extract_commonsense_answer("boolq", None)

    def test_score_joins_by_id_and_keeps_invalid_predictions_in_denominator(self):
        predictions = [{"sample_id": "b", "generated_text": ""}, {"sample_id": "a", "generated_text": "TRUE"}, {"sample_id": "c", "generated_text": "true or false"}]
        references = [{"sample_id": "a", "answer": "true"}, {"sample_id": "b", "answer": "false"}, {"sample_id": "c", "answer": "true"}]
        result = score_commonsense("boolq", predictions, references)
        self.assertEqual((result["correct"], result["total"]), (1, 3))
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual([row["sample_id"] for row in result["rows"]], ["b", "a", "c"])
        self.assertEqual([row["parse_status"] for row in result["rows"]], ["no_label", "valid", "ambiguous"])
        changed = copy.deepcopy(references)
        changed[0]["answer"] = "false"
        self.assertEqual(score_commonsense("boolq", predictions, changed)["correct"], 0)
        self.assertEqual(predictions[1]["generated_text"], "TRUE")

    def test_duplicate_missing_or_invalid_reference_and_extra_fields_fail(self):
        prediction = {"sample_id": "a", "generated_text": "true"}
        reference = {"sample_id": "a", "answer": "true"}
        cases = (([prediction, prediction], [reference]), ([prediction], []),
                 ([prediction], [{**reference, "sample_id": "b"}]),
                 ([prediction], [{**reference, "answer": "yes"}]),
                 ([{**prediction, "prompt": "Do not parse prompt options"}], [reference]),
                 ([{**prediction, "generated_text": None}], [reference]))
        for predictions, references in cases:
            with self.subTest(predictions=predictions, references=references), self.assertRaises(ValueError):
                score_commonsense("boolq", predictions, references)

    def test_eight_task_macro_is_not_weighted_by_example_count(self):
        scores = [{"task": task, "correct": 1 if task != "openbookqa" else 0,
                   "total": 1 if task != "openbookqa" else 500,
                   "accuracy": 1.0 if task != "openbookqa" else 0.0} for task in TASK_LABELS]
        result = aggregate_commonsense(scores)
        self.assertEqual(result["macro_accuracy"], 7 / 8)
        self.assertEqual(result["total_examples"], 507)
        for broken in (scores[:-1], scores + [scores[0]], [{**row, "accuracy": 0.5} for row in scores]):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                aggregate_commonsense(broken)


if __name__ == "__main__":
    unittest.main()
