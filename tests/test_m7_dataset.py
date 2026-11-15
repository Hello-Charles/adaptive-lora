"""M7离线数据审计的合成JSON测试，不下载/读取真实benchmark或验证真实tokenizer。

夹具只证明字节、字段、抽样与文本身份控制流；不能证明真实内容权利、语义隔离或模型效果。
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from adaptive_lora.m7_dataset import (audit_dataset, canonical_sha256, overlap_summary,
    read_verified_rows, sample_indices, selected_records, source_path, validate_rows)
from adaptive_lora.m7_protocol import TASK_LABELS


def save_source(root, path, rows):
    """自建小型JSON与Git blob，原字节仅由本测试控制，不来自网络。"""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    target.write_bytes(raw)
    return {"path": path, "size_bytes": len(raw),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}


def dataset_fixture(root, *, overlapping=False):
    """八任务夹具各四条，抽两条；训练八条抽五条，校准来自其中三条。"""
    train = [{"instruction": f"training {i}", "input": "", "output": "synthetic answer"} for i in range(8)]
    spec = {"revision": "f" * 40, "train_file": save_source(root, "train.json", train),
            "train_samples": 5, "calibration_samples": 3, "train_sampling_seed": 42,
            "calibration_sampling_seed": 42, "evaluation_sampling_seed": 42, "evaluation_files": []}
    for task, labels in TASK_LABELS.items():
        rows = [{"instruction": f"{task} question {i}", "answer": labels[0]} for i in range(4)]
        if overlapping:
            # 所有候选都复用已选训练首行输入，确保原抽样无论选哪条都会暴露直接重叠。
            picked = sample_indices(8, 5, 42)[0]
            rows = [{"instruction": train[picked]["instruction"], "answer": labels[0]} for _ in range(4)]
        spec["evaluation_files"].append({**save_source(root, task + ".json", rows), "task": task, "sample_count": 2})
    return spec


class M7DatasetTests(unittest.TestCase):
    """测试源路径/JSON/身份和局部RNG，不用真实模型或网络替代品冒充实际运行验收。"""
    def test_git_blob_and_sha256_are_distinct_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = save_source(root, "x.json", [{"instruction": "x", "output": "a"}])
            rows, info = read_verified_rows(root, spec)
            self.assertEqual(len(rows), 1)
            self.assertEqual(info["git_blob_sha1"], spec["git_blob_sha1"])
            self.assertEqual(len(info["sha256"]), 64)

    def test_bad_size_or_blob_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec=save_source(root,"x.json",[1])
            for changed in ({**spec,"size_bytes":spec["size_bytes"]+1},{**spec,"git_blob_sha1":"0"*40}):
                with self.assertRaises(ValueError):read_verified_rows(root,changed)

    def test_duplicate_keys_nonfinite_and_nonlist_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for raw in (b'[{"instruction":"a","instruction":"b"}]',b'[NaN]',b'{}',b'[]'):
                (root/"x.json").write_bytes(raw)
                spec={"path":"x.json","size_bytes":len(raw),"git_blob_sha1":hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
                with self.subTest(raw=raw),self.assertRaises(ValueError):read_verified_rows(root,spec)

    def test_paths_reject_traversal_drive_absolute_and_empty_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../x","/x","C:/x","a\\b","a//b","a/./b",""):
                with self.subTest(name=name),self.assertRaises(ValueError):source_path(directory,name)
            self.assertEqual(source_path(directory,"safe/x.json"),Path(directory)/"safe/x.json")

    def test_size_bound_and_bool_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in (True,0,-1,100_000_001):
                with self.assertRaises(ValueError):read_verified_rows(directory,{"path":"x","size_bytes":size})

    def test_sampling_matches_local_random_and_preserves_global_rng(self):
        state=random.getstate()
        self.assertEqual(sample_indices(100,10,42),random.Random(42).sample(range(100),10))
        self.assertEqual(random.getstate(),state)
        self.assertEqual(len(set(sample_indices(100,10,42))),10)

    def test_invalid_sampling_does_not_silently_reduce_count(self):
        for values in ((2,3,42),(0,1,42),(3,0,42),(3,True,42),(3,2,-1)):
            with self.subTest(values=values),self.assertRaises(ValueError):sample_indices(*values)

    def test_training_output_and_input_types_are_validated(self):
        for row in ({"instruction":"","output":"a"},{"instruction":"q","input":[] ,"output":"a"},{"instruction":"q","answer":"a"},{"instruction":"q","output":" "}):
            with self.assertRaises(ValueError):validate_rows([row])
        self.assertEqual(validate_rows([{"instruction":"q","input":None,"output":"a"}]),{"input,instruction,output":1})

    def test_eval_uses_answer_not_output_and_exact_task_labels(self):
        self.assertEqual(validate_rows([{"instruction":"q","answer":"true","output":"ignored metadata"}],"boolq"),{"answer,instruction,output":1})
        for label in ("yes","True"," true ","solution1"):
            with self.assertRaises(ValueError):validate_rows([{"instruction":"q","answer":label}],"boolq")

    def test_answer_changes_row_hash_but_not_user_prompt_identity(self):
        rows=[{"instruction":"question","input":"context","output":"gold one"}]
        first=selected_records(rows,[0],"train.json","a"*40)[0]
        rows[0]["output"]="gold two"
        second=selected_records(rows,[0],"train.json","a"*40)[0]
        self.assertEqual(first["prompt_sha256"],second["prompt_sha256"])
        self.assertNotEqual(first["row_sha256"],second["row_sha256"])
        self.assertNotIn("output",first)

    def test_calibration_uses_training_positions_and_original_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=dataset_fixture(Path(directory));result=audit_dataset(spec,directory)
            sample=result["sampling"]
            self.assertEqual(sample["calibration_source_indices"],[sample["train_indices"][i] for i in sample["calibration_training_positions"]])
            self.assertTrue(result["overlap"]["calibration_is_training_subset"])
            self.assertEqual(len(result["train"]),5)
            self.assertEqual(len(result["calibration"]),3)

    def test_repeat_audit_is_identical_without_tokenizer_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=dataset_fixture(Path(directory))
            first=audit_dataset(spec,directory);second=audit_dataset(spec,directory)
            self.assertEqual(first,second)
            self.assertEqual(first["status"],"TEXT_GATE_PASS_TOKENIZER_PENDING")
            self.assertFalse(first["tokenizer_validated"])
            self.assertFalse(first["content_rights_individually_cleared"])

    def test_text_overlap_blocks_but_does_not_resample(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=dataset_fixture(Path(directory),overlapping=True);result=audit_dataset(spec,directory)
            self.assertEqual(result["status"],"BLOCKED_TEXT_OVERLAP")
            self.assertEqual(result["sampling"]["train_indices"],sample_indices(8,5,42))
            self.assertEqual(len(result["evaluation"]["boolq"]),2)
            self.assertEqual(len(result["overlap"]["per_task"]["boolq"]["train_prompt_sha256_intersection"]),1)
            self.assertEqual(result["overlap"]["per_task"]["boolq"]["train_sample_id_intersection"],[])

    def test_same_prompt_different_source_id_is_not_missed(self):
        train=[{"sample_id":"train:1","prompt_sha256":"a"}]
        evaluation=[{"sample_id":"eval:2","prompt_sha256":"a"}]
        result=overlap_summary(train,train,{"task":evaluation})
        self.assertFalse(result["text_disjoint"])

    def test_duplicate_task_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=dataset_fixture(Path(directory));spec["evaluation_files"].append(copy.deepcopy(spec["evaluation_files"][0]))
            with self.assertRaisesRegex(ValueError,"Duplicate evaluation task"):audit_dataset(spec,directory)

    def test_canonical_hash_ignores_dict_key_order_not_list_order(self):
        self.assertEqual(canonical_sha256({"a":1,"b":2}),canonical_sha256({"b":2,"a":1}))
        self.assertNotEqual(canonical_sha256([1,2]),canonical_sha256([2,1]))


if __name__=="__main__":
    unittest.main()
