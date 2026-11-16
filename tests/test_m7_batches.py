"""M7 数据→micro-batch 装配层的测试；不训练、不用 GPU、不下载。

测试能力边界：

- 顺序与切分逻辑用合成样本验证，不依赖 tokenizer。
- `prepare_supervised_examples` 用本地已缓存的固定 0.5B 真实分词器验证 BPE 边界与
  样本身份补全；缺缓存时明确跳过并计数，不伪造通过。
- 本文件不加载模型、不做前向、不写文件，因此不能证明批训练数值正确。
"""

import random
from pathlib import Path
import unittest

from adaptive_lora.m7_batches import (
    ORDER_POLICIES,
    epoch_order,
    iter_epoch_micro_batches,
    plan_epoch_accumulation,
    prepare_supervised_examples,
)
from adaptive_lora.m7_protocol import build_commonsense_prompt

CACHE = Path(__file__).resolve().parents[1] / "data/models/qwen2_5_0_5b_060db6499f32faf8b98477b0a26969ef7d8b9987"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def load_tokenizer():
    """只在本地固定缓存可用时加载；返回 None 表示无法运行真实分词测试。"""
    if not CACHE.is_dir():
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", revision=REVISION, cache_dir=str(CACHE),
        local_files_only=True, use_fast=True,
    )


TOKENIZER = load_tokenizer()
REQUIRES_TOKENIZER = unittest.skipUnless(TOKENIZER is not None, "cached fixed tokenizer unavailable")


def fake_examples(count):
    """构造只含身份的最小样本；用于顺序与切分测试，不含真实 token。"""
    return [{"sample_id": f"sample-{index:03d}"} for index in range(count)]


class EpochOrderTests(unittest.TestCase):
    """顺序策略必须是显式的、可复现的，且不依赖全局 random 状态。"""

    def test_none_policy_keeps_original_order(self):
        for epoch in (1, 2, 3):
            self.assertEqual(epoch_order(5, epoch=epoch, policy="none"), [0, 1, 2, 3, 4])

    def test_shuffle_returns_a_permutation(self):
        order = epoch_order(20, epoch=1, policy="per_epoch_shuffle", seed=42)
        self.assertEqual(sorted(order), list(range(20)))
        self.assertNotEqual(order, list(range(20)))

    def test_different_epochs_shuffle_differently(self):
        first = epoch_order(50, epoch=1, policy="per_epoch_shuffle", seed=42)
        second = epoch_order(50, epoch=2, policy="per_epoch_shuffle", seed=42)
        self.assertNotEqual(first, second, "per-epoch shuffling must differ across epochs")

    def test_same_seed_and_epoch_reproduce(self):
        left = epoch_order(30, epoch=2, policy="per_epoch_shuffle", seed=42)
        right = epoch_order(30, epoch=2, policy="per_epoch_shuffle", seed=42)
        self.assertEqual(left, right)

    def test_different_seeds_differ(self):
        left = epoch_order(30, epoch=1, policy="per_epoch_shuffle", seed=42)
        right = epoch_order(30, epoch=1, policy="per_epoch_shuffle", seed=43)
        self.assertNotEqual(left, right)

    def test_independent_of_global_random_state(self):
        """局部 RNG：调用方是否动过全局 random 都不得改变结果。"""
        random.seed(999)
        first = epoch_order(25, epoch=1, policy="per_epoch_shuffle", seed=42)
        random.seed(1)
        second = epoch_order(25, epoch=1, policy="per_epoch_shuffle", seed=42)
        self.assertEqual(first, second)

    def test_invalid_arguments_rejected(self):
        for count in (0, -1, 1.0, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                epoch_order(count, epoch=1)
        for epoch in (0, -1, 1.0, True):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                epoch_order(5, epoch=epoch)
        with self.assertRaisesRegex(ValueError, "Unknown order policy"):
            epoch_order(5, epoch=1, policy="shuffle-everything")
        for seed in (-1, 2 ** 32, 1.0, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                epoch_order(5, epoch=1, policy="per_epoch_shuffle", seed=seed)
        self.assertIn("none", ORDER_POLICIES)


class MicroBatchTests(unittest.TestCase):
    """切分必须保持样本不丢、顺序可核，且余量在装配阶段就拒绝。"""

    def test_even_split(self):
        batches, plan = iter_epoch_micro_batches(
            fake_examples(16), epoch=1, micro_batch_size=8)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 8 for batch in batches))
        self.assertEqual(plan["short_micro_batch_sizes"], [])
        self.assertEqual(plan["samples"], 16)
        self.assertEqual(plan["full_micro_batches"], 2)

    def test_no_sample_is_lost_or_duplicated(self):
        examples = fake_examples(17)
        batches, plan = iter_epoch_micro_batches(
            examples, epoch=1, micro_batch_size=8, final_partial_group="step")
        seen = [example["sample_id"] for batch in batches for example in batch]
        self.assertEqual(sorted(seen), sorted(example["sample_id"] for example in examples))
        self.assertEqual(len(seen), len(set(seen)))

    def test_error_policy_rejects_short_final_batch(self):
        # error 必须仍然可用：默认已改为 drop，因此这里显式传入。
        with self.assertRaisesRegex(ValueError, "short micro-batch|would hold"):
            iter_epoch_micro_batches(
                fake_examples(10), epoch=1, micro_batch_size=8, final_partial_group="error")

    def test_drop_policy_discards_with_an_auditable_list(self):
        """已确认决策：drop 真正丢弃末尾不足一批的样本，并留下完整清单。

        "丢了哪几条"必须能事后追溯，否则"用了 10,000 条"无法核对。
        """
        examples = fake_examples(10)
        batches, plan = iter_epoch_micro_batches(
            examples, epoch=1, micro_batch_size=8, final_partial_group="drop")
        self.assertEqual([len(batch) for batch in batches], [8])
        self.assertEqual(plan["samples_used"], 8)
        self.assertEqual(plan["samples_dropped"], 2)
        # 抽样顺序策略下被丢的是末尾两条。
        self.assertEqual(plan["dropped_sample_ids"], ["sample-008", "sample-009"])
        seen = [example["sample_id"] for batch in batches for example in batch]
        self.assertEqual(seen, ["sample-000", "sample-001", "sample-002", "sample-003",
                                "sample-004", "sample-005", "sample-006", "sample-007"])
        # 清单本身也要有指纹，便于与训练日志交叉核对。
        self.assertEqual(len(plan["dropped_sample_ids_sha256"]), 64)
        self.assertNotEqual(plan["used_order_sha256"], plan["order_sha256"])

    def test_drop_is_the_confirmed_default(self):
        batches, plan = iter_epoch_micro_batches(fake_examples(10), epoch=1, micro_batch_size=8)
        self.assertEqual([len(batch) for batch in batches], [8])
        self.assertEqual(plan["final_partial_group"], "drop")
        self.assertEqual(plan["samples_dropped"], 2)

    def test_step_policy_allows_short_final_batch(self):
        batches, plan = iter_epoch_micro_batches(
            fake_examples(10), epoch=1, micro_batch_size=8, final_partial_group="step")
        self.assertEqual([len(batch) for batch in batches], [8, 2])
        self.assertEqual(plan["short_micro_batch_sizes"], [2])
        # step 不丢样本，清单必须为空。
        self.assertEqual(plan["samples_dropped"], 0)
        self.assertEqual(plan["dropped_sample_ids"], [])

    def test_order_sha256_reflects_the_actual_order(self):
        examples = fake_examples(8)
        _, sequential = iter_epoch_micro_batches(examples, epoch=1, micro_batch_size=8)
        _, shuffled = iter_epoch_micro_batches(
            examples, epoch=1, micro_batch_size=8, order_policy="per_epoch_shuffle", seed=42)
        # 样本集合相同，但访问顺序不同 → 两个指纹必须不同。
        self.assertEqual(sequential["sample_ids_sha256"], shuffled["sample_ids_sha256"])
        self.assertNotEqual(sequential["order_sha256"], shuffled["order_sha256"])

    def test_paper_shape_plan(self):
        """10,000 条 / micro 8 = 1,250 个 micro-batch，累积 8 → 156 组 + 余 2。"""
        examples = fake_examples(10000)
        batches, plan = iter_epoch_micro_batches(
            examples, epoch=1, micro_batch_size=8, final_partial_group="step")
        self.assertEqual(plan["micro_batches"], 1250)
        self.assertEqual(plan["short_micro_batch_sizes"], [])
        accumulation = plan_epoch_accumulation(plan, accumulation=8, final_partial_group="step")
        self.assertEqual(accumulation["accumulation_groups"], 157)
        self.assertEqual(accumulation["last_group_micro_batches"], 2)
        self.assertFalse(accumulation["divides_evenly"])
        self.assertEqual(accumulation["samples_effective_batch_nominal"], 64)

    def test_paper_shape_under_confirmed_drop_policy(self):
        """已确认 drop 下的论文形状：每 epoch 156 步、丢 16 条，3 epochs 共丢 48 条。"""
        _, plan = iter_epoch_micro_batches(fake_examples(10000), epoch=1, micro_batch_size=8)
        # 10,000 条正好切成 1,250 个满 micro-batch，micro-batch 层不丢样本。
        self.assertEqual(plan["micro_batches"], 1250)
        self.assertEqual(plan["samples_dropped"], 0)
        accumulation = plan_epoch_accumulation(plan, accumulation=8)
        self.assertEqual(accumulation["final_partial_group"], "drop")
        self.assertEqual(accumulation["accumulation_groups"], 156)
        self.assertEqual(accumulation["optimizer_steps"], 156)
        self.assertEqual(accumulation["micro_batches_dropped"], 2)
        self.assertEqual(accumulation["samples_used_by_accumulation"], 156 * 8 * 8)
        self.assertEqual(accumulation["samples_dropped_by_accumulation"], 16)
        # 训练实际使用的样本数必须与"10,000 条"分开陈述。
        self.assertNotEqual(accumulation["samples_used_by_accumulation"], accumulation["samples"])

    def test_accumulation_counts_actual_batch_sizes_not_nominal(self):
        """`step` 策略下末尾是短批，样本数必须按实际大小求和。

        10 条 / micro 8 → 批大小 [8,2]。若用"批数 × 名义批大小"估算会得到 16 条，
        把参与训练的样本数多算 6 条；这里断言三项守恒：已用 + 已丢 + 未记账 = 总数。
        """
        examples = fake_examples(10)
        _, plan = iter_epoch_micro_batches(
            examples, epoch=1, micro_batch_size=8, final_partial_group="step")
        self.assertEqual(plan["micro_batch_sizes"], [8, 2])
        accumulation = plan_epoch_accumulation(plan, accumulation=8, final_partial_group="step")
        self.assertEqual(accumulation["samples_used_by_accumulation"], 10)
        self.assertEqual(accumulation["samples_dropped_by_accumulation"], 0)
        self.assertEqual(accumulation["samples_unaccounted"], 0)
        # 与"名义批大小 × 批数"的错误算法明确区分。
        self.assertNotEqual(accumulation["samples_used_by_accumulation"], len(plan["micro_batch_sizes"]) * 8)

    def test_accumulation_error_policy_surfaces_leftover(self):
        examples = fake_examples(10000)
        _, plan = iter_epoch_micro_batches(examples, epoch=1, micro_batch_size=8)
        with self.assertRaisesRegex(ValueError, "do not divide into groups"):
            plan_epoch_accumulation(plan, accumulation=8, final_partial_group="error")

    def test_invalid_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one example"):
            iter_epoch_micro_batches([], epoch=1, micro_batch_size=8)
        for size in (0, -1, 1.0, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                iter_epoch_micro_batches(fake_examples(8), epoch=1, micro_batch_size=size)
        with self.assertRaises(ValueError):
            plan_epoch_accumulation({"micro_batches": 4}, accumulation=0)


@REQUIRES_TOKENIZER
class PrepareExamplesTests(unittest.TestCase):
    """真实分词器下的样本准备：身份补全、训练目标字段与错误不静默跳过。"""

    def _rows(self, count=3):
        return [
            {"instruction": f"Reply with the digit {index}.", "input": "", "output": str(index),
             "answer": "ignored-for-training"}
            for index in range(count)
        ]

    def _records(self, count=3):
        return [{"sample_id": f"rev:path:{index}", "source_index": index,
                 "prompt_sha256": f"hash-{index}"} for index in range(count)]

    def test_adds_sample_identity_and_uses_output_not_answer(self):
        """训练目标必须是 output；评估用的 answer 不得混入。"""
        rows, records = self._rows(), self._records()
        prepared = prepare_supervised_examples(rows, records, TOKENIZER)
        self.assertEqual([item["sample_id"] for item in prepared], [r["sample_id"] for r in records])
        self.assertEqual([item["source_index"] for item in prepared], [0, 1, 2])
        # 每条样本的监督区间都必须以 EOS 收尾，说明确实按 output+EOS 构造。
        for item in prepared:
            valid = item["prompt_token_count"] + item["supervised_token_count"]
            self.assertEqual(item["labels"][valid - 1], TOKENIZER.eos_token_id)

    def test_answer_field_is_not_used_as_the_target(self):
        """把 answer 改成非规范值后，分词结果不应变化——证明它没参与构造。"""
        rows = self._rows()
        baseline = prepare_supervised_examples(rows, self._records(), TOKENIZER)
        mutated = [dict(row, answer="totally-different-label") for row in rows]
        after = prepare_supervised_examples(mutated, self._records(), TOKENIZER)
        self.assertEqual([item["full_ids_sha256"] for item in baseline],
                         [item["full_ids_sha256"] for item in after])

    def test_prefix_matches_protocol_rendering(self):
        """装配层必须使用协议模板渲染前缀，不能自行拼接。"""
        rows, records = self._rows(1), self._records(1)
        prepared = prepare_supervised_examples(rows, records, TOKENIZER)
        expected = build_commonsense_prompt(rows[0]["instruction"], rows[0]["input"])
        from adaptive_lora.m7_tokenization import tokenize_ids
        prefix_ids = tokenize_ids(TOKENIZER, expected, "expected prefix")
        self.assertEqual(prepared[0]["prompt_token_count"], len(prefix_ids))

    def test_missing_or_duplicate_sample_id_rejected(self):
        rows = self._rows()
        with self.assertRaisesRegex(ValueError, "lacks a non-empty sample_id"):
            prepare_supervised_examples(rows, [{}, {}, {}], TOKENIZER)
        duplicated = self._records()
        duplicated[2]["sample_id"] = duplicated[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            prepare_supervised_examples(rows, duplicated, TOKENIZER)

    def test_empty_training_target_rejected(self):
        rows = self._rows()
        rows[1]["output"] = "   "
        with self.assertRaisesRegex(ValueError, "non-empty training output"):
            prepare_supervised_examples(rows, self._records(), TOKENIZER)

    def test_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            prepare_supervised_examples(self._rows(3), self._records(2), TOKENIZER)
        with self.assertRaisesRegex(ValueError, "equal length"):
            prepare_supervised_examples([], [], TOKENIZER)

    def test_bad_boundary_is_not_silently_skipped(self):
        """截断到极小上限导致无监督 token 时必须整批失败，不能返回较少的样本。"""
        rows, records = self._rows(2), self._records(2)
        with self.assertRaisesRegex(ValueError, "no supervised token remains"):
            prepare_supervised_examples(rows, records, TOKENIZER, max_length=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
