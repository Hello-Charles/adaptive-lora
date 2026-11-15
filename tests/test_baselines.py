"""M6 层集合规划测试：纯索引/RNG 契约，不加载模型，也不声称检查了真实参数量。"""

import random
import unittest

from adaptive_lora.baselines import plan_baselines


class BaselinePlanningTests(unittest.TestCase):
    """Full 全层，Random 均匀不放回，Adaptive 保留已验收排名；层号全部零基。"""

    def test_three_methods_use_full_random_and_supplied_adaptive_layers(self):
        adaptive = [0, 1, 2, 5, 3, 8, 10]
        plans = plan_baselines(24, 7, adaptive, layer_seed=42)
        self.assertEqual([plan.method for plan in plans], ["full", "random_k", "adaptive_k"])
        self.assertEqual(plans[0].indices, tuple(range(24)))
        self.assertEqual(plans[1].indices, tuple(random.Random(42).sample(range(24), 7)))
        self.assertEqual(plans[2].indices, tuple(adaptive))
        self.assertEqual(len(set(plans[1].indices)), 7)
        saved = plans[1].to_dict()
        self.assertEqual(saved["selection_seed"], 42)
        self.assertEqual(saved["index_base"], 0)
        self.assertEqual(saved["selected_layer_indices_ascending"], sorted(plans[1].indices))
        self.assertIsNone(plans[0].selection_seed)
        self.assertIsNone(plans[2].selection_seed)

    def test_layer_rng_is_local_and_repeats_without_touching_global_training_rng(self):
        state = random.getstate()
        first = plan_baselines(24, 7, range(7), layer_seed=20261115)
        second = plan_baselines(24, 7, range(7), layer_seed=20261115)
        self.assertEqual(first, second)
        self.assertEqual(state, random.getstate())

    def test_changing_layer_seed_only_changes_random_plan(self):
        first = plan_baselines(24, 7, range(7), layer_seed=42)
        second = plan_baselines(24, 7, range(7), layer_seed=7)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])
        self.assertNotEqual(first[1].indices, second[1].indices)

    def test_all_layer_budget_is_same_set_not_three_independent_parameter_budgets(self):
        plans = plan_baselines(4, 4, [3, 2, 1, 0], layer_seed=42)
        self.assertTrue(all(set(plan.indices) == set(range(4)) for plan in plans))
        self.assertTrue(all(plan.to_dict()["selected_layer_count"] == 4 for plan in plans))

    def test_invalid_layer_counts_k_seeds_and_adaptive_indices_are_rejected(self):
        cases = ((0, 1, [0], 42), (True, 1, [0], 42), (4, 0, [], 42), (4, 5, range(5), 42),
                 (4, True, [0], 42), (4, 2, [0], 42), (4, 2, [0, 0], 42), (4, 2, [-1, 1], 42),
                 (4, 2, [0, 4], 42), (4, 2, [True, 1], 42), (4, 2, [0, 1], -1), (4, 2, [0, 1], True))
        for layers, k, adaptive, seed in cases:
            with self.subTest(layers=layers, k=k, adaptive=adaptive, seed=seed), self.assertRaises(ValueError):
                plan_baselines(layers, k, adaptive, layer_seed=seed)

    def test_input_mutation_cannot_change_saved_plan(self):
        indices = [0, 2]
        plan = plan_baselines(4, 2, indices, layer_seed=42)[2]
        indices[0] = 3
        self.assertEqual(plan.indices, (0, 2))
        saved = plan.to_dict()
        saved["selected_layer_indices"][0] = 3
        self.assertEqual(plan.indices, (0, 2))


if __name__ == "__main__":
    unittest.main()
