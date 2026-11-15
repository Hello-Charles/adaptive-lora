"""Top-K 的纯数学测试；不加载模型，不把排序正确等同于校准分数正确。"""

import math
import unittest

from adaptive_lora.selection import rank_layers, select_top_k


class LayerSelectionTests(unittest.TestCase):
    """分数数组下标就是零基层索引；相等分数按小索引优先是明确的工程约定。"""

    def test_descending_scores_keep_zero_based_layer_indices(self):
        scores = [0.2, 0.8, 0.1, 0.6]
        self.assertEqual(rank_layers(scores), [1, 3, 0, 2])
        self.assertEqual(select_top_k(scores, 2), [1, 3])
        self.assertEqual(scores, [0.2, 0.8, 0.1, 0.6])

    def test_ties_are_stable_and_use_ascending_layer_index(self):
        self.assertEqual(rank_layers([0.3, 0.3, 0.1, 0.3]), [0, 1, 3, 2])
        self.assertEqual(select_top_k([0.0] * 24, 7), list(range(7)))

    def test_top_seven_is_unique_legal_and_exactly_seven(self):
        scores = [index / 100 for index in range(24)]
        selected = select_top_k(scores, 7)
        self.assertEqual(selected, [23, 22, 21, 20, 19, 18, 17])
        self.assertEqual(len(set(selected)), 7)
        self.assertTrue(all(0 <= index < 24 for index in selected))
        self.assertEqual(select_top_k(scores, 24), list(reversed(range(24))))

    def test_rejects_empty_nonfinite_negative_or_non_numeric_scores(self):
        for scores in ([], [math.nan], [math.inf], [-0.1], ["0.1"], [True]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rank_layers(scores)

    def test_rejects_invalid_k_instead_of_silently_clipping(self):
        for k in (0, -1, 3, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                select_top_k([0.1, 0.2], k)


if __name__ == "__main__":
    unittest.main()
