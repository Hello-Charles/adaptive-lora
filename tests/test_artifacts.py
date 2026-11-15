"""M4 结果文件/图形结构测试；不生成真实校准数值，不替代浏览器渲染检查。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

from adaptive_lora.artifacts import (
    render_importance_html,
    render_importance_svg,
    write_calibration_artifacts,
    write_json,
)


def fixture_result(scores=None):
    """只模拟完成校准后的均值，用人工递增数组检查排序和输出，不调用核心聚合器。"""
    scores = [index / 100 for index in range(24)] if scores is None else scores
    return {
        "n": 2, "layer_count": len(scores), "samples": [],
        "layer_importance": [
            {"layer_index": index, "importance": value,
             "attention_mean_abs_delta": value / 2, "mlp_mean_abs_delta": value / 2}
            for index, value in enumerate(scores)
        ],
    }


class CalibrationArtifactTests(unittest.TestCase):
    """检查零基索引、独占写入、图表可访问数据及 XML 合法性，不宣称布局已看过。"""

    def test_exports_csv_json_and_top_k_without_losing_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            result = fixture_result()
            selection = write_calibration_artifacts(Path(directory), result, top_k=7)
            self.assertEqual(selection["selected_layer_indices"], [23, 22, 21, 20, 19, 18, 17])
            self.assertEqual(selection["selected_layer_indices_ascending"], list(range(17, 24)))
            saved = json.loads((Path(directory) / "selected_layers.json").read_text(encoding="utf-8"))
            self.assertEqual(selection, saved)
            self.assertEqual(saved["index_base"], 0)
            with (Path(directory) / "layer_importance.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 24)
            self.assertEqual(int(rows[23]["rank"]), 1)
            self.assertAlmostEqual(float(rows[23]["importance"]), 0.23)
            self.assertEqual(rows[23]["selected"], "True")
            exported = json.loads((Path(directory) / "layer_importance.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["n"], 2)
            self.assertEqual(len(exported["layers"]), 24)
            self.assertTrue((Path(directory) / "importance.svg").is_file())
            self.assertTrue((Path(directory) / "importance.html").is_file())

    def test_json_exclusive_creation_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "invalid.json", {"value": float("nan")})

    def test_artifact_rewrite_is_rejected_and_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_calibration_artifacts(path, fixture_result(), top_k=7)
            before = {file.name: file.read_bytes() for file in path.iterdir()}
            with self.assertRaises(FileExistsError):
                write_calibration_artifacts(path, fixture_result([0.1] * 24), top_k=1)
            self.assertEqual(before, {file.name: file.read_bytes() for file in path.iterdir()})

    def test_svg_has_zero_baseline_one_bar_and_tooltip_per_layer(self):
        svg = render_importance_svg(fixture_result(), list(range(17, 24)))
        root = ElementTree.fromstring(svg)
        ns = {"s": "http://www.w3.org/2000/svg"}
        marks = [group for group in root.findall(".//s:g", ns) if group.get("class") == "mark"]
        self.assertEqual(len(marks), 24)
        for index, mark in enumerate(marks):
            self.assertEqual(mark.get("tabindex"), "0")
            self.assertIn(f"Layer {index}", mark.get("data-tooltip"))
            self.assertIsNotNone(mark.find("s:title", ns))
            self.assertIsNotNone(mark.find("s:rect[@class='hit']", ns))
        self.assertIn('data-baseline="0"', svg)
        self.assertIn("#2a78d6", svg)
        self.assertIn("#3987e5", svg)

    def test_html_has_full_table_keyboard_tooltip_and_explicit_theme_texture(self):
        html = render_importance_html(fixture_result(), list(range(17, 24)))
        self.assertIn("<table", html)
        self.assertEqual(html.count('scope="row"'), 24)
        self.assertIn("textContent", html)
        self.assertNotIn("innerHTML", html)
        self.assertIn("focusin", html)
        self.assertIn('id="theme-toggle"', html)
        self.assertIn('id="texture-toggle"', html)
        self.assertIn("forced-colors", html)
        self.assertNotIn('src="https://', html)

    def test_all_zero_scores_render_without_division_by_zero(self):
        root = ElementTree.fromstring(render_importance_svg(fixture_result([0.0] * 24), list(range(7))))
        self.assertEqual(root.get("data-baseline"), "0")

    def test_wrong_layer_order_or_invalid_scores_are_rejected(self):
        result = fixture_result()
        result["layer_importance"][0]["layer_index"] = 1
        with self.assertRaises(ValueError):
            render_importance_svg(result, [0])
        for bad in (float("nan"), -0.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                render_importance_svg(fixture_result([bad]), [0])


if __name__ == "__main__":
    unittest.main()
