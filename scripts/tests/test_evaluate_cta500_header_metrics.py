#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_cta500_header_metrics.py"
SPEC = importlib.util.spec_from_file_location("header_metrics", SCRIPT)
hm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hm
SPEC.loader.exec_module(hm)


class HeaderMetricTests(unittest.TestCase):
    def test_split_cell_tag_is_reconstructed(self):
        tokens = ["<tr>", "<td", ' rowspan="2"', ' colspan="3"', ">", "</td>", "</tr>"]
        self.assertEqual(
            hm.reconstructed_tokens(tokens),
            ["<tr>", '<td rowspan="2" colspan="3">', "</td>", "</tr>"],
        )
        self.assertEqual(hm.parse_cells(tokens), {(0, 0, 2, 3)})

    def test_first_rows_uses_logical_rows_not_thead_marker(self):
        tokens = [
            "<tbody>", "<tr>", "<td></td>", "</tr>",
            "<tr>", '<td colspan="2"></td>', "</tr>", "</tbody>",
        ]
        self.assertEqual(
            hm.first_rows(tokens, 1),
            ["<thead>", "<tr>", "<td></td>", "</tr>", "</thead>"],
        )

    def test_first_rows_matches_direct_thead_when_depth_matches(self):
        tokens = [
            "<thead>",
            "<tr>", '<td colspan="2"></td>', "</tr>",
            "<tr>", "<td></td>", "<td></td>", "</tr>",
            "</thead>",
            "<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>",
        ]
        depth = hm.header_depth(tokens)
        self.assertEqual(depth, 2)
        self.assertEqual(hm.first_rows(tokens, depth), hm.thead_tokens(tokens))

    def test_span_crossing_header_boundary_is_not_clipped(self):
        tokens = [
            "<thead>", "<tr>", '<td rowspan="3"></td>', "<td></td>", "</tr>", "</thead>",
            "<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>",
        ]
        cells = hm.parse_cells(tokens)
        self.assertIn((0, 0, 3, 1), cells)
        header_spans = {cell for cell in cells if cell[0] < 1 and (cell[2] > 1 or cell[3] > 1)}
        self.assertEqual(header_spans, {(0, 0, 3, 1)})

    def test_missing_predicted_thead_remains_evaluable(self):
        gt = ["<thead>", "<tr>", "<td></td>", "</tr>", "</thead>"]
        pred = ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]
        self.assertIsNone(hm.header_depth(pred))
        self.assertEqual(hm.teds(hm.first_rows(pred, 1), hm.first_rows(gt, 1)), 1.0)

    def test_empty_span_sets_have_perfect_f1_by_protocol(self):
        pred = {cell for cell in hm.parse_cells(["<tr>", "<td></td>", "</tr>"]) if cell[2] > 1 or cell[3] > 1}
        gt = set()
        tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        self.assertEqual(score, 1.0)

    def test_tree_metric_penalizes_span_attribute_error(self):
        gt = ["<thead>", "<tr>", '<td colspan="2"></td>', "</tr>", "</thead>"]
        pred = ["<thead>", "<tr>", "<td></td>", "</tr>", "</thead>"]
        self.assertLess(hm.teds(pred, gt), 1.0)


if __name__ == "__main__":
    unittest.main()
