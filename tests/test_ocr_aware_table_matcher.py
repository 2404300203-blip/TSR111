import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppstructure.table.matcher import TableMatch, normalize_xyxy_box


def test_normalize_reversed_xyxy_box():
    assert normalize_xyxy_box([10, 20, 1, 2]) == [1.0, 2.0, 10.0, 20.0]


def test_ocr_inside_cell_matches_cell():
    matcher = TableMatch(match_mode="ocr_aware_hungarian")
    matched, diagnostics = matcher.match_result_ocr_aware(
        dt_boxes=[[2, 2, 8, 8]],
        pred_bboxes=[[0, 0, 10, 10]],
        rec_res=[("abc", 0.9)],
    )
    assert matched == {0: [0]}
    assert diagnostics["num_primary_matches"] == 1


def test_boundary_ocr_prefers_better_covering_cell():
    matcher = TableMatch(match_mode="ocr_aware_hungarian")
    matched, _ = matcher.match_result_ocr_aware(
        dt_boxes=[[8, 1, 11, 9]],
        pred_bboxes=[[0, 0, 10, 10], [10, 0, 20, 10]],
        rec_res=[("edge", 0.9)],
    )
    assert matched == {0: [0]}


def test_multiple_ocr_tokens_are_aggregated_in_reading_order():
    matcher = TableMatch(match_mode="ocr_aware_hungarian")
    structure = (["<td></td>"], [[0, 0, 100, 40]])
    dt_boxes = [[2, 22, 20, 32], [2, 2, 20, 12], [30, 2, 50, 12]]
    rec_res = [("second", 0.9), ("first", 0.9), ("line", 0.9)]
    html = matcher(structure, dt_boxes, rec_res)
    assert html == "<td>first line second</td>"


def test_empty_inputs_do_not_crash():
    matcher = TableMatch(match_mode="ocr_aware_hungarian")
    matched, diagnostics = matcher.match_result_ocr_aware([], [], [])
    assert matched == {}
    assert diagnostics["num_cells"] == 0
    assert diagnostics["num_ocr_boxes"] == 0


def test_missing_ocr_confidence_defaults_to_one():
    matcher = TableMatch(match_mode="ocr_aware_hungarian")
    matched, diagnostics = matcher.match_result_ocr_aware(
        dt_boxes=[[2, 2, 8, 8]],
        pred_bboxes=[[0, 0, 10, 10]],
        rec_res=[("abc",)],
    )
    assert matched == {0: [0]}
    assert diagnostics["matches"][0]["features"]["ocr_conf"] == 1.0


def test_legacy_mode_keeps_existing_match_result():
    matcher = TableMatch(match_mode="legacy")
    matched = matcher.match_result(
        dt_boxes=[[2, 2, 8, 8]],
        pred_bboxes=[[0, 0, 10, 10]],
    )
    assert matched == {0: [0]}
