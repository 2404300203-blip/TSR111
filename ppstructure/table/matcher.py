# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import numpy as np
from ppstructure.table.table_master_match import deal_eb_token, deal_bb
import html

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


DEFAULT_OCR_AWARE_MATCHER_CONFIG = {
    "alpha": 0.20,
    "beta": 0.20,
    "gamma": 0.30,
    "delta": 0.15,
    "epsilon": 0.05,
    "zeta": 0.10,
    "min_iou": 0.01,
    "min_cover_ocr": 0.20,
    "cell_expand_ratio": 0.03,
    "min_match_score": 0.25,
    "secondary_assign_score": 0.30,
    "secondary_margin": 0.00,
    "unmatched_cost": 0.60,
    "large_cost": 1e6,
}


def normalize_xyxy_box(box):
    if len(box) != 4:
        return box
    x1, y1, x2, y2 = box
    return [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]


def polygon_to_xyxy(box):
    if box is None:
        return None
    box = np.asarray(box).reshape([-1]).tolist()
    if len(box) == 8:
        return [
            float(np.min(box[0::2])),
            float(np.min(box[1::2])),
            float(np.max(box[0::2])),
            float(np.max(box[1::2])),
        ]
    if len(box) == 4:
        return normalize_xyxy_box(box)
    return None


def box_area(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = normalize_xyxy_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_1, box_2):
    if box_1 is None or box_2 is None:
        return 0.0
    x1, y1, x2, y2 = normalize_xyxy_box(box_1)
    x3, y3, x4, y4 = normalize_xyxy_box(box_2)
    left = max(x1, x3)
    top = max(y1, y3)
    right = min(x2, x4)
    bottom = min(y2, y4)
    if left >= right or top >= bottom:
        return 0.0
    return (right - left) * (bottom - top)


def expand_box(box, ratio):
    x1, y1, x2, y2 = normalize_xyxy_box(box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return [
        x1 - width * ratio,
        y1 - height * ratio,
        x2 + width * ratio,
        y2 + height * ratio,
    ]


def box_center(box):
    x1, y1, x2, y2 = normalize_xyxy_box(box)
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = normalize_xyxy_box(box)
    return x1 <= x <= x2 and y1 <= y <= y2


def coverage(inner_box, outer_box):
    area = box_area(inner_box)
    if area <= 0:
        return 0.0
    return intersection_area(inner_box, outer_box) / area


def compute_center_sim(cell_box, ocr_box):
    cx, cy = box_center(cell_box)
    ox, oy = box_center(ocr_box)
    x1, y1, x2, y2 = normalize_xyxy_box(cell_box)
    diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / diag)


def compute_inside_score(cell_box, ocr_box):
    center = box_center(ocr_box)
    if not point_in_box(center, cell_box):
        return 0.0
    x, y = center
    x1, y1, x2, y2 = normalize_xyxy_box(cell_box)
    half_w = max(1.0, (x2 - x1) * 0.5)
    half_h = max(1.0, (y2 - y1) * 0.5)
    cx, cy = box_center(cell_box)
    edge_margin = min((x - x1) / half_w, (x2 - x) / half_w, (y - y1) / half_h, (y2 - y) / half_h)
    center_bonus = compute_center_sim(cell_box, ocr_box)
    return max(0.0, min(1.0, 0.5 * edge_margin + 0.5 * center_bonus))


def compute_direction_prior(cell_box, ocr_box):
    center = box_center(ocr_box)
    if point_in_box(center, cell_box):
        return 1.0
    if intersection_area(expand_box(cell_box, 0.03), ocr_box) > 0:
        return 0.5
    return 0.0


def ocr_confidence(rec):
    if rec is None or len(rec) < 2:
        return 1.0
    try:
        conf = float(rec[1])
    except Exception:
        return 1.0
    if np.isnan(conf):
        return 1.0
    return max(0.0, min(1.0, conf))


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def distance(box_1, box_2):
    box_1 = normalize_xyxy_box(box_1)
    box_2 = normalize_xyxy_box(box_2)
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    rec1 = normalize_xyxy_box(rec1)
    rec2 = normalize_xyxy_box(rec2)
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


class TableMatch:
    def __init__(
        self,
        filter_ocr_result=False,
        use_master=False,
        match_mode="legacy",
        matcher_config=None,
        diagnostics_path=None,
    ):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master
        self.match_mode = match_mode
        self.matcher_config = DEFAULT_OCR_AWARE_MATCHER_CONFIG.copy()
        if matcher_config:
            self.matcher_config.update(matcher_config)
        self.diagnostics_path = diagnostics_path
        if self.diagnostics_path:
            os.makedirs(os.path.dirname(self.diagnostics_path), exist_ok=True)

    def __call__(self, structure_res, dt_boxes, rec_res):
        pred_structures, pred_bboxes = structure_res
        if self.filter_ocr_result:
            dt_boxes, rec_res = self._filter_ocr_result(pred_bboxes, dt_boxes, rec_res)
        if self.match_mode == "ocr_aware_hungarian":
            matched_index, diagnostics = self.match_result_ocr_aware(
                dt_boxes, pred_bboxes, rec_res
            )
            self._dump_diagnostics(diagnostics)
        else:
            matched_index = self.match_result(dt_boxes, pred_bboxes)
        if self.use_master:
            pred_html, pred = self.get_pred_html_master(
                pred_structures, matched_index, rec_res
            )
        else:
            pred_html, pred = self.get_pred_html(
                pred_structures, matched_index, rec_res
            )
        return pred_html

    def _prepare_boxes(self, boxes):
        if boxes is None:
            return []
        return [polygon_to_xyxy(box) for box in boxes if polygon_to_xyxy(box) is not None]

    def match_result(self, dt_boxes, pred_bboxes):
        matched = {}
        for i, gt_box in enumerate(dt_boxes):
            gt_box = normalize_xyxy_box(gt_box)
            distances = []
            for j, pred_box in enumerate(pred_bboxes):
                if len(pred_box) == 8:
                    pred_box = [
                        np.min(pred_box[0::2]),
                        np.min(pred_box[1::2]),
                        np.max(pred_box[0::2]),
                        np.max(pred_box[1::2]),
                    ]
                pred_box = normalize_xyxy_box(pred_box)
                distances.append(
                    (distance(gt_box, pred_box), 1.0 - compute_iou(gt_box, pred_box))
                )  # compute iou and l1 distance
            sorted_distances = distances.copy()
            # select det box by iou and l1 distance
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0])
            )
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        return matched

    def _pair_features(self, cell_box, ocr_box, rec=None):
        inter = intersection_area(cell_box, ocr_box)
        union = box_area(cell_box) + box_area(ocr_box) - inter
        iou = inter / union if union > 0 else 0.0
        cover_ocr = coverage(ocr_box, cell_box)
        cover_cell = coverage(cell_box, ocr_box)
        center_sim = compute_center_sim(cell_box, ocr_box)
        inside_score = compute_inside_score(cell_box, ocr_box)
        direction_prior = compute_direction_prior(cell_box, ocr_box)
        conf = ocr_confidence(rec)
        return {
            "iou": iou,
            "cover_ocr": cover_ocr,
            "cover_cell": cover_cell,
            "center_sim": center_sim,
            "inside_score": inside_score,
            "direction_prior": direction_prior,
            "ocr_conf": conf,
        }

    def _pair_score(self, features):
        cfg = self.matcher_config
        return (
            cfg["alpha"] * features["iou"]
            + cfg["beta"] * features["center_sim"]
            + cfg["gamma"] * features["cover_ocr"]
            + cfg["delta"] * features["inside_score"]
            + cfg["epsilon"] * features["direction_prior"]
            + cfg["zeta"] * features["ocr_conf"]
        )

    def _is_candidate(self, cell_box, ocr_box, features):
        cfg = self.matcher_config
        if point_in_box(box_center(ocr_box), cell_box):
            return True
        if features["cover_ocr"] >= cfg["min_cover_ocr"]:
            return True
        if features["iou"] >= cfg["min_iou"]:
            return True
        return intersection_area(expand_box(cell_box, cfg["cell_expand_ratio"]), ocr_box) > 0

    def match_result_ocr_aware(self, dt_boxes, pred_bboxes, rec_res=None):
        if linear_sum_assignment is None:
            raise ImportError("scipy.optimize.linear_sum_assignment is required for ocr_aware_hungarian")
        cell_boxes = self._prepare_boxes(pred_bboxes)
        ocr_boxes = self._prepare_boxes(dt_boxes)
        rec_res = rec_res or []
        diagnostics = {
            "num_cells": len(cell_boxes),
            "num_ocr_boxes": len(ocr_boxes),
            "num_primary_matches": 0,
            "num_secondary_assignments": 0,
            "num_unmatched_cells": 0,
            "num_unmatched_ocr": 0,
            "mean_match_score": 0.0,
            "matches": [],
        }
        if not cell_boxes or not ocr_boxes:
            diagnostics["num_unmatched_cells"] = len(cell_boxes)
            diagnostics["num_unmatched_ocr"] = len(ocr_boxes)
            return {}, diagnostics

        cfg = self.matcher_config
        score_matrix = np.full((len(cell_boxes), len(ocr_boxes)), -1.0, dtype="float32")
        feature_matrix = {}
        cost_matrix = np.full(score_matrix.shape, cfg["large_cost"], dtype="float32")
        for cell_id, cell_box in enumerate(cell_boxes):
            for ocr_id, ocr_box in enumerate(ocr_boxes):
                rec = rec_res[ocr_id] if ocr_id < len(rec_res) else None
                features = self._pair_features(cell_box, ocr_box, rec)
                if not self._is_candidate(cell_box, ocr_box, features):
                    continue
                score = self._pair_score(features)
                score_matrix[cell_id, ocr_id] = score
                feature_matrix[(cell_id, ocr_id)] = features
                cost_matrix[cell_id, ocr_id] = 1.0 - score

        dummy_cost = cfg["unmatched_cost"]
        padded_cost = np.full(
            (len(cell_boxes) + len(ocr_boxes), len(ocr_boxes) + len(cell_boxes)),
            cfg["large_cost"],
            dtype="float32",
        )
        padded_cost[: len(cell_boxes), : len(ocr_boxes)] = cost_matrix
        padded_cost[len(cell_boxes) :, len(ocr_boxes) :] = 0.0
        for cell_id in range(len(cell_boxes)):
            padded_cost[cell_id, len(ocr_boxes) + cell_id] = dummy_cost
        for ocr_id in range(len(ocr_boxes)):
            padded_cost[len(cell_boxes) + ocr_id, ocr_id] = dummy_cost

        rows, cols = linear_sum_assignment(padded_cost)
        matched = {}
        assigned_ocr = set()
        accepted_scores = []
        for row, col in zip(rows, cols):
            if row >= len(cell_boxes) or col >= len(ocr_boxes):
                continue
            score = float(score_matrix[row, col])
            if score < cfg["min_match_score"]:
                continue
            matched.setdefault(row, []).append(col)
            assigned_ocr.add(col)
            accepted_scores.append(score)
            diagnostics["matches"].append(
                {
                    "cell_id": row,
                    "ocr_id": col,
                    "score": score,
                    "is_primary": True,
                    "features": feature_matrix.get((row, col), {}),
                }
            )

        diagnostics["num_primary_matches"] = len(accepted_scores)

        for ocr_id, ocr_box in enumerate(ocr_boxes):
            if ocr_id in assigned_ocr:
                continue
            scored_cells = []
            for cell_id, cell_box in enumerate(cell_boxes):
                rec = rec_res[ocr_id] if ocr_id < len(rec_res) else None
                features = self._pair_features(cell_box, ocr_box, rec)
                if not self._is_candidate(cell_box, ocr_box, features):
                    continue
                scored_cells.append((self._pair_score(features), cell_id, features))
            if not scored_cells:
                continue
            scored_cells.sort(reverse=True, key=lambda item: item[0])
            best_score, best_cell, best_features = scored_cells[0]
            second_score = scored_cells[1][0] if len(scored_cells) > 1 else 0.0
            if (
                best_score >= cfg["secondary_assign_score"]
                and best_score - second_score >= cfg["secondary_margin"]
            ):
                matched.setdefault(best_cell, []).append(ocr_id)
                assigned_ocr.add(ocr_id)
                accepted_scores.append(float(best_score))
                diagnostics["matches"].append(
                    {
                        "cell_id": best_cell,
                        "ocr_id": ocr_id,
                        "score": float(best_score),
                        "is_primary": False,
                        "features": best_features,
                    }
                )

        for cell_id, ocr_ids in list(matched.items()):
            matched[cell_id] = self._sort_ocr_ids_reading_order(ocr_ids, ocr_boxes)

        diagnostics["num_secondary_assignments"] = len(assigned_ocr) - diagnostics["num_primary_matches"]
        diagnostics["num_unmatched_cells"] = len([i for i in range(len(cell_boxes)) if i not in matched])
        diagnostics["num_unmatched_ocr"] = len(ocr_boxes) - len(assigned_ocr)
        diagnostics["mean_match_score"] = float(np.mean(accepted_scores)) if accepted_scores else 0.0
        diagnostics["empty_cell_count"] = diagnostics["num_unmatched_cells"]
        return matched, diagnostics

    def _sort_ocr_ids_reading_order(self, ocr_ids, ocr_boxes):
        if len(ocr_ids) <= 1:
            return ocr_ids
        heights = [max(1.0, ocr_boxes[i][3] - ocr_boxes[i][1]) for i in ocr_ids]
        line_threshold = max(2.0, float(np.median(heights)) * 0.6)
        items = sorted(
            [(i, box_center(ocr_boxes[i])[0], box_center(ocr_boxes[i])[1]) for i in ocr_ids],
            key=lambda item: item[2],
        )
        lines = []
        for item in items:
            if not lines or abs(item[2] - np.mean([x[2] for x in lines[-1]])) > line_threshold:
                lines.append([item])
            else:
                lines[-1].append(item)
        sorted_ids = []
        for line in lines:
            sorted_ids.extend([item[0] for item in sorted(line, key=lambda item: item[1])])
        return sorted_ids

    def _dump_diagnostics(self, diagnostics):
        if not self.diagnostics_path:
            return
        with open(self.diagnostics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(diagnostics), ensure_ascii=False) + "\n")

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if "</td>" in tag:
                if "<td></td>" == tag:
                    end_html.extend("<td>")
                if td_index in matched_index.keys():
                    b_with = False
                    if (
                        "<b>" in ocr_contents[matched_index[td_index][0]]
                        and len(matched_index[td_index]) > 1
                    ):
                        b_with = True
                        end_html.extend("<b>")
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == " ":
                                content = content[1:]
                            if "<b>" in content:
                                content = content[3:]
                            if "</b>" in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if (
                                i != len(matched_index[td_index]) - 1
                                and " " != content[-1]
                            ):
                                content += " "
                        # escape content
                        content = html.escape(content)
                        end_html.extend(content)
                    if b_with:
                        end_html.extend("</b>")
                if "<td></td>" == tag:
                    end_html.append("</td>")
                else:
                    end_html.append(tag)
                td_index += 1
            else:
                end_html.append(tag)
        return "".join(end_html), end_html

    def get_pred_html_master(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for token in pred_structures:
            if "</td>" in token:
                txt = ""
                b_with = False
                if td_index in matched_index.keys():
                    if (
                        "<b>" in ocr_contents[matched_index[td_index][0]]
                        and len(matched_index[td_index]) > 1
                    ):
                        b_with = True
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == " ":
                                content = content[1:]
                            if "<b>" in content:
                                content = content[3:]
                            if "</b>" in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if (
                                i != len(matched_index[td_index]) - 1
                                and " " != content[-1]
                            ):
                                content += " "
                        txt += content
                if b_with:
                    txt = "<b>{}</b>".format(txt)
                if "<td></td>" == token:
                    token = "<td>{}</td>".format(txt)
                else:
                    token = "{}</td>".format(txt)
                td_index += 1
            token = deal_eb_token(token)
            end_html.append(token)
        html = "".join(end_html)
        html = deal_bb(html)
        return html, end_html

    def _filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
        y1 = pred_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
