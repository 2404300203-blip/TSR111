# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from dataclasses import dataclass, field

import numpy as np
from ppocr.metrics.det_metric import DetMetric


@dataclass
class _TEDSNode:
    label: str
    children: list = field(default_factory=list)


def _normalize_teds_token(token):
    token = str(token).strip()
    if token.startswith("</"):
        return token
    if token.startswith("<") and token.endswith(">"):
        return token
    return "#text"


def _build_teds_tree(tokens):
    root = _TEDSNode("root")
    stack = [root]
    for raw in tokens:
        token = _normalize_teds_token(raw)
        if not token:
            continue
        if token.startswith("</"):
            if len(stack) > 1:
                stack.pop()
            continue
        node = _TEDSNode(token)
        stack[-1].children.append(node)
        if token.startswith("<") and not token.startswith("</") and token not in {
            "<br>",
            "<hr>",
            "<img>",
        }:
            stack.append(node)
    return root


def _teds_tree_size(node):
    total = 0
    stack = [node]
    while stack:
        cur = stack.pop()
        total += 1
        stack.extend(cur.children)
    return total


def _teds_tree_distance(a, b, memo):
    key = (id(a), id(b))
    if key in memo:
        return memo[key]

    rename = 0 if a.label == b.label else 1
    ac = a.children
    bc = b.children
    dp = [[0] * (len(bc) + 1) for _ in range(len(ac) + 1)]

    for i in range(1, len(ac) + 1):
        dp[i][0] = dp[i - 1][0] + _teds_tree_size(ac[i - 1])
    for j in range(1, len(bc) + 1):
        dp[0][j] = dp[0][j - 1] + _teds_tree_size(bc[j - 1])

    for i in range(1, len(ac) + 1):
        for j in range(1, len(bc) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _teds_tree_size(ac[i - 1]),
                dp[i][j - 1] + _teds_tree_size(bc[j - 1]),
                dp[i - 1][j - 1] + _teds_tree_distance(ac[i - 1], bc[j - 1], memo),
            )

    value = rename + dp[len(ac)][len(bc)]
    memo[key] = value
    return value


def table_teds_score(pred_tokens, gt_tokens):
    pred_tree = _build_teds_tree(pred_tokens)
    gt_tree = _build_teds_tree(gt_tokens)
    denom = max(_teds_tree_size(pred_tree), _teds_tree_size(gt_tree), 1)
    dist = _teds_tree_distance(pred_tree, gt_tree, {})
    return max(0.0, 1.0 - dist / denom)


class TableStructureMetric(object):
    def __init__(
        self,
        main_indicator="acc",
        eps=1e-6,
        del_thead_tbody=False,
        compute_teds=False,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.eps = eps
        self.del_thead_tbody = del_thead_tbody
        self.compute_teds = compute_teds
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        pred_structure_batch_list = preds["structure_batch_list"]
        gt_structure_batch_list = labels["structure_batch_list"]
        correct_num = 0
        all_num = 0
        for (pred, pred_conf), target in zip(
            pred_structure_batch_list, gt_structure_batch_list
        ):
            pred_str = "".join(pred)
            target_str = "".join(target)
            if self.del_thead_tbody:
                pred_str = (
                    pred_str.replace("<thead>", "")
                    .replace("</thead>", "")
                    .replace("<tbody>", "")
                    .replace("</tbody>", "")
                )
                target_str = (
                    target_str.replace("<thead>", "")
                    .replace("</thead>", "")
                    .replace("<tbody>", "")
                    .replace("</tbody>", "")
                )
            if pred_str == target_str:
                correct_num += 1
            if self.compute_teds:
                pred_tokens = [
                    token
                    for token in pred
                    if token not in ("<html>", "</html>", "<body>", "</body>")
                ]
                target_tokens = [
                    token
                    for token in target
                    if token not in ("<html>", "</html>", "<body>", "</body>")
                ]
                try:
                    self.total_teds += table_teds_score(pred_tokens, target_tokens)
                except RecursionError:
                    pass
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        result = {"acc": acc}
        if self.compute_teds:
            result["teds"] = 1.0 * self.total_teds / (self.all_num + self.eps)
        self.reset()
        return result

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.total_teds = 0.0
        self.len_acc_num = 0
        self.token_nums = 0
        self.anys_dict = dict()


class TableMetric(object):
    def __init__(
        self,
        main_indicator="acc",
        compute_bbox_metric=False,
        compute_teds=False,
        box_format="xyxy",
        del_thead_tbody=False,
        **kwargs,
    ):
        """

        @param sub_metrics: configs of sub_metric
        @param main_matric: main_matric for save best_model
        @param kwargs:
        """
        self.structure_metric = TableStructureMetric(
            del_thead_tbody=del_thead_tbody,
            compute_teds=compute_teds or main_indicator == "teds",
        )
        self.bbox_metric = DetMetric() if compute_bbox_metric else None
        self.main_indicator = main_indicator
        self.box_format = box_format
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        self.structure_metric(pred_label)
        if self.bbox_metric is not None:
            self.bbox_metric(*self.prepare_bbox_metric_input(pred_label))

    def prepare_bbox_metric_input(self, pred_label):
        pred_bbox_batch_list = []
        gt_ignore_tags_batch_list = []
        gt_bbox_batch_list = []
        preds, labels = pred_label

        batch_num = len(preds["bbox_batch_list"])
        for batch_idx in range(batch_num):
            # pred
            pred_bbox_list = [
                self.format_box(pred_box)
                for pred_box in preds["bbox_batch_list"][batch_idx]
            ]
            pred_bbox_batch_list.append({"points": pred_bbox_list})

            # gt
            gt_bbox_list = []
            gt_ignore_tags_list = []
            for gt_box in labels["bbox_batch_list"][batch_idx]:
                gt_bbox_list.append(self.format_box(gt_box))
                gt_ignore_tags_list.append(0)
            gt_bbox_batch_list.append(gt_bbox_list)
            gt_ignore_tags_batch_list.append(gt_ignore_tags_list)

        return [
            pred_bbox_batch_list,
            [0, 0, gt_bbox_batch_list, gt_ignore_tags_batch_list],
        ]

    def get_metric(self):
        structure_metric = self.structure_metric.get_metric()
        if self.bbox_metric is None:
            return structure_metric
        bbox_metric = self.bbox_metric.get_metric()
        if self.main_indicator == self.bbox_metric.main_indicator:
            output = bbox_metric
            for sub_key in structure_metric:
                output["structure_metric_{}".format(sub_key)] = structure_metric[
                    sub_key
                ]
        else:
            output = structure_metric
            for sub_key in bbox_metric:
                output["bbox_metric_{}".format(sub_key)] = bbox_metric[sub_key]
        return output

    def reset(self):
        self.structure_metric.reset()
        if self.bbox_metric is not None:
            self.bbox_metric.reset()

    def format_box(self, box):
        if self.box_format == "xyxy":
            x1, y1, x2, y2 = box
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif self.box_format == "xywh":
            x, y, w, h = box
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif self.box_format == "xyxyxyxy":
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return box
