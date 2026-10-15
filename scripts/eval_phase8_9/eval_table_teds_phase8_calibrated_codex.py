import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import paddle

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
import tools.program as program


@dataclass
class Node:
    label: str
    children: list = field(default_factory=list)


def normalize_token(token):
    token = str(token).strip()
    if token.startswith("</"):
        return token
    if token.startswith("<") and token.endswith(">"):
        return token
    return "#text"


def build_tree(tokens):
    root = Node("root")
    stack = [root]
    for raw in tokens:
        token = normalize_token(raw)
        if not token:
            continue
        if token.startswith("</"):
            if len(stack) > 1:
                stack.pop()
            continue
        node = Node(token)
        stack[-1].children.append(node)
        if token.startswith("<") and not token.startswith("</") and token not in {
            "<br>",
            "<hr>",
            "<img>",
        }:
            stack.append(node)
    return root


def tree_size(node):
    total = 0
    stack = [node]
    while stack:
        cur = stack.pop()
        total += 1
        stack.extend(cur.children)
    return total


def tree_distance(a, b, memo):
    key = (id(a), id(b))
    if key in memo:
        return memo[key]

    rename = 0 if a.label == b.label else 1
    ac = a.children
    bc = b.children
    dp = [[0] * (len(bc) + 1) for _ in range(len(ac) + 1)]

    for i in range(1, len(ac) + 1):
        dp[i][0] = dp[i - 1][0] + tree_size(ac[i - 1])
    for j in range(1, len(bc) + 1):
        dp[0][j] = dp[0][j - 1] + tree_size(bc[j - 1])

    for i in range(1, len(ac) + 1):
        for j in range(1, len(bc) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + tree_size(ac[i - 1]),
                dp[i][j - 1] + tree_size(bc[j - 1]),
                dp[i - 1][j - 1] + tree_distance(ac[i - 1], bc[j - 1], memo),
            )

    value = rename + dp[len(ac)][len(bc)]
    memo[key] = value
    return value


def teds_score(pred_tokens, gt_tokens):
    pred_tree = build_tree(pred_tokens)
    gt_tree = build_tree(gt_tokens)
    denom = max(tree_size(pred_tree), tree_size(gt_tree), 1)
    dist = tree_distance(pred_tree, gt_tree, {})
    return max(0.0, 1.0 - dist / denom)


def strip_wrappers(tokens):
    return [
        token
        for token in tokens
        if token not in ("<html>", "</html>", "<body>", "</body>")
    ]


def is_span_token(token):
    token = str(token)
    return "rowspan" in token or "colspan" in token


def span_values(token):
    row = 1
    col = 1
    row_m = re.search(r'rowspan="(\d+)"', str(token))
    col_m = re.search(r'colspan="(\d+)"', str(token))
    if row_m:
        row = int(row_m.group(1))
    if col_m:
        col = int(col_m.group(1))
    return row, col


def row_segments(tokens):
    rows = []
    cur = None
    in_tr = False
    for i, token in enumerate(tokens):
        if token == "<tr>":
            in_tr = True
            cur = []
        elif token == "</tr>":
            if in_tr and cur is not None:
                rows.append(cur)
            in_tr = False
            cur = None
        elif in_tr and cur is not None:
            cur.append((i, token))
    return rows


def iter_cells_in_row(row):
    cells = []
    pos = 0
    while pos < len(row):
        idx, token = row[pos]
        token = str(token)
        if token == "<td></td>":
            cells.append((idx, idx + 1, 1, 1, False))
            pos += 1
            continue
        if token == "<td":
            parts = [token]
            end = idx + 1
            pos += 1
            while pos < len(row):
                next_idx, next_token = row[pos]
                parts.append(str(next_token))
                end = next_idx + 1
                pos += 1
                if str(next_token) == "</td>":
                    break
            row_span, col_span = span_values(" ".join(parts))
            cells.append((idx, end, row_span, col_span, row_span > 1 or col_span > 1))
            continue
        pos += 1
    return cells


def iter_cells(tokens):
    for row in row_segments(tokens):
        yield from iter_cells_in_row(row)


def row_widths(tokens):
    widths = []
    for row in row_segments(tokens):
        width = 0
        for _start, _end, _row_span, col_span, _is_span in iter_cells_in_row(row):
            width += col_span
        widths.append(width)
    return widths


def modal_width(widths):
    valid = [width for width in widths if width > 0]
    if not valid:
        return 0
    counts = {}
    for width in valid:
        counts[width] = counts.get(width, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def calibrate_tokens(
    tokens,
    scores,
    mode="none",
    span_conf_threshold=0.0,
    margins=None,
    span_margin_threshold=0.0,
):
    if mode == "none":
        return list(tokens), 0

    calibrated = list(tokens)
    suppressions = 0
    target_width = modal_width(row_widths(calibrated))

    for start, end, _row_span, _col_span, cell_has_span in reversed(
        list(iter_cells(calibrated))
    ):
        if not cell_has_span:
            continue

        cell_scores = scores[start:end] if start < len(scores) else []
        score = min(cell_scores) if cell_scores else 1.0
        suppress = False

        if mode in ("confidence", "hybrid") and score < span_conf_threshold:
            suppress = True

        cell_margins = margins[start:end] if margins is not None and start < len(margins) else []
        numeric_margins = [value for value in cell_margins if value is not None]
        margin = min(numeric_margins) if numeric_margins else None
        if mode in ("margin", "margin_shape") and margin is not None:
            if margin < span_margin_threshold:
                suppress = True

        if mode in ("shape", "hybrid") and target_width > 0:
            trial = list(calibrated)
            trial[start:end] = ["<td></td>"]
            old_bad = sum(abs(width - target_width) for width in row_widths(calibrated))
            new_bad = sum(abs(width - target_width) for width in row_widths(trial))
            if new_bad < old_bad:
                suppress = True

        if mode == "margin_shape" and target_width > 0:
            trial = list(calibrated)
            trial[start:end] = ["<td></td>"]
            old_bad = sum(abs(width - target_width) for width in row_widths(calibrated))
            new_bad = sum(abs(width - target_width) for width in row_widths(trial))
            if new_bad > old_bad:
                suppress = False

        if suppress:
            calibrated[start:end] = ["<td></td>"]
            suppressions += 1

    return calibrated, suppressions


class CalibratedTEDSMetric:
    def __init__(
        self,
        post_process_class,
        mode="none",
        span_conf_threshold=0.0,
        span_margin_threshold=0.0,
        diagnostics_path=None,
    ):
        self.post_process_class = post_process_class
        self.mode = mode
        self.span_conf_threshold = span_conf_threshold
        self.span_margin_threshold = span_margin_threshold
        self.diagnostics_path = diagnostics_path
        self.fp = None
        if diagnostics_path:
            os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
            self.fp = open(diagnostics_path, "w", encoding="utf-8")
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds = pred_label
        post_result, labels = self.post_process_class(preds, batch)
        pred_structures = post_result["structure_batch_list"]
        gt_structures = labels["structure_batch_list"]

        structure_probs = preds["structure_probs"]
        if isinstance(structure_probs, paddle.Tensor):
            structure_probs = structure_probs.numpy()
        token_scores = structure_probs.max(axis=2)
        pred_indices = structure_probs.argmax(axis=2)
        ignored = set(int(token) for token in self.post_process_class.get_ignored_tokens())
        end_idx = self.post_process_class.dict[self.post_process_class.end_str]
        plain_td_idx = self.post_process_class.dict.get("<td></td>")

        for batch_idx, ((pred_tokens_raw, _conf), gt_tokens_raw) in enumerate(
            zip(pred_structures, gt_structures)
        ):
            scores = []
            margins = []
            for token_idx in range(len(pred_indices[batch_idx])):
                char_idx = int(pred_indices[batch_idx][token_idx])
                if token_idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored:
                    continue
                scores.append(float(token_scores[batch_idx][token_idx]))
                if plain_td_idx is None:
                    margins.append(None)
                else:
                    margins.append(
                        float(
                            structure_probs[batch_idx][token_idx][char_idx]
                            - structure_probs[batch_idx][token_idx][plain_td_idx]
                        )
                    )

            pred_tokens = strip_wrappers(pred_tokens_raw)
            gt_tokens = strip_wrappers(gt_tokens_raw)
            calibrated_tokens, suppressions = calibrate_tokens(
                pred_tokens,
                scores,
                self.mode,
                self.span_conf_threshold,
                margins=margins,
                span_margin_threshold=self.span_margin_threshold,
            )

            try:
                score = teds_score(calibrated_tokens, gt_tokens)
            except RecursionError:
                score = 0.0
            exact = "".join(calibrated_tokens) == "".join(gt_tokens)

            self.total_teds += score
            self.exact += int(exact)
            self.count += 1
            self.suppressions += suppressions

            if self.fp:
                span_debug = []
                for start, end, _row_span, _col_span, cell_has_span in iter_cells(
                    pred_tokens
                ):
                    if cell_has_span:
                        span_debug.append(
                            {
                                "tokens": pred_tokens[start:end],
                                "min_score": min(scores[start:end])
                                if start < len(scores) and scores[start:end]
                                else None,
                                "min_margin_vs_plain_td": min(
                                    [
                                        value
                                        for value in margins[start:end]
                                        if value is not None
                                    ]
                                )
                                if start < len(margins)
                                and [
                                    value
                                    for value in margins[start:end]
                                    if value is not None
                                ]
                                else None,
                            }
                        )
                self.fp.write(
                    json.dumps(
                        {
                            "sample_index": self.count - 1,
                            "teds": score,
                            "exact": exact,
                            "suppressions": suppressions,
                            "pred_tokens": pred_tokens,
                            "calibrated_tokens": calibrated_tokens,
                            "gt_tokens": gt_tokens,
                            "span_debug": span_debug,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def get_metric(self):
        if self.fp and not self.fp.closed:
            self.fp.flush()
        count = max(self.count, 1)
        result = {
            "teds": self.total_teds / count,
            "structure_acc": self.exact / count,
            "samples": self.count,
            "span_suppressions": self.suppressions,
        }
        self.reset()
        return result

    def reset(self):
        self.total_teds = 0.0
        self.exact = 0
        self.count = 0
        self.suppressions = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="none",
        choices=["none", "confidence", "shape", "hybrid", "margin", "margin_shape"],
    )
    parser.add_argument("--span_conf_threshold", type=float, default=0.0)
    parser.add_argument("--span_margin_threshold", type=float, default=0.0)
    parser.add_argument("--diagnostics_path", default="")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    config, device, logger, _ = program.preprocess()
    global_config = config["Global"]
    set_signal_handlers()

    valid_dataloader = build_dataloader(config, "Eval", device, logger)
    post_process_class = build_post_process(config["PostProcess"], global_config)

    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])

    metric = program.eval(
        model,
        valid_dataloader,
        None,
        CalibratedTEDSMetric(
            post_process_class,
            mode=args.mode,
            span_conf_threshold=args.span_conf_threshold,
            span_margin_threshold=args.span_margin_threshold,
            diagnostics_path=args.diagnostics_path or None,
        ),
        config["Architecture"]["model_type"],
        False,
        None,
        "O2",
        global_config.get("amp_custom_black_list", []),
    )

    logger.info("phase8 calibrated structure TEDS eval ***************")
    logger.info("mode:%s", args.mode)
    logger.info("span_conf_threshold:%s", args.span_conf_threshold)
    logger.info("span_margin_threshold:%s", args.span_margin_threshold)
    for key, value in metric.items():
        logger.info("%s:%s", key, value)


if __name__ == "__main__":
    main()
