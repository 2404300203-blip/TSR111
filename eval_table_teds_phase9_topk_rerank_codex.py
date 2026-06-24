import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

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
    return max(0.0, 1.0 - tree_distance(pred_tree, gt_tree, {}) / denom)


def span_values(token):
    row = 1
    col = 1
    token = str(token)
    if "rowspan" in token:
        row = int(token.split('rowspan="', 1)[1].split('"', 1)[0])
    if "colspan" in token:
        col = int(token.split('colspan="', 1)[1].split('"', 1)[0])
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
            cells.append((idx, idx + 1, False))
            pos += 1
            continue
        if token == "<td":
            end = idx + 1
            cell_has_span = False
            pos += 1
            while pos < len(row):
                next_idx, next_token = row[pos]
                next_token = str(next_token)
                end = next_idx + 1
                pos += 1
                if "rowspan" in next_token or "colspan" in next_token:
                    cell_has_span = True
                if next_token == "</td>":
                    break
            cells.append((idx, end, cell_has_span))
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
        for start, end, _cell_has_span in iter_cells_in_row(row):
            attr = " ".join(str(token) for _idx, token in row if start <= _idx < end)
            _row_span, col_span = span_values(attr)
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


def shape_badness(tokens):
    widths = row_widths(tokens)
    target = modal_width(widths)
    if target <= 0:
        return 0.0
    return float(sum(abs(width - target) for width in widths if width > 0))


def decode_with_positions(post_process_class, structure_probs_batch):
    ignored = set(int(token) for token in post_process_class.get_ignored_tokens())
    end_idx = post_process_class.dict[post_process_class.end_str]
    pred_indices = structure_probs_batch.argmax(axis=1)
    pred_scores = structure_probs_batch.max(axis=1)
    decoded = []
    for raw_idx, char_idx_raw in enumerate(pred_indices):
        char_idx = int(char_idx_raw)
        if raw_idx > 0 and char_idx == end_idx:
            break
        if char_idx in ignored:
            continue
        token = post_process_class.character[char_idx]
        if token in ("<html>", "</html>", "<body>", "</body>"):
            continue
        decoded.append(
            {
                "token": token,
                "raw_idx": raw_idx,
                "char_idx": char_idx,
                "score": float(pred_scores[raw_idx]),
            }
        )
    return decoded


def top_attr_alternatives(character, probs, attr_indices, topk):
    ranked = sorted(
        ((float(probs[idx]), idx, character[idx]) for idx in attr_indices),
        key=lambda item: item[0],
        reverse=True,
    )
    return ranked[:topk]


def rerank_tokens(decoded, structure_probs_batch, post_process_class, topk, shape_lambda):
    tokens = [item["token"] for item in decoded]
    raw_indices = [item["raw_idx"] for item in decoded]
    char_indices = [item["char_idx"] for item in decoded]
    eps = 1e-12
    base_logprob = sum(
        math.log(max(float(structure_probs_batch[raw_idx][char_idx]), eps))
        for raw_idx, char_idx in zip(raw_indices, char_indices)
    )
    base_bad = shape_badness(tokens)
    best_tokens = tokens
    best_score = base_logprob - shape_lambda * base_bad
    changes = 0

    attr_indices = [
        idx
        for idx, token in enumerate(post_process_class.character)
        if "rowspan" in str(token) or "colspan" in str(token)
    ]
    if not attr_indices:
        return tokens, changes

    for start, end, cell_has_span in iter_cells(tokens):
        if not cell_has_span:
            continue
        for pos in range(start, end):
            token = tokens[pos]
            if "rowspan" not in token and "colspan" not in token:
                continue
            raw_idx = raw_indices[pos]
            old_char_idx = char_indices[pos]
            old_logprob = math.log(
                max(float(structure_probs_batch[raw_idx][old_char_idx]), eps)
            )
            for prob, alt_idx, alt_token in top_attr_alternatives(
                post_process_class.character,
                structure_probs_batch[raw_idx],
                attr_indices,
                topk,
            ):
                if alt_idx == old_char_idx:
                    continue
                candidate = list(tokens)
                candidate[pos] = alt_token
                candidate_bad = shape_badness(candidate)
                candidate_logprob = base_logprob - old_logprob + math.log(
                    max(prob, eps)
                )
                candidate_score = candidate_logprob - shape_lambda * candidate_bad
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_tokens = candidate
                    changes = 1

    return best_tokens, changes


class TopKRerankTEDSMetric:
    def __init__(
        self,
        post_process_class,
        topk=3,
        shape_lambda=0.0,
        diagnostics_path=None,
    ):
        self.post_process_class = post_process_class
        self.topk = topk
        self.shape_lambda = shape_lambda
        self.fp = None
        if diagnostics_path:
            os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
            self.fp = open(diagnostics_path, "w", encoding="utf-8")
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds = pred_label
        _post_result, labels = self.post_process_class(preds, batch)
        gt_structures = labels["structure_batch_list"]
        structure_probs = preds["structure_probs"]
        if isinstance(structure_probs, paddle.Tensor):
            structure_probs = structure_probs.numpy()

        for batch_idx, gt_tokens_raw in enumerate(gt_structures):
            decoded = decode_with_positions(
                self.post_process_class, structure_probs[batch_idx]
            )
            pred_tokens = [item["token"] for item in decoded]
            reranked_tokens, changes = rerank_tokens(
                decoded,
                structure_probs[batch_idx],
                self.post_process_class,
                self.topk,
                self.shape_lambda,
            )
            gt_tokens = [
                token
                for token in gt_tokens_raw
                if token not in ("<html>", "</html>", "<body>", "</body>")
            ]
            try:
                score = teds_score(reranked_tokens, gt_tokens)
            except RecursionError:
                score = 0.0
            exact = "".join(reranked_tokens) == "".join(gt_tokens)
            self.total_teds += score
            self.exact += int(exact)
            self.count += 1
            self.changed += changes
            if self.fp:
                self.fp.write(
                    json.dumps(
                        {
                            "sample_index": self.count - 1,
                            "teds": score,
                            "exact": exact,
                            "changed": changes,
                            "pred_tokens": pred_tokens,
                            "reranked_tokens": reranked_tokens,
                            "gt_tokens": gt_tokens,
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
            "changed_samples": self.changed,
        }
        self.reset()
        return result

    def reset(self):
        self.total_teds = 0.0
        self.exact = 0
        self.count = 0
        self.changed = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--shape_lambda", type=float, default=0.0)
    parser.add_argument("--diagnostics_path", default="")
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    config, device, logger, _ = program.preprocess()
    global_config = config["Global"]
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, "Eval", device, logger)
    post_process_class = build_post_process(config["PostProcess"], global_config)
    if hasattr(post_process_class, "character"):
        config["Architecture"]["Head"]["out_channels"] = len(
            post_process_class.character
        )
    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    metric = program.eval(
        model,
        valid_dataloader,
        None,
        TopKRerankTEDSMetric(
            post_process_class,
            topk=args.topk,
            shape_lambda=args.shape_lambda,
            diagnostics_path=args.diagnostics_path or None,
        ),
        config["Architecture"]["model_type"],
        False,
        None,
        "O2",
        global_config.get("amp_custom_black_list", []),
    )
    logger.info("phase9 top-k rerank structure TEDS eval ***************")
    logger.info("topk:%s", args.topk)
    logger.info("shape_lambda:%s", args.shape_lambda)
    for key, value in metric.items():
        logger.info("%s:%s", key, value)


if __name__ == "__main__":
    main()
