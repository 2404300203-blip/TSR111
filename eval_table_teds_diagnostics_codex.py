import json
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
    dist = tree_distance(pred_tree, gt_tree, {})
    return max(0.0, 1.0 - dist / denom)


def load_eval_records(config):
    dataset_cfg = config["Eval"]["dataset"]
    records = []
    for label_file in dataset_cfg["label_file_list"]:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                info = json.loads(line)
                records.append(
                    {
                        "filename": info.get("filename", ""),
                        "image_path": os.path.join(
                            dataset_cfg.get("data_dir", ""), info.get("filename", "")
                        ),
                    }
                )
    return records


class StructureTEDSDiagnostics:
    def __init__(self, output_path, records=None, main_indicator="teds", **kwargs):
        self.output_path = output_path
        self.records = records or []
        self.main_indicator = main_indicator
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.fp = open(output_path, "w", encoding="utf-8")
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        pred_structures = preds["structure_batch_list"]
        gt_structures = labels["structure_batch_list"]
        for (pred, conf), gt in zip(pred_structures, gt_structures):
            sample_index = self.count
            record = self.records[sample_index] if sample_index < len(self.records) else {}
            pred_tokens = [
                t for t in pred if t not in ("<html>", "</html>", "<body>", "</body>")
            ]
            gt_tokens = [
                t for t in gt if t not in ("<html>", "</html>", "<body>", "</body>")
            ]
            try:
                score = teds_score(pred_tokens, gt_tokens)
            except RecursionError:
                score = 0.0
            exact = "".join(pred_tokens) == "".join(gt_tokens)
            self.total_teds += score
            self.exact += int(exact)
            self.count += 1
            row = {
                "sample_index": sample_index,
                "filename": record.get("filename", ""),
                "image_path": record.get("image_path", ""),
                "teds": score,
                "exact": exact,
                "confidence": float(conf) if conf is not None else None,
                "pred_len": len(pred_tokens),
                "gt_len": len(gt_tokens),
                "pred_tokens": pred_tokens,
                "gt_tokens": gt_tokens,
            }
            self.fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def get_metric(self):
        if not self.fp.closed:
            self.fp.flush()
            self.fp.close()
        count = max(self.count, 1)
        result = {
            "teds": self.total_teds / count,
            "structure_acc": self.exact / count,
            "samples": self.count,
            "diagnostics_path": self.output_path,
        }
        self.reset()
        return result

    def reset(self):
        self.total_teds = 0.0
        self.count = 0
        self.exact = 0


def main():
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

    diagnostics_path = global_config.get(
        "diagnostics_path",
        os.path.join(global_config["save_model_dir"], "eval", "sample_teds.jsonl"),
    )
    metric = program.eval(
        model,
        valid_dataloader,
        post_process_class,
        StructureTEDSDiagnostics(diagnostics_path, load_eval_records(config)),
        config["Architecture"]["model_type"],
        False,
        None,
        "O2",
        global_config.get("amp_custom_black_list", []),
    )

    logger.info("structure TEDS diagnostics eval ***************")
    for key, value in metric.items():
        logger.info("%s:%s", key, value)


if __name__ == "__main__":
    main()
