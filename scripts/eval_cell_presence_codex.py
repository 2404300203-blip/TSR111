import json
import os
import sys

import numpy as np
import paddle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
import tools.program as program


def sigmoid(x):
    return 1.0 / (1.0 + paddle.exp(-x))


def binary_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype="float32")
    labels = np.asarray(labels, dtype="float32")
    pred = scores >= threshold
    target = labels >= 0.5
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    config, device, logger, _ = program.preprocess()
    global_config = config["Global"]
    set_signal_handlers()

    valid_dataloader = build_dataloader(config, "Eval", device, logger)
    post_process_class = build_post_process(config["PostProcess"], global_config)
    if hasattr(post_process_class, "character"):
        config["Architecture"]["Head"]["out_channels"] = len(post_process_class.character)

    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()

    all_scores = []
    all_labels = []
    per_batch = []
    dropped_positions = 0

    with paddle.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader):
            images = batch[0]
            preds = model(images, data=batch[1:])
            if "cell_has_text_logits" not in preds:
                raise RuntimeError("Model output does not contain cell_has_text_logits.")

            max_len = int(batch[-2].max().numpy())
            logits = preds["cell_has_text_logits"][:, : max_len + 1]
            bbox_mask = batch[3].astype("float32")[:, : max_len + 1]
            target = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)
            common_len = min(logits.shape[1], target.shape[1])
            dropped_positions += int((logits.shape[1] - common_len) * logits.shape[0])
            dropped_positions += int((target.shape[1] - common_len) * target.shape[0])
            logits = logits[:, :common_len]
            target = target[:, :common_len]
            score = sigmoid(logits)

            scores_np = score.flatten().numpy()
            labels_np = target.flatten().numpy()
            all_scores.extend(float(x) for x in scores_np)
            all_labels.extend(float(x) for x in labels_np)
            per_batch.append(
                {
                    "batch_idx": batch_idx,
                    "positions": int(labels_np.size),
                    "positive": int((labels_np >= 0.5).sum()),
                    "score_mean": float(scores_np.mean()) if scores_np.size else 0.0,
                }
            )

    thresholds = [i / 100.0 for i in range(5, 96, 5)]
    metrics = [binary_metrics(all_scores, all_labels, t) for t in thresholds]
    at_05 = binary_metrics(all_scores, all_labels, 0.5)
    best = max(metrics, key=lambda x: x["f1"])

    positives = sum(1 for x in all_labels if x >= 0.5)
    total = len(all_labels)
    result = {
        "samples": len(valid_dataloader.dataset) if hasattr(valid_dataloader, "dataset") else None,
        "positions": total,
        "dropped_positions_due_to_length_mismatch": dropped_positions,
        "positives": positives,
        "negatives": total - positives,
        "positive_rate": positives / max(total, 1),
        "threshold_0.5": at_05,
        "best_threshold": best,
        "threshold_sweep": metrics,
        "per_batch": per_batch,
    }

    output_path = global_config.get(
        "presence_diagnostics_path",
        "output/cell_presence_metrics/cell_presence_metrics.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("cell presence metrics ***************")
    logger.info("presence_diagnostics_path:%s", output_path)
    logger.info("positions:%s positives:%s positive_rate:%s", total, positives, result["positive_rate"])
    logger.info("threshold_0.5:%s", at_05)
    logger.info("best_threshold:%s", best)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
