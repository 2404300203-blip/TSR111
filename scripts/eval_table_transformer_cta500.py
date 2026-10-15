#!/usr/bin/env python3
"""Evaluate Microsoft's Table Transformer v1.0 structure model on CTA-500.

The official Microsoft preprocessing and structural postprocessing are used.
Only the final cell-grid serialization is adapted to PaddleOCR's PubTabNet
token grammar so that the existing TEDS and exact-match evaluator can be used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import TableTransformerConfig, TableTransformerForObjectDetection

from eval_unitable_cta500 import (
    paddle_merge_no_span_structure,
    teds_score,
)


CLASS_IDX_TO_NAME = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
    6: "no object",
}


def cells_to_paddle_tokens(cells: Sequence[dict]) -> list[str]:
    """Serialize official TATR cells as PaddleOCR PubTabNet structure tokens."""
    if not cells:
        return []

    max_row = max(max(cell["row_nums"]) for cell in cells)
    cells_by_start_row: dict[int, list[dict]] = {row: [] for row in range(max_row + 1)}
    header_rows: set[int] = set()
    for cell in cells:
        start_row = min(cell["row_nums"])
        cells_by_start_row[start_row].append(cell)
        if cell.get("column header", False):
            header_rows.update(cell["row_nums"])

    # PubTabNet uses one leading thead section; any later isolated header
    # detections are treated as body rows to preserve valid section ordering.
    leading_header_rows: set[int] = set()
    for row in range(max_row + 1):
        if row in header_rows:
            leading_header_rows.add(row)
        else:
            break

    tokens: list[str] = []
    active_section = None
    for row in range(max_row + 1):
        section = "thead" if row in leading_header_rows else "tbody"
        if section != active_section:
            if active_section is not None:
                tokens.append(f"</{active_section}>")
            tokens.append(f"<{section}>")
            active_section = section
        tokens.append("<tr>")
        for cell in sorted(cells_by_start_row[row], key=lambda item: min(item["column_nums"])):
            rowspan = len(cell["row_nums"])
            colspan = len(cell["column_nums"])
            if rowspan == 1 and colspan == 1:
                tokens.append("<td></td>")
                continue
            tokens.append("<td")
            if rowspan > 1:
                tokens.append(f' rowspan="{rowspan}"')
            if colspan > 1:
                tokens.append(f' colspan="{colspan}"')
            tokens.extend([">", "</td>"])
        tokens.append("</tr>")
    if active_section is not None:
        tokens.append(f"</{active_section}>")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microsoft-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    microsoft_src = args.microsoft_root / "src"
    sys.path.insert(0, str(args.microsoft_root / "detr"))
    sys.path.insert(0, str(microsoft_src))

    import inference as microsoft_inference

    device = torch.device("cuda:0")
    config = TableTransformerConfig(
        backbone="resnet18",
        use_pretrained_backbone=False,
        num_queries=125,
        num_labels=6,
        eos_coefficient=0.4,
        auxiliary_loss=False,
    )
    model = TableTransformerForObjectDetection(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    with args.labels.open("r", encoding="utf-8") as label_handle:
        records = [json.loads(line) for line in label_handle if line.strip()]
    if args.limit > 0:
        records = records[: args.limit]

    def infer(record: dict) -> tuple[list[str], float, int, int, int]:
        start = time.perf_counter()
        image = Image.open(args.image_dir / record["filename"]).convert("RGB")
        pixel_values = microsoft_inference.structure_transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(pixel_values=pixel_values)
        raw_output = {"pred_logits": output.logits, "pred_boxes": output.pred_boxes}
        objects = microsoft_inference.outputs_to_objects(
            raw_output, image.size, CLASS_IDX_TO_NAME
        )
        structures = microsoft_inference.objects_to_structures(
            objects, [], microsoft_inference.structure_class_thresholds
        )
        if structures:
            cells, _ = microsoft_inference.structure_to_cells(structures[0], [])
            pred_tokens = cells_to_paddle_tokens(cells)
        else:
            cells = []
            pred_tokens = []
        torch.cuda.synchronize()
        return pred_tokens, time.perf_counter() - start, len(objects), len(structures), len(cells)

    for warmup_record in records[: min(args.warmup, len(records))]:
        infer(warmup_record)

    rows = []
    total_teds = 0.0
    total_exact = 0
    total_seconds = 0.0
    no_structure = 0
    output_path = args.output_dir / "predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            pred_tokens, seconds, num_objects, num_structures, num_cells = infer(record)
            gt_tokens = paddle_merge_no_span_structure(record["html"]["structure"]["tokens"])
            try:
                score = teds_score(pred_tokens, gt_tokens)
            except (RecursionError, ZeroDivisionError, ValueError):
                score = 0.0
            exact = int("".join(pred_tokens) == "".join(gt_tokens))
            no_structure += int(num_structures == 0)
            row = {
                "index": index - 1,
                "filename": record["filename"],
                "pred_tokens": pred_tokens,
                "gt_tokens": gt_tokens,
                "teds": score,
                "exact": exact,
                "seconds": seconds,
                "num_objects": num_objects,
                "num_structures": num_structures,
                "num_cells": num_cells,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            total_teds += score
            total_exact += exact
            total_seconds += seconds
            if index == 1 or index % 10 == 0 or index == len(records):
                print(
                    f"[{index}/{len(records)}] TEDS={total_teds/index:.6f} "
                    f"Exact={total_exact/index:.6f} FPS={index/total_seconds:.3f} "
                    f"NoStructure={no_structure}",
                    flush=True,
                )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "model": "Table Transformer v1.0 (structure, R18)",
        "status": "zero-shot external baseline",
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "teds": total_teds / max(len(rows), 1),
        "exact_structure_accuracy": total_exact / max(len(rows), 1),
        "fps": len(rows) / max(total_seconds, 1e-12),
        "mean_latency_ms": 1000.0 * total_seconds / max(len(rows), 1),
        "parameters": parameters,
        "no_structure_samples": no_structure,
        "warmup_samples": min(args.warmup, len(records)),
        "batch_size": 1,
        "max_image_side": 1000,
        "object_threshold": 0.5,
        "timing_scope": "image load + RGB conversion + preprocessing + GPU inference + official Microsoft postprocessing + token serialization",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
