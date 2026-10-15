#!/usr/bin/env python3
"""Evaluate the official UniTable Large structure checkpoint on CTA-500.

The metric implementation and exact-match token convention intentionally
match ``eval_table_teds_codex.py`` and PaddleOCR's TableLabelEncode with
``merge_no_span_structure=true``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import tokenizers as tk
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms


@dataclass
class Node:
    label: str
    children: list["Node"] = field(default_factory=list)


def normalize_token(token: object) -> str:
    token = str(token).strip()
    if token.startswith("</"):
        return token
    if token.startswith("<") and token.endswith(">"):
        return token
    return "#text"


def build_tree(tokens: Sequence[str]) -> Node:
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


def tree_size(node: Node) -> int:
    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        total += 1
        stack.extend(current.children)
    return total


def tree_distance(a: Node, b: Node, memo: dict[tuple[int, int], int]) -> int:
    key = (id(a), id(b))
    if key in memo:
        return memo[key]
    rename = 0 if a.label == b.label else 1
    ac, bc = a.children, b.children
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


def teds_score(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    pred_tree = build_tree(pred_tokens)
    gt_tree = build_tree(gt_tokens)
    denominator = max(tree_size(pred_tree), tree_size(gt_tree), 1)
    distance = tree_distance(pred_tree, gt_tree, {})
    return max(0.0, 1.0 - distance / denominator)


def paddle_merge_no_span_structure(tokens: Sequence[str]) -> list[str]:
    """Mirror TableLabelEncode._merge_no_span_structure."""
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "<td>":
            merged.append("<td></td>")
            index += 2
        else:
            if token:
                if "span" in token and not token.startswith(" "):
                    token = " " + token
                merged.append(token)
            index += 1
    return merged


def unitable_to_paddle_tokens(tokens: Sequence[str]) -> list[str]:
    """Map UniTable's equivalent cell tokens to PaddleOCR's token grammar."""
    output: list[str] = []
    for token in tokens:
        stripped = token.strip()
        # The tokenizers decoder inserts a separator before a vocab token that
        # already starts with a space. html_str_to_token_list therefore emits
        # an extra whitespace-only item before rowspan/colspan attributes.
        if not stripped:
            continue
        if stripped.startswith(("rowspan=", "colspan=")):
            output.append(" " + stripped)
        elif token == "<td>[]</td>":
            output.append("<td></td>")
        elif token in {">[]</td>", "></td>"}:
            output.extend([">", "</td>"])
        elif token not in {"<html>", "</html>", "<body>", "</body>"}:
            output.append(token)
    return output


def autoregressive_decode(
    model: nn.Module,
    image: Tensor,
    prefix: Sequence[int],
    max_decode_len: int,
    eos_id: int,
    token_whitelist: Sequence[int],
    device: torch.device,
    subsequent_mask,
) -> Tensor:
    model.eval()
    with torch.inference_mode():
        memory = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.long, device=device).repeat(image.shape[0], 1)
        whitelist = torch.tensor(token_whitelist, dtype=torch.long, device=device)
        for _ in range(max_decode_len):
            if torch.all(torch.any(context == eos_id, dim=1)).item():
                break
            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=causal_mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]
            allowed = torch.full_like(logits, -torch.inf)
            allowed[:, whitelist] = logits[:, whitelist]
            next_tokens = allowed.argmax(dim=-1, keepdim=True)
            context = torch.cat([context, next_tokens], dim=1)
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitable-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-decode-len", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.unitable_root))

    from src.model import Decoder, Encoder, EncoderDecoder, ImgLinearBackbone
    from src.utils import html_str_to_token_list, subsequent_mask
    from src.vocab import HTML_TOKENS

    device = torch.device("cuda:0")
    d_model = 768
    dropout = 0.2
    backbone = ImgLinearBackbone(d_model=d_model, patch_size=16)
    encoder = Encoder(
        d_model=d_model,
        nhead=12,
        dropout=dropout,
        activation="gelu",
        norm_first=True,
        nlayer=12,
        ff_ratio=4,
    )
    decoder = Decoder(
        d_model=d_model,
        nhead=12,
        dropout=dropout,
        activation="gelu",
        norm_first=True,
        nlayer=4,
        ff_ratio=4,
    )
    vocab = tk.Tokenizer.from_file(str(args.unitable_root / "vocab" / "vocab_html.json"))
    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=784,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.86597056, 0.88463002, 0.87491087],
                std=[0.20686628, 0.18201602, 0.18485524],
            ),
        ]
    )
    prefix = [vocab.token_to_id("[html]")]
    eos_id = vocab.token_to_id("<eos>")
    whitelist = [vocab.token_to_id(token) for token in ["<eos>", *HTML_TOKENS]]

    # Do not use str.splitlines(): PubTabNet cell text can contain the Unicode
    # line-separator character U+2028 inside an otherwise valid JSONL record.
    with args.labels.open("r", encoding="utf-8") as label_handle:
        records = [json.loads(line) for line in label_handle if line.strip()]
    if args.limit > 0:
        records = records[: args.limit]

    def infer(record: dict) -> tuple[list[str], float]:
        start = time.perf_counter()
        image = Image.open(args.image_dir / record["filename"]).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        pred_ids = autoregressive_decode(
            model,
            image_tensor,
            prefix,
            args.max_decode_len,
            eos_id,
            whitelist,
            device,
            subsequent_mask,
        )
        pred_text = vocab.decode(pred_ids.detach().cpu().numpy()[0], skip_special_tokens=False)
        pred_tokens = unitable_to_paddle_tokens(html_str_to_token_list(pred_text))
        torch.cuda.synchronize()
        return pred_tokens, time.perf_counter() - start

    for warmup_record in records[: min(args.warmup, len(records))]:
        infer(warmup_record)

    rows = []
    output_path = args.output_dir / "predictions.jsonl"
    total_teds = 0.0
    total_exact = 0
    total_seconds = 0.0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            pred_tokens, seconds = infer(record)
            gt_tokens = paddle_merge_no_span_structure(record["html"]["structure"]["tokens"])
            try:
                score = teds_score(pred_tokens, gt_tokens)
            except RecursionError:
                score = 0.0
            exact = int("".join(pred_tokens) == "".join(gt_tokens))
            row = {
                "index": index - 1,
                "filename": record["filename"],
                "pred_tokens": pred_tokens,
                "gt_tokens": gt_tokens,
                "teds": score,
                "exact": exact,
                "seconds": seconds,
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
                    f"Exact={total_exact/index:.6f} FPS={index/total_seconds:.3f}",
                    flush=True,
                )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "model": "UniTable Large",
        "status": "zero-shot external baseline",
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "teds": total_teds / max(len(rows), 1),
        "exact_structure_accuracy": total_exact / max(len(rows), 1),
        "fps": len(rows) / max(total_seconds, 1e-12),
        "mean_latency_ms": 1000.0 * total_seconds / max(len(rows), 1),
        "parameters": parameters,
        "warmup_samples": min(args.warmup, len(records)),
        "batch_size": 1,
        "image_size": [448, 448],
        "max_decode_len": args.max_decode_len,
        "timing_scope": "image load + RGB conversion + preprocessing + GPU encode/decode + token postprocessing",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
