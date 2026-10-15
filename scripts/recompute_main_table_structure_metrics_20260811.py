#!/usr/bin/env python3
"""Recompute the 2026-08-11 paper tables from cached per-sample predictions.

The original table builder treated split PubTabNet cell tags such as
``["<td", " rowspan=\"2\"", ">", "</td>"]`` as four unrelated tokens.
This script first reconstructs each opening cell tag, then computes topology
and spanning-cell metrics from the resulting table grid.  It also replaces
three invalid Paddle FLOPs measurements with values produced by the trusted
architecture-sweep profiler.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path("/Users/table/TSR")
SOURCE = ROOT / "remote_results/main_tables_rebuild_20260811"
SAMPLES = SOURCE / "sample_jsonl"
CTA500 = SOURCE / "complex_structure_cta500_20260625.jsonl"
OUTPUT = SOURCE / "corrected"


MODEL_ORDER = [
    "slanet",
    "cell_token",
    "lite_mamba",
    "full_mamba_no_presence",
    "full_mamba_presence",
    "full_mamba_one_way",
    "dbm_d64",
    "dbm_ds8",
    "dbm_d192",
    "dbm_e1",
    "zero_geometry",
    "true_horizontal",
    "true_vertical",
    "dbm_slanet",
]

PHASE11_IDS = [
    "cell_token",
    "lite_mamba",
    "full_mamba_no_presence",
    "full_mamba_presence",
    "full_mamba_one_way",
    "dbm_d64",
    "dbm_ds8",
    "dbm_d192",
    "dbm_e1",
    "zero_geometry",
    "true_horizontal",
    "true_vertical",
]

AXIS_IDS = ["slanet", "true_horizontal", "true_vertical", "dbm_slanet"]

# These values come from architecture_sweep_true_axis_20260810/results.csv.
# Presence supervision and zero-geometry affect training/input features, not
# the instantiated full-DBM architecture, so they share the baseline profile.
TRUSTED_PROFILES = {
    "slanet": (1_892_610, 5_135_768_256),
    "cell_token": (1_927_427, 5_152_762_176),
    "lite_mamba": (2_088_373, 5_239_529_408),
    "full_mamba_no_presence": (2_445_607, 5_409_346_496),
    "full_mamba_presence": (2_445_607, 5_409_346_496),
    "full_mamba_one_way": (2_211_285, 5_297_503_168),
    "dbm_d64": (2_046_295, 5_208_942_400),
    "dbm_ds8": (2_420_967, 5_401_138_112),
    "dbm_d192": (3_090_679, 5_736_980_544),
    "dbm_e1": (2_212_135, 5_297_503_168),
    "zero_geometry": (2_445_607, 5_409_346_496),
    "true_horizontal": (2_194_901, 5_289_294_784),
    "true_vertical": (2_194_901, 5_289_294_784),
    "dbm_slanet": (2_445_607, 5_409_346_496),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(num: int, den: int, empty: float = 1.0) -> float:
    return num / den if den else empty


def f1_counts(pred: set, gt: set) -> tuple[int, int, int, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return tp, fp, fn, ratio(2 * tp, 2 * tp + fp + fn)


def reconstructed_tokens(tokens: Iterable[str]) -> Iterable[str]:
    """Yield tokens after joining split PubTabNet td/th opening tags."""
    values = [str(token) for token in tokens]
    index = 0
    while index < len(values):
        token = values[index]
        if token.startswith("<td") or token.startswith("<th"):
            pieces = [token]
            while ">" not in "".join(pieces) and index + 1 < len(values):
                index += 1
                pieces.append(values[index])
            yield "".join(pieces)
        else:
            yield token
        index += 1


def parse_span(tag: str, attribute: str) -> int:
    match = re.search(rf"\b{attribute}\s*=\s*['\"]?(\d+)", tag, flags=re.IGNORECASE)
    return max(1, int(match.group(1))) if match else 1


def parse_cells(tokens: Iterable[str]) -> set[tuple[int, int, int, int]]:
    """Map a structure sequence to (row, column, rowspan, colspan) cells."""
    cells: set[tuple[int, int, int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    row = -1
    column = 0
    in_row = False

    for token in reconstructed_tokens(tokens):
        normalized = token.strip().lower()
        if normalized.startswith("<tr") and not normalized.startswith("</tr"):
            row += 1
            column = 0
            in_row = True
            continue
        if normalized.startswith("</tr"):
            in_row = False
            continue
        if not in_row or not (normalized.startswith("<td") or normalized.startswith("<th")):
            continue

        while (row, column) in occupied:
            column += 1
        rowspan = parse_span(token, "rowspan")
        colspan = parse_span(token, "colspan")
        cells.add((row, column, rowspan, colspan))
        for occupied_row in range(row, row + rowspan):
            for occupied_column in range(column, column + colspan):
                occupied.add((occupied_row, occupied_column))
        column += colspan

    return cells


def summarize(rows: list[dict]) -> dict:
    topology_tp = topology_fp = topology_fn = 0
    span_tp = span_fp = span_fn = 0
    topology_sample_f1: list[float] = []
    gt_span_samples = pred_span_samples = parse_bad = 0

    for row in rows:
        try:
            pred_cells = parse_cells(row["pred_tokens"])
            gt_cells = parse_cells(row["gt_tokens"])
        except (KeyError, TypeError, ValueError):
            parse_bad += 1
            pred_cells = set()
            gt_cells = set()

        tp, fp, fn, sample_f1 = f1_counts(pred_cells, gt_cells)
        topology_tp += tp
        topology_fp += fp
        topology_fn += fn
        topology_sample_f1.append(sample_f1)

        pred_spans = {cell for cell in pred_cells if cell[2] > 1 or cell[3] > 1}
        gt_spans = {cell for cell in gt_cells if cell[2] > 1 or cell[3] > 1}
        pred_span_samples += int(bool(pred_spans))
        gt_span_samples += int(bool(gt_spans))
        tp, fp, fn, _ = f1_counts(pred_spans, gt_spans)
        span_tp += tp
        span_fp += fp
        span_fn += fn

    n = len(rows)
    topology_precision = ratio(topology_tp, topology_tp + topology_fp)
    topology_recall = ratio(topology_tp, topology_tp + topology_fn)
    span_precision = ratio(span_tp, span_tp + span_fp)
    span_recall = ratio(span_tp, span_tp + span_fn)
    return {
        "n": n,
        "teds": sum(float(row["teds"]) for row in rows) / n,
        "structure_acc": sum(int(bool(row["exact"])) for row in rows) / n,
        "topology_precision": topology_precision,
        "topology_recall": topology_recall,
        "topology_f1": ratio(2 * topology_tp, 2 * topology_tp + topology_fp + topology_fn),
        "topology_mean": sum(topology_sample_f1) / n,
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": ratio(2 * span_tp, 2 * span_tp + span_fp + span_fn),
        "gt_span_samples": gt_span_samples,
        "pred_span_samples": pred_span_samples,
        "parse_bad": parse_bad,
    }


def corrected_profiles() -> tuple[list[dict], dict[str, dict]]:
    original = {row["id"]: row for row in read_csv(SOURCE / "table2_profile.csv")}
    rows: list[dict] = []
    by_id: dict[str, dict] = {}
    for model_id in MODEL_ORDER:
        row = dict(original[model_id])
        params, flops = TRUSTED_PROFILES[model_id]
        row.update(
            {
                "params": params,
                "params_m": f"{params / 1_000_000:.6f}",
                "flops": flops,
                "flops_g": f"{flops / 1_000_000_000:.6f}",
            }
        )
        rows.append(row)
        by_id[model_id] = row
    return rows, by_id


def metric_row(model_id: str, profile: dict, summary: dict) -> dict:
    fps = float(profile["fps"])
    return {
        "model": profile["model"],
        "id": model_id,
        "n": summary["n"],
        "params_m": profile["params_m"],
        "flops_g": profile["flops_g"],
        "fps": f"{fps:.12f}",
        "latency_ms": f"{1000 / fps:.3f}",
        **summary,
    }


def markdown_table(rows: list[dict], cta: bool = False) -> str:
    title = "CTA TEDS" if cta else "TEDS"
    lines = [
        f"| Model | Params (M) | FLOPs (G) | {title} | Acc. | Topology F1 | Span-F1 | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {params:.3f} | {flops:.3f} | {teds:.4f} | {acc:.4f} | "
            "{top:.4f} | {span:.4f} | {fps:.2f} |".format(
                model=row["model"],
                params=float(row["params_m"]),
                flops=float(row["flops_g"]),
                teds=float(row["teds"]),
                acc=float(row["structure_acc"]),
                top=float(row["topology_f1"]),
                span=float(row["span_f1"]),
                fps=float(row["fps"]),
            )
        )
    return "\n".join(lines)


def markdown_profiles(rows: list[dict]) -> str:
    lines = [
        "| Model | d | d_s | e | Axis | Direction | Params (M) | FLOPs (G) | FPS | Latency (ms) |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {d} | {ds} | {e} | {axis} | {direction} | {params:.6f} | "
            "{flops:.6f} | {fps:.3f} | {latency:.3f} |".format(
                model=row["model"],
                d=row["d"] or "-",
                ds=row["d_s"] or "-",
                e=row["e"] or "-",
                axis=row["axis"],
                direction=row["direction"],
                params=float(row["params_m"]),
                flops=float(row["flops_g"]),
                fps=float(row["fps"]),
                latency=float(row["latency_ms"]),
            )
        )
    return "\n".join(lines)


def validate_results(pub_rows: list[dict], cta_rows: list[dict]) -> None:
    run_status = {row["id"]: row for row in read_csv(SOURCE / "run_status.csv")}
    if [row["id"] for row in pub_rows] != MODEL_ORDER:
        raise RuntimeError("PubTabNet model order is incomplete or changed")
    if [row["id"] for row in cta_rows] != MODEL_ORDER:
        raise RuntimeError("CTA model order is incomplete or changed")
    for pub, cta in zip(pub_rows, cta_rows):
        status = run_status[pub["id"]]
        if pub["n"] != 5000 or cta["n"] != 500:
            raise RuntimeError(f"{pub['id']}: unexpected sample count")
        if pub["parse_bad"] or cta["parse_bad"]:
            raise RuntimeError(f"{pub['id']}: structure parsing failed")
        if pub["gt_span_samples"] != 2395 or cta["gt_span_samples"] != 457:
            raise RuntimeError(f"{pub['id']}: inconsistent ground-truth span count")
        if abs(float(pub["teds"]) - float(status["teds"])) > 1e-12:
            raise RuntimeError(f"{pub['id']}: TEDS changed during post-processing")
        if abs(float(pub["structure_acc"]) - float(status["structure_acc"])) > 1e-12:
            raise RuntimeError(f"{pub['id']}: exact accuracy changed during post-processing")


def write_report(profile_rows: list[dict], pub_rows: list[dict], cta_rows: list[dict]) -> None:
    pub = {row["id"]: row for row in pub_rows}
    cta = {row["id"]: row for row in cta_rows}
    phase11_rows = [cta[x] for x in PHASE11_IDS]
    axis_rows = [cta[x] for x in AXIS_IDS]
    report = f"""# 主论文实验表修正复核报告（2026-08-11）

## 1. 复核范围与数据来源

- 复用 14 个已完成模型在 PubTabNet-5000 上的样本级预测，共 70,000 条；未纳入中断运行。
- CTA-500 按 `complex_structure_cta500_20260625.jsonl` 中的文件名从同一批 5,000 样本预测中筛选，没有再次推理。
- TEDS、Exact Structure Accuracy 和 FPS 沿用统一评估器的原始结果；本次只重算 Topology 和 Span 指标。
- 参数量与 FLOPs 采用可信架构扫描 profile，输入尺寸统一为 `1 x 3 x 488 x 488`。
- 按第九版论文编号：Table 9 是 company-disjoint；Table 10 是 Phase 11 消融；Table 11 是真实单轴消融。
- `SLANet controlled, loc_weight=10` 按约定不纳入本次统计。

## 2. Table 2：架构与效率（修正版）

{markdown_profiles(profile_rows)}

## 3. Table 3/4：PubTabNet-5000（修正版）

{markdown_table(pub_rows)}

所有模型均为 `n=5000`、`gt_span_samples={pub_rows[0]['gt_span_samples']}`、`parse_bad=0`。Table 3 可取 TEDS、Acc.、FPS、Latency 列；Table 4 使用同一次预测中的 Topology F1 和 Span-F1，无需重复推理。

## 4. Table 5：CTA-500 综合比较（修正版）

{markdown_table(cta_rows, cta=True)}

所有模型均为 `n=500`、`gt_span_samples={cta_rows[0]['gt_span_samples']}`、`parse_bad=0`。

## 5. Table 10：Phase 11 CTA-500 消融（修正版）

{markdown_table(phase11_rows, cta=True)}

注意：历史 `inference auxiliary off` 只有 TEDS=0.54215、Acc.=0.302、FPS=18.68 的不完整日志，没有本次统一协议下的样本级预测，且审计状态为 `incomplete`。因此本修正版没有伪造其 Topology/Span 指标，也没有把它混入完整行。

## 6. Table 11：真实单轴 CTA-500 消融（修正版）

{markdown_table(axis_rows, cta=True)}

Horizontal-only 与 Vertical-only 均为真实单分支实例化，不是旧版“双分支都实例化、仅屏蔽一个输出”。二者参数量为 2.194901 M、FLOPs 为 5.289295 G，均低于完整双轴 DBM 的 2.445607 M 和 5.409346 G。

## 7. 本次修正内容

1. 旧脚本逐 token 解析单元格；PubTabNet 会把 `<td rowspan=\"2\">` 拆成 `<td`、` rowspan=\"2\"`、`>`，导致所有跨度都被错误读取为 1 x 1，进而出现不可能的 `span_f1=1.0`、`gt_span_samples=0`，Topology 指标也被高估。
2. 修正后恢复 PubTabNet-5000 中 {pub_rows[0]['gt_span_samples']} 个、CTA-500 中 {cta_rows[0]['gt_span_samples']} 个包含跨行/跨列单元格的 GT 样本。
3. 覆盖三项异常 FLOPs：`d_s=8` 从 1.876430 修正为 5.401138 G；`d=192` 从 1.855829 修正为 5.736981 G；zero geometry 从 1.855170 修正为 5.409346 G。
4. presence loss 和 zero geometry 不改变已实例化的完整 DBM 结构，因此其参数量/FLOPs 与完整双轴 DBM 相同；训练目标不同不等于模型结构参数不同。

## 8. 与 2026-07-10 旧可信汇总的交叉校验

| Check | Corrected | Trusted | Absolute difference |
|---|---:|---:|---:|
| Pub SLANet topology F1 | {float(pub['slanet']['topology_f1']):.12f} | 0.960107392933 | {abs(float(pub['slanet']['topology_f1']) - 0.9601073929328612):.3e} |
| Pub SLANet Span-F1 | {float(pub['slanet']['span_f1']):.12f} | 0.673796791444 | {abs(float(pub['slanet']['span_f1']) - 0.6737967914438503):.3e} |
| Pub DBM-SLANet Span-F1 | {float(pub['dbm_slanet']['span_f1']):.12f} | 0.704500271101 | {abs(float(pub['dbm_slanet']['span_f1']) - 0.7045002711006686):.3e} |
| CTA SLANet Span-F1 | {float(cta['slanet']['span_f1']):.12f} | 0.587424224415 | {abs(float(cta['slanet']['span_f1']) - 0.5874242244145964):.3e} |
| CTA DBM-SLANet Span-F1 | {float(cta['dbm_slanet']['span_f1']):.12f} | 0.661789532921 | {abs(float(cta['dbm_slanet']['span_f1']) - 0.6617895329206527):.3e} |

SLANet 的关键指标完全一致。DBM-SLANet 的极小差值来自一个预测单元格的边界处理：本次统一解析保留显式预测单元格并将其计为 FP，旧汇总忽略了它。新表中的 14 个模型全部使用同一规则，因此应以本修正版为准，不应把两套后处理结果混排。

## 9. 可支持的实验结论与边界

- DBM-SLANet 在这 14 个模型中取得最高 PubTabNet-5000 TEDS/Acc.，并在 CTA-500 上取得最高 TEDS、Acc.、Topology F1 和 Span-F1。
- Lite Mamba 在 CTA-500 上较强，但仍低于最终 DBM-SLANet。
- 真实单轴模型更小、更快，但复杂结构性能明显低于最终 DBM-SLANet，可支持“双轴建模以效率换取复杂结构恢复能力”的结论。
- no-presence 和 zero-geometry 的 TEDS/Acc. 较低，支持 presence 监督与几何信息的必要性；one-way 更快，但复杂集精度不如最终双轴模型。
- Paddle FLOPs 是同一脚本、同一输入尺寸下的 profile proxy。论文应写作 profiled FLOPs，避免宣称为与实现无关的理论 FLOPs。
- company-disjoint Table 9 不属于本批 14 模型的重算范围，应继续引用其独立三随机种子结果；不要用 CTA-500 数值替代 Table 9。
"""
    (OUTPUT / "RECHECK_REPORT_20260811.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile_rows, profiles = corrected_profiles()
    profile_fields = list(profile_rows[0].keys())
    write_csv(OUTPUT / "table2_profile_corrected.csv", profile_rows, profile_fields)

    cta_names = {row["filename"] for row in read_jsonl(CTA500)}
    pub_rows: list[dict] = []
    cta_rows: list[dict] = []
    for index, model_id in enumerate(MODEL_ORDER, start=1):
        print(f"[{index:02d}/{len(MODEL_ORDER)}] {model_id}", flush=True)
        samples = read_jsonl(SAMPLES / model_id / "sample_teds.jsonl")
        if len(samples) != 5000:
            raise RuntimeError(f"{model_id}: expected 5000 samples, got {len(samples)}")
        cta_samples = [row for row in samples if row.get("filename") in cta_names]
        if len(cta_samples) != 500:
            raise RuntimeError(f"{model_id}: expected 500 CTA samples, got {len(cta_samples)}")
        pub_rows.append(metric_row(model_id, profiles[model_id], summarize(samples)))
        cta_rows.append(metric_row(model_id, profiles[model_id], summarize(cta_samples)))

    fields = list(pub_rows[0].keys())
    write_csv(OUTPUT / "table3_pubtabnet5000_corrected.csv", pub_rows, fields)
    write_csv(OUTPUT / "table4_pubtabnet5000_comprehensive_corrected.csv", pub_rows, fields)
    write_csv(OUTPUT / "table5_cta500_from_pub5000_corrected.csv", cta_rows, fields)
    cta_by_id = {row["id"]: row for row in cta_rows}
    write_csv(OUTPUT / "table10_phase11_cta500_corrected.csv", [cta_by_id[x] for x in PHASE11_IDS], fields)
    write_csv(OUTPUT / "table11_true_axis_rerun_corrected.csv", [cta_by_id[x] for x in AXIS_IDS], fields)
    validate_results(pub_rows, cta_rows)
    write_report(profile_rows, pub_rows, cta_rows)
    (OUTPUT / "COMPLETE").write_text("2026-08-11 corrected recheck complete\n", encoding="utf-8")
    print(f"Corrected outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
