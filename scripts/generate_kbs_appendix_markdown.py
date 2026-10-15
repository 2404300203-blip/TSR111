#!/usr/bin/env python3
"""Assemble the dense, traceable KBS-style supplementary experiment Markdown."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文材料/kbs_appendix_experiments_20260803"
TAB = OUT / "tables"
NEW_RESULTS = TAB / "N00_appendix_new_experiment_results.csv"
DEST = OUT / "KBS_SUPPLEMENTARY_EXPERIMENTS_FULL_20260803.md"
MANIFEST = OUT / "APPENDIX_EXPERIMENT_MANIFEST.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def fmt(value: str, column: str) -> str:
    parsed = num(value)
    if parsed is None:
        return value or "–"
    if any(key in column.lower() for key in ("samples", "count", "rows", "logs", "memory", "repeat")):
        return str(int(round(parsed)))
    if "fps" in column.lower() or "wall" in column.lower():
        return f"{parsed:.2f}"
    if any(key in column.lower() for key in ("rate", "exact", "acc", "teds", "confidence", "gain", "delta", "mean", "std", "min", "max")):
        return f"{parsed:.4f}"
    return f"{parsed:g}"


def markdown_table(rows: list[dict[str, str]], columns: list[str] | None = None,
                   labels: dict[str, str] | None = None) -> str:
    if not rows:
        return "_No valid records._"
    columns = columns or list(rows[0])
    labels = labels or {}
    head = "| " + " | ".join(labels.get(col, col) for col in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(str(row.get(col, "")), col) for col in columns) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def relative_path(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def new_tables(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    completed = [row for row in rows if row["status"] == "completed"]
    clean_reference = {
        "SLANet": {"teds": 0.4843354130430532, "structure_acc": 0.208},
        "DBM-SLANet": {"teds": 0.640672654391009, "structure_acc": 0.432},
    }
    for row in completed:
        reference = clean_reference[row["model"]]
        row["delta_teds_vs_clean"] = str(num(row["teds"]) - reference["teds"])
        row["delta_acc_vs_clean"] = str(num(row["structure_acc"]) - reference["structure_acc"])
    result = {}
    for category in ("resolution", "batch_size", "robustness", "timing_repeat"):
        result[category] = [row for row in completed if row["category"] == category]
    return result


def timing_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for model in ("SLANet", "DBM-SLANet"):
        part = [row for row in rows if row["model"] == model]
        fps = [num(row["fps"]) for row in part]
        wall = [num(row["wall_time_s"]) for row in part]
        memory = [num(row["peak_memory_mib"]) for row in part]
        if not fps:
            continue
        mean = sum(fps) / len(fps)
        std = math.sqrt(sum((value - mean) ** 2 for value in fps) / max(1, len(fps) - 1))
        result.append({
            "model": model, "repeats": str(len(part)), "fps_mean": str(mean), "fps_std": str(std),
            "fps_min": str(min(fps)), "fps_max": str(max(fps)),
            "wall_time_mean_s": str(sum(wall) / len(wall)),
            "peak_memory_mib": str(max(memory)),
        })
    return result


def search_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline = 0.8308172189543931
    # The lone 3,000-sample extension is not directly comparable to the 1,000-sample search baseline.
    rows = [row for row in rows if num(row.get("samples")) == 1000]
    grouped = {}
    for row in rows:
        family = row["search_family"]
        grouped.setdefault(family, []).append(row)
    result = []
    for family, part in grouped.items():
        scores = [num(row["teds"]) for row in part]
        best = max(part, key=lambda row: num(row["teds"]))
        result.append({
            "search_family": family, "evaluated_settings": str(len(part)),
            "best_run": best["run"], "best_teds": best["teds"],
            "best_delta_vs_phase5e": str(num(best["teds"]) - baseline),
            "worst_teds": str(min(scores)),
            "conclusion": "no consistent improvement over Phase-5e",
        })
    return result


def build_manifest(new_rows: list[dict[str, str]], historical: list[dict[str, str]]) -> None:
    fields = [
        "experiment_id", "category", "status", "new_or_historical", "overlap_with_main",
        "dataset", "model", "config", "checkpoint", "samples", "teds", "exact", "fps",
        "source_log", "output_table", "output_figure", "validity_notes",
    ]
    rows = []
    for row in new_rows:
        rows.append({
            "experiment_id": row["experiment_id"], "category": row["category"], "status": row["status"],
            "new_or_historical": "new_20260803", "overlap_with_main": "no",
            "dataset": "CTA-500", "model": row["model"], "config": row["config"], "checkpoint": "see config",
            "samples": row["samples"], "teds": row["teds"], "exact": row["structure_acc"], "fps": row["fps"],
            "source_log": row["log"], "output_table": "tables/N00_appendix_new_experiment_results.csv",
            "output_figure": "figures/N01-N04", "validity_notes": row["error_note"] or "single-variable control",
        })
    for index, row in enumerate(historical, start=1):
        rows.append({
            "experiment_id": f"HIST-{index:03d}", "category": row["category"], "status": row["status"],
            "new_or_historical": "historical", "overlap_with_main": row["overlap_with_main"],
            "dataset": row["dataset"], "model": row["model_head"], "config": "", "checkpoint": row["checkpoint"],
            "samples": row["samples"], "teds": row["teds"], "exact": row["structure_acc"], "fps": row["fps"],
            "source_log": row["log_path"], "output_table": "tables/H01_historical_log_classification.csv",
            "output_figure": "", "validity_notes": row["validity"],
        })
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    new_rows = read_csv(NEW_RESULTS)
    historical = read_csv(TAB / "H01_historical_log_classification.csv")
    search = read_csv(TAB / "H03_historical_calibration_and_reranking_search.csv")
    trajectory = read_csv(TAB / "H04_progressive_training_trajectory.csv")
    validation = read_csv(TAB / "H05_validation_scale_control.csv")
    groups = new_tables(new_rows)
    build_manifest(new_rows, historical)

    analysis_tables = [TAB / f"A{index:02d}_{name}.csv" for index, name in []]
    appendix_analyses = sorted(TAB.glob("A*.csv"))
    diagnostic_sections = []
    for path in appendix_analyses:
        if path.name == "A00_cta500_merged_sample_records.csv":
            continue
        rows = read_csv(path)
        display = rows[:40]
        diagnostic_sections.append(
            f"### {path.stem}\n\n原始表：`{relative_path(path)}`；记录数：{len(rows)}。\n\n"
            + markdown_table(display)
            + ("\n\n> 表内记录超过 40 行，Markdown 仅展示前 40 行，完整数据见 CSV。" if len(rows) > 40 else "")
        )

    completed_new = sum(row["status"] == "completed" for row in new_rows)
    valid_historical = sum(
        row["overlap_with_main"] == "not_in_main_text" and row["validity"] == "eligible_after_source_check"
        for row in historical
    )
    text = f"""# DBM-SLANet Supplementary Experiments and Reproducibility Appendix

> 版本：2026-08-03；目标期刊版式参考：Knowledge-Based Systems (KBS) 补充材料。本文档只汇总**正文未展示的控制实验、历史搜索、误差分析和定性案例**。所有指标来自日志、逐样本 JSONL 或本轮重新推理，未对结果进行平滑、补点或人工改写。

## A. 附录范围、计数口径与非重合声明

正文《第七版》已有 10 张表和 9 幅图，覆盖 PubTabNet-5000、Complex-4086、CTA-500、FinTabNet.c 主结果，固定 Phase-11 消融、方向消融、复杂类别结果、效率主表、bootstrap、Topology/Span-F1、定性图和训练曲线。上述内容均进入禁重复清单，不在本附录中作为新增证据重复排版。

| 证据层 | 设置/记录数 | 是否计作新增工作量 | 说明 |
|---|---:|---|---|
| 2026-08-03 新增独立推理控制 | {completed_new}/{len(new_rows)} | 是 | 分辨率、batch size、输入退化、重复测速 |
| 历史日志总账 | {len(historical)} | 否，按搜索族汇总 | 其中正文外且待源日志核对的记录 {valid_historical} 条 |
| 历史后处理搜索 | {len(search)} | 是，按两类搜索族计 | 不把相同输出拆成独立贡献 |
| 渐进训练/检查点轨迹 | {len(trajectory)} | 是，按训练阶段计 | 包含失败或退化阶段，完整报告 |
| CTA-500 逐样本诊断表 | {len(appendix_analyses)-1} 类 | 是，按分析问题计 | 6 模型 × 500 样本的派生分析 |
| 新增定性案例 | 8 例 | 是 | 正文未使用的输入文件，含成功与失败 |

计数原则：同一 checkpoint 的重复日志、debug/smoke、正文同表复测和完全相同的阈值输出不作为独立贡献。按保守口径，正文约 19 个表图级实验单元；本附录包含 34 个新推理条件、20 类逐样本分析、43 条历史搜索相关记录和 14 条阶段轨迹，实验设置数量已超过正文三倍，但论证时按控制变量族组织，而不是人为拆成大量“小实验”。

## B. 正文禁重复清单

| 正文项目 | 已覆盖内容 | 附录处理 |
|---|---|---|
| Table 1 | 数据集与拆分 | 仅补充样本属性分层，不复述主表 |
| Table 2 | 参数量、FLOPs、FPS | 仅补 batch/分辨率/显存控制 |
| Tables 3–5 | PubTabNet、Complex-4086、CTA-500 主比较 | 不重复主排名 |
| Table 6 | CTA-500 类别 TEDS | 改做四分位、交叉标签和恢复率 |
| Tables 7–9 | FinTabNet.c、方向消融、Phase-11 消融 | 仅列历史域适配边界与负面搜索 |
| Table 10 | Topology F1、Span-F1 | 不重复聚合值 |
| Figures 6–8 | 主定性及消融案例 | 使用 8 个不同文件名的正负案例 |
| Figure 9 | 训练 loss/accuracy 曲线 | 改列 checkpoint 轨迹表，不复制曲线 |

## C. 复现环境与完整参数

| 类别 | 参数 | 值 |
|---|---|---|
| 硬件 | GPU | NVIDIA GeForce RTX 4090 D, 24 GB |
| 框架 | PaddlePaddle / CUDA / cuDNN | 2.6.2 / 11.8 / 8.6 |
| 主干 | Backbone | PP-LCNet x1.0, SSLD pretrained |
| 颈部 | Neck | CSP-PAN, 96 output channels |
| 解码器 | hidden size / max length | 256 / 500 |
| cell token | dimension / dropout | 128 / 0.1 |
| selective scan | d_state / expand / d_conv / dt_rank | 16 / 2 / 3 / 8 (auto) |
| 方向 | scan blocks | horizontal forward/backward + vertical forward/backward |
| 图像 | decode / resize / pad | BGR / long side <= 488 / 488 x 488 |
| 归一化 | scale / mean / std | 1/255 / [0.485,0.456,0.406] / [0.229,0.224,0.225] |
| 几何 | box format | xyxy |
| 标签 | merge no-span / learn empty box | true / false |
| 优化器 | Adam beta1 / beta2 | 0.9 / 0.999 |
| 优化 | clip norm / L2 | 5.0 / 0 |
| 损失 | structure / localization / presence | 1.0 / 10.0 / 0.2 |
| progressive LR | 500–10k / 30k / final 30k | 2e-5 / 1e-5 / 5e-6 |
| 评测 | fixed batch / workers | 4 / 0，除 batch-size 控制外不变 |

本轮所有新实验固定 CTA-500 标签文件、模型 checkpoint、HTML 后处理、TEDS 实现和 GPU，只改变一个显式控制变量。CTA-500 是从 PubTabNet 验证集按复杂结构规则选出的诊断子集，不是独立训练集，也不用于训练。

## D. 历史实验总账与搜索过程

完整 209 条分类表见 `tables/H01_historical_log_classification.csv`，汇总见 `tables/H02_historical_log_classification_summary.csv`。需注意，早期日志解析器有时从 Train 配置抓取 `label_file_list`；任何具体数据集结论均以对应 Eval 配置和源日志为准。

### D.1 渐进训练与 checkpoint 轨迹

{markdown_table(trajectory, ["run","teds","structure_acc","samples","fps","source_log"])}

轨迹显示，Full Mamba 直接引入时曾出现 TEDS 0.0742 的严重失稳；通过轻量阶段初始化、扩大数据规模并降低学习率后恢复到约 0.83。最终 Phase-5e 比继续降低学习率的 Phase-5f 高 0.0023 TEDS，而后者仅有 0.001 的 exact 优势，因此不存在“训练越久必然越好”的单调关系。

### D.2 验证规模一致性

{markdown_table(validation)}

从 3,000 到 5,000 个验证样本，SLANet TEDS 变化约 -0.0003，DBM-SLANet 变化约 +0.0026；结果排序保持稳定。该表用于验证样本规模稳定性，不替代正文 PubTabNet-5000 主比较。

### D.3 阈值校准与候选重排序搜索汇总

{markdown_table(search_summary(search))}

完整 43 条搜索相关记录见 `tables/H03_historical_calibration_and_reranking_search.csv`，其中 42 条为可与 1,000 样本 Phase-5e 直接比较的设置，另 1 条是 3,000 样本扩展评测，不参与上表差值。Phase-8 中多组 confidence/margin 阈值完全不改变解码输出，表明触发条件没有作用于多数样本；margin-v2 加强后反而下降。Phase-9 中简单 top-k 权重多数保持原结果，cell-pattern rerank 的最好 TEDS 0.8317 仅带来约 +0.00085，但 exact 从 0.693 降到 0.688；更大的 shape 权重使 TEDS 降至 0.8205。因此最终模型不采用这些后处理。

## E. 新增输入尺度实验

{markdown_table(groups["resolution"], ["model","resolution","samples","teds","structure_acc","delta_teds_vs_clean","delta_acc_vs_clean","fps","peak_memory_mib"])}

488 是训练和主评测的匹配尺度。低于 488 时细粒度网格与跨度线索丢失；高于 488 时，未做尺度再训练的特征和解码分布发生偏移，同样退化。因此该实验不支持“更高分辨率自然更优”，而支持训练-测试尺度一致性。

![Input-resolution sensitivity](figures/N01_resolution_sensitivity.png)

## F. 新增 batch-size、吞吐与显存实验

{markdown_table(groups["batch_size"], ["model","batch_size","samples","teds","structure_acc","fps","wall_time_s","peak_memory_mib"])}

SLANet 在 batch size 1/2/8 下的 TEDS 和 exact 完全一致。DBM-SLANet 在 batch 1/2 时为 0.6390/43.0%，batch 8 时为 0.6407/43.2%，范围分别为 0.001675 TEDS 和 0.2 个百分点。该微小差异表明当前 DBM 解码存在 batch-dependent 数值或序列路径敏感性，不能当作 batch 8 的性能收益；正式比较应固定正文使用的 batch size 4。吞吐随 batch 增大明显提升，同时峰值显存上升。

![Batch-size efficiency](figures/N02_batch_efficiency.png)

## G. 新增输入退化鲁棒性实验

{markdown_table(groups["robustness"], ["model","variant","samples","teds","structure_acc","delta_teds_vs_clean","delta_acc_vs_clean","fps"])}

扰动直接施加在 500 张输入图像的副本上，标签、模型和后处理不变。JPEG 质量、Gaussian blur、亮度与对比度分别考察压缩伪影、边缘扩散和整体光度变化。轻微光度变化可能因阈值和归一化偶然改善部分样本，不应解释为数据增强收益；模糊退化更直接反映线条与单元格边界信息的损失。

![Robustness controls](figures/N03_robustness_controls.png)

## H. 新增重复测速稳定性

{markdown_table(timing_summary(groups["timing_repeat"]))}

![Repeated timing](figures/N04_timing_stability.png)

重复测速用于报告均值和离散度，而非从一次 FPS 中过度推断。峰值显存由 `nvidia-smi` 每 0.2 s 轮询记录，属于进程级近似值，不等同于框架内部精确峰值分配。

## I. CTA-500 逐样本复杂度、置信度与互补性分析

以下分析统一使用 6 个模型在相同 500 个样本上的逐样本预测：SLANet、Cell Token、Mamba lambda=0、Horizontal、Vertical 和 DBM-SLANet。文件名顺序 SHA-256 一致，避免因样本错位产生伪差异。

{"\n\n".join(diagnostic_sections)}

## J. 新增非正文定性案例

所有案例均展示真实 CTA-500 输入和真实预测 token。绿色表示与 GT cell tuple 一致，红色表示错误预测，橙色虚线表示 GT 中存在但预测漏掉的单元格。正文已用过的 `PMC2898048`、`PMC3126717`、`PMC3496644`、`PMC4225768` 等案例未在此复用。

### J.1 DBM 精确恢复

![Two additional exact recoveries](effect_figures/QP_success.png)

### J.2 双方向互补

![Two dual-branch complementarity cases](effect_figures/QP_complement.png)

### J.3 高置信度失败

![Two high-confidence failures](effect_figures/QP_confidence_failure.png)

### J.4 DBM 退化边界

![Two DBM regression cases](effect_figures/QP_regression.png)

这些失败案例说明 decoder token confidence 不能作为结构正确性的充分条件。特别是长表截断、序列长度偏差和跨度闭合错误可能在高 token confidence 下同时发生，因此附录保留 selective prediction 分析，但不把 confidence 当作校准后的结构概率。

## K. 排除项、无效指标与方法边界

1. `Cell Presence F1=1.0` 已排除。历史评估把 GT `bbox_masks` 传入 forward，存在标签泄漏；移除 targets 后输出退化为全正预测。presence 分支只通过 TEDS/exact/FPS 的受控消融评价。
2. debug、smoke、启动失败和被中断日志保留在总账，但不进入性能结论。
3. 相同 TEDS/exact 的阈值结果作为“未触发或无效搜索”报告，不拆分成多项贡献。
4. CTA-500 是困难诊断子集；其收益不能外推为全部普通表格都统一占优。无显式 span 的样本上 DBM 可能低于 SLANet。
5. FinTabNet.c zero-shot 中 DBM 不优于 baseline，说明金融域适配是必要条件；本文不能声称零样本跨域优势。
6. FPS 依赖 batch、驱动、热身、数据加载和输入长度；正文与附录仅在同一服务器、同一评测驱动内比较。

## L. 复现文件和真源索引

| 文件 | 用途 |
|---|---|
| `APPENDIX_EXPERIMENT_MANIFEST.csv` | 新增和历史记录的统一机器清单 |
| `tables/N00_appendix_new_experiment_results.csv` | 34 个本轮推理条件原始结果 |
| `tables/H01_historical_log_classification.csv` | 209 份历史日志分类 |
| `tables/H03_historical_calibration_and_reranking_search.csv` | 43 条历史搜索明细 |
| `tables/A00_cta500_merged_sample_records.csv` | 6 模型 × 500 样本合并记录 |
| `figures/` | 新增统计图的 PNG/PDF/SVG |
| `effect_figures/` | 8 个真实案例及 4 张组合图 |
| `case_lists/` | 成功、退化、互补和高置信失败的候选清单 |

## M. 总结

补充实验表明，DBM-SLANet 的主要价值集中在复杂跨度、长程依赖和多级表头，而不是所有简单表格上的一致优势。双方向建模能恢复大量 SLANet 非 exact 样本，但存在速度、尺度敏感性和高置信结构失败。历史负面搜索进一步说明，简单置信度阈值或候选重排序不能替代几何感知的训练期建模。以上结论与正文创新点一致，同时明确了可复现条件和适用边界。
"""
    DEST.write_text(text, encoding="utf-8")
    print(json.dumps({"markdown": str(DEST), "lines": len(text.splitlines()), "manifest_rows": len(new_rows) + len(historical)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
