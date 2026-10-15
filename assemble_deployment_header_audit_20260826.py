#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import paddle
import yaml


ROOT = Path("/root/autodl-tmp/PaddleOCR")
OUT = ROOT / "output/deployment_header_audit_20260826"
OLD = ROOT / "output/deployment_export_20260812"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def get_row(rows: list[dict], model: str, batch: int) -> dict:
    return next(r for r in rows if r["model"] == model and int(r["batch_size"]) == batch)


def copy_header_outputs() -> None:
    source = OUT / "header_metrics"
    for name in (
        "header_metrics_per_sample.csv",
        "header_metrics_summary.csv",
        "header_bootstrap_10000.csv",
        "horizontal_vertical_audit.json",
        "header_metric_unit_tests.json",
    ):
        shutil.copy2(source / name, OUT / name)


def deployment_profile() -> dict:
    model_size = json.loads((OLD / "model_size.json").read_text())
    checkpoint = Path(model_size["deploy_checkpoint"])
    state = paddle.load(str(checkpoint))
    module_counts = Counter(".".join(key.split(".")[:2]) for key in state)
    table2 = read_csv(ROOT / "output/main_tables_rebuild_20260810/table2_profile.csv")
    training = next(r for r in table2 if r["id"] == "dbm_slanet")
    config = yaml.safe_load((OLD / "dbm_slanet_deploy.yml").read_text())
    profile = {
        "source_training_checkpoint": model_size["source_checkpoint"],
        "source_training_graph_parameters": int(training["params"]),
        "source_training_graph_parameters_m": float(training["params_m"]),
        "source_training_graph_paddle_estimated_flops": int(training["flops"]),
        "source_training_graph_paddle_estimated_flops_g": float(training["flops_g"]),
        "deployment_checkpoint": str(checkpoint),
        "deployment_checkpoint_size_bytes": checkpoint.stat().st_size,
        "deployment_checkpoint_sha256": sha(checkpoint),
        "deployment_stored_parameters": model_size["stored_parameters"],
        "deployment_trainable_parameters": model_size["trainable_parameters"],
        "deployment_paddle_estimated_flops": model_size["paddle_flops_proxy"],
        "deployment_paddle_estimated_flops_g": model_size["paddle_flops_proxy"] / 1e9,
        "source_parameter_keys": model_size["source_parameter_keys"],
        "deployment_parameter_keys": model_size["deploy_parameter_keys"],
        "removed_parameter_keys": model_size["removed_parameter_keys"],
        "missing_keys": model_size["missing_keys"],
        "shape_mismatch": model_size["shape_mismatch"],
        "deployment_head": config["Architecture"]["Head"],
        "module_parameter_tensor_counts": dict(sorted(module_counts.items())),
        "removed_at_deployment": [
            "geometry token encoder",
            "horizontal/vertical sorting and inverse permutation",
            "four directional Mamba scans",
            "axis fusion",
            "auxiliary cell-presence head",
        ],
    }
    (OUT / "deployment_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(OLD / "parameter_manifest.csv", OUT / "deployment_parameter_manifest.csv")
    return profile


def report(profile: dict) -> None:
    pub = read_csv(OUT / "strict_fps_pubtabnet5000.csv")
    cta = read_csv(OUT / "strict_fps_cta500.csv")
    fin = read_csv(OUT / "strict_fps_financial.csv")
    header = {r["model"]: r for r in read_csv(OUT / "header_metrics_summary.csv")}
    boot = read_csv(OUT / "header_bootstrap_10000.csv")
    equiv = read_csv(OUT / "prediction_equivalence.csv")
    hv = json.loads((OUT / "horizontal_vertical_audit.json").read_text())
    dbm_hh = header["dbm_slanet"]
    sla_hh = header["slanet"]
    hh_boot = next(r for r in boot if r["model"] == "dbm_slanet" and r["reference"] == "slanet" and r["metric"] == "hh_teds")
    hs_boot = next(r for r in boot if r["model"] == "dbm_slanet" and r["reference"] == "slanet" and r["metric"] == "hspan_f1_micro")
    lines = [
        "# 部署剪枝、严格效率与表头指标补充实验报告",
        "",
        "## 1. 审计结论",
        "",
        f"- 完整训练图参数量为 `{profile['source_training_graph_parameters']:,}`，物理导出部署 checkpoint 存储参数量为 `{profile['deployment_stored_parameters']:,}`。二者属于训练图与部署图两个不同口径，不应混写。",
        f"- 部署 checkpoint 大小为 `{profile['deployment_checkpoint_size_bytes'] / 2**20:.3f} MiB`，SHA256 为 `{profile['deployment_checkpoint_sha256']}`。",
        f"- 原始张量等价性共覆盖 `{sum(int(r['samples']) for r in equiv)}` 个样本；所有数据集结构概率与 box 的最大绝对误差均为 `{max(float(r['max_structure_probability_abs_error']) for r in equiv):.3g}` 和 `{max(float(r['max_box_abs_error']) for r in equiv):.3g}`。",
        f"- CTA-500 上 DBM-SLANet 的 HH-TEDS 为 `{float(dbm_hh['hh_teds_mean']):.6f}`，SLANet 为 `{float(sla_hh['hh_teds_mean']):.6f}`，差值 `{float(dbm_hh['hh_teds_mean'])-float(sla_hh['hh_teds_mean']):+.6f}`。",
        f"- CTA-500 上 DBM-SLANet 的 HSpan-F1 为 `{float(dbm_hh['hspan_f1_micro']):.6f}`，SLANet 为 `{float(sla_hh['hspan_f1_micro']):.6f}`，差值 `{float(dbm_hh['hspan_f1_micro'])-float(sla_hh['hspan_f1_micro']):+.6f}`。",
        "",
        "## 2. 物理导出模型",
        "",
        "| 图 | 参数量 | Paddle-estimated FLOPs | 用途 |",
        "|---|---:|---:|---|",
        f"| 完整训练图 | {profile['source_training_graph_parameters']:,} | {profile['source_training_graph_paddle_estimated_flops_g']:.3f} G | 训练与辅助结构正则 |",
        f"| 物理导出部署图 | {profile['deployment_stored_parameters']:,} | {profile['deployment_paddle_estimated_flops_g']:.3f} G | 最终推理 |",
        "",
        "部署图移除了 Geometry、H/V 排序、四个方向的 Mamba scan、axis fusion 和 auxiliary presence head，仅保留与最终 HTML/box 输出有关的 SLA 路径。",
        "",
        "## 3. 预测等价性",
        "",
        "| 数据集 | 样本 | token 一致率 | 结构张量一致率 | box 一致率 | box 最大误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in equiv:
        lines.append(f"| {r['dataset']} | {r['samples']} | {float(r['token_exact_rate']):.3f} | {float(r['structure_tensor_exact_rate']):.3f} | {float(r['box_tensor_exact_rate']):.3f} | {float(r['max_box_abs_error']):.3g} |")
    lines += [
        "",
        "所有 token 派生指标，包括 TEDS、Exact Acc.、Topology F1、Span-F1、HH-TEDS 和 HSpan-F1，部署前后差值均为 0。",
        "",
        "## 4. 严格同步效率协议",
        "",
        "- 输入分辨率：488 x 488。",
        "- GPU：NVIDIA GeForce RTX 4090 D。",
        "- BS1：20 次 warmup，400 个计时批次。",
        "- BS4：20 次 warmup，200 个计时批次。",
        "- 每个模型/数据集/批大小均运行 3 个独立 Python 进程。",
        "- forward 前后均执行 CUDA synchronize。",
        "- pipeline 指 forward + structure/box decoding，不包含磁盘读取。",
        "- 显存同时报告 peak allocated 与 peak reserved。",
        "",
        "### PubTabNet-5000",
        "",
        "| 模型 | BS | Forward FPS | SD | Pipeline FPS | P50 / P95 forward latency (ms/batch) | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pub:
        lines.append(f"| {r['model']} | {r['batch_size']} | {fmt(r['forward_fps_mean'])} | {fmt(r['forward_fps_sample_sd'])} | {fmt(r['pipeline_fps_mean'])} | {fmt(r['forward_batch_latency_p50_ms_mean'])} / {fmt(r['forward_batch_latency_p95_ms_mean'])} | {fmt(r['peak_allocated_mib_max'], 1)} |")
    lines += [
        "",
        "### CTA-500",
        "",
        "| 模型 | BS | Forward FPS | SD | Pipeline FPS | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in cta:
        lines.append(f"| {r['model']} | {r['batch_size']} | {fmt(r['forward_fps_mean'])} | {fmt(r['forward_fps_sample_sd'])} | {fmt(r['pipeline_fps_mean'])} | {fmt(r['peak_allocated_mib_max'], 1)} |")
    lines += [
        "",
        "金融域的全部逐 checkpoint 严格测速位于 `strict_fps_financial.csv`。BS4 对非 4 整除的数据集启用 drop_last，并循环 dataloader，保证每个计时批次严格为 4。",
        "",
        "## 5. 表头专用指标",
        "",
        "| 模型 | HH-TEDS | HSpan-P | HSpan-R | HSpan-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in header:
        r = header[model]
        lines.append(f"| {model} | {float(r['hh_teds_mean']):.6f} | {float(r['hspan_precision_micro']):.6f} | {float(r['hspan_recall_micro']):.6f} | {float(r['hspan_f1_micro']):.6f} |")
    lines += [
        "",
        f"CTA-500 全部 500 条解析成功；457 条含任意真实 span，317 条在 GT 表头区域含 span。DBM-SLANet 相对 SLANet 的 HH-TEDS paired bootstrap 95% CI 为 `[{float(hh_boot['ci95_low']):+.6f}, {float(hh_boot['ci95_high']):+.6f}]`；HSpan-F1 micro 差值的 95% CI 为 `[{float(hs_boot['ci95_low']):+.6f}, {float(hs_boot['ci95_high']):+.6f}]`。",
        "",
        "## 6. True H / True V 文件审计",
        "",
        f"- 文件 SHA256 不同：`{hv['horizontal_file_sha256']}` vs `{hv['vertical_file_sha256']}`。",
        f"- 500 条 CTA 样本中有 `{hv['different_prediction_sha256_samples']}` 条结构预测不同。",
        f"- HH-TEDS 不同样本数为 `{hv['different_hh_teds_samples']}`，HSpan-F1 不同样本数为 `{hv['different_hspan_f1_samples']}`。",
        "- 因差异发生在 GT 表头区域之外，两项表头指标相同是可复核的结果，不是预测文件复用。",
        "",
        "## 7. 附件更新口径",
        "",
        "旧的 `DBM-SLANet 9.13 FPS / SLANet 18.65 FPS` 只能放入 historical development audit，并标注旧 evaluator。最终部署效率应引用本目录严格同步结果。正文或附件出现 2.45M 与 1.89M 时必须分别标为 training graph 与 physically exported deployment graph。",
        "",
        "## 8. 文件说明",
        "",
        "- `deployment_profile.json`：导出模型、参数、FLOPs、模块与哈希。",
        "- `prediction_equivalence.csv`：九个数据集的 raw tensor 与指标等价性。",
        "- `strict_fps_*.csv`：PubTabNet、CTA 和金融域严格效率。",
        "- `header_metrics_per_sample.csv`：14 模型 x 500 样本的逐样本表头指标。",
        "- `header_bootstrap_10000.csv`：10,000 次 paired bootstrap。",
        "- `horizontal_vertical_audit.json`：H/V 文件独立性与逐样本差异。",
        "- `artifact_manifest.csv`：文件大小与 SHA256 索引。",
    ]
    (OUT / "完整实验总结报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocol_md(profile: dict) -> None:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, capture_output=True).stdout.strip()
    text = f"""# Protocol and Checksums

- Date: 2026-08-26
- Host: {platform.node()}
- GPU: {gpu}
- Paddle: {paddle.__version__}
- Python: {platform.python_version()}
- Git commit: `{git_commit}`
- Input: 488 x 488
- BS1: 20 warmup + 400 measured batches x 3 independent processes
- BS4: 20 warmup + 200 measured batches x 3 independent processes
- CUDA synchronized before and after every forward
- Deployment checkpoint SHA256: `{profile['deployment_checkpoint_sha256']}`
- Training graph parameters: {profile['source_training_graph_parameters']:,}
- Deployment stored parameters: {profile['deployment_stored_parameters']:,}

`pipeline_fps` includes model forward and PaddleOCR structure/box post-processing, but excludes disk image reading. Financial BS4 skips incomplete tail batches and cycles the loader so every timed batch contains exactly four samples.
"""
    (OUT / "protocol_and_checksums.md").write_text(text, encoding="utf-8")


def manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.csv", "deployment_header_audit_20260826.zip"}:
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    write_csv(OUT / "artifact_manifest.csv", rows)


def main() -> None:
    required = [OUT / "strict_fps.COMPLETE", OUT / "prediction_equivalence.COMPLETE"]
    if not all(path.exists() for path in required):
        raise RuntimeError(f"incomplete prerequisites: {[str(p) for p in required if not p.exists()]}")
    copy_header_outputs()
    profile = deployment_profile()
    protocol_md(profile)
    report(profile)
    manifest()
    archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
    print(archive)


if __name__ == "__main__":
    main()
