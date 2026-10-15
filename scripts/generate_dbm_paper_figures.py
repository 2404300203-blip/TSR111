#!/usr/bin/env python3
"""Generate paper-ready DBM-SLANet figures from collected experiment results."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image, ImageOps, ImageDraw


ROOT = Path("/Users/table/TSR")
OUT = ROOT / "论文材料" / "generated_figures_20260710"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "blue": "#3568B8",
    "teal": "#2A9D8F",
    "orange": "#E58B2A",
    "red": "#C75146",
    "purple": "#7E57C2",
    "gray": "#6B7280",
    "green": "#4C956C",
    "light": "#F3F6FA",
    "dark": "#223047",
}


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc="#F7FAFC", ec="#334155", lw=1.2, fs=9, weight="normal"):
    x, y = xy
    w, h = wh
    p = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, wrap=True)
    return p


def arrow(ax, start, end, color="#334155", lw=1.4):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=4, shrinkB=4),
    )


def fig1_overall_framework():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.84)
    ax.axis("off")
    ax.set_title("Fig. 1. Overall Framework of DBM-SLANet", pad=10, weight="bold")

    box(ax, (0.03, 0.43), (0.11, 0.16), "Table\nImage", COLORS["light"], COLORS["dark"], fs=10, weight="bold")
    box(ax, (0.18, 0.43), (0.13, 0.16), "SLANet\nBackbone", "#EAF2FF", COLORS["blue"], fs=10, weight="bold")
    box(ax, (0.35, 0.43), (0.13, 0.16), "CSPPAN\nNeck", "#EDF7F4", COLORS["teal"], fs=10, weight="bold")
    box(ax, (0.52, 0.54), (0.13, 0.14), "Structure\nDecoder", "#FFF6E8", COLORS["orange"], fs=9, weight="bold")
    box(ax, (0.52, 0.34), (0.13, 0.14), "Location\nDecoder", "#FFF6E8", COLORS["orange"], fs=9, weight="bold")
    box(ax, (0.69, 0.43), (0.14, 0.16), "Geometry-aware\nCell Tokens", "#F2ECFF", COLORS["purple"], fs=9, weight="bold")
    box(ax, (0.86, 0.54), (0.11, 0.14), "HTML\nTokens", "#F4F4F5", COLORS["gray"], fs=9, weight="bold")
    box(ax, (0.86, 0.34), (0.11, 0.14), "Cell\nBBoxes", "#F4F4F5", COLORS["gray"], fs=9, weight="bold")

    box(ax, (0.36, 0.10), (0.42, 0.13), "Dual-Branch Mamba Cell Relation Modeling\nhorizontal scan + vertical scan + fusion", "#F7F0FF", COLORS["purple"], fs=10, weight="bold")
    box(ax, (0.80, 0.10), (0.16, 0.13), "Cell Presence\nAuxiliary Loss\n(training only)", "#FFEDEC", COLORS["red"], fs=9, weight="bold")

    arrow(ax, (0.14, 0.51), (0.18, 0.51))
    arrow(ax, (0.31, 0.51), (0.35, 0.51))
    arrow(ax, (0.48, 0.51), (0.52, 0.61))
    arrow(ax, (0.48, 0.51), (0.52, 0.41))
    arrow(ax, (0.65, 0.61), (0.86, 0.61))
    arrow(ax, (0.65, 0.41), (0.69, 0.51))
    arrow(ax, (0.83, 0.51), (0.86, 0.41))
    arrow(ax, (0.76, 0.43), (0.61, 0.23))
    arrow(ax, (0.57, 0.34), (0.54, 0.23))
    arrow(ax, (0.78, 0.165), (0.80, 0.165), COLORS["red"])

    ax.text(0.58, 0.78, "Main prediction path", color=COLORS["dark"], fontsize=10, weight="bold")
    ax.text(0.51, 0.04, "Auxiliary branch regularizes cell-level geometry during training; it can be disabled at inference.", color=COLORS["gray"], ha="center", fontsize=9)
    save(fig, "fig1_overall_framework")


def fig2_dbm_module():
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Fig. 2. Dual-Branch Mamba Cell Modeling Module", pad=12, weight="bold")

    box(ax, (0.04, 0.42), (0.13, 0.16), "loc_preds\n(x1,y1,x2,y2)", "#EAF2FF", COLORS["blue"], fs=9, weight="bold")
    box(ax, (0.22, 0.42), (0.16, 0.16), "Geometry Token\nEncoder\ncenter, size, area,\naspect, left/top", "#F2ECFF", COLORS["purple"], fs=8.5, weight="bold")
    box(ax, (0.43, 0.62), (0.17, 0.14), "Horizontal\nMamba Scan\n(row-wise order)", "#EDF7F4", COLORS["teal"], fs=9, weight="bold")
    box(ax, (0.43, 0.25), (0.17, 0.14), "Vertical\nMamba Scan\n(column-wise order)", "#EDF7F4", COLORS["teal"], fs=9, weight="bold")
    box(ax, (0.66, 0.43), (0.14, 0.16), "Fusion\nconcat + linear\nLayerNorm + GELU", "#FFF6E8", COLORS["orange"], fs=8.5, weight="bold")
    box(ax, (0.85, 0.55), (0.11, 0.14), "Cell\nRelation\nTokens", COLORS["light"], COLORS["dark"], fs=9, weight="bold")
    box(ax, (0.85, 0.29), (0.11, 0.14), "Presence\nLogits", "#FFEDEC", COLORS["red"], fs=9, weight="bold")

    arrow(ax, (0.17, 0.50), (0.22, 0.50))
    arrow(ax, (0.38, 0.50), (0.43, 0.69))
    arrow(ax, (0.38, 0.50), (0.43, 0.32))
    arrow(ax, (0.60, 0.69), (0.66, 0.54))
    arrow(ax, (0.60, 0.32), (0.66, 0.48))
    arrow(ax, (0.80, 0.51), (0.85, 0.62))
    arrow(ax, (0.80, 0.48), (0.85, 0.36), COLORS["red"])

    ax.text(0.51, 0.83, "Bidirectional scan improves structural accuracy", ha="center", color=COLORS["teal"], weight="bold")
    ax.text(0.50, 0.12, "Ablations: geometry zero, bidirectional=false, lambda_pres=0, auxiliary-off inference", ha="center", color=COLORS["gray"])
    save(fig, "fig2_dbm_module_structure")


def fig3_quantitative():
    labels = ["Pilot\nheldout75", "Company-\ndisjoint100", "Cached-\ntest500", "Company-\nheldout100"]
    base_teds = np.array([0.4165, 0.4739, 0.4396, 0.4211])
    dbm_teds = np.array([0.5520, 0.6127, 0.4877, 0.4558])
    base_acc = np.array([0.1467, 0.2000, 0.0540, 0.0200])
    dbm_acc = np.array([0.2667, 0.3500, 0.0660, 0.0700])

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, b, d, title, ylabel in [
        (axes[0], base_teds, dbm_teds, "TEDS", "TEDS"),
        (axes[1], base_acc, dbm_acc, "Structure Accuracy", "Structure Acc"),
    ]:
        ax.bar(x - width / 2, b, width, label="SLANet baseline", color="#A7B1C2")
        ax.bar(x + width / 2, d, width, label="DBM-R2", color=COLORS["blue"])
        for i, (bv, dv) in enumerate(zip(b, d)):
            ax.text(i + width / 2, dv + 0.015, f"+{dv-bv:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["blue"], weight="bold")
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(d) + 0.12)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Fig. 3. Quantitative Comparison on Financial Table Benchmarks", weight="bold", y=1.03)
    save(fig, "fig3_quantitative_comparison")


def fig5_tradeoff():
    data = [
        ("Cell Token", 20.33, 0.54364, 0.284, COLORS["gray"]),
        ("Mamba λ=0", 9.40, 0.53921, 0.282, COLORS["orange"]),
        ("Mamba λ=0.2", 9.54, 0.54215, 0.302, COLORS["blue"]),
        ("Bidir=false", 14.14, 0.53040, 0.272, COLORS["teal"]),
        ("Geom zero", 11.12, 0.53921, 0.282, COLORS["purple"]),
        ("Aux off", 18.68, 0.54215, 0.302, COLORS["green"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, yi, ylabel in [(axes[0], 2, "TEDS"), (axes[1], 3, "Structure Acc")]:
        for name, fps, teds, acc, color in data:
            y = teds if yi == 2 else acc
            ax.scatter(fps, y, s=95, color=color, edgecolor="white", linewidth=1.0, zorder=3)
            dx, dy, ha = 0.24, 0.0, "left"
            if name == "Mamba λ=0":
                dx, dy, ha = -0.15, 0.00035, "right"
            elif name == "Geom zero":
                dx, dy, ha = 0.25, -0.00025, "left"
            elif name == "Mamba λ=0.2":
                dx, dy, ha = 0.24, 0.0007, "left"
            ax.text(fps + dx, y + dy, name, va="center", ha=ha, fontsize=8)
        ax.set_xlim(8.2, 21.2)
        ax.set_xlabel("FPS")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title("TEDS vs FPS", weight="bold")
    axes[1].set_title("Structure Accuracy vs FPS", weight="bold")
    fig.suptitle("Fig. 5. Accuracy-Efficiency Trade-off of DBM-SLANet Variants", weight="bold", y=1.02)
    save(fig, "fig5_accuracy_efficiency_tradeoff")


def fig_s1_dataset_splits():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Fig. S1. Dataset Construction and Evaluation Splits", pad=12, weight="bold")

    box(ax, (0.04, 0.62), (0.17, 0.14), "FinTabNet / FinTabNet.c\nPDF annotations", "#EAF2FF", COLORS["blue"], fs=9, weight="bold")
    box(ax, (0.29, 0.62), (0.16, 0.14), "PDF rendering\n+ table crop\n+ label conversion", "#EDF7F4", COLORS["teal"], fs=9, weight="bold")
    box(ax, (0.53, 0.70), (0.18, 0.12), "Pilot split\ntrain 350 / val 75\nheldout 75", "#FFF6E8", COLORS["orange"], fs=8.5, weight="bold")
    box(ax, (0.53, 0.50), (0.18, 0.12), "Company-disjoint\ntrain 349 / val 51\nheldout 100", "#FFF6E8", COLORS["orange"], fs=8.5, weight="bold")
    box(ax, (0.78, 0.62), (0.17, 0.14), "Evaluation\nTEDS / Structure Acc\nFPS", COLORS["light"], COLORS["dark"], fs=9, weight="bold")

    box(ax, (0.29, 0.18), (0.18, 0.13), "Independent\ntrain2000\npublic cropped images", "#F2ECFF", COLORS["purple"], fs=9, weight="bold")
    box(ax, (0.56, 0.18), (0.18, 0.13), "External tests\ncached-test500\ncompany-heldout100", "#F2ECFF", COLORS["purple"], fs=9, weight="bold")

    arrow(ax, (0.21, 0.69), (0.29, 0.69))
    arrow(ax, (0.45, 0.69), (0.53, 0.76))
    arrow(ax, (0.45, 0.69), (0.53, 0.56))
    arrow(ax, (0.71, 0.76), (0.78, 0.69))
    arrow(ax, (0.71, 0.56), (0.78, 0.69))
    arrow(ax, (0.47, 0.245), (0.56, 0.245))
    arrow(ax, (0.65, 0.31), (0.78, 0.62))
    ax.text(0.50, 0.40, "Zero-shot tests are reported separately to demonstrate the domain gap.", ha="center", color=COLORS["gray"])
    save(fig, "fig_s1_dataset_construction_splits")


def fig_s2_ablation_bar():
    names = ["Cell\nToken", "Mamba\nλ=0", "Mamba\nλ=0.2", "Bidir\nfalse", "Geom\nzero", "Aux\noff"]
    teds = [0.54364, 0.53921, 0.54215, 0.53040, 0.53921, 0.54215]
    acc = [0.284, 0.282, 0.302, 0.272, 0.282, 0.302]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.bar(x - width / 2, teds, width, label="TEDS", color=COLORS["blue"])
    ax.bar(x + width / 2, acc, width, label="Structure Acc", color=COLORS["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylabel("Score")
    ax.set_title("Fig. S2. Complete Ablation Study on CTA500", weight="bold")
    for i, v in enumerate(teds):
        ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(acc):
        ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "fig_s2_complete_ablation_bar")


LOGS = {
    "Cell Token": ROOT / "remote_results/phase11_controlled_ablation_20260708/E11A_cell_token_only_train.log",
    "Mamba λ=0": ROOT / "remote_results/phase11_controlled_ablation_20260708/E11B_full_mamba_lambda0_train.log",
    "Mamba λ=0.2": ROOT / "remote_results/phase11_extra_logs_20260710/phase11c_lambda02_train.log",
    "Bidir=false": ROOT / "remote_results/phase11_extra_logs_20260710/phase11e_bidirfalse_train.log",
    "Geom zero": ROOT / "remote_results/phase11_extra_logs_20260710/phase11f_geomzero_train.log",
}


def parse_train_log(path: Path):
    pattern = re.compile(
        r"global_step:\s*(\d+).*?loss:\s*([0-9.]+).*?structure_loss:\s*([0-9.]+).*?loc_loss:\s*([0-9.]+).*?cell_presence_loss:\s*([0-9.]+)"
    )
    eval_pat = re.compile(r"global_step:\s*(\d+).*?")  # placeholder
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    return np.array(rows, dtype=float) if rows else np.zeros((0, 5))


def parse_eval_acc(path: Path):
    vals = []
    step = None
    step_pat = re.compile(r"global_step:\s*(\d+)")
    metric_pat = re.compile(r"cur metric,\s*acc:\s*([0-9.]+)")
    for line in path.read_text(errors="ignore").splitlines():
        sm = step_pat.search(line)
        if sm:
            step = int(sm.group(1))
        mm = metric_pat.search(line)
        if mm:
            vals.append((step if step else len(vals) + 1, float(mm.group(1))))
    return vals


def smooth(y, window=7):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def fig_s4_loss_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    selected = ["Mamba λ=0.2", "Bidir=false", "Geom zero"]
    for name in selected:
        arr = parse_train_log(LOGS[name])
        if arr.size == 0:
            continue
        step = arr[:, 0]
        axes[0, 0].plot(step, smooth(arr[:, 1]), label=name)
        axes[0, 1].plot(step, smooth(arr[:, 2]), label=name)
        axes[1, 0].plot(step, smooth(arr[:, 3]), label=name)
        axes[1, 1].plot(step, smooth(arr[:, 4]), label=name)
    titles = ["Total loss", "Structure loss", "Localization loss", "Cell presence loss"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Global step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fig. S4. Training Loss Curves of DBM-SLANet Variants", weight="bold", y=1.02)
    save(fig, "fig_s4_training_loss_curves")


def fig_s4_single_loss_curves():
    selected = ["Mamba λ=0.2", "Bidir=false", "Geom zero"]
    panels = [
        ("fig_s4a_total_loss_curve", "Fig. S4a. Total Loss", 1, "Total loss"),
        ("fig_s4b_structure_loss_curve", "Fig. S4b. Structure Loss", 2, "Structure loss"),
        ("fig_s4c_localization_loss_curve", "Fig. S4c. Localization Loss", 3, "Localization loss"),
        ("fig_s4d_cell_presence_loss_curve", "Fig. S4d. Cell Presence Loss", 4, "Cell presence loss"),
    ]
    for filename, title, col, ylabel in panels:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for name in selected:
            arr = parse_train_log(LOGS[name])
            if arr.size == 0:
                continue
            ax.plot(arr[:, 0], smooth(arr[:, col]), label=name, linewidth=1.7)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Global step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        save(fig, filename)


def fig_s5_validation_curves():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    any_vals = False
    for name, path in LOGS.items():
        vals = parse_eval_acc(path)
        if not vals:
            continue
        any_vals = True
        xs, ys = zip(*vals)
        ax.plot(xs, ys, marker="o", label=name)
    if not any_vals:
        ax.text(0.5, 0.5, "Validation metric points were not found in available logs.", ha="center", va="center")
    ax.set_title("Fig. S5. Validation Structure Accuracy Curves During Fine-tuning", weight="bold")
    ax.set_xlabel("Approx. global step")
    ax.set_ylabel("Validation structure accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig_s5_validation_performance_curves")


def fig_s6_data_scale():
    sizes = np.array([350, 2000])
    baseline = np.array([0.4165, 0.4396])
    dbm = np.array([0.5520, 0.4877])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(sizes, baseline, marker="o", linewidth=2, label="SLANet baseline", color="#A7B1C2")
    ax.plot(sizes, dbm, marker="o", linewidth=2, label="DBM-R2", color=COLORS["blue"])
    for x, b, d in zip(sizes, baseline, dbm):
        ax.text(x, d + 0.015, f"+{d-b:.3f}", color=COLORS["blue"], ha="center", weight="bold")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["350", "2000"])
    ax.set_xlabel("Training samples")
    ax.set_ylabel("TEDS")
    ax.set_title("Fig. S6. Performance under Different Financial Training Scales", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.5, -0.22, "Note: this plot uses two available financial-domain training scales; treat as supplementary evidence.", transform=ax.transAxes, ha="center", color=COLORS["gray"], fontsize=8)
    save(fig, "fig_s6_data_scale_curve")


def fig4_qualitative_montage():
    qdir = ROOT / "论文材料/qualitative_figures"
    files = [
        qdir / "case_885_success_PMC3126717_005_00.png",
        qdir / "case_531_success_PMC2898048_011_00.png",
        qdir / "case_650_success_PMC3496644_001_00.png",
        qdir / "case_48_failure_PMC4225768_004_00.png",
    ]
    imgs = []
    labels = ["Success case 1", "Success case 2", "Success case 3", "Failure case"]
    for path, label in zip(files, labels):
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((900, 560))
        canvas = Image.new("RGB", (920, 620), "white")
        canvas.paste(img, ((920 - img.width) // 2, 50))
        d = ImageDraw.Draw(canvas)
        d.text((20, 18), label, fill=(34, 48, 71))
        imgs.append(canvas)
    if not imgs:
        return
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2 + 60), "white")
    d = ImageDraw.Draw(grid)
    d.text((20, 18), "Fig. 4. Qualitative Visualization of Table Structure Recovery", fill=(34, 48, 71))
    for i, img in enumerate(imgs[:4]):
        x = (i % 2) * w
        y = 60 + (i // 2) * h
        grid.paste(ImageOps.expand(img, border=1, fill=(220, 226, 235)), (x, y))
    grid.save(OUT / "fig4_qualitative_visualization_montage.png")
    # Store as PDF via PIL.
    grid.save(OUT / "fig4_qualitative_visualization_montage.pdf", "PDF", resolution=300)


def write_index():
    files = sorted(p.name for p in OUT.glob("*.png"))
    lines = [
        "# Generated DBM-SLANet Paper Figures",
        "",
        "Generated on 2026-07-10.",
        "",
        "## Files",
        "",
    ]
    for f in files:
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## Notes",
        "",
        "- Figures are generated from the currently available experiment summaries and logs.",
        "- Cell Presence PR/F1 curves are intentionally omitted because the audited F1=1.0 metric has label leakage.",
        "- `fig4_qualitative_visualization_montage` uses existing qualitative case images under `论文材料/qualitative_figures/`.",
        "- `fig_s6_data_scale_curve` has only two training-scale points and should be treated as supplementary evidence.",
        "",
    ]
    (OUT / "README_generated_figures.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    fig1_overall_framework()
    fig2_dbm_module()
    fig3_quantitative()
    fig4_qualitative_montage()
    fig5_tradeoff()
    fig_s1_dataset_splits()
    fig_s2_ablation_bar()
    fig_s4_loss_curves()
    fig_s4_single_loss_curves()
    fig_s5_validation_curves()
    fig_s6_data_scale()
    write_index()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
