#!/usr/bin/env python3
"""Clean, audit and plot the completed appendix control experiments."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_results/kbs_appendix_new_experiments_20260803/appendix_new_experiment_results.csv"
OUT = ROOT / "论文材料/kbs_appendix_experiments_20260803"
TAB = OUT / "tables"
FIG = OUT / "figures"

COLORS = {"SLANet": "#A9B4BC", "DBM-SLANet": "#164F49"}
ACCENT = "#D0A13B"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.75, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig: plt.Figure, stem: str) -> None:
    for ext, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04, **kwargs)
    plt.close(fig)


def style(ax: plt.Axes) -> None:
    ax.grid(True, color="#DCE1E3", lw=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#42515A")
    ax.spines["bottom"].set_color("#42515A")


def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(SOURCE)
    failures = raw[raw.status != "completed"].copy()
    completed = raw[raw.status == "completed"].drop_duplicates("experiment_id", keep="last").copy()
    completed = completed.sort_values(["category", "model", "experiment_id"]).reset_index(drop=True)
    if len(completed) != 34:
        raise ValueError(f"Expected 34 completed experiment IDs, found {len(completed)}")
    if not (completed.samples == 500).all():
        raise ValueError("Not all completed experiments contain 500 samples")
    completed.to_csv(TAB / "N00_appendix_new_experiment_results.csv", index=False)
    failures.to_csv(TAB / "N00_setup_failures_excluded.csv", index=False)
    return completed, failures


def plot_resolution(frame: pd.DataFrame) -> None:
    part = frame[frame.category == "resolution"].copy()
    references = pd.DataFrame([
        {"model": "SLANet", "resolution": 488, "teds": 0.4843354130430532, "structure_acc": 0.208, "fps": 18.6527},
        {"model": "DBM-SLANet", "resolution": 488, "teds": 0.640672654391009, "structure_acc": 0.432, "fps": 9.1346},
    ])
    part = pd.concat([part, references], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.05), facecolor="white")
    for model in COLORS:
        rows = part[part.model == model].sort_values("resolution")
        axes[0].plot(rows.resolution, rows.teds, "o-", color=COLORS[model], lw=1.5, ms=4, label=model)
        axes[1].plot(rows.resolution, rows.structure_acc * 100, "o-", color=COLORS[model], lw=1.5, ms=4)
        axes[2].plot(rows.resolution, rows.fps, "o-", color=COLORS[model], lw=1.5, ms=4)
    for ax, ylabel in zip(axes, ("TEDS", "Structure exact match (%)", "FPS")):
        ax.axvline(488, color=ACCENT, lw=1, ls="--")
        ax.set(xlabel="Input resolution (pixels)", ylabel=ylabel)
        style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.4)
    save(fig, "N01_resolution_sensitivity")


def plot_batch(frame: pd.DataFrame) -> None:
    part = frame[frame.category == "batch_size"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.05), facecolor="white")
    for model in COLORS:
        rows = part[part.model == model].sort_values("batch_size")
        axes[0].plot(rows.batch_size, rows.fps, "o-", color=COLORS[model], lw=1.5, ms=4, label=model)
        axes[1].plot(rows.batch_size, rows.peak_memory_mib, "o-", color=COLORS[model], lw=1.5, ms=4)
        axes[2].plot(rows.batch_size, rows.teds, "o-", color=COLORS[model], lw=1.5, ms=4)
    for ax, ylabel in zip(axes, ("FPS", "Peak memory (MiB)", "TEDS")):
        ax.set(xlabel="Evaluation batch size", ylabel=ylabel, xticks=[1, 2, 8])
        style(ax)
    axes[0].legend(frameon=False)
    axes[2].set_ylim(0.47, 0.65)
    fig.tight_layout(w_pad=1.4)
    save(fig, "N02_batch_efficiency")


def plot_robustness(frame: pd.DataFrame) -> None:
    order = ["clean", "jpeg_q75", "jpeg_q50", "blur_sigma0p8", "blur_sigma1p2", "brightness0p75", "brightness1p25", "contrast0p75"]
    labels = ["Clean", "JPEG 75", "JPEG 50", "Blur 0.8", "Blur 1.2", "Bright 0.75", "Bright 1.25", "Contrast 0.75"]
    part = frame[frame.category == "robustness"].copy()
    clean = pd.DataFrame([
        {"model": "SLANet", "variant": "clean", "teds": 0.4843354130430532, "structure_acc": 0.208},
        {"model": "DBM-SLANet", "variant": "clean", "teds": 0.640672654391009, "structure_acc": 0.432},
    ])
    part = pd.concat([clean, part], ignore_index=True)
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.3), facecolor="white", sharex=True)
    for model in COLORS:
        rows = part[part.model == model].set_index("variant").reindex(order)
        axes[0].plot(x, rows.teds, "o-", color=COLORS[model], lw=1.5, ms=4, label=model)
        axes[1].plot(x, rows.structure_acc * 100, "o-", color=COLORS[model], lw=1.5, ms=4)
    axes[0].set_ylabel("TEDS")
    axes[1].set_ylabel("Structure exact match (%)")
    axes[1].set_xlabel("Input condition")
    axes[1].set_xticks(x, labels, rotation=23, ha="right")
    for ax in axes:
        style(ax)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout(h_pad=0.9)
    save(fig, "N03_robustness_controls")


def plot_timing(frame: pd.DataFrame) -> None:
    part = frame[frame.category == "timing_repeat"].copy()
    summary = part.groupby("model").fps.agg(["mean", "std", "min", "max"]).reindex(list(COLORS))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.05), facecolor="white")
    for model in COLORS:
        rows = part[part.model == model].sort_values("repeat")
        axes[0].plot(rows.repeat, rows.fps, "o-", color=COLORS[model], lw=1.5, ms=4, label=model)
    axes[0].set(xlabel="Independent repeat", ylabel="FPS", xticks=[1, 2, 3])
    x = np.arange(2)
    axes[1].bar(x, summary["mean"], yerr=summary["std"], width=0.58,
                color=[COLORS[name] for name in summary.index], capsize=4)
    axes[1].set(xlabel="Model", ylabel="Mean FPS ± SD", xticks=x, xticklabels=summary.index)
    for ax in axes:
        style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.7)
    save(fig, "N04_timing_stability")


def audit(frame: pd.DataFrame, failures: pd.DataFrame) -> None:
    batch = frame[frame.category == "batch_size"]
    timing = frame[frame.category == "timing_repeat"]
    report = {
        "raw_rows": int(len(pd.read_csv(SOURCE))),
        "excluded_setup_failures": int(len(failures)),
        "completed_unique_experiments": int(len(frame)),
        "all_completed_samples_500": bool((frame.samples == 500).all()),
        "categories": frame.category.value_counts().to_dict(),
        "batch_metric_ranges": {
            model: {
                "teds_range": float(part.teds.max() - part.teds.min()),
                "exact_range": float(part.structure_acc.max() - part.structure_acc.min()),
            }
            for model, part in batch.groupby("model")
        },
        "timing_metric_ranges": {
            model: {
                "teds_range": float(part.teds.max() - part.teds.min()),
                "exact_range": float(part.structure_acc.max() - part.structure_acc.min()),
                "fps_mean": float(part.fps.mean()),
                "fps_std": float(part.fps.std(ddof=1)),
            }
            for model, part in timing.groupby("model")
        },
    }
    (OUT / "new_experiment_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame, failures = clean()
    plot_resolution(frame)
    plot_batch(frame)
    plot_robustness(frame)
    plot_timing(frame)
    audit(frame, failures)


if __name__ == "__main__":
    main()
