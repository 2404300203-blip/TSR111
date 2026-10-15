#!/usr/bin/env python3
"""Generate non-overlapping CTA-500 appendix diagnostics from sample-level truth."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_results" / "kbs_appendix_audit_20260803" / "cta500_sample_truth"
AXIS = ROOT / "remote_results" / "cta500_axis_ablation_20260707" / "cta500_axis_sample_delta.jsonl"
OUT = ROOT / "论文材料" / "kbs_appendix_experiments_20260803"
FIG = OUT / "figures"
TAB = OUT / "tables"

MODEL_FILES = {
    "SLANet": "baseline_cta500_sample_teds.jsonl",
    "Cell Token": "phase11a_cell_token_only_cta500_sample_teds.jsonl",
    "Mamba lambda=0": "phase11b_full_mamba_lambda0_cta500_sample_teds.jsonl",
    "Horizontal": "expC_horizontal_cta500_sample_teds.jsonl",
    "Vertical": "expD_vertical_cta500_sample_teds.jsonl",
    "DBM-SLANet": "phase5e_cta500_sample_teds.jsonl",
}

COLORS = {
    "SLANet": "#AEB9C2",
    "Cell Token": "#D5A84E",
    "Mamba lambda=0": "#96AABE",
    "Horizontal": "#65A89D",
    "Vertical": "#387F84",
    "DBM-SLANet": "#153F3A",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def savefig(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"dpi": 500}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            FIG / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.04,
            **kwargs,
        )
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9DFE2", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#41505A")
    ax.spines["bottom"].set_color("#41505A")


def load_frame() -> pd.DataFrame:
    metadata = pd.DataFrame(read_jsonl(AXIS)).rename(columns={"sample_index_cta": "sample_index"})
    keep = [
        "sample_index", "filename", "tags", "row_count", "header_rows", "max_width",
        "span_tokens", "rowspan_tokens", "colspan_tokens", "gt_len", "complexity_score",
    ]
    frame = metadata[keep].copy()

    labels = read_jsonl(SOURCE / "cta500_labels.jsonl")
    label_map = {row["filename"]: row for row in labels}
    frame["cell_count"] = frame["filename"].map(lambda name: len(label_map[name]["html"]["cells"]))
    frame["span_density"] = frame["span_tokens"] / frame["cell_count"].clip(lower=1)
    frame["span_type"] = np.select(
        [
            (frame["rowspan_tokens"] > 0) & (frame["colspan_tokens"] > 0),
            frame["rowspan_tokens"] > 0,
            frame["colspan_tokens"] > 0,
        ],
        ["rowspan+colspan", "rowspan only", "colspan only"],
        default="no explicit span",
    )

    for model, filename in MODEL_FILES.items():
        records = read_jsonl(SOURCE / filename)
        if [row["filename"] for row in records] != frame["filename"].tolist():
            raise ValueError(f"Sample order mismatch: {model}")
        prefix = model.lower().replace(" ", "_").replace("=", "").replace("-", "_")
        frame[f"{prefix}_teds"] = [row["teds"] for row in records]
        frame[f"{prefix}_exact"] = [bool(row["exact"]) for row in records]
        frame[f"{prefix}_confidence"] = [row.get("confidence") for row in records]
        frame[f"{prefix}_pred_len"] = [row.get("pred_len") for row in records]

    frame["dbm_length_error"] = frame["dbm_slanet_pred_len"] - frame["gt_len"]
    frame["dbm_abs_length_error"] = frame["dbm_length_error"].abs()
    frame["dbm_gain_vs_slanet"] = frame["dbm_slanet_teds"] - frame["slanet_teds"]
    frame["dbm_gain_vs_best_single"] = frame["dbm_slanet_teds"] - frame[
        ["horizontal_teds", "vertical_teds"]
    ].max(axis=1)
    frame["axis_gap"] = (frame["horizontal_teds"] - frame["vertical_teds"]).abs()
    return frame


def metric_columns() -> tuple[list[str], list[str]]:
    teds = []
    exact = []
    for model in MODEL_FILES:
        prefix = model.lower().replace(" ", "_").replace("=", "").replace("-", "_")
        teds.append(f"{prefix}_teds")
        exact.append(f"{prefix}_exact")
    return teds, exact


def write_overall(frame: pd.DataFrame) -> None:
    rows = []
    for model in MODEL_FILES:
        prefix = model.lower().replace(" ", "_").replace("=", "").replace("-", "_")
        rows.append(
            {
                "model": model,
                "samples": len(frame),
                "mean_teds": frame[f"{prefix}_teds"].mean(),
                "median_teds": frame[f"{prefix}_teds"].median(),
                "std_teds": frame[f"{prefix}_teds"].std(ddof=1),
                "exact_rate": frame[f"{prefix}_exact"].mean(),
                "mean_confidence": frame[f"{prefix}_confidence"].mean(),
                "mean_pred_len": frame[f"{prefix}_pred_len"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(TAB / "A01_overall_sample_statistics.csv", index=False)


def stratified_table(frame: pd.DataFrame, variable: str, labels: list[str], stem: str) -> pd.DataFrame:
    work = frame.copy()
    work["bucket"] = pd.qcut(work[variable], q=4, labels=labels, duplicates="drop")
    rows = []
    for bucket, part in work.groupby("bucket", observed=True):
        row = {
            "bucket": str(bucket),
            "samples": len(part),
            f"{variable}_min": part[variable].min(),
            f"{variable}_mean": part[variable].mean(),
            f"{variable}_max": part[variable].max(),
        }
        for model in ("SLANet", "Cell Token", "Horizontal", "Vertical", "DBM-SLANet"):
            prefix = model.lower().replace(" ", "_").replace("-", "_")
            row[f"{prefix}_teds"] = part[f"{prefix}_teds"].mean()
            row[f"{prefix}_exact"] = part[f"{prefix}_exact"].mean()
        row["dbm_gain_vs_slanet"] = part["dbm_gain_vs_slanet"].mean()
        row["dbm_gain_vs_best_single"] = part["dbm_gain_vs_best_single"].mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TAB / f"{stem}.csv", index=False)
    return result


def plot_stratified(table: pd.DataFrame, xlabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), facecolor="white")
    x = np.arange(len(table))
    for model in ("SLANet", "Cell Token", "Horizontal", "Vertical", "DBM-SLANet"):
        prefix = model.lower().replace(" ", "_").replace("-", "_")
        axes[0].plot(
            x, table[f"{prefix}_teds"], marker="o", ms=4, lw=1.4,
            color=COLORS[model], label=model,
        )
        axes[1].plot(
            x, table[f"{prefix}_exact"] * 100, marker="o", ms=4, lw=1.4,
            color=COLORS[model], label=model,
        )
    for ax in axes:
        ax.set_xticks(x, table["bucket"])
        ax.set_xlabel(xlabel)
        style_axis(ax)
    axes[0].set_ylabel("Mean TEDS")
    axes[1].set_ylabel("Structure exact match (%)")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(w_pad=2.0)
    savefig(fig, stem)


def categorical_table(frame: pd.DataFrame, variable: str, order: list, stem: str) -> pd.DataFrame:
    rows = []
    for value in order:
        part = frame[frame[variable] == value]
        if part.empty:
            continue
        rows.append(
            {
                variable: value,
                "samples": len(part),
                "slanet_teds": part["slanet_teds"].mean(),
                "cell_token_teds": part["cell_token_teds"].mean(),
                "horizontal_teds": part["horizontal_teds"].mean(),
                "vertical_teds": part["vertical_teds"].mean(),
                "dbm_slanet_teds": part["dbm_slanet_teds"].mean(),
                "dbm_slanet_exact": part["dbm_slanet_exact"].mean(),
                "dbm_gain_vs_slanet": part["dbm_gain_vs_slanet"].mean(),
                "dbm_gain_vs_best_single": part["dbm_gain_vs_best_single"].mean(),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TAB / f"{stem}.csv", index=False)
    return result


def confidence_analysis(frame: pd.DataFrame) -> None:
    conf = "dbm_slanet_confidence"
    work = frame.copy()
    work["confidence_decile"] = pd.qcut(work[conf], 10, labels=False, duplicates="drop") + 1
    reliability = work.groupby("confidence_decile").agg(
        samples=(conf, "size"),
        confidence_mean=(conf, "mean"),
        confidence_min=(conf, "min"),
        confidence_max=(conf, "max"),
        teds_mean=("dbm_slanet_teds", "mean"),
        exact_rate=("dbm_slanet_exact", "mean"),
        abs_length_error=("dbm_abs_length_error", "mean"),
    ).reset_index()
    reliability.to_csv(TAB / "A10_dbm_confidence_deciles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), facecolor="white")
    axes[0].plot(reliability["confidence_mean"], reliability["teds_mean"], "o-", color="#153F3A")
    axes[1].plot(reliability["confidence_mean"], reliability["exact_rate"] * 100, "o-", color="#153F3A")
    axes[0].set(xlabel="Mean decoder confidence", ylabel="Mean TEDS")
    axes[1].set(xlabel="Mean decoder confidence", ylabel="Structure exact match (%)")
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(w_pad=2.0)
    savefig(fig, "F10_dbm_confidence_reliability")

    rows = []
    ordered = work.sort_values(conf, ascending=False)
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        kept = ordered.iloc[: max(1, round(len(ordered) * coverage))]
        rows.append(
            {
                "coverage": coverage,
                "samples": len(kept),
                "confidence_threshold": kept[conf].min(),
                "mean_teds": kept["dbm_slanet_teds"].mean(),
                "exact_rate": kept["dbm_slanet_exact"].mean(),
                "mean_abs_length_error": kept["dbm_abs_length_error"].mean(),
            }
        )
    selective = pd.DataFrame(rows)
    selective.to_csv(TAB / "A11_dbm_selective_prediction.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(5.8, 3.4), facecolor="white")
    ax2 = ax1.twinx()
    ax1.plot(selective["coverage"] * 100, selective["mean_teds"], "o-", color="#153F3A", label="TEDS")
    ax2.plot(selective["coverage"] * 100, selective["exact_rate"] * 100, "s-", color="#D5A84E", label="Exact match")
    ax1.set(xlabel="Retained samples by confidence (%)", ylabel="Mean TEDS")
    ax2.set_ylabel("Structure exact match (%)")
    ax1.invert_xaxis()
    style_axis(ax1)
    ax2.spines["top"].set_visible(False)
    ax1.legend(frameon=False, loc="lower left")
    ax2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, "F11_dbm_selective_prediction")


def relationship_analysis(frame: pd.DataFrame) -> None:
    teds_cols, exact_cols = metric_columns()
    names = list(MODEL_FILES)
    corr = frame[teds_cols].corr(method="spearman")
    corr.index = names
    corr.columns = names
    corr.to_csv(TAB / "A12_teds_spearman_correlation.csv")

    fig, ax = plt.subplots(figsize=(5.4, 4.7), facecolor="white")
    image = ax.imshow(corr, vmin=0.35, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(len(names)), names, rotation=38, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.7,
                    color="white" if corr.iloc[i, j] > 0.78 else "#24313A")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Spearman rho")
    fig.tight_layout()
    savefig(fig, "F12_teds_error_correlation")

    wins = pd.DataFrame(index=names, columns=names, dtype=float)
    ties = pd.DataFrame(index=names, columns=names, dtype=float)
    exact_disagreement = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, left in enumerate(teds_cols):
        for j, right in enumerate(teds_cols):
            delta = frame[left] - frame[right]
            wins.iloc[i, j] = (delta > 1e-12).mean()
            ties.iloc[i, j] = (delta.abs() <= 1e-12).mean()
            exact_disagreement.iloc[i, j] = (frame[exact_cols[i]] != frame[exact_cols[j]]).mean()
    wins.to_csv(TAB / "A13_pairwise_teds_win_rate.csv")
    ties.to_csv(TAB / "A14_pairwise_teds_tie_rate.csv")
    exact_disagreement.to_csv(TAB / "A15_pairwise_exact_disagreement.csv")

    fig, ax = plt.subplots(figsize=(5.4, 4.7), facecolor="white")
    image = ax.imshow(wins * 100, vmin=0, vmax=60, cmap="GnBu")
    ax.set_xticks(range(len(names)), names, rotation=38, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{wins.iloc[i, j] * 100:.1f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Row model wins (%)")
    fig.tight_layout()
    savefig(fig, "F13_pairwise_teds_win_rate")


def length_error_analysis(frame: pd.DataFrame) -> None:
    bins = [-np.inf, -50, -20, -5, 5, 20, 50, np.inf]
    labels = ["<-50", "-50:-21", "-20:-6", "-5:5", "6:20", "21:50", ">50"]
    work = frame.copy()
    work["length_error_bin"] = pd.cut(work["dbm_length_error"], bins=bins, labels=labels)
    table = work.groupby("length_error_bin", observed=True).agg(
        samples=("filename", "size"),
        mean_length_error=("dbm_length_error", "mean"),
        mean_teds=("dbm_slanet_teds", "mean"),
        exact_rate=("dbm_slanet_exact", "mean"),
        mean_confidence=("dbm_slanet_confidence", "mean"),
        mean_gt_len=("gt_len", "mean"),
    ).reset_index()
    table.to_csv(TAB / "A16_dbm_prediction_length_error.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), facecolor="white")
    axes[0].hist(frame["dbm_length_error"], bins=35, color="#4F8D85", edgecolor="white")
    axes[0].axvline(0, color="#24313A", lw=0.9)
    axes[0].set(xlabel="Predicted length - GT length", ylabel="Samples")
    axes[1].scatter(
        frame["dbm_abs_length_error"], frame["dbm_slanet_teds"],
        s=12, alpha=0.42, color="#2F746D", edgecolors="none",
    )
    rho = spearmanr(frame["dbm_abs_length_error"], frame["dbm_slanet_teds"]).statistic
    axes[1].text(0.97, 0.95, f"Spearman rho = {rho:.3f}", transform=axes[1].transAxes,
                 ha="right", va="top", color="#4A5860")
    axes[1].set(xlabel="Absolute sequence-length error", ylabel="TEDS")
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(w_pad=2.0)
    savefig(fig, "F14_dbm_sequence_length_error")


def bootstrap_analysis(frame: pd.DataFrame) -> None:
    delta = frame["dbm_gain_vs_slanet"].to_numpy()
    rng = np.random.default_rng(20260803)
    rows = []
    for sample_size in (50, 100, 200, 300, 400, 500):
        estimates = np.empty(3000)
        for i in range(len(estimates)):
            estimates[i] = rng.choice(delta, size=sample_size, replace=True).mean()
        rows.append(
            {
                "bootstrap_sample_size": sample_size,
                "resamples": len(estimates),
                "mean_delta": estimates.mean(),
                "ci95_low": np.quantile(estimates, 0.025),
                "ci95_high": np.quantile(estimates, 0.975),
                "p_delta_le_zero": (estimates <= 0).mean(),
                "std": estimates.std(ddof=1),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "A17_bootstrap_sample_size_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.7, 3.35), facecolor="white")
    ax.errorbar(
        table["bootstrap_sample_size"], table["mean_delta"],
        yerr=[table["mean_delta"] - table["ci95_low"], table["ci95_high"] - table["mean_delta"]],
        fmt="o-", capsize=3, color="#153F3A",
    )
    ax.axhline(0, color="#A4ADB3", lw=0.8)
    ax.set(xlabel="Bootstrap sample size", ylabel="DBM-SLANet - SLANet mean TEDS")
    style_axis(ax)
    fig.tight_layout()
    savefig(fig, "F15_bootstrap_sample_size_stability")


def gain_and_tags(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), facecolor="white")
    axes[0].hist(frame["dbm_gain_vs_slanet"], bins=35, color="#3E837B", edgecolor="white")
    axes[0].axvline(0, color="#24313A", lw=0.8)
    axes[0].set(xlabel="Per-sample TEDS gain over SLANet", ylabel="Samples")
    axes[1].scatter(
        frame["complexity_score"], frame["dbm_gain_vs_slanet"],
        s=12, alpha=0.42, color="#356F77", edgecolors="none",
    )
    rho = spearmanr(frame["complexity_score"], frame["dbm_gain_vs_slanet"]).statistic
    axes[1].text(0.97, 0.95, f"Spearman rho = {rho:.3f}", transform=axes[1].transAxes,
                 ha="right", va="top", color="#4A5860")
    axes[1].axhline(0, color="#A4ADB3", lw=0.8)
    axes[1].set(xlabel="Complexity score", ylabel="TEDS gain over SLANet")
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(w_pad=2.0)
    savefig(fig, "F16_gain_distribution_and_complexity")

    all_tags = sorted({tag for tags in frame["tags"] for tag in tags})
    matrix = np.zeros((len(all_tags), len(all_tags)), dtype=int)
    for tags in frame["tags"]:
        present = set(tags)
        for i, left in enumerate(all_tags):
            for j, right in enumerate(all_tags):
                matrix[i, j] += int(left in present and right in present)
    pd.DataFrame(matrix, index=all_tags, columns=all_tags).to_csv(TAB / "A18_tag_cooccurrence_counts.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor="white")
    image = ax.imshow(matrix, cmap="YlGnBu")
    labels = [tag.replace("_", " ") for tag in all_tags]
    ax.set_xticks(range(len(labels)), labels, rotation=42, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6.2,
                    color="white" if matrix[i, j] > matrix.max() * 0.58 else "#24313A")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="Samples")
    fig.tight_layout()
    savefig(fig, "F17_complex_tag_cooccurrence")

    combos = Counter(tuple(sorted(tags)) for tags in frame["tags"])
    rows = []
    for combo, count in combos.most_common(15):
        mask = frame["tags"].map(lambda tags: tuple(sorted(tags)) == combo)
        part = frame[mask]
        rows.append(
            {
                "tag_combination": " + ".join(combo),
                "samples": count,
                "slanet_teds": part["slanet_teds"].mean(),
                "dbm_slanet_teds": part["dbm_slanet_teds"].mean(),
                "dbm_gain_vs_slanet": part["dbm_gain_vs_slanet"].mean(),
                "dbm_gain_vs_best_single": part["dbm_gain_vs_best_single"].mean(),
                "dbm_exact": part["dbm_slanet_exact"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(TAB / "A19_top_tag_intersections.csv", index=False)


def recovery_analysis(frame: pd.DataFrame) -> None:
    baseline_fail = ~frame["slanet_exact"]
    rows = []
    for model in ("Cell Token", "Mamba lambda=0", "Horizontal", "Vertical", "DBM-SLANet"):
        prefix = model.lower().replace(" ", "_").replace("=", "").replace("-", "_")
        rows.append(
            {
                "model": model,
                "baseline_inexact_samples": int(baseline_fail.sum()),
                "recovered_exact_count": int((baseline_fail & frame[f"{prefix}_exact"]).sum()),
                "recovered_exact_rate": (baseline_fail & frame[f"{prefix}_exact"]).sum() / baseline_fail.sum(),
                "mean_teds_on_baseline_failures": frame.loc[baseline_fail, f"{prefix}_teds"].mean(),
                "catastrophic_teds_zero_rate": (frame.loc[baseline_fail, f"{prefix}_teds"] == 0).mean(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "A20_recovery_on_slanet_failures.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.45), facecolor="white")
    x = np.arange(len(table))
    ax.bar(x, table["recovered_exact_rate"] * 100, color=[COLORS[m] for m in table["model"]])
    ax.set_xticks(x, table["model"], rotation=24, ha="right")
    ax.set(ylabel="Exact recovery among SLANet failures (%)")
    style_axis(ax)
    fig.tight_layout()
    savefig(fig, "F18_recovery_on_slanet_failures")


def export_case_lists(frame: pd.DataFrame) -> None:
    case_dir = OUT / "case_lists"
    case_dir.mkdir(parents=True, exist_ok=True)
    selections = {
        "top_dbm_improvements": frame.nlargest(20, "dbm_gain_vs_slanet"),
        "largest_dbm_regressions": frame.nsmallest(20, "dbm_gain_vs_slanet"),
        "strong_axis_complementarity": frame.nlargest(20, "dbm_gain_vs_best_single"),
        "high_confidence_failures": frame[~frame["dbm_slanet_exact"]].nlargest(20, "dbm_slanet_confidence"),
        "exact_complex_successes": frame[frame["dbm_slanet_exact"]].nlargest(20, "complexity_score"),
    }
    columns = [
        "sample_index", "filename", "tags", "row_count", "max_width", "span_tokens",
        "gt_len", "complexity_score", "slanet_teds", "horizontal_teds", "vertical_teds",
        "dbm_slanet_teds", "dbm_gain_vs_slanet", "dbm_gain_vs_best_single",
        "dbm_slanet_confidence", "dbm_length_error",
    ]
    for name, subset in selections.items():
        subset[columns].to_csv(case_dir / f"{name}.csv", index=False)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    frame.to_csv(TAB / "A00_cta500_merged_sample_records.csv", index=False)
    write_overall(frame)

    specifications = [
        ("gt_len", ["Q1 shortest", "Q2", "Q3", "Q4 longest"], "A02_gt_length_quartiles", "GT structure length quartile", "F02_performance_by_gt_length"),
        ("cell_count", ["Q1 fewest", "Q2", "Q3", "Q4 most"], "A03_cell_count_quartiles", "Cell-count quartile", "F03_performance_by_cell_count"),
        ("row_count", ["Q1 fewest", "Q2", "Q3", "Q4 most"], "A04_row_count_quartiles", "Row-count quartile", "F04_performance_by_row_count"),
        ("max_width", ["Q1 narrowest", "Q2", "Q3", "Q4 widest"], "A05_width_quartiles", "Maximum-width quartile", "F05_performance_by_table_width"),
        ("span_tokens", ["Q1 fewest", "Q2", "Q3", "Q4 most"], "A06_span_count_quartiles", "Span-count quartile", "F06_performance_by_span_count"),
        ("span_density", ["Q1 sparse", "Q2", "Q3", "Q4 dense"], "A07_span_density_quartiles", "Span-density quartile", "F07_performance_by_span_density"),
        ("complexity_score", ["Q1 lower", "Q2", "Q3", "Q4 higher"], "A08_complexity_quartiles", "Complexity-score quartile", "F08_performance_by_complexity"),
    ]
    for variable, labels, table_stem, xlabel, figure_stem in specifications:
        table = stratified_table(frame, variable, labels, table_stem)
        plot_stratified(table, xlabel, figure_stem)

    span_table = categorical_table(
        frame,
        "span_type",
        ["no explicit span", "rowspan only", "colspan only", "rowspan+colspan"],
        "A09_span_type_comparison",
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.45), facecolor="white")
    x = np.arange(len(span_table))
    width = 0.16
    models = ("SLANet", "Cell Token", "Horizontal", "Vertical", "DBM-SLANet")
    for index, model in enumerate(models):
        prefix = model.lower().replace(" ", "_").replace("-", "_")
        ax.bar(x + (index - 2) * width, span_table[f"{prefix}_teds"], width,
               color=COLORS[model], label=model)
    ax.set_xticks(x, span_table["span_type"], rotation=15, ha="right")
    ax.set_ylabel("Mean TEDS")
    ax.legend(frameon=False, ncol=3)
    style_axis(ax)
    fig.tight_layout()
    savefig(fig, "F09_performance_by_span_type")

    confidence_analysis(frame)
    relationship_analysis(frame)
    length_error_analysis(frame)
    bootstrap_analysis(frame)
    gain_and_tags(frame)
    recovery_analysis(frame)
    export_case_lists(frame)

    manifest = {
        "samples": len(frame),
        "models": list(MODEL_FILES),
        "tables": len(list(TAB.glob("*.csv"))),
        "figures_png": len(list(FIG.glob("*.png"))),
        "figures_pdf": len(list(FIG.glob("*.pdf"))),
        "case_lists": len(list((OUT / "case_lists").glob("*.csv"))),
    }
    (OUT / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
