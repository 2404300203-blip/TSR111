#!/usr/bin/env python3
"""Build Fig. 1 from unified CTA-500 measurements only."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文材料" / "fig1_updated_bubble_20260811"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 10.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
    }
)


# Internal rows come from corrected Table 2/Table 5. External rows come from
# the official checkpoints evaluated on the identical 500-image CTA protocol.
MEASURED = [
    {"model": "SLANet", "label": "SLANet", "family": "Baseline", "status": "internal", "teds": 0.4843354130, "acc": 0.208, "fps": 24.6357935924, "params": 1892610},
    {"model": "Cell Token only", "label": "Cell Token", "family": "Lightweight", "status": "internal", "teds": 0.5436438417, "acc": 0.284, "fps": 24.0375704776, "params": 1927427},
    {"model": "Lite Mamba", "label": "Lite Mamba", "family": "Lightweight", "status": "internal", "teds": 0.6057106970, "acc": 0.386, "fps": 16.4232018235, "params": 2088373},
    {"model": "Full Mamba, no presence", "label": "Mamba, no presence", "family": "Ablation", "status": "internal", "teds": 0.5392074837, "acc": 0.282, "fps": 12.4587209652, "params": 2445607},
    {"model": "Full Mamba, presence", "label": "Mamba, presence", "family": "Ablation", "status": "internal", "teds": 0.5421486913, "acc": 0.302, "fps": 12.4278706769, "params": 2445607},
    {"model": "Full Mamba, one-way", "label": "Mamba, one-way", "family": "Ablation", "status": "internal", "teds": 0.5466726603, "acc": 0.300, "fps": 17.5790122725, "params": 2211285},
    {"model": "Full DBM, d=64", "label": "DBM d=64", "family": "Ablation", "status": "internal", "teds": 0.5437838209, "acc": 0.290, "fps": 13.6432054117, "params": 2046295},
    {"model": "Full DBM, d_s=8", "label": "DBM d_s=8", "family": "Ablation", "status": "internal", "teds": 0.5493114122, "acc": 0.300, "fps": 13.5347063901, "params": 2420967},
    {"model": "Full DBM, d=192", "label": "DBM d=192", "family": "Ablation", "status": "internal", "teds": 0.5468112900, "acc": 0.302, "fps": 13.3842998613, "params": 3090679},
    {"model": "Full DBM, e=1", "label": "DBM e=1", "family": "Ablation", "status": "internal", "teds": 0.5493633086, "acc": 0.302, "fps": 13.4851102094, "params": 2212135},
    {"model": "Full Mamba, zero geometry", "label": "Mamba, zero geometry", "family": "Ablation", "status": "internal", "teds": 0.5392074837, "acc": 0.282, "fps": 12.6263297130, "params": 2445607},
    {"model": "True Horizontal-only", "label": "Horizontal-only", "family": "Single-axis", "status": "internal", "teds": 0.5460396438, "acc": 0.294, "fps": 17.2840983167, "params": 2194901},
    {"model": "True Vertical-only", "label": "Vertical-only", "family": "Single-axis", "status": "internal", "teds": 0.5456143933, "acc": 0.296, "fps": 17.2466630346, "params": 2194901},
    {"model": "DBM-SLANet", "label": "DBM-SLANet (ours)", "family": "Proposed", "status": "internal", "teds": 0.6406726544, "acc": 0.432, "fps": 12.4545526331, "params": 2445607},
    {"model": "UniTable Large", "label": "UniTable Large\n(zero-shot external)", "family": "External", "status": "zero-shot external baseline", "teds": 0.7129328478, "acc": 0.538, "fps": 2.1354772544, "params": 124146491},
    {"model": "Table Transformer v1.0", "label": "Table Transformer\n(zero-shot external)", "family": "External", "status": "zero-shot external baseline", "teds": 0.2551554703, "acc": 0.004, "fps": 13.0688583587, "params": 28828619},
]

MEMORY_MIB = {
    "SLANet": 724,
    "Cell Token only": 724,
    "Lite Mamba": 726,
    "Full Mamba, no presence": 728,
    "Full Mamba, presence": 728,
    "Full Mamba, one-way": 726,
    "Full DBM, d=64": 726,
    "Full DBM, d_s=8": 728,
    "Full DBM, d=192": 736,
    "Full DBM, e=1": 726,
    "Full Mamba, zero geometry": 728,
    "True Horizontal-only": 726,
    "True Vertical-only": 726,
    "DBM-SLANet": 728,
    "UniTable Large": 1018,
    "Table Transformer v1.0": 1016,
}

for measured_row in MEASURED:
    measured_row["memory_mib"] = MEMORY_MIB[measured_row["model"]]

COLORS = {
    "Baseline": "#5B8FF9",
    "Lightweight": "#E3A83B",
    "Ablation": "#7D97B5",
    "Single-axis": "#46A98B",
    "Proposed": "#D9534F",
    "External": "#266E75",
}

OFFSETS = {
    "SLANet": (-12, 20),
    "Cell Token only": (11, 18),
    "Lite Mamba": (10, 18),
    "Full Mamba, no presence": (-34, -43),
    "Full Mamba, presence": (-45, -13),
    "Full Mamba, one-way": (-12, 47),
    "Full DBM, d=64": (-14, 56),
    "Full DBM, d_s=8": (31, 50),
    "Full DBM, d=192": (36, 17),
    "Full DBM, e=1": (38, -13),
    "Full Mamba, zero geometry": (-47, -7),
    "True Horizontal-only": (-32, -27),
    "True Vertical-only": (-18, -49),
    "DBM-SLANet": (-14, -25),
    "UniTable Large": (-16, 25),
    "Table Transformer v1.0": (12, 22),
}

MEMORY_OFFSETS = {
    "SLANet": (-15, 20),
    "Cell Token only": (-18, 20),
    "Lite Mamba": (10, -25),
    "Full Mamba, no presence": (-50, -38),
    "Full Mamba, presence": (-52, -63),
    "Full Mamba, one-way": (-45, -88),
    "Full DBM, d=64": (-30, -113),
    "Full DBM, d_s=8": (48, -113),
    "Full DBM, d=192": (50, -88),
    "Full DBM, e=1": (48, -63),
    "Full Mamba, zero geometry": (-55, -138),
    "True Horizontal-only": (-35, -163),
    "True Vertical-only": (30, -163),
    "DBM-SLANet": (10, -28),
    "UniTable Large": (-14, 27),
    "Table Transformer v1.0": (10, 27),
}


def bubble_area(structure_accuracy: float) -> float:
    return 95.0 + 1320.0 * structure_accuracy


def parameter_bubble_area(parameters: int) -> float:
    log_min = math.log10(min(row["params"] for row in MEASURED))
    log_max = math.log10(max(row["params"] for row in MEASURED))
    normalized = (math.log10(parameters) - log_min) / (log_max - log_min)
    return 180.0 + 850.0 * normalized


def write_source_csv() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig1_updated_bubble_source.csv"
    fields = [
        "model",
        "label",
        "family",
        "status",
        "teds_cta500",
        "structure_accuracy_cta500",
        "fps_cta500",
        "peak_gpu_memory_mib",
        "parameters",
        "source_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in MEASURED:
            external = row["family"] == "External"
            writer.writerow(
                {
                    "model": row["model"],
                    "label": row["label"].replace("\n", " "),
                    "family": row["family"],
                    "status": row["status"],
                    "teds_cta500": f'{row["teds"]:.10f}',
                    "structure_accuracy_cta500": f'{row["acc"]:.3f}',
                    "fps_cta500": f'{row["fps"]:.6f}',
                    "peak_gpu_memory_mib": row["memory_mib"],
                    "parameters": row["params"],
                    "source_note": (
                        "official checkpoint; unified zero-shot CTA-500 evaluation"
                        if external
                        else "corrected Table 2/Table 5; unified CTA-500 evaluation"
                    ),
                }
            )


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.4, 6.25), facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.14, top=0.965)
    ax.set_facecolor("#FBFDFE")

    for row in sorted(MEASURED, key=lambda item: item["acc"], reverse=True):
        proposed = row["family"] == "Proposed"
        external = row["family"] == "External"
        ax.scatter(
            row["teds"],
            row["fps"],
            s=bubble_area(row["acc"]),
            color=COLORS[row["family"]],
            alpha=0.96 if proposed else 0.88,
            edgecolor="#A92F2B" if proposed else ("#164D52" if external else "white"),
            linewidth=2.0 if proposed else (1.35 if external else 1.0),
            zorder=5 if proposed or external else 3,
        )

    for row in MEASURED:
        dx, dy = OFFSETS[row["model"]]
        proposed = row["family"] == "Proposed"
        external = row["family"] == "External"
        ax.annotate(
            row["label"],
            xy=(row["teds"], row["fps"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=7.45 if not proposed else 8.1,
            fontweight="bold" if proposed or external else "normal",
            color="#B23834" if proposed else ("#18575D" if external else "#26343C"),
            linespacing=1.15,
            bbox={"facecolor": "#FBFDFE", "edgecolor": "none", "pad": 0.7, "alpha": 0.90},
            arrowprops={
                "arrowstyle": "-",
                "color": "#6F8791",
                "lw": 0.55,
                "shrinkA": 2,
                "shrinkB": 8,
                "connectionstyle": "arc3,rad=0",
            },
            zorder=7,
        )

    ax.set_xlim(0.22, 0.745)
    ax.set_ylim(0.0, 26.0)
    ax.set_xticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("TEDS score on CTA-500 (n = 500)", labelpad=9, color="#24323A")
    ax.set_ylabel("Inference speed on CTA-500 (FPS; higher = faster)", labelpad=9, color="#24323A")
    ax.grid(True, color="#D8E7EE", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3D4A51")
    ax.spines["bottom"].set_color("#3D4A51")
    ax.tick_params(colors="#3D4A51", length=3)
    return fig


def build_memory_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.4, 6.25), facecolor="white")
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.14, top=0.965)
    ax.set_facecolor("#FBFDFE")

    for row in sorted(MEASURED, key=lambda item: item["acc"], reverse=True):
        proposed = row["family"] == "Proposed"
        external = row["family"] == "External"
        ax.scatter(
            row["teds"],
            row["memory_mib"],
            s=parameter_bubble_area(row["params"]),
            color=COLORS[row["family"]],
            alpha=0.96 if proposed else 0.88,
            edgecolor="#A92F2B" if proposed else ("#164D52" if external else "white"),
            linewidth=2.0 if proposed else (1.35 if external else 1.0),
            zorder=5 if proposed or external else 3,
        )

    for row in MEASURED:
        dx, dy = MEMORY_OFFSETS[row["model"]]
        proposed = row["family"] == "Proposed"
        external = row["family"] == "External"
        ax.annotate(
            row["label"],
            xy=(row["teds"], row["memory_mib"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=7.35 if not proposed else 8.05,
            fontweight="bold" if proposed or external else "normal",
            color="#B23834" if proposed else ("#18575D" if external else "#26343C"),
            linespacing=1.15,
            bbox={"facecolor": "#FBFDFE", "edgecolor": "none", "pad": 0.7, "alpha": 0.90},
            arrowprops={
                "arrowstyle": "-",
                "color": "#6F8791",
                "lw": 0.55,
                "shrinkA": 2,
                "shrinkB": 8,
                "connectionstyle": "arc3,rad=0",
            },
            zorder=7,
        )

    ax.set_xlim(0.22, 0.745)
    ax.set_ylim(1060, 680)
    ax.set_xticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    ax.set_yticks([700, 750, 800, 850, 900, 950, 1000, 1050])
    ax.set_xlabel("TEDS score on CTA-500 (n = 500)", labelpad=9, color="#24323A")
    ax.set_ylabel("Peak GPU memory (MiB; lower = better)", labelpad=9, color="#24323A")
    ax.grid(True, color="#D8E7EE", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3D4A51")
    ax.spines["bottom"].set_color("#3D4A51")
    ax.tick_params(colors="#3D4A51", length=3)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_source_csv()
    for stem_name in ("fig1_updated_bubble", "fig1_updated_bubble_clean"):
        fig = build_figure()
        stem = OUT / stem_name
        fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.04)
        fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.04)
        plt.close(fig)

    memory_fig = build_memory_figure()
    memory_stem = OUT / "fig1_updated_bubble_memory"
    memory_fig.savefig(f"{memory_stem}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    memory_fig.savefig(f"{memory_stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    memory_fig.savefig(f"{memory_stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(memory_fig)

    notes = (
        "# Fig. 1 measured update\n\n"
        "- All plotted coordinates are measured on the same 500-image CTA-500 set.\n"
        "- UniTable Large and Table Transformer use official public checkpoints and are marked as zero-shot external baselines.\n"
        "- UniTabNet and VAST are excluded because no reproducible official checkpoint was available.\n"
        "- x-axis: TEDS; y-axis: end-to-end FPS; bubble area: exact structure accuracy.\n"
        "- The rendered figure keeps only axis explanations and direct model labels; it has no legend or reference-only markers.\n"
    )
    (OUT / "FIG1_UPDATE_NOTES.md").write_text(notes, encoding="utf-8")
    (OUT / "FIG1_CLEAN_NOTES.md").write_text(notes, encoding="utf-8")
    (OUT / "FIG1_MEMORY_NOTES.md").write_text(
        "# Fig. 1 peak-memory version\n\n"
        "- x-axis: unified CTA-500 TEDS.\n"
        "- y-axis: process-level peak GPU memory in MiB; lower is better.\n"
        "- The y-axis is inverted so lower memory appears higher in the figure.\n"
        "- Every model was run in a separate process on one RTX 4090 D.\n"
        "- Batch size 1, the same first 10 CTA-500 images, and official model-specific preprocessing.\n"
        "- Memory was sampled from nvidia-smi every 0.02 seconds and includes the framework CUDA context.\n"
        "- Bubble area represents parameter count using log10 scaling; larger means more parameters.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
