#!/usr/bin/env python3
"""Generate a compact half-column efficiency table for the manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文材料" / "lightweight_efficiency_half_table_20260801"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


ROWS = [
    ("SLANet", 0.833090033952271, 68.98, 0.6737967914438503, 23.50),
    ("Cell Token only", 0.832059540507130, 69.62, 0.6792141020932795, 24.25),
    ("Mamba lambda=0", 0.830760878290048, 69.38, 0.6758406998997540, 12.06),
    ("Mamba lambda=0.2", 0.834450897294001, 70.16, 0.6867079826103043, 12.16),
    ("Lite Mamba", 0.836650785177051, 70.84, 0.6915871410352259, 15.99),
    ("Mamba stage6", 0.843538834821512, 71.78, 0.7035918698454189, 12.13),
    ("Span-aware", 0.844973530001792, 71.92, 0.7027492967970238, 12.32),
    ("Horizontal Mamba", 0.845012611516246, 71.94, 0.7042598557365151, 11.68),
    ("Vertical Mamba", 0.844861222048308, 71.88, 0.7027762656505171, 12.00),
    ("DBM-SLANet", 0.843985613353128, 71.82, 0.7045002711006686, 9.65),
]


def write_source_csv() -> None:
    with (OUT / "lightweight_efficiency_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "teds", "structure_accuracy_percent", "span_f1", "fps"])
        writer.writerows(ROWS)


def write_latex() -> None:
    lines = [
        "% Half-column table; requires booktabs and xcolor.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy, merged-cell recovery, and inference-efficiency comparison on PubTabNet-5000.}",
        r"\label{tab:lightweight_efficiency}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Model & TEDS & Acc. (\\%) & Span-F1 & FPS \\\\",
        r"\midrule",
    ]
    for model, teds, accuracy, span_f1, fps in ROWS:
        if model == "DBM-SLANet":
            lines.append(r"\rowcolor{green!10}")
            lines.append(
                f"\\textbf{{{model}}} & \\textbf{{{teds:.5f}}} & \\textbf{{{accuracy:.2f}}} & \\textbf{{{span_f1:.4f}}} & \\textbf{{{fps:.2f}}} \\\\"
            )
        else:
            lines.append(f"{model} & {teds:.5f} & {accuracy:.2f} & {span_f1:.4f} & {fps:.2f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\\[-1mm]",
            r"\parbox{\columnwidth}{\footnotesize\textit{Note:} Span-F1 is the micro F1 for cells with rowspan or colspan greater than one. All models use the same PubTabNet-5000 validation protocol.}",
            r"\end{table}",
            "",
        ]
    )
    (OUT / "lightweight_efficiency_half_table.tex").write_text("\n".join(lines), encoding="utf-8")


def build_table() -> plt.Figure:
    fig = plt.figure(figsize=(3.50, 3.25), facecolor="white")
    ax = fig.add_axes([0.02, 0.25, 0.96, 0.70])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column widths are tuned for a single-column manuscript placement.
    widths = [0.30, 0.18, 0.19, 0.16, 0.17]
    x_edges = [0.0]
    for width in widths:
        x_edges.append(x_edges[-1] + width)
    header_h = 0.11
    row_h = 0.078
    table_top = 0.96
    table_bottom = table_top - header_h - row_h * len(ROWS)

    # Light row bands keep the table scannable after reduction to half-column size.
    row_colors = ["#F4F5F6", "#EEF3F8", "#F4F5F6", "#EEF3F8", "#F4F5F6",
                  "#EEF3F8", "#F4F5F6", "#EEF3F8", "#F4F5F6", "#E6F2EA"]
    for row_idx, color in enumerate(row_colors):
        y = table_top - header_h - row_h * (row_idx + 1)
        ax.add_patch(Rectangle((0, y), 1, row_h, facecolor=color, edgecolor="none", zorder=0))

    ax.add_patch(Rectangle((0, table_top - header_h), 1, header_h, facecolor="#34495E", edgecolor="none", zorder=0))
    ax.plot([0, 1], [table_top, table_top], color="#263442", linewidth=1.0, solid_capstyle="butt")
    ax.plot([0, 1], [table_bottom, table_bottom], color="#263442", linewidth=1.0, solid_capstyle="butt")
    ax.plot([0, 1], [table_top - header_h, table_top - header_h], color="#8A959F", linewidth=0.55, solid_capstyle="butt")

    headers = ["Model", "TEDS", "Acc. (%)", "Span-F1", "FPS"]
    for col_idx, header in enumerate(headers):
        x0, x1 = x_edges[col_idx], x_edges[col_idx + 1]
        ax.text(
            (x0 + x1) / 2,
            table_top - header_h / 2,
            header,
            ha="center",
            va="center",
            color="white",
            fontsize=5.7,
            fontweight="bold",
            linespacing=0.9,
        )

    for row_idx, (model, teds, accuracy, span_f1, fps) in enumerate(ROWS):
        y = table_top - header_h - row_h * (row_idx + 0.5)
        values = [model, f"{teds:.5f}", f"{accuracy:.2f}", f"{span_f1:.4f}", f"{fps:.2f}"]
        for col_idx, value in enumerate(values):
            x0, x1 = x_edges[col_idx], x_edges[col_idx + 1]
            is_ours = model == "DBM-SLANet"
            ax.text(
                x0 + 0.012 if col_idx == 0 else (x0 + x1) / 2,
                y,
                value,
                ha="left" if col_idx == 0 else "center",
                va="center",
                color="#1F2D3A",
                fontsize=5.9,
                fontweight="bold" if is_ours else "normal",
            )

    delta_acc = ROWS[-1][2] - ROWS[0][2]
    delta_teds = ROWS[-1][1] - ROWS[0][1]
    delta_span = ROWS[-1][3] - ROWS[0][3]
    delta_fps = ROWS[-1][4] - ROWS[0][4]
    fig.text(
        0.03,
        0.145,
        f"DBM-SLANet vs SLANet: +{delta_acc:.2f} pp Acc. | +{delta_span:.4f} Span-F1 | {delta_fps:.2f} FPS",
        ha="left",
        va="center",
        fontsize=5.1,
        color="#34495E",
    )
    fig.text(
        0.03,
        0.065,
        "PubTabNet-5000; Span-F1 evaluates merged-cell recovery; all rows use the same validation protocol.",
        ha="left",
        va="center",
        fontsize=4.6,
        color="#66717E",
    )
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_source_csv()
    write_latex()
    fig = build_table()
    stem = OUT / "lightweight_efficiency_half_table"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(f"{stem}.png", dpi=800, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
