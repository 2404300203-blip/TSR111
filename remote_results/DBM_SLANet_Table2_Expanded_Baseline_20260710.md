# Table 2 Expanded: Overall Comparison on the Fixed 5,000-sample Validation Split

Date: 2026-07-10  
Split: `train_data/table/pubtabnet/phase10_val_5000.jsonl`  

## Recommended Table

| Method | Role | TEDS | Structure Acc. | FPS |
|---|---|---:|---:|---:|
| SLANet | original baseline | 0.8331 | 0.6898 | **23.50** |
| SLANet + Horizontal Mamba | single-axis baseline / ablation | **0.8450** | **0.7194** | 11.68 |
| SLANet + Vertical Mamba | single-axis baseline / ablation | 0.8449 | 0.7188 | 12.00 |
| DBM-SLANet | dual-branch main model | 0.8440 | 0.7182 | 9.65 |

## LaTeX Version

```latex
\begin{table}[t]
\centering
\caption{Overall comparison on the fixed 5,000-sample validation split.}
\label{tab:overall_val5000}
\begin{tabular}{lccc}
\toprule
Method & TEDS & Structure Acc. & FPS \\
\midrule
SLANet & 0.8331 & 0.6898 & \textbf{23.50} \\
SLANet + Horizontal Mamba & \textbf{0.8450} & \textbf{0.7194} & 11.68 \\
SLANet + Vertical Mamba & 0.8449 & 0.7188 & 12.00 \\
DBM-SLANet & 0.8440 & 0.7182 & 9.65 \\
\bottomrule
\end{tabular}
\end{table}
```

## Interpretation

This expanded comparison includes the original SLANet baseline and two single-axis Mamba variants. Compared with SLANet, all Mamba-enhanced variants improve both TEDS and structure accuracy on the fixed 5,000-sample validation split, confirming that sequence-style structural modeling is beneficial for table structure recognition.

The newly evaluated horizontal-only and vertical-only variants slightly outperform the previous dual-branch DBM-SLANet result on this generic PubTabNet validation split. Therefore, this table should be interpreted carefully: it supports the usefulness of Mamba-based row/column dependency modeling, but it does not by itself prove that the dual-branch fusion is always superior on the generic validation set.

For the paper, a safer statement is:

> Mamba-enhanced structural heads consistently outperform the original SLANet baseline on the fixed 5,000-sample validation split. Among the tested variants, single-axis Mamba achieves the highest PubTabNet-5000 score, while DBM-SLANet remains the main model for studying dual-axis row-column dependency fusion and should be further justified using complex-structure and financial-domain results.

## Data Sources

Original existing results:

- `output/phase10_large_val_20260624/baseline_val5000_eval.log`
- `output/phase10_large_val_20260624/phase5e_val5000_eval.log`

Newly evaluated on 2026-07-10:

- `output/phase10_axis_ablation_val5000_20260710/expC_horizontal_val5000_teds.log`
- `output/phase10_axis_ablation_val5000_20260710/expD_vertical_val5000_teds.log`
- `output/phase10_axis_ablation_val5000_20260710/expC_horizontal_val5000_sample_teds.jsonl`
- `output/phase10_axis_ablation_val5000_20260710/expD_vertical_val5000_sample_teds.jsonl`

