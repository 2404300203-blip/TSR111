# Expanded Tables 2 and 3

Date: 2026-07-10  

This draft expands both tables shown in the manuscript page:

- Table 2: overall comparison on the fixed 5,000-sample validation split.
- Table 3: complex-structure diagnostic comparison.

For Table 2, all listed models were evaluated on the same PubTabNet-5000 split. Additional 5,000-sample diagnostics were generated on the remote server under `output/phase10_more_models_val5000_20260710/`.

---

## Table 2: Overall Comparison on the Fixed 5,000-sample Validation Split

Split: `train_data/table/pubtabnet/phase10_val_5000.jsonl`

| Method | Role | TEDS | Structure Acc. | FPS |
|---|---|---:|---:|---:|
| SLANet | original baseline | 0.8331 | 0.6898 | 23.50 |
| Cell Token only | geometry-token baseline | 0.8321 | 0.6962 | **24.25** |
| Full Mamba, $\lambda_{pres}=0$ | no presence loss | 0.8308 | 0.6938 | 12.06 |
| Full Mamba, bidirectional=false | single-direction scan | 0.8309 | 0.6944 | 16.19 |
| Full Mamba, geometry zero | no geometry input | 0.8308 | 0.6936 | 12.32 |
| Full Mamba, $\lambda_{pres}=0.2$ | presence-loss variant | 0.8345 | 0.7016 | 12.16 |
| Lite Mamba | lightweight Mamba baseline | 0.8367 | 0.7084 | 15.99 |
| Full Mamba stage4 | full Mamba intermediate | 0.8432 | 0.7178 | 12.20 |
| Full Mamba stage6 | lower-LR continuation | 0.8435 | 0.7178 | 12.13 |
| TEDS-select continuation | checkpoint selection variant | 0.8426 | 0.7144 | 12.22 |
| Span-aware continuation | span-aware loss variant | **0.8450** | 0.7192 | 12.32 |
| SLANet + Horizontal Mamba | horizontal single-axis | **0.8450** | **0.7194** | 11.68 |
| SLANet + Vertical Mamba | vertical single-axis | 0.8449 | 0.7188 | 12.00 |
| DBM-SLANet | dual branch | 0.8440 | 0.7182 | 9.65 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Overall comparison on the fixed 5,000-sample validation split.}
\label{tab:overall_val5000}
\begin{tabular}{llccc}
\toprule
Method & Role & TEDS & Structure Acc. & FPS \\
\midrule
SLANet & original & 0.8331 & 0.6898 & 23.50 \\
Cell Token only & geometry token & 0.8321 & 0.6962 & \textbf{24.25} \\
Full Mamba, $\lambda_{pres}=0$ & no pres. loss & 0.8308 & 0.6938 & 12.06 \\
Full Mamba, bidirectional=false & single-dir. & 0.8309 & 0.6944 & 16.19 \\
Full Mamba, geometry zero & no geometry & 0.8308 & 0.6936 & 12.32 \\
Full Mamba, $\lambda_{pres}=0.2$ & pres. loss & 0.8345 & 0.7016 & 12.16 \\
Lite Mamba & lightweight & 0.8367 & 0.7084 & 15.99 \\
Full Mamba stage4 & intermediate & 0.8432 & 0.7178 & 12.20 \\
Full Mamba stage6 & lower-LR cont. & 0.8435 & 0.7178 & 12.13 \\
TEDS-select continuation & selection & 0.8426 & 0.7144 & 12.22 \\
Span-aware continuation & span-aware & \textbf{0.8450} & 0.7192 & 12.32 \\
SLANet + Horizontal Mamba & horizontal & \textbf{0.8450} & \textbf{0.7194} & 11.68 \\
SLANet + Vertical Mamba & vertical & 0.8449 & 0.7188 & 12.00 \\
DBM-SLANet & dual branch & 0.8440 & 0.7182 & 9.65 \\
\bottomrule
\end{tabular}
\end{table}
```

### Safe Interpretation

Mamba-enhanced structural heads consistently outperform the original SLANet baseline on the fixed 5,000-sample validation split. The single-axis variants are slightly stronger than the current dual-branch DBM-SLANet on this generic validation split, so this table should be used to support the general value of Mamba-style structural modeling rather than claiming that dual-branch fusion is always best on PubTabNet-5000.

---

## Table 3a: Complex-4086 Diagnostic Subset

Subset: `complex_structure_special_full_4086_20260625`

| Model | N | TEDS | Structure Acc. |
|---|---:|---:|---:|
| SLANet | 4,086 | 0.8026 | 0.6356 |
| SLANet + Horizontal Mamba* | 826 | 0.7972 | 0.6453 |
| SLANet + Vertical Mamba* | 826 | 0.7971 | 0.6441 |
| DBM-SLANet | 4,086 | **0.8156** | **0.6684** |

`*` The horizontal/vertical rows are filtered subsets available in the existing complex-structure analysis, not full 4,086-sample reruns. For strict full-subset comparison, rerun these models on all 4,086 samples.

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Comparison on the Complex-4086 diagnostic subset.}
\label{tab:complex4086}
\begin{tabular}{lccc}
\toprule
Model & N & TEDS & Structure Acc. \\
\midrule
SLANet & 4,086 & 0.8026 & 0.6356 \\
SLANet + Horizontal Mamba$^{*}$ & 826 & 0.7972 & 0.6453 \\
SLANet + Vertical Mamba$^{*}$ & 826 & 0.7971 & 0.6441 \\
DBM-SLANet & 4,086 & \textbf{0.8156} & \textbf{0.6684} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3b: CTA-500 Complex-Structure Subset

Subset: `complex_structure_cta500_20260625.jsonl`

| Method | Role | TEDS | Structure Acc. | FPS |
|---|---|---:|---:|---:|
| SLANet | original baseline | 0.4843 | 0.208 | 16.30 |
| Cell Token only | geometry token baseline | 0.5436 | 0.284 | **20.33** |
| Full Mamba, $\lambda_{pres}=0$ | no presence loss | 0.5392 | 0.282 | 9.40 |
| Full Mamba, $\lambda_{pres}=0.2$ | controlled full Mamba | 0.5421 | 0.302 | 11.00 |
| Full Mamba, aux off | inference auxiliary disabled | 0.5421 | 0.302 | 18.68 |
| SLANet + Horizontal Mamba | horizontal single-axis | 0.6276 | 0.420 | 8.65 |
| SLANet + Vertical Mamba | vertical single-axis | 0.6258 | 0.416 | 9.76 |
| DBM-SLANet | dual branch | **0.6407** | **0.432** | 8.14 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Comparison on the CTA-500 complex-structure diagnostic subset.}
\label{tab:cta500_expanded}
\begin{tabular}{llccc}
\toprule
Method & Role & TEDS & Structure Acc. & FPS \\
\midrule
SLANet & original & 0.4843 & 0.208 & 16.30 \\
Cell Token only & geometry token & 0.5436 & 0.284 & \textbf{20.33} \\
Full Mamba, $\lambda_{pres}=0$ & no pres. loss & 0.5392 & 0.282 & 9.40 \\
Full Mamba, $\lambda_{pres}=0.2$ & controlled full & 0.5421 & 0.302 & 11.00 \\
Full Mamba, aux off & inference aux off & 0.5421 & 0.302 & 18.68 \\
SLANet + Horizontal Mamba & horizontal & 0.6276 & 0.420 & 8.65 \\
SLANet + Vertical Mamba & vertical & 0.6258 & 0.416 & 9.76 \\
DBM-SLANet & dual branch & \textbf{0.6407} & \textbf{0.432} & 8.14 \\
\bottomrule
\end{tabular}
\end{table}
```

### Safe Interpretation

On the generic PubTabNet-5000 split, single-axis Mamba variants are slightly stronger than the current dual-branch model. However, on the CTA-500 complex-structure subset, DBM-SLANet achieves the best TEDS and structure accuracy, indicating that dual-axis fusion is more useful for difficult tables involving multi-level headers, row/column spans, and long-range structural dependencies.
