# DBM-SLANet Tables Updated with New Metrics

Date: 2026-07-10

This version updates the manuscript tables with the newly added metrics:

- `Topology F1`: GriTS-Top-style topology proxy from reconstructed grid-cell tuples.
- `Span-F1`: micro F1 over cells with `rowspan > 1` or `colspan > 1`.
- `Params / FLOPs / Latency`: model complexity and efficiency indicators.

Note: `Topology F1` is an HTML-token-derived proxy, not the official GriTS evaluator. In the paper, use the wording **GriTS-Top-style Topology F1**.

---

## Updated Table 2: PubTabNet-5000 with Topology and Span Metrics

Split: `train_data/table/pubtabnet/phase10_val_5000.jsonl`

| Method | TEDS | Structure Acc. | Topology F1 | Span-F1 | FPS |
|---|---:|---:|---:|---:|---:|
| SLANet | 0.8331 | 0.6898 | 0.9601 | 0.6738 | 23.50 |
| Cell Token only | 0.8321 | 0.6962 | 0.9592 | 0.6792 | **24.25** |
| Full Mamba, $\lambda_{pres}=0$ | 0.8308 | 0.6938 | 0.9587 | 0.6758 | 12.06 |
| Full Mamba, $\lambda_{pres}=0.2$ | 0.8345 | 0.7016 | 0.9602 | 0.6867 | 12.16 |
| Lite Mamba | 0.8367 | 0.7084 | 0.9576 | 0.6916 | 15.99 |
| Full Mamba stage6 | 0.8435 | 0.7178 | 0.9603 | 0.7036 | 12.13 |
| Span-aware | 0.8450 | 0.7192 | **0.9618** | 0.7027 | 12.32 |
| SLANet + Horizontal Mamba | **0.8450** | **0.7194** | **0.9618** | 0.7043 | 11.68 |
| SLANet + Vertical Mamba | 0.8449 | 0.7188 | **0.9618** | 0.7028 | 12.00 |
| DBM-SLANet | 0.8440 | 0.7182 | 0.9610 | **0.7045** | 9.65 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Overall comparison on the fixed 5,000-sample validation split with additional topology and span metrics.}
\label{tab:overall_val5000_extra}
\begin{tabular}{lccccc}
\toprule
Method & TEDS & Struct. Acc. & Topology F1 & Span-F1 & FPS \\
\midrule
SLANet & 0.8331 & 0.6898 & 0.9601 & 0.6738 & 23.50 \\
Cell Token only & 0.8321 & 0.6962 & 0.9592 & 0.6792 & \textbf{24.25} \\
Full Mamba, $\lambda_{pres}=0$ & 0.8308 & 0.6938 & 0.9587 & 0.6758 & 12.06 \\
Full Mamba, $\lambda_{pres}=0.2$ & 0.8345 & 0.7016 & 0.9602 & 0.6867 & 12.16 \\
Lite Mamba & 0.8367 & 0.7084 & 0.9576 & 0.6916 & 15.99 \\
Full Mamba stage6 & 0.8435 & 0.7178 & 0.9603 & 0.7036 & 12.13 \\
Span-aware & 0.8450 & 0.7192 & \textbf{0.9618} & 0.7027 & 12.32 \\
SLANet + Horizontal Mamba & \textbf{0.8450} & \textbf{0.7194} & \textbf{0.9618} & 0.7043 & 11.68 \\
SLANet + Vertical Mamba & 0.8449 & 0.7188 & \textbf{0.9618} & 0.7028 & 12.00 \\
DBM-SLANet & 0.8440 & 0.7182 & 0.9610 & \textbf{0.7045} & 9.65 \\
\bottomrule
\end{tabular}
\end{table}
```

### Interpretation

On PubTabNet-5000, Horizontal Mamba and Span-aware variants obtain the highest TEDS and topology scores, while DBM-SLANet obtains the best Span-F1. This means DBM-SLANet is not the single best model on every generic validation metric, but it has the strongest span-structure recovery signal among the reported variants.

---

## Updated Table 3: CTA-500 Complex-Structure Subset

Subset: `complex_structure_cta500_20260625.jsonl`

| Method | TEDS | Structure Acc. | Topology F1 | Span-F1 | FPS |
|---|---:|---:|---:|---:|---:|
| SLANet | 0.4843 | 0.208 | 0.8916 | 0.5874 | 16.30 |
| Cell Token only | 0.5436 | 0.284 | 0.8990 | 0.6220 | **20.33** |
| Full Mamba, $\lambda_{pres}=0$ | 0.5392 | 0.282 | 0.8977 | 0.6171 | 9.40 |
| SLANet + Horizontal Mamba | 0.6276 | 0.420 | 0.9038 | 0.6563 | 8.65 |
| SLANet + Vertical Mamba | 0.6258 | 0.416 | 0.9038 | 0.6523 | 9.76 |
| DBM-SLANet | **0.6407** | **0.432** | **0.9044** | **0.6618** | 8.14 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Comparison on the CTA-500 complex-structure diagnostic subset with topology and span metrics.}
\label{tab:cta500_extra}
\begin{tabular}{lccccc}
\toprule
Method & TEDS & Struct. Acc. & Topology F1 & Span-F1 & FPS \\
\midrule
SLANet & 0.4843 & 0.208 & 0.8916 & 0.5874 & 16.30 \\
Cell Token only & 0.5436 & 0.284 & 0.8990 & 0.6220 & \textbf{20.33} \\
Full Mamba, $\lambda_{pres}=0$ & 0.5392 & 0.282 & 0.8977 & 0.6171 & 9.40 \\
SLANet + Horizontal Mamba & 0.6276 & 0.420 & 0.9038 & 0.6563 & 8.65 \\
SLANet + Vertical Mamba & 0.6258 & 0.416 & 0.9038 & 0.6523 & 9.76 \\
DBM-SLANet & \textbf{0.6407} & \textbf{0.432} & \textbf{0.9044} & \textbf{0.6618} & 8.14 \\
\bottomrule
\end{tabular}
\end{table}
```

### Interpretation

CTA-500 is the strongest table for supporting DBM-SLANet. DBM-SLANet is best on all structure-oriented metrics: TEDS, Structure Accuracy, Topology F1, and Span-F1. This directly supports the claim that dual-axis Mamba modeling is useful for difficult structures such as multi-level headers, row/column spans, and long-range table dependencies.

---

## Efficiency Table: Params / FLOPs / Latency

FLOPs are estimated by `paddle.flops` with input size `1 x 3 x 488 x 488`. Several unsupported operators are counted as zero by Paddle, so FLOPs should be described as a Paddle-estimated proxy. Params and latency are more reliable.

| Method | Params (M) | Paddle-est. FLOPs (G) | FPS | Latency (ms/img) |
|---|---:|---:|---:|---:|
| SLANet | 1.89 | 5.14 | 23.50 | 42.6 |
| Cell Token only | 1.93 | 5.15 | 24.25 | 41.2 |
| Lite Mamba | 2.09 | 5.24 | 15.99 | 62.5 |
| Span-aware | 2.45 | 5.41 | 12.32 | 81.2 |
| SLANet + Horizontal Mamba | 2.45 | 5.41 | 11.68 | 85.6 |
| SLANet + Vertical Mamba | 2.45 | 5.41 | 12.00 | 83.3 |
| DBM-SLANet | 2.45 | 5.41 | 9.65 | 103.6 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Model complexity and inference efficiency. FLOPs are Paddle-estimated with input size $1\times3\times488\times488$.}
\label{tab:efficiency}
\begin{tabular}{lcccc}
\toprule
Method & Params (M) & FLOPs (G) & FPS & Latency (ms/img) \\
\midrule
SLANet & 1.89 & 5.14 & 23.50 & 42.6 \\
Cell Token only & 1.93 & 5.15 & \textbf{24.25} & \textbf{41.2} \\
Lite Mamba & 2.09 & 5.24 & 15.99 & 62.5 \\
Span-aware & 2.45 & 5.41 & 12.32 & 81.2 \\
SLANet + Horizontal Mamba & 2.45 & 5.41 & 11.68 & 85.6 \\
SLANet + Vertical Mamba & 2.45 & 5.41 & 12.00 & 83.3 \\
DBM-SLANet & 2.45 & 5.41 & 9.65 & 103.6 \\
\bottomrule
\end{tabular}
\end{table}
```

### Recommended Paper Wording

> Beyond TEDS and exact structure accuracy, we report a GriTS-Top-style topology F1 and Span-F1 to evaluate row/column topology and rowspan/colspan recovery. On the CTA-500 complex-structure subset, DBM-SLANet achieves the best Topology F1 and Span-F1, confirming that dual-axis Mamba modeling improves complex table-structure recovery. The method increases parameters moderately from 1.89M to 2.45M, but introduces additional latency due to row/column structural modeling.

