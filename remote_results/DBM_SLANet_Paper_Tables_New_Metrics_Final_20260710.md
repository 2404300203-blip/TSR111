# DBM-SLANet 新指标版论文表格（更新版）

日期：2026-07-10  
用途：根据新增结构指标更新论文主表、消融表、附表和效率表。  

## 指标说明

| 指标 | 含义 | 推荐放置 |
|---|---|---|
| TEDS | HTML tree edit distance similarity，表格结构识别通用指标 | 主表 |
| Structure Acc. | 预测 HTML 与 GT 完全一致的比例 | 主表 |
| GriTS-Top-style Topology F1 | 从 HTML token 解析出的表格拓扑 F1，属于 GriTS-Top 风格 proxy，不等同于官方 GriTS | 主表或附表 |
| Span-F1 | 对 `rowspan > 1` 或 `colspan > 1` 单元格的 micro F1 | 主表 |
| FPS / Latency | 推理效率 | 效率表 |
| Params / Paddle-est. FLOPs | 模型规模与计算量。FLOPs 为 Paddle 估计值，部分不支持算子会按 0 计 | 效率表 |

论文中建议写作：`GriTS-Top-style Topology F1` 或 `HTML-token-derived topology proxy`，不要直接写成 official GriTS。

---

## 推荐正文主表：Baseline 与关键模型对比

这张表最适合放正文。它把原始 baseline、单轴 Mamba 和最终 DBM-SLANet 放在一起，能直接说明：通用集上 Mamba 系列有效，复杂结构集上 DBM-SLANet 最强。

| Dataset | Method | TEDS | Structure Acc. | Topology F1 | Span-F1 | FPS |
|---|---|---:|---:|---:|---:|---:|
| PubTabNet-5000 | SLANet | 0.8331 | 0.6898 | 0.9601 | 0.6738 | **23.50** |
| PubTabNet-5000 | SLANet + Horizontal Mamba | **0.8450** | **0.7194** | **0.9618** | 0.7043 | 11.68 |
| PubTabNet-5000 | SLANet + Vertical Mamba | 0.8449 | 0.7188 | **0.9618** | 0.7028 | 12.00 |
| PubTabNet-5000 | DBM-SLANet | 0.8440 | 0.7182 | 0.9610 | **0.7045** | 9.65 |
| CTA-500 | SLANet | 0.4843 | 0.208 | 0.8916 | 0.5874 | **16.30** |
| CTA-500 | SLANet + Horizontal Mamba | 0.6276 | 0.420 | 0.9038 | 0.6563 | 8.65 |
| CTA-500 | SLANet + Vertical Mamba | 0.6258 | 0.416 | 0.9038 | 0.6523 | 9.76 |
| CTA-500 | DBM-SLANet | **0.6407** | **0.432** | **0.9044** | **0.6618** | 8.14 |

### 主表可写结论

相对于原始 SLANet，Mamba 系列模型在 PubTabNet-5000 和 CTA-500 上均带来结构识别提升。PubTabNet-5000 上单轴 Mamba 的总体指标略高，而 DBM-SLANet 的 Span-F1 最高；CTA-500 上 DBM-SLANet 在 TEDS、Structure Acc.、Topology F1 和 Span-F1 上均为最优，说明双轴建模的优势主要体现在复杂表格结构恢复。

### 推荐正文主表 LaTeX

```latex
\begin{table}[t]
\centering
\caption{Comparison between the baseline and key Mamba-enhanced variants on generic and complex-structure subsets.}
\label{tab:key_model_comparison}
\begin{tabular}{llccccc}
\toprule
Dataset & Method & TEDS & Struct. Acc. & Topology F1 & Span-F1 & FPS \\
\midrule
PubTabNet-5000 & SLANet & 0.8331 & 0.6898 & 0.9601 & 0.6738 & \textbf{23.50} \\
PubTabNet-5000 & Horizontal Mamba & \textbf{0.8450} & \textbf{0.7194} & \textbf{0.9618} & 0.7043 & 11.68 \\
PubTabNet-5000 & Vertical Mamba & 0.8449 & 0.7188 & \textbf{0.9618} & 0.7028 & 12.00 \\
PubTabNet-5000 & DBM-SLANet & 0.8440 & 0.7182 & 0.9610 & \textbf{0.7045} & 9.65 \\
\midrule
CTA-500 & SLANet & 0.4843 & 0.208 & 0.8916 & 0.5874 & \textbf{16.30} \\
CTA-500 & Horizontal Mamba & 0.6276 & 0.420 & 0.9038 & 0.6563 & 8.65 \\
CTA-500 & Vertical Mamba & 0.6258 & 0.416 & 0.9038 & 0.6523 & 9.76 \\
CTA-500 & DBM-SLANet & \textbf{0.6407} & \textbf{0.432} & \textbf{0.9044} & \textbf{0.6618} & 8.14 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2：PubTabNet-5000 总体对比

数据集：`train_data/table/pubtabnet/phase10_val_5000.jsonl`

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

### 表 2 可写结论

在 PubTabNet-5000 通用验证集上，Mamba 系列结构头整体优于原始 SLANet。Horizontal Mamba 和 Span-aware 变体取得最高 TEDS/Topology F1，DBM-SLANet 取得最高 Span-F1。该表更适合证明“结构建模和 span 建模有效”，不建议用来声称 DBM-SLANet 在通用集上所有指标第一。

### Table 2 LaTeX

```latex
\begin{table}[t]
\centering
\caption{Overall comparison on the fixed 5,000-sample PubTabNet validation split with additional topology and span metrics.}
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

---

## Table 3：CTA-500 复杂结构子集对比

数据集：`complex_structure_cta500_20260625.jsonl`

| Method | TEDS | Structure Acc. | Topology F1 | Span-F1 | FPS |
|---|---:|---:|---:|---:|---:|
| SLANet | 0.4843 | 0.208 | 0.8916 | 0.5874 | 16.30 |
| Cell Token only | 0.5436 | 0.284 | 0.8990 | 0.6220 | **20.33** |
| Full Mamba, $\lambda_{pres}=0$ | 0.5392 | 0.282 | 0.8977 | 0.6171 | 9.40 |
| SLANet + Horizontal Mamba | 0.6276 | 0.420 | 0.9038 | 0.6563 | 8.65 |
| SLANet + Vertical Mamba | 0.6258 | 0.416 | 0.9038 | 0.6523 | 9.76 |
| DBM-SLANet | **0.6407** | **0.432** | **0.9044** | **0.6618** | 8.14 |

### 表 3 可写结论

CTA-500 是目前最适合作为正文核心证据的表。DBM-SLANet 在 TEDS、Structure Acc.、Topology F1 和 Span-F1 上均为最高，说明双分支行列建模更适合复杂表格，尤其是多级表头、跨行跨列单元格和长距离结构依赖。

### Table 3 LaTeX

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

---

## Table 4：效率与复杂度

输入尺寸：`1 x 3 x 488 x 488`。FLOPs 由 `paddle.flops` 估计。

| Method | Params (M) | Paddle-est. FLOPs (G) | FPS | Latency (ms/img) |
|---|---:|---:|---:|---:|
| SLANet | 1.89 | 5.14 | 23.50 | 42.6 |
| Cell Token only | 1.93 | 5.15 | **24.25** | **41.2** |
| Lite Mamba | 2.09 | 5.24 | 15.99 | 62.5 |
| Span-aware | 2.45 | 5.41 | 12.32 | 81.2 |
| SLANet + Horizontal Mamba | 2.45 | 5.41 | 11.68 | 85.6 |
| SLANet + Vertical Mamba | 2.45 | 5.41 | 12.00 | 83.3 |
| DBM-SLANet | 2.45 | 5.41 | 9.65 | 103.6 |

### 效率表可写结论

DBM-SLANet 的参数量从 SLANet 的 1.89M 增加到 2.45M，FLOPs proxy 从 5.14G 增加到 5.41G，模型规模增加不大；主要代价体现在推理延迟，FPS 从 23.50 下降到 9.65。这说明双轴结构建模能提升复杂结构恢复能力，但需要用效率表主动说明速度折中。

---

## Supplementary Table S1：PubTabNet-5000 完整消融结果

这张表建议放附录。部分控制实验目前只有 TEDS、Structure Acc. 和 FPS，尚未同步计算 Topology F1 与 Span-F1，因此不建议混入正文主表。

| Method | Role | TEDS | Structure Acc. | FPS | New structure metrics |
|---|---|---:|---:|---:|---|
| SLANet | original baseline | 0.8331 | 0.6898 | 23.50 | available |
| Cell Token only | geometry-token baseline | 0.8321 | 0.6962 | **24.25** | available |
| Full Mamba, $\lambda_{pres}=0$ | no presence loss | 0.8308 | 0.6938 | 12.06 | available |
| Full Mamba, bidirectional=false | single-direction scan | 0.8309 | 0.6944 | 16.19 | not yet computed |
| Full Mamba, geometry zero | no geometry input | 0.8308 | 0.6936 | 12.32 | not yet computed |
| Full Mamba, $\lambda_{pres}=0.2$ | presence-loss variant | 0.8345 | 0.7016 | 12.16 | available |
| Lite Mamba | lightweight Mamba baseline | 0.8367 | 0.7084 | 15.99 | available |
| Full Mamba stage4 | full Mamba intermediate | 0.8432 | 0.7178 | 12.20 | not yet computed |
| Full Mamba stage6 | lower-LR continuation | 0.8435 | 0.7178 | 12.13 | available |
| TEDS-select continuation | checkpoint selection variant | 0.8426 | 0.7144 | 12.22 | not yet computed |
| Span-aware continuation | span-aware loss variant | **0.8450** | 0.7192 | 12.32 | available |
| SLANet + Horizontal Mamba | horizontal single-axis | **0.8450** | **0.7194** | 11.68 | available |
| SLANet + Vertical Mamba | vertical single-axis | 0.8449 | 0.7188 | 12.00 | available |
| DBM-SLANet | dual branch | 0.8440 | 0.7182 | 9.65 | available |

### S1 可写结论

完整消融显示，Mamba 系列结构头整体优于原始 SLANet，但不同设计的收益并不完全一致。`Horizontal Mamba` 和 `Span-aware continuation` 在 PubTabNet-5000 上达到最高 TEDS，`DBM-SLANet` 在 Span-F1 上表现最好。由于部分控制实验尚未同步新结构指标，正文建议只放指标完整的关键模型。

---

## Supplementary Table S2：CTA-500 完整消融结果

这张表用于补充说明 Phase 11 控制实验。正文可使用 Table 3，附录保留这张完整消融表。

| Method | Role | TEDS | Structure Acc. | FPS | New structure metrics |
|---|---|---:|---:|---:|---|
| SLANet | original baseline | 0.4843 | 0.208 | 16.30 | available |
| Cell Token only | geometry-token baseline | 0.5436 | 0.284 | **20.33** | available |
| Full Mamba, $\lambda_{pres}=0$ | no presence loss | 0.5392 | 0.282 | 9.40 | available |
| Full Mamba, $\lambda_{pres}=0.2$ | controlled full Mamba | 0.5421 | 0.302 | 11.00 | not yet computed |
| Full Mamba, aux off | inference auxiliary disabled | 0.5421 | 0.302 | 18.68 | not yet computed |
| SLANet + Horizontal Mamba | horizontal single-axis | 0.6276 | 0.420 | 8.65 | available |
| SLANet + Vertical Mamba | vertical single-axis | 0.6258 | 0.416 | 9.76 | available |
| DBM-SLANet | dual branch | **0.6407** | **0.432** | 8.14 | available |

### S2 可写结论

CTA-500 完整消融更清楚地体现了复杂结构任务上的收益：单独加入 cell token 已明显优于原始 SLANet，单轴 Mamba 进一步提升，而 DBM-SLANet 在 TEDS 和 Structure Acc. 上达到最高。`Full Mamba, aux off` 的精度与 aux on 保持一致但 FPS 更高，可作为推理效率优化的附加观察，不宜替代最终模型主结论。

---

## 推荐正文表述

英文：

> Besides TEDS and exact structure accuracy, we report a GriTS-Top-style topology F1 and Span-F1 to evaluate row/column topology and rowspan/colspan recovery. On the generic PubTabNet-5000 split, Mamba-enhanced variants consistently improve structural recognition over SLANet, while DBM-SLANet obtains the best Span-F1. On the CTA-500 complex-structure subset, DBM-SLANet achieves the best TEDS, structure accuracy, topology F1, and Span-F1, indicating that dual-axis Mamba modeling is particularly effective for complex table structures.

中文：

> 除 TEDS 和结构完全匹配率外，本文进一步报告 GriTS-Top 风格的拓扑 F1 和 Span-F1，用于衡量行列拓扑关系及跨行跨列单元格恢复能力。在 PubTabNet-5000 通用验证集上，Mamba 系列结构头整体优于原始 SLANet，其中 DBM-SLANet 取得最高 Span-F1。在 CTA-500 复杂结构子集上，DBM-SLANet 在 TEDS、结构准确率、Topology F1 和 Span-F1 上均取得最优结果，说明双轴 Mamba 建模对复杂表格结构恢复更有效。

## 论文使用建议

1. 正文主表优先放 Table 3，因为它最能体现 DBM-SLANet 的创新点。
2. PubTabNet-5000 表用于说明整体性能和 ablation，不要把它包装成 DBM 全面第一。
3. Topology F1 必须注明是 proxy 或 GriTS-Top-style 指标。
4. Cell Presence F1 不建议作为核心证据，可放审计说明或附录。
5. 效率表建议保留，主动说明速度代价，比被审稿人追问更稳。
