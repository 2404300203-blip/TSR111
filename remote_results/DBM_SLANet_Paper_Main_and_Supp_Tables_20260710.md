# DBM-SLANet 论文主表与附表整理

日期：2026-07-10  
数据集：CTA500  
评估指标：TEDS、Structure Acc、FPS  
实验目的：整理本轮 DBM-SLANet 补充对比实验与消融实验，形成可直接写入论文的主表和附表。

## 表 1 主要消融实验结果

该表建议放在论文正文实验部分，用于回答 DBM-SLANet 的核心模块是否有效。

| Method | Cell Token | Full Mamba | Geometry | Bidirectional | Cell Presence Loss | Inference Aux | TEDS | Structure Acc | FPS |
|---|---|---|---|---|---|---|---:|---:|---:|
| Cell Token only | yes | no | full | no | 0.2 | on | 0.54364 | 0.284 | 20.33 |
| Full Mamba w/o presence loss | yes | yes | full | yes | 0.0 | on | 0.53921 | 0.282 | 9.40 |
| Full Mamba | yes | yes | full | yes | 0.2 | on | 0.54215 | 0.302 | 9.54 |
| Full Mamba w/o bidirectional scan | yes | yes | full | no | 0.2 | on | 0.53040 | 0.272 | 14.14 |
| Full Mamba w/o geometry input | yes | yes | zero | yes | 0.2 | on | 0.53921 | 0.282 | 11.12 |
| Full Mamba w/ aux off at inference | yes | yes | full | yes | 0.2 | off | 0.54215 | 0.302 | 18.68 |

### 正文表格解读

从表 1 可以得到四个核心结论：

1. 单独加入 Full Mamba 并不一定提升 TEDS，说明序列建模模块需要与几何 token 和辅助监督共同工作。
2. 加入 `cell presence` 辅助损失后，Structure Acc 从 `0.282` 提升到 `0.302`，说明该监督有助于结构完整性。
3. 关闭 bidirectional 后，TEDS 和 Structure Acc 均明显下降，说明双向行列依赖建模对结构恢复有效。
4. 推理阶段关闭辅助分支后，TEDS 和 Structure Acc 不变，FPS 从 `9.54` 提升到 `18.68`，说明辅助分支主要发挥训练期正则化作用。

## 表 2 模块贡献汇总

该表可放在正文，也可作为主表后的解释表，用于突出每个设计对应的实验依据。

| 模块 | 对比设置 | 主要变化 | 结论 |
|---|---|---|---|
| Cell presence auxiliary loss | `lambda_pres=0` vs `lambda_pres=0.2` | TEDS `0.53921 -> 0.54215`；Structure Acc `0.282 -> 0.302` | 辅助监督提升结构完整性 |
| Bidirectional Mamba | `bidirectional=true` vs `false` | TEDS `0.54215 -> 0.53040`；Structure Acc `0.302 -> 0.272` | 双向行列依赖建模有效 |
| Geometry token | full geometry vs zero geometry | TEDS `0.54215 -> 0.53921`；Structure Acc `0.302 -> 0.282` | 几何信息有助于结构恢复 |
| Inference auxiliary branch | aux on vs aux off | 精度不变；FPS `9.54 -> 18.68` | 推理阶段可关闭辅助分支提速 |

## 表 3 论文附表：完整 Phase 11 消融结果

该表建议放在附录或实验补充材料，展示本轮所有 Phase 11 控制实验。

| Experiment ID | Experiment | Training Setting | Evaluation Setting | TEDS | Structure Acc | FPS | Status |
|---|---|---|---|---:|---:|---:|---|
| E11A | Cell Token only | baseline init, 30k samples | CTA500 | 0.54364 | 0.284 | 20.33 | done |
| E11B | Full Mamba, `lambda_pres=0` | baseline init, 30k samples | CTA500 | 0.53921 | 0.282 | 9.40 | done |
| E11C | Full Mamba, `lambda_pres=0.2` | baseline init, 30k samples | CTA500 | 0.54215 | 0.302 | 9.54 | done |
| E11D | Inference auxiliary off | E11C checkpoint | CTA500, aux off | 0.54215 | 0.302 | 18.68 | done |
| E11E | Full Mamba, `bidirectional=false` | baseline init, 30k samples | CTA500 | 0.53040 | 0.272 | 14.14 | done |
| E11F | Full Mamba, geometry zero | baseline init, 30k samples | CTA500 | 0.53921 | 0.282 | 11.12 | done |

## 表 4 论文附表：Empty-td 辅助指标

该表建议作为附表使用，不建议作为主要性能表。它适合辅助说明不同模型在空单元格预测数量上的偏差。

| Model | Samples | TEDS | Exact | Pred Empty TD Total | GT Empty TD Total | Pred Empty TD / Image | GT Empty TD / Image | Empty TD / GT Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 500 | 0.48434 | 0.208 | 57685 | 55966 | 115.37 | 111.93 | 1.0307 |
| Horizontal only | 500 | 0.62764 | 0.420 | 58330 | 55966 | 116.66 | 111.93 | 1.0422 |
| Vertical only | 500 | 0.62576 | 0.416 | 58315 | 55966 | 116.63 | 111.93 | 1.0420 |
| Dual branch | 500 | 0.64067 | 0.432 | 58148 | 55966 | 116.30 | 111.93 | 1.0390 |

## LaTeX 主表版本

```latex
\begin{table}[t]
\centering
\caption{Main ablation results of DBM-SLANet on CTA500.}
\label{tab:main_ablation}
\begin{tabular}{lccccccccc}
\hline
Method & Cell & Mamba & Geo. & Bi-dir. & $\lambda_{pres}$ & Aux. & TEDS & Struct. Acc & FPS \\
\hline
Cell Token only & \checkmark & -- & full & -- & 0.2 & on & 0.54364 & 0.284 & 20.33 \\
Full Mamba w/o pres. & \checkmark & \checkmark & full & \checkmark & 0.0 & on & 0.53921 & 0.282 & 9.40 \\
Full Mamba & \checkmark & \checkmark & full & \checkmark & 0.2 & on & 0.54215 & 0.302 & 9.54 \\
w/o bidirectional & \checkmark & \checkmark & full & -- & 0.2 & on & 0.53040 & 0.272 & 14.14 \\
w/o geometry input & \checkmark & \checkmark & zero & \checkmark & 0.2 & on & 0.53921 & 0.282 & 11.12 \\
Aux. off at inference & \checkmark & \checkmark & full & \checkmark & 0.2 & off & 0.54215 & 0.302 & 18.68 \\
\hline
\end{tabular}
\end{table}
```

## LaTeX 附表版本

```latex
\begin{table}[t]
\centering
\caption{Complete Phase 11 controlled ablation experiments on CTA500.}
\label{tab:phase11_complete_ablation}
\begin{tabular}{llllccc}
\hline
ID & Experiment & Training Setting & Evaluation Setting & TEDS & Struct. Acc & FPS \\
\hline
E11A & Cell Token only & baseline init, 30k & CTA500 & 0.54364 & 0.284 & 20.33 \\
E11B & Full Mamba, $\lambda_{pres}=0$ & baseline init, 30k & CTA500 & 0.53921 & 0.282 & 9.40 \\
E11C & Full Mamba, $\lambda_{pres}=0.2$ & baseline init, 30k & CTA500 & 0.54215 & 0.302 & 9.54 \\
E11D & Aux. off at inference & E11C checkpoint & CTA500, aux off & 0.54215 & 0.302 & 18.68 \\
E11E & Full Mamba, bidirectional=false & baseline init, 30k & CTA500 & 0.53040 & 0.272 & 14.14 \\
E11F & Full Mamba, geometry zero & baseline init, 30k & CTA500 & 0.53921 & 0.282 & 11.12 \\
\hline
\end{tabular}
\end{table}
```

## 推荐写法

正文中建议这样描述主表：

> As shown in Table~\ref{tab:main_ablation}, simply introducing the Full Mamba branch does not consistently improve TEDS over the lightweight cell-token baseline. However, when combined with geometry-aware cell tokens and the cell-presence auxiliary objective, the model achieves a higher structure accuracy. The bidirectional scan is critical for modeling row-wise and column-wise dependencies, while the geometry ablation confirms that explicit cell geometry provides useful structural cues. Moreover, disabling the auxiliary branch during inference preserves both TEDS and structure accuracy while substantially improving FPS, indicating that the auxiliary branch mainly acts as a training-time regularizer.

中文说明：

> 如表所示，单独引入 Full Mamba 并不能稳定超过轻量 Cell Token baseline；但当其与几何感知 cell token 和 cell presence 辅助目标结合时，结构完全匹配率得到提升。双向扫描对行列依赖建模至关重要，几何特征消融进一步证明显式单元格几何信息能够提供有效结构线索。此外，推理阶段关闭辅助分支不会降低 TEDS 和结构准确率，并能显著提升 FPS，说明该辅助分支主要发挥训练期正则化作用。

