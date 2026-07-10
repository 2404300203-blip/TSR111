# DBM-SLANet 在 FinTabNet / FinTabNet.c 上的 Baseline 对比表

日期：2026-07-10  
任务：金融表格结构识别  
Baseline：SLANet  
最终模型：DBM-R2 / DBM-SLANet with Dual-Branch Mamba  

## 1. 论文正文主表：金融域微调后最终模型 vs Baseline

该表建议放入论文正文，用于说明最终模型在金融表格数据上的主要效果。

| Setting | Test Set | Model | TEDS | Structure Acc | FPS | Samples | TEDS Gain | Struct. Acc Gain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Pilot split | FinTabNet.c heldout75 | SLANet baseline-ft | 0.4165 | 0.1467 | 33.74 | 75 | - | - |
| Pilot split | FinTabNet.c heldout75 | DBM-R2 | 0.5520 | 0.2667 | 16.59 | 75 | +0.1355 | +0.1200 |
| Company-disjoint | FinTabNet.c unseen-company100 | SLANet baseline-ft | 0.4739 | 0.2000 | 36.37 | 100 | - | - |
| Company-disjoint | FinTabNet.c unseen-company100 | DBM-R2 | 0.6127 | 0.3500 | 12.11 | 100 | +0.1388 | +0.1500 |
| Independent train2000 | FinTabNet.c cached-test500 | SLANet baseline train2000 | 0.4396 | 0.0540 | 32.76 | 500 | - | - |
| Independent train2000 | FinTabNet.c cached-test500 | DBM-R2 train2000 | 0.4877 | 0.0660 | 18.85 | 500 | +0.0481 | +0.0120 |
| Independent train2000 | Company-heldout100 | SLANet baseline train2000 | 0.4211 | 0.0200 | 22.24 | 100 | - | - |
| Independent train2000 | Company-heldout100 | DBM-R2 train2000 | 0.4558 | 0.0700 | 17.69 | 100 | +0.0347 | +0.0500 |

## 2. 主表结论

金融域微调后，DBM-R2 在所有主要评估设置中均超过对应 SLANet baseline：

| Test Set | TEDS Gain | Structure Acc Gain | 结论 |
|---|---:|---:|---|
| FinTabNet.c heldout75 | +0.1355 | +0.1200 | 小样本金融域适配后提升明显 |
| FinTabNet.c unseen-company100 | +0.1388 | +0.1500 | 对未见公司具备更好的结构泛化能力 |
| FinTabNet.c cached-test500 | +0.0481 | +0.0120 | 独立 train2000 设置下仍稳定提升 |
| Company-heldout100 | +0.0347 | +0.0500 | 外部公司测试集上仍优于 baseline |

核心结论：

> DBM-SLANet 不是 zero-shot 直接迁移型增强；它需要金融域微调。但在完成领域适配后，DBM-R2 相比 SLANet baseline 能稳定提升金融表格结构识别效果，尤其在 company-disjoint 设置下提升更明显。

## 3. 附表 A：Zero-shot 跨域测试

该表建议放入附录或实验分析部分，用于说明金融域微调的必要性。

| Dataset | Model | TEDS | Structure Acc | FPS | Samples |
|---|---|---:|---:|---:|---:|
| FinTabNet original test500 | SLANet baseline | 0.2269 | 0.0000 | 28.16 | 500 |
| FinTabNet original test500 | DBM-SLANet | 0.2049 | 0.0000 | 11.86 | 500 |
| FinTabNet.c cached-test500 | SLANet baseline | 0.1928 | 0.0000 | 36.20 | 500 |
| FinTabNet.c cached-test500 | DBM-SLANet | 0.1783 | 0.0000 | 12.75 | 500 |

Zero-shot 结论：

- 通用表格识别模型直接迁移到金融表格场景效果较弱；
- 未经金融域微调时，DBM-SLANet 没有超过 baseline；
- 因此金融域微调是该任务的必要步骤。

## 4. 附表 B：Pilot 随机划分完整结果

数据划分：

- train：350
- val：75
- heldout：75

| Model / Strategy | Heldout TEDS | Structure Acc | FPS |
|---|---:|---:|---:|
| baseline pre-ft | 0.2091 | 0.0000 | 30.87 |
| baseline ft | 0.4165 | 0.1467 | 33.74 |
| original DBM ft | 0.1894 | 0.0000 | 11.78 |
| DBM-R1 | 0.4442 | 0.2000 | 13.11 |
| DBM-R2 | 0.5520 | 0.2667 | 16.59 |

Pilot split 结论：

- baseline 经过金融域微调后显著优于 pre-ft；
- original DBM 直接微调效果不好；
- DBM-R2，即从金融域 baseline checkpoint 初始化 DBM，再用稳定学习率继续微调，是当前最有效策略。

## 5. 附表 C：Company-disjoint 泛化实验

数据划分：

- train：349，35 个公司
- val：51，5 个公司
- heldout：100，10 个未见公司

| Model / Strategy | Heldout TEDS | Structure Acc | FPS | Samples |
|---|---:|---:|---:|---:|
| baseline pre-ft | 0.1626 | 0.0000 | 26.72 | 100 |
| DBM pre-ft | 0.1590 | 0.0000 | 11.66 | 100 |
| baseline ft | 0.4739 | 0.2000 | 36.37 | 100 |
| DBM-R2 ft | 0.6127 | 0.3500 | 12.11 | 100 |

Company-disjoint 结论：

- DBM-R2 在未见公司测试集上仍明显优于 baseline；
- TEDS 提升 `+0.1388`；
- Structure Acc 提升 `+0.1500`；
- 说明 DBM-R2 不只是记忆训练样本，而是对金融表格结构关系具有一定泛化能力。

## 6. 附表 D：Independent train2000 实验

该实验使用独立公开金融表格图像数据构建 train2000 训练集，并在两个外部测试集上评估。

| Model | Test Set | TEDS | Structure Acc | FPS | Samples |
|---|---|---:|---:|---:|---:|
| baseline train2000 | cached-test500 | 0.4396 | 0.0540 | 32.76 | 500 |
| DBM-R2 train2000 | cached-test500 | 0.4877 | 0.0660 | 18.85 | 500 |
| baseline train2000 | company-heldout100 | 0.4211 | 0.0200 | 22.24 | 100 |
| DBM-R2 train2000 | company-heldout100 | 0.4558 | 0.0700 | 17.69 | 100 |

Independent train2000 结论：

- 在独立训练数据上，DBM-R2 仍稳定超过 baseline；
- cached-test500 上 TEDS 提升 `+0.0481`；
- company-heldout100 上 TEDS 提升 `+0.0347`；
- 这说明 DBM-R2 的收益不是 pilot 小样本划分带来的偶然结果。

## 7. 速度与精度权衡

DBM-R2 相比 baseline 的主要代价是推理速度下降。

| Setting | Baseline FPS | DBM-R2 FPS | 速度变化 |
|---|---:|---:|---:|
| Pilot heldout75 | 33.74 | 16.59 | -50.8% |
| Company-disjoint heldout100 | 36.37 | 12.11 | -66.7% |
| cached-test500 train2000 | 32.76 | 18.85 | -42.5% |
| company-heldout100 train2000 | 22.24 | 17.69 | -20.5% |

论文中需要如实说明：DBM-R2 属于以推理速度为代价换取结构识别精度的模型。

## 8. LaTeX 主表版本

```latex
\begin{table}[t]
\centering
\caption{Comparison between SLANet baseline and the final DBM-R2 model on financial table benchmarks.}
\label{tab:fintabnet_main}
\begin{tabular}{llrrrrr}
\hline
Setting & Model & TEDS & Struct. Acc & FPS & Samples & TEDS Gain \\
\hline
Pilot heldout75 & SLANet baseline-ft & 0.4165 & 0.1467 & 33.74 & 75 & -- \\
Pilot heldout75 & DBM-R2 & 0.5520 & 0.2667 & 16.59 & 75 & +0.1355 \\
Company-disjoint100 & SLANet baseline-ft & 0.4739 & 0.2000 & 36.37 & 100 & -- \\
Company-disjoint100 & DBM-R2 & 0.6127 & 0.3500 & 12.11 & 100 & +0.1388 \\
Cached-test500 & SLANet train2000 & 0.4396 & 0.0540 & 32.76 & 500 & -- \\
Cached-test500 & DBM-R2 train2000 & 0.4877 & 0.0660 & 18.85 & 500 & +0.0481 \\
Company-heldout100 & SLANet train2000 & 0.4211 & 0.0200 & 22.24 & 100 & -- \\
Company-heldout100 & DBM-R2 train2000 & 0.4558 & 0.0700 & 17.69 & 100 & +0.0347 \\
\hline
\end{tabular}
\end{table}
```

## 9. 论文正文推荐表述

英文：

> Table~\ref{tab:fintabnet_main} compares the proposed DBM-R2 model with the SLANet baseline on financial table benchmarks. The results show that direct zero-shot transfer is insufficient for both models, indicating a clear domain gap between generic table datasets and financial tables. After financial-domain adaptation, DBM-R2 consistently outperforms the fine-tuned SLANet baseline across pilot, company-disjoint, and independent train2000 settings. In particular, on the company-disjoint split, DBM-R2 improves TEDS from 0.4739 to 0.6127 and structure accuracy from 0.2000 to 0.3500, suggesting better generalization to unseen companies. The improvement comes with lower inference speed, which reflects the accuracy-efficiency trade-off introduced by the dual-branch Mamba module.

中文：

> 表中对比了最终 DBM-R2 模型与 SLANet baseline 在金融表格数据集上的表现。实验结果表明，两个模型在 zero-shot 跨域条件下表现均较弱，说明通用表格数据与金融表格之间存在明显分布差异。经过金融域微调后，DBM-R2 在 pilot、company-disjoint 和 independent train2000 设置下均稳定超过对应 baseline。尤其在 company-disjoint 设置中，DBM-R2 将 TEDS 从 0.4739 提升到 0.6127，将 Structure Acc 从 0.2000 提升到 0.3500，说明其对未见公司表格具有更好的结构泛化能力。与此同时，DBM-R2 的 FPS 低于 baseline，体现出双分支 Mamba 结构带来的精度与效率权衡。

## 10. 最终结论

金融表格实验可形成如下论文结论：

> DBM-SLANet 在金融域微调后能够稳定优于 SLANet baseline，尤其在 company-disjoint 泛化测试中提升明显。这说明 Dual-Branch Mamba 对金融表格中的行列关系和单元格结构关系建模具有实际贡献。但该方法依赖金融域适配，且推理速度低于 baseline，需要在论文中作为限制说明。

