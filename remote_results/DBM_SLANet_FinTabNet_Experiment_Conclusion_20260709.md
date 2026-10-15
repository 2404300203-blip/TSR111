# DBM-SLANet 金融表格实验结论

## 1. 总体结论

当前实验表明：**DBM-SLANet 在经过金融表格数据微调后，相比 SLANet baseline 能稳定提升 FinTabNet.c 上的表格结构识别效果。**

该方法不是零样本直接迁移型增强。实验结果显示，未经金融域微调时，SLANet baseline 和 DBM-SLANet 在 FinTabNet / FinTabNet.c 上表现都较弱；但在完成金融域适配后，DBM-R2 在多个测试设置下均超过对应 baseline。

因此，目前可以形成的核心结论是：

> 通用表格识别模型直接迁移到金融表格场景效果有限；通过金融域微调后，引入 Dual-Branch Mamba 的 DBM-SLANet 能够进一步提升表格结构识别能力，尤其对金融表格中的行列关系和单元格结构关系建模具有帮助。

## 2. Zero-shot 跨域测试

| 数据集 | 模型 | TEDS | Structure Acc | FPS | 样本数 |
|---|---|---:|---:|---:|---:|
| FinTabNet original test500 | SLANet baseline | 0.2269 | 0.0000 | 28.16 | 500 |
| FinTabNet original test500 | DBM-SLANet | 0.2049 | 0.0000 | 11.86 | 500 |
| FinTabNet.c cached-test500 | SLANet baseline | 0.1928 | 0.0000 | 36.20 | 500 |
| FinTabNet.c cached-test500 | DBM-SLANet | 0.1783 | 0.0000 | 12.75 | 500 |

### 结论

Zero-shot 条件下，两个模型表现都较弱，DBM-SLANet 甚至略低于 baseline。这说明：

- 金融表格与原始训练域存在明显分布差异；
- 直接迁移不能充分发挥 DBM 结构建模能力；
- 金融域微调是后续实验的必要前提。

## 3. Pilot 随机划分实验

数据集来自 cached FinTabNet.c test500 的 pilot 划分：

- train：350
- val：75
- heldout：75

| 模型/策略 | Heldout TEDS | Structure Acc | FPS |
|---|---:|---:|---:|
| baseline pre-ft | 0.2091 | 0.0000 | 30.87 |
| baseline ft | 0.4165 | 0.1467 | 33.74 |
| original DBM ft | 0.1894 | 0.0000 | 11.78 |
| DBM-R1 | 0.4442 | 0.2000 | 13.11 |
| DBM-R2 | 0.5520 | 0.2667 | 16.59 |

### 结论

金融域微调显著提升 baseline。更重要的是，DBM-R2 在 baseline 金融微调结果的基础上继续提升：

- TEDS 从 `0.4165` 提升到 `0.5520`；
- Structure Acc 从 `0.1467` 提升到 `0.2667`。

这说明 DBM-R2 的训练策略有效。相比直接用原始 DBM 微调，从金融域 baseline checkpoint 初始化 DBM，并采用较小学习率与更稳定的 loss 权重，是更合理的适配方式。

## 4. Company-disjoint 泛化实验

该实验使用 company-disjoint 划分，测试公司不出现在训练集中：

- train：349，35 个公司
- val：51，5 个公司
- heldout：100，10 个未见公司

| 模型/策略 | Heldout TEDS | Structure Acc | FPS | 样本数 |
|---|---:|---:|---:|---:|
| baseline pre-ft | 0.1626 | 0.0000 | 26.72 | 100 |
| DBM pre-ft | 0.1590 | 0.0000 | 11.66 | 100 |
| baseline ft | 0.4739 | 0.2000 | 36.37 | 100 |
| DBM-R2 ft | 0.6127 | 0.3500 | 12.11 | 100 |

### 结论

在测试公司与训练公司不重叠的设置下，DBM-R2 仍然明显优于 baseline：

- TEDS 从 `0.4739` 提升到 `0.6127`；
- Structure Acc 从 `0.2000` 提升到 `0.3500`。

这说明 DBM-R2 并不是简单记忆训练样本，而是对金融表格中更一般的结构关系具有一定泛化能力。

## 5. 独立 train2000 实验

为了进一步排除 pilot 小样本划分带来的偶然性，使用独立的 `doduytoan/fintabnet2kimg` cropped image 数据构建 train2000 训练集，并在两个外部测试集上评估：

- cached-test500
- company-heldout100

| 模型 | 测试集 | TEDS | Structure Acc | FPS | 样本数 |
|---|---|---:|---:|---:|---:|
| baseline train2000 | cached-test500 | 0.4396 | 0.054 | 32.76 | 500 |
| DBM-R2 train2000 | cached-test500 | 0.4877 | 0.066 | 18.85 | 500 |
| baseline train2000 | company-heldout100 | 0.4211 | 0.020 | 22.24 | 100 |
| DBM-R2 train2000 | company-heldout100 | 0.4558 | 0.070 | 17.69 | 100 |

### 结论

在独立 train2000 金融表格数据上训练后，DBM-R2 仍然超过 baseline：

- cached-test500：TEDS 提升 `+0.0481`；
- company-heldout100：TEDS 提升 `+0.0346`；
- company-heldout100 Structure Acc 从 `0.020` 提升到 `0.070`。

这进一步说明 DBM-R2 的提升具有稳定性，不只是单次小样本实验中的偶然结果。

## 6. 方法有效性判断

综合所有实验，可以得到以下判断：

1. **Zero-shot 不成立**
   DBM-SLANet 不能在不适配金融域的情况下直接超过 baseline。

2. **金融域微调是关键**
   baseline 经过金融域微调后已有明显提升，说明数据域适配是金融表格识别任务的基础。

3. **DBM-R2 是当前最有效策略**
   先训练金融域 baseline，再从该 checkpoint 初始化 DBM，并使用较小学习率继续微调，能够稳定获得更好的 TEDS 和 Structure Acc。

4. **Dual-Branch Mamba 的贡献主要体现在结构建模**
   实验提升主要出现在 TEDS 和 Structure Acc 上，说明 DBM 的价值不是 OCR 层面的文本识别，而是增强单元格级别的行列关系建模。

5. **具备一定跨公司泛化能力**
   company-disjoint 实验中 DBM-R2 仍然优于 baseline，说明模型对未见公司表格具有一定泛化价值。

6. **代价是推理速度下降**
   DBM-R2 的 FPS 低于 baseline。该方法属于以一定推理速度开销换取结构识别精度提升。

## 7. 当前论文可用结论

当前实验已经可以支撑以下论文结论：

> 针对金融表格识别任务，仅依赖通用表格识别模型难以获得理想效果。通过金融域微调后，本文提出的 DBM-SLANet 能够在 FinTabNet.c 测试集和公司不重叠测试集上持续优于 SLANet baseline，提升 TEDS 与结构准确率。实验结果说明，双分支 Mamba 单元格关系建模能够有效增强模型对金融表格行列结构的表达能力，并具备一定跨公司泛化能力。

## 8. 局限性

当前实验仍存在以下局限：

- FinTabNet.c 官方完整数据下载与构建受限，部分实验基于 cached subset 和公开镜像数据；
- 独立 train2000 数据的验证集分布不够理想，外部测试集结果更有参考价值；
- DBM-R2 推理速度低于 baseline，需要在论文中如实报告；
- 当前结果已能证明方法有效，但若目标是更高水平论文，仍建议补充更完整的官方数据评估和更多消融实验。

## 9. 最终一句话结论

**DBM-SLANet 在金融域微调后能够稳定优于 SLANet baseline，说明 Dual-Branch Mamba 对金融表格结构关系建模有效；该方法具备论文可写性，但需要承认其依赖领域适配且推理速度较慢。**
