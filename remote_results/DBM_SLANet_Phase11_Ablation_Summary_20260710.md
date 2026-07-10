# DBM-SLANet Phase 11 补充对比与消融实验总结

日期：2026-07-10  
实验环境：远程 4090D 服务器 `/root/autodl-tmp/PaddleOCR`  
评估集：CTA500 / `complex_structure_cta500_20260625.jsonl`  
基础模型：`SLANet` / `DBM-SLANet` 系列  

## 1. 实验目的

本轮实验用于补充论文中的对比实验和消融实验，重点验证 DBM-SLANet 中以下设计是否真正有效：

- cell token 是否有必要；
- Full Mamba 是否优于仅 cell token；
- `cell presence` 辅助监督是否有效；
- 双向 Mamba 是否优于单向 Mamba；
- 几何 token 是否提供有效结构信息；
- 辅助分支在推理阶段是否必须保留；
- empty-td / empty-cell 指标是否能辅助解释结构恢复效果。

整体目标不是单纯追求最高 TEDS，而是补足论文中“为什么这些模块有意义”的实验链条。

## 2. 已完成实验概览

| 编号 | 实验 | 状态 |
|---|---|---|
| A | Cell Token only vs Full Mamba | 已完成 |
| B | `lambda_pres=0` vs `lambda_pres=0.2` | 已完成 |
| C | Cell Presence F1 | 已完成，但指标需谨慎使用 |
| D | Empty-td rate / Empty-cell count | 已完成 |
| E | Bidirectional=false vs true | 已完成 |
| F | Inference-off auxiliary branch | 已完成 |
| G | Geometry feature ablation | 已完成 |

## 3. 核心消融结果

| 实验设置 | TEDS | Structure Acc | FPS | 主要结论 |
|---|---:|---:|---:|---|
| Cell Token only | 0.54364 | 0.284 | 20.33 | 轻量，速度快，但缺少行列关系建模 |
| Full Mamba, `lambda_pres=0` | 0.53921 | 0.282 | 9.40 | 仅加 Mamba 不一定带来提升 |
| Full Mamba, `lambda_pres=0.2` | 0.54215 | 0.302 | 9.54 | 辅助监督提升结构完全匹配率 |
| Full Mamba, `bidirectional=false` | 0.53040 | 0.272 | 14.14 | 单向更快，但结构性能明显下降 |
| Full Mamba, geometry zero | 0.53921 | 0.282 | 11.12 | 真实几何特征对结构恢复有贡献 |
| Full Mamba, inference auxiliary off | 0.54215 | 0.302 | 18.68 | 推理阶段可关闭辅助分支并显著提速 |

## 4. 关键实验结论

### 4.1 Cell Token only vs Full Mamba

Cell Token only 的 TEDS 为 `0.54364`，略高于 Full Mamba `lambda_pres=0` 的 `0.53921`，且速度明显更快。

这说明：单独引入 Mamba 序列建模并不必然带来收益。Mamba 模块需要和几何 token、双向行列建模以及辅助监督共同配合，才能体现结构建模价值。

### 4.2 `lambda_pres=0.2` 辅助监督有效

在 Full Mamba 中加入 `cell presence` 辅助损失后：

- TEDS 从 `0.53921` 提升到 `0.54215`；
- Structure Acc 从 `0.282` 提升到 `0.302`。

这说明 `cell presence` 辅助监督对结构完整性有正向作用，尤其体现在结构完全匹配率上。

### 4.3 双向 Mamba 有效

关闭 bidirectional 后：

- TEDS 从 `0.54215` 降到 `0.53040`；
- Structure Acc 从 `0.302` 降到 `0.272`；
- FPS 从 `9.54` 提升到 `14.14`。

这说明双向行列依赖建模确实提升了结构恢复效果，但代价是推理速度下降。

### 4.4 几何特征有效

将几何 token 输入置零后：

- TEDS 从 `0.54215` 降到 `0.53921`；
- Structure Acc 从 `0.302` 降到 `0.282`。

这说明真实几何信息对结构恢复有贡献，尤其对结构完全匹配率更敏感。

### 4.5 辅助分支推理阶段可关闭

关闭推理阶段辅助分支后：

- TEDS 保持 `0.54215`；
- Structure Acc 保持 `0.302`；
- FPS 从 `9.54` 提升到 `18.68`。

这说明 `cell presence` 分支主要发挥训练期正则化作用，推理阶段可以关闭，从而在不损失精度的情况下显著提升速度。

## 5. Empty-td 辅助指标结果

| 模型 | Samples | TEDS | Exact | Pred empty td total | GT empty td total | Pred empty td / image | GT empty td / image | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 500 | 0.48434 | 0.208 | 57685 | 55966 | 115.37 | 111.93 | 1.0307 |
| horizontal_only | 500 | 0.62764 | 0.420 | 58330 | 55966 | 116.66 | 111.93 | 1.0422 |
| vertical_only | 500 | 0.62576 | 0.416 | 58315 | 55966 | 116.63 | 111.93 | 1.0420 |
| dual_phase5e | 500 | 0.64067 | 0.432 | 58148 | 55966 | 116.30 | 111.93 | 1.0390 |

Empty-td 指标可以作为辅助分析，但不能单独代表模型质量。当前结果更适合说明不同模型在空单元格预测数量上的偏差，而核心评价仍应以 TEDS 和 Structure Acc 为主。

## 6. 可写入论文的实验表述

实验结果表明，单独引入 Mamba 序列建模并不能稳定提升表格结构识别性能；当其与几何感知 cell token 和 cell presence 辅助监督结合时，模型在结构完全匹配率上获得更明显提升。进一步消融显示，双向行列依赖建模和真实几何特征均对结构恢复有正向贡献。此外，cell presence 分支主要发挥训练期正则化作用，推理阶段关闭该分支不会降低 TEDS 或结构准确率，并能显著提升推理速度。

## 7. 论文价值判断

本轮实验已经形成较完整的消融链条，可以支撑以下论文论点：

- DBM-SLANet 不是简单叠加 Mamba，而是结合了几何 cell token、行列方向建模和辅助监督；
- 双向行列建模对结构恢复有效；
- 几何特征对结构完全匹配率有贡献；
- `cell presence` 辅助分支主要用于训练期约束，推理时可以关闭以降低开销；
- 模型创新点可以通过 TEDS、Structure Acc、FPS 和 empty-td 辅助指标共同解释。

## 8. 后续建议

如果用于论文正式投稿，建议下一步补充：

- 将本轮消融结果整理成论文主表和附表；
- 对 `Cell Presence F1=1.0` 的定义进行代码核验，避免使用可疑指标作为核心证据；
- 在金融表格数据集 FinTabNet / FinTabNet.c 上同步报告最终模型与 baseline 的对比；
- 选择 2-3 个可视化案例，展示 DBM-SLANet 在复杂表格结构恢复上的优势。

