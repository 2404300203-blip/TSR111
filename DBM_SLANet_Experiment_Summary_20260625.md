# DBM-SLANet 当前实验总结

日期：2026-06-25  
项目：表格结构识别 / Table Structure Recognition  
主模型：Phase5-E Full Mamba  
主要 baseline：Original SLANet

---

## 1. 实验总体结论

当前实验已经形成较完整的论文实验链路：

```text
Original SLANet baseline
→ Cell token / geometry-aware head
→ Dual-axis Mamba structure modeling
→ Phase5-E Full Mamba final model
→ PubTabNet 3000/5000 大样本验证
→ CTA-500 复杂结构专项验证
→ bootstrap 统计检验
→ 消融实验与失败案例分析
```

核心结论：

> Phase5-E Full Mamba 在普通大样本验证集和复杂结构专项子集上均显著优于 Original SLANet baseline，尤其在多级表头、跨行跨列、横向/纵向依赖和长距离结构依赖样本上提升明显。

---

## 2. Baseline 设置

baseline 使用原始 SLANet 结构识别模型：

```text
Original SLANet baseline / original SLAHead
```

baseline checkpoint：

```text
output/baseline_package_20260621/model/best_accuracy.pdparams
```

baseline 主要作用：

- 作为所有后续 Mamba 改进的对照；
- 验证模型结构改进是否真实有效；
- 在 PubTabNet 和 CTA-500 上提供统一比较基准。

---

## 3. 主模型：Phase5-E Full Mamba

最终主模型：

```text
Phase5-E Full Mamba
```

checkpoint：

```text
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams
```

核心结构设计：

- 基于 SLANet 表格结构识别框架；
- 引入 cell geometry token；
- 使用 horizontal order 建模横向单元格关系；
- 使用 vertical order 建模纵向单元格关系；
- 使用 bidirectional selective scan；
- 融合 original / horizontal / vertical 三路 cell token；
- 用 Full Mamba-style state-space block 增强 HTML structure token 解码。

关键代码：

```text
ppocr/modeling/heads/cell_token_mlp_head.py
ppocr/modeling/heads/cell_dual_branch_mamba_head.py
ppocr/modeling/heads/cell_full_mamba_head.py
```

---

## 4. PubTabNet 5000 样本主结果

验证集：

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 5000 | 0.8330900340 | 0.6898 | 23.4966 |
| Phase5-E Full Mamba | 5000 | 0.8439856134 | 0.7182 | 9.6547 |

提升：

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.8330900340 | 0.8439856134 | +0.0108955794 |
| Structure Acc | 0.6898 | 0.7182 | +0.0284 |
| FPS | 23.4966 | 9.6547 | -13.8419 |

paired bootstrap：

```text
TEDS delta: +0.0108955794
95% CI: [+0.0059055140, +0.0161684765]
p(delta <= 0): 0.0000
```

结论：

> Phase5-E 在 5000 样本验证集上相对 Original SLANet 获得稳定、统计显著的 TEDS 和 structure accuracy 提升。

---

## 5. CTA-500 复杂结构专项验证

CTA-500 是从 PubTabNet-style validation 中筛选出的复杂结构诊断子集，覆盖：

- 多级表头；
- `rowspan` / `colspan`；
- 横向依赖；
- 纵向依赖；
- 宽表；
- 长表；
- 长距离结构依赖。

CTA-500 文件：

```text
train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl
```

### 5.1 CTA-500 主结果

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 500 | 0.4843354130 | 0.2080 | 16.3023 |
| Phase5-E Full Mamba | 500 | 0.6406726544 | 0.4320 | 8.1429 |

提升：

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.4843354130 | 0.6406726544 | +0.1563372413 |
| Structure Acc | 0.2080 | 0.4320 | +0.2240 |
| FPS | 16.3023 | 8.1429 | -8.1594 |

paired bootstrap：

```text
TEDS delta: +0.1563372413
95% CI: [+0.1118128127, +0.2007433510]
p(delta <= 0): 0.0000
```

样本级对比：

| Category | Count |
|---|---:|
| Phase5-E better than baseline | 244 |
| Phase5-E worse than baseline | 114 |
| Same TEDS | 142 |

结论：

> CTA-500 上 Phase5-E 的提升远大于普通验证集，说明 Full Mamba 对复杂表格结构建模更有效。

### 5.2 按复杂结构标签拆分

| Tag | Samples | Baseline TEDS | Phase5-E TEDS | Gain |
|---|---:|---:|---:|---:|
| hierarchical_span_header | 274 | 0.475252 | 0.675940 | +0.200688 |
| multi_level_header | 325 | 0.460398 | 0.652324 | +0.191926 |
| vertical_dependency | 286 | 0.449572 | 0.638929 | +0.189357 |
| horizontal_dependency | 376 | 0.458076 | 0.646133 | +0.188058 |
| row_col_span | 457 | 0.462662 | 0.647945 | +0.185283 |
| wide_table | 149 | 0.541695 | 0.708302 | +0.166606 |
| long_range_dependency | 473 | 0.489143 | 0.630415 | +0.141272 |
| tall_table | 341 | 0.491260 | 0.570912 | +0.079652 |

可写入论文的结论：

> CTA-500 证明模型在多级表头、层次化跨行跨列表头、横向/纵向结构依赖等复杂场景下具有更强识别能力。

注意：

```text
CTA-500 应描述为 complex-structure diagnostic subset / curated benchmark，
不要表述为完全独立采集的新数据集。
```

---

## 6. 分阶段训练与数据构建

实验中构建了多个训练和验证子集，用于稳定推进模型训练：

```text
phase4_smoke_train_100.jsonl
phase4_stage1_train_500.jsonl
phase4_stage2_train_2000.jsonl
phase4_stage4_train_10000.jsonl
phase4_stage5_train_30000.jsonl
phase4_stage5_val_1000.jsonl
phase10_val_3000.jsonl
phase10_val_5000.jsonl
complex_structure_cta500_20260625.jsonl
complex_structure_special_full_4086_20260625.jsonl
```

数据集相关创新点现在可以这样定位：

> 构建复杂结构诊断子集 CTA-500，用于专门评估模型在多级表头、跨行跨列、横向/纵向依赖和长距离结构关系上的表现。

---

## 7. 主要消融实验

### 7.1 Lite Mamba vs Full Mamba

| Method | TEDS | Structure Acc | Decision |
|---|---:|---:|---|
| Original SLANet baseline | 0.8280000534 | 0.668 | baseline |
| Phase4 Lite Mamba | 0.8294241307 | 0.691 | improved |
| Phase5-E Full Mamba | 0.8308172190 | 0.693 | selected |

结论：

> Lite Mamba 已能提升 baseline，Full Mamba 进一步提升，说明 Mamba-style structure modeling 是有效方向。

### 7.2 学习率 continuation

| Method | TEDS | Structure Acc | Decision |
|---|---:|---:|---|
| Phase5-D lr=1e-5 | 0.8306299579 | 0.693 | improved |
| Phase5-E lr=5e-6 | 0.8308172190 | 0.693 | selected |
| Phase5-F lr=2e-6 | 0.8285197073 | 0.694 | rejected |

结论：

> 更高 structure accuracy 不一定带来更高 TEDS。Phase5-F 的 exact accuracy 更高，但 TEDS 更低。

---

## 8. 失败实验与负向结果

这些实验没有成为最终方法，但可以作为论文消融或误差分析。

| Experiment | Result | Interpretation |
|---|---|---|
| Phase6 TEDS-based checkpoint selection | 没超过 Phase5-E | 证明 TEDS 与 exact acc 选择会不一致 |
| Phase7 span-aware loss | acc 提高，TEDS 下降 | 简单 span 加权不可靠 |
| Phase8 span calibration | 无提升 | 错误 span 往往不是低置信度错误 |
| Phase9-C candidate reranking | 1000 样本提升，3000 样本不稳 | 后处理启发式泛化不足 |
| OCR-aware matching | 直接替换不稳定 | OCR-cell matching 仍需谨慎设计 |

关键发现：

```text
1. rowspan/colspan 是主要错误来源。
2. 错误 span 往往也是高置信度预测。
3. exact structure accuracy 与 TEDS 会明显不一致。
4. 简单后处理不能稳定解决复杂结构错误。
```

---

## 9. 当前可写入论文的贡献点

建议论文贡献点写成：

1. 提出 Dual-Axis Full-Mamba table structure head，用于增强 SLANet 的 HTML structure token 建模能力。
2. 引入 cell geometry token 与 horizontal/vertical sequence modeling，增强表格单元格结构关系表示。
3. 构建 CTA-500 complex-structure diagnostic subset，用于专项评估复杂表格结构识别能力。
4. 在 PubTabNet 5000 和 CTA-500 上验证模型有效性，并通过 bootstrap、标签拆分和定性案例分析证明提升可靠。

---

## 10. 当前实验水平评估

| 维度 | 当前水平 |
|---|---|
| 方法创新 | 较强 |
| 主实验结果 | 已成立 |
| 复杂结构专项验证 | 已成立 |
| 统计检验 | 已完成 |
| 消融实验 | 较完整 |
| 定性分析 | 已有材料 |
| 外部 SOTA 对比 | 仍不足 |
| 英文论文初稿 | 尚未完成 |

综合判断：

```text
当前实验已经达到 SCI 3 区论文雏形水平。
```

但还不能说已经稳投，主要短板是：

```text
1. 外部公开方法对比不足；
2. Phase5-E 速度明显低于 baseline；
3. CTA-500 是筛选诊断子集，不是独立采集的新数据集；
4. 还需要正式英文论文初稿。
```

---

## 11. 下一步建议

优先级：

1. 写英文论文初稿；
2. 整理最终实验表；
3. 补公开方法对比表；
4. 将 CTA-500 写成复杂结构诊断子集贡献；
5. 将速度下降写入 limitation；
6. 选 4-6 个定性案例图放入论文；
7. 如时间允许，补 full 9115 validation 或第二数据集。

---

## 12. 推荐论文题目

```text
DBM-SLANet: Dual-Axis Mamba-Enhanced Structure Decoding for Table Structure Recognition
```

备选：

```text
Dual-Axis Mamba Structure Modeling for Robust Table Structure Recognition
```

---

## 13. 推荐论文主张

建议主张：

> 本文针对 SLANet 在复杂表格结构识别中对多级表头、跨行跨列和长距离结构依赖建模不足的问题，提出 Dual-Axis Full-Mamba table structure head。该方法在 PubTabNet 5000 样本验证集上稳定提升 TEDS，并在 CTA-500 复杂结构诊断子集上获得更显著提升，说明其对复杂结构关系建模更有效。

需要避免的表述：

```text
不要说达到 SOTA。
不要说解决了所有复杂表格结构问题。
不要说速度更快。
不要把 CTA-500 描述成独立采集的新数据集。
不要把 Phase9-C 写成最终方法。
```

