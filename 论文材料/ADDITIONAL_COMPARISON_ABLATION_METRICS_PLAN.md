# 补充对比实验、消融实验与新增指标建议

整理时间：2026-07-07  
项目：DBM-SLANet / 复杂表格结构识别  
目标：基于现有实验和已锁定指标，补强论文中的对比实验、消融实验和复杂结构专项评价指标。

---

## 1. 当前创新点与实验证据对应关系

| 创新点 | 当前已有证据 | 仍需补强的证据 |
|---|---|---|
| 双分支 Mamba 复杂结构关系建模 | Baseline vs Phase5-E；Horizontal-only / Vertical-only；CTA-500 提升明显 | 更清楚的 axis ablation 表、bidirectional ablation、inference-off 辅助分支实验 |
| 复杂表格专项评估子集 | CTA-500；复杂结构标签；按标签拆分结果 | 增加复杂结构专项指标，如 Span-F1、Header-TEDS、Complex-Gain |
| 融合几何信息的单元格表示 | GeometryTokenEncoder、cell token、cell presence auxiliary loss | Geometry feature ablation、cell presence loss weight ablation |

---

## 2. 建议保留的主对比实验

### 2.1 普通验证集主对比

目的：证明本文方法在标准 PubTabNet-style 验证集上优于原始 SLANet。

已有结果：

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 5000 | 0.8330900340 | 0.6898 | 23.4966 |
| Phase5-E Full Mamba | 5000 | 0.8439856134 | 0.7182 | 9.6547 |

建议论文表述：

> 在 5000 样本 PubTabNet 验证集上，Phase5-E 相比 Original SLANet baseline 获得 +0.0109 TEDS 和 +0.0284 structure accuracy 的提升，但推理速度下降。

### 2.2 复杂结构专项子集主对比

目的：证明本文方法在复杂结构表格上提升更明显，更能支撑创新点一和创新点二。

已有结果：

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 500 | 0.4843354130 | 0.2080 | 16.3023 |
| Phase5-E Full Mamba | 500 | 0.6406726544 | 0.4320 | 8.1429 |

建议突出：

```text
普通 5000 样本：TEDS +0.0109
CTA-500 复杂样本：TEDS +0.1563
```

这说明本文方法的主要优势集中在复杂结构关系建模，而不是简单表格上的微小扰动。

### 2.3 按复杂结构标签拆分对比

目的：把“多级表头、跨行跨列、横向/纵向依赖、长距离依赖”这些创新点落到可量化结果。

已有结果：

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

建议将这张表作为论文重点表之一。

---

## 3. 建议补充的消融实验

### 3.1 Axis Ablation：横向分支、纵向分支与双分支

目的：直接支撑“双分支 Mamba 同时建模横向和纵向结构关系”的创新点。

已有配置：

```text
DBM_SLANet_expC_horizontal_mamba_10000_from_phase5e_codex.yml
DBM_SLANet_expD_vertical_mamba_10000_from_phase5e_codex.yml
```

已有 1000 样本结果：

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Horizontal Mamba only | 1000 | 0.8279904875 | 0.697 | 8.4171 |
| Vertical Mamba only | 1000 | 0.8279144473 | 0.696 | 8.3946 |
| Dual-Branch Full Mamba | 1000 | 0.8308172190 | 0.693 | 9.5650 |

建议补跑：

| 补充实验 | 数据集 | 目的 |
|---|---|---|
| Horizontal-only on CTA-500 | CTA-500 | 看横向分支对 colspan / horizontal dependency 的贡献 |
| Vertical-only on CTA-500 | CTA-500 | 看纵向分支对 rowspan / vertical dependency 的贡献 |
| Dual vs single-axis by tag | CTA-500 tags | 证明双分支在复杂标签上互补 |

新增指标：

```text
Dual-Axis Gain = TEDS_dual - max(TEDS_horizontal, TEDS_vertical)
```

论文中可以写：

> Dual-Axis Gain 衡量完整双分支模型相对最强单轴模型的增益，用于验证横向与纵向结构建模的互补性。

### 3.2 Cell Geometry Token Ablation

目的：支撑“融合几何信息的单元格表示方法”。

建议对比：

| Variant | GeometryTokenEncoder | Mamba Branch | Cell Presence Loss | 目的 |
|---|---|---|---|---|
| Baseline | 无 | 无 | 无 | 原始 SLANet |
| Cell Token only | 有 | 无 | 有 | 验证几何 token 和辅助监督是否有效 |
| Full Mamba | 有 | 有 | 有 | 最终模型 |

如果已有 Cell Token only 结果，可以整理；如果没有，建议补跑一个 1000 / CTA-500 小规模评估。

新增指标：

```text
Geometry Gain = TEDS_cell_token - TEDS_baseline
Mamba Gain = TEDS_full_mamba - TEDS_cell_token
```

这样可以把“几何表示贡献”和“Mamba 关系建模贡献”拆开。

### 3.3 Geometry Feature Ablation

目的：进一步证明几何特征设计不是随意堆特征。

当前 GeometryTokenEncoder 输入为 8 维：

```text
x_center, y_center, width, height, area, aspect, left, top
```

建议消融：

| Variant | Geometry Features | 目的 |
|---|---|---|
| Center only | `x_center, y_center` | 只用位置 |
| Basic box | `x_center, y_center, width, height` | 位置 + 尺寸 |
| Full geometry | `x_center, y_center, width, height, area, aspect, left, top` | 当前完整版本 |

建议只在 1000 samples 或 CTA-500 上跑，不必全量。

新增指标：

```text
Geo-Complex Gain = TEDS_full_geometry_CTA500 - TEDS_basic_box_CTA500
```

### 3.4 Cell Presence Loss Weight Ablation

目的：当前代码核验显示 fused token 主要进入 `cell_presence_head`，因此必须解释这个辅助损失是否真的必要。

当前配置：

```text
cell_presence_weight = 0.2
```

建议对比：

| Variant | λ_pres | 目的 |
|---|---:|---|
| no presence loss | 0.0 | 验证辅助监督是否必要 |
| weak presence loss | 0.1 | 检查低权重 |
| current | 0.2 | 当前 Phase5-E 设置 |
| strong presence loss | 0.5 | 检查过强辅助监督是否伤害 TEDS |

建议指标：

```text
TEDS
Structure Acc
Cell Presence Acc
Cell Presence F1
CTA-500 TEDS
```

其中 `Cell Presence F1` 可以定义为：

```text
预测该 step 是否存在有效 bbox / text target，与 bbox_mask 构造的 y_cell 对比。
```

这能直接体现 `cell_presence_head` 的价值。

### 3.5 Bidirectional Selective Scan Ablation

目的：支撑“通过双轴 Selective Scan 建立单元格之间全局联系”的表述。

当前配置：

```text
bidirectional: true
```

建议补跑：

| Variant | bidirectional | 目的 |
|---|---|---|
| Unidirectional scan | false | 只保留单向依赖 |
| Bidirectional scan | true | 当前模型 |

建议指标：

```text
TEDS
CTA-500 TEDS
Long-range dependency TEDS
```

如果 bidirectional 在 `long_range_dependency` 标签上提升更明显，就很适合写入论文。

### 3.6 Inference-Off Auxiliary Branch 实验

目的：当前代码中 Mamba fused token 主要用于 cell presence auxiliary loss。需要确认推理阶段关闭辅助分支后，TEDS 是否不变、速度是否恢复。

建议对比：

| Variant | Training | Inference | 目的 |
|---|---|---|---|
| Phase5-E full inference | 开启 DBM + cell presence | 推理也执行 DBM branch | 当前结果 |
| Phase5-E aux-off inference | 使用同一 checkpoint | 推理跳过 DBM / cell_presence branch | 验证 DBM 是否主要是训练期正则 |

建议指标：

```text
TEDS
Structure Acc
FPS
Output difference rate
```

如果 TEDS 完全一致或几乎一致、FPS 提升，则论文需要谨慎表述：

```text
DBM branch acts as a training-time auxiliary regularization module.
```

如果 TEDS 下降，则说明推理分支仍然影响输出路径，需要进一步核代码或实现。

---

## 4. 建议新增的专项指标

当前已有指标：

```text
TEDS
Structure Acc
FPS
Bootstrap CI
按复杂标签拆分 TEDS
```

为了更贴合三个创新点，建议新增以下指标。

### 4.1 Complex-TEDS

定义：

```text
Complex-TEDS = CTA-500 上的平均 TEDS
```

作用：

- 专门衡量复杂结构表格识别能力；
- 比普通 PubTabNet 验证集更能体现复杂金融表格场景。

### 4.2 Span-TEDS

定义：

```text
Span-TEDS = row_col_span 标签样本上的平均 TEDS
```

作用：

- 专门衡量 rowspan / colspan 场景；
- 对应复杂金融表格中的跨行跨列结构。

已有可用值：

```text
Baseline: 0.462662
Phase5-E: 0.647945
Gain: +0.185283
```

### 4.3 Header-TEDS

定义：

```text
Header-TEDS = multi_level_header 或 hierarchical_span_header 标签样本上的平均 TEDS
```

作用：

- 专门评价多级表头恢复能力；
- 对应金融报表常见的层次化表头。

已有可用值：

```text
multi_level_header Gain: +0.191926
hierarchical_span_header Gain: +0.200688
```

### 4.4 Axis Dependency TEDS

定义：

```text
Horizontal-TEDS = horizontal_dependency 样本上的平均 TEDS
Vertical-TEDS = vertical_dependency 样本上的平均 TEDS
```

作用：

- 分别衡量横向 colspan 和纵向 rowspan 依赖；
- 对应双分支 Mamba 的横向/纵向设计。

已有可用值：

```text
Horizontal dependency Gain: +0.188058
Vertical dependency Gain: +0.189357
```

### 4.5 Long-Range-TEDS

定义：

```text
Long-Range-TEDS = long_range_dependency 标签样本上的平均 TEDS
```

作用：

- 衡量长距离结构依赖恢复能力；
- 对应 Mamba / selective scan 的长序列建模动机。

已有可用值：

```text
Baseline: 0.489143
Phase5-E: 0.630415
Gain: +0.141272
```

### 4.6 Cell Presence F1

定义：

```text
Cell Presence F1 = cell_presence_head 对有效 bbox/text cell step 的二分类 F1
```

标签：

```text
y_cell = clip(sum(bbox_mask), 0, 1)
```

预测：

```text
p_cell = sigmoid(cell_has_text_logits)
```

作用：

- 直接评价 `cell_presence_head` 是否学到了有效 cell 表示；
- 支撑几何 cell token 和 fused representation 的有效性。

### 4.7 Empty-Cell Rate / Empty-td Count

定义：

```text
empty_td_count = 预测 HTML 中空 <td></td> 数量
empty_td_rate = empty_td_count / total_td_count
```

作用：

- 对应 OCR-aware matching 和单元格内容恢复问题；
- 如果作为主线较弱，可以放入补充实验或 error analysis。

---

## 5. 推荐最终实验表结构

### Table 1：Main Results

| Method | PubTabNet-5000 TEDS | PubTabNet-5000 Acc | CTA-500 TEDS | CTA-500 Acc | FPS |
|---|---:|---:|---:|---:|---:|
| Original SLANet | 0.8331 | 0.6898 | 0.4843 | 0.2080 | 23.50 / 16.30 |
| Phase5-E Full Mamba | 0.8440 | 0.7182 | 0.6407 | 0.4320 | 9.65 / 8.14 |

### Table 2：Complex Structure Breakdown

| Tag | Samples | Baseline TEDS | Phase5-E TEDS | Gain |
|---|---:|---:|---:|---:|
| hierarchical_span_header | 274 | 0.4753 | 0.6759 | +0.2007 |
| multi_level_header | 325 | 0.4604 | 0.6523 | +0.1919 |
| vertical_dependency | 286 | 0.4496 | 0.6389 | +0.1894 |
| horizontal_dependency | 376 | 0.4581 | 0.6461 | +0.1881 |
| row_col_span | 457 | 0.4627 | 0.6479 | +0.1853 |
| long_range_dependency | 473 | 0.4891 | 0.6304 | +0.1413 |

### Table 3：Ablation Study

| Variant | Main Change | TEDS | CTA-500 TEDS | Structure Acc | FPS |
|---|---|---:|---:|---:|---:|
| Baseline | Original SLAHead | 已有 | 已有 | 已有 | 已有 |
| Cell Token only | + geometry token | 待补 | 待补 | 待补 | 待补 |
| Horizontal only | + horizontal Mamba | 已有 1000 | 待补 | 已有 | 已有 |
| Vertical only | + vertical Mamba | 已有 1000 | 待补 | 已有 | 已有 |
| Dual-axis Full Mamba | + horizontal + vertical | 已有 | 已有 | 已有 | 已有 |
| No cell-presence loss | λ_pres=0 | 待补 | 待补 | 待补 | 待补 |
| Unidirectional scan | bidirectional=false | 待补 | 待补 | 待补 | 待补 |

### Table 4：Auxiliary Loss Ablation

| λ_pres | TEDS | CTA-500 TEDS | Cell Presence F1 | Structure Acc |
|---:|---:|---:|---:|---:|
| 0.0 | 待补 | 待补 | 待补 | 待补 |
| 0.1 | 待补 | 待补 | 待补 | 待补 |
| 0.2 | 当前设置 | 当前设置 | 待补 | 当前设置 |
| 0.5 | 待补 | 待补 | 待补 | 待补 |

---

## 6. 最建议优先补的 4 个实验

如果时间有限，只补下面 4 个，性价比最高：

1. **Horizontal-only / Vertical-only / Dual on CTA-500**  
   支撑双分支 Mamba 和复杂结构专项子集。

2. **Cell Token only vs Full Mamba on CTA-500**  
   拆分几何 token 和 Mamba 关系建模的贡献。

3. **λ_pres=0 vs λ_pres=0.2**  
   解释 `cell_presence_loss` 是否有效，避免方法逻辑被质疑。

4. **bidirectional=false vs true**  
   支撑双向 selective scan 对长距离依赖的作用。

---

## 7. 最终建议写法

可以把实验贡献改写成：

> 除常规 PubTabNet 验证集外，本文进一步构建 CTA-500 复杂结构专项评估子集，并提出 Span-TEDS、Header-TEDS、Axis Dependency TEDS 和 Long-Range-TEDS 等标签化评价指标，用于衡量模型在多级表头、跨行跨列、横纵向依赖和长距离结构关系下的恢复能力。实验表明，Phase5-E 在普通验证集上取得稳定提升，而在 CTA-500 及其复杂结构标签子集上提升更为显著，说明双分支 Mamba 更适合复杂表格结构关系建模。

需要避免：

```text
不要把 CTA-500 写成完全独立采集的新金融表格数据集。
不要说已经超过所有 SOTA。
不要说 fused token 直接修正 structure logits。
不要隐藏速度下降问题。
```

