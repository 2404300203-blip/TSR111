# 特殊数据集构建与实验操作总结报告（用于第二个创新点）

日期：2026-06-25  
项目路径：`/root/autodl-tmp/PaddleOCR`  
数据集目录：`论文材料/complex_structure_special_dataset_20260625/`

## 1. 实验目的

本次实验的目标是在已有 Phase10 大规模验证结果基础上，构建一个面向复杂表格结构识别的特殊数据集，用于支撑论文中的第二个创新点。

该特殊数据集重点覆盖以下复杂表格类型：

- 多级表头；
- 跨行单元格，即 `rowspan`；
- 跨列单元格，即 `colspan`；
- 横向长距离结构依赖；
- 纵向长距离结构依赖；
- 宽表、长表以及层次化表头结构。

构建该数据集的核心目的不是单纯挑选好看的案例图，而是通过已有模型实验结果和 GT HTML 结构标注，自动筛选出能够体现复杂结构建模难点的样本，为后续论文中的方法动机、对比实验和可视化分析提供数据支撑。

## 2. 使用的数据来源

本次特殊数据集来自 Phase10 阶段已经跑完的 5000 样本验证集和逐样本诊断结果。

使用的主要文件如下：

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl
output/phase10_axis_ablation_20260624/expC_horizontal_sample_teds.jsonl
output/phase10_axis_ablation_20260624/expD_vertical_sample_teds.jsonl
```

其中：

| 文件 | 用途 |
|---|---|
| `phase10_val_5000.jsonl` | 提供验证集样本和 GT HTML 结构信息 |
| `baseline_val5000_sample_teds.jsonl` | 提供 baseline 逐样本 TEDS 表现 |
| `phase5e_val5000_sample_teds.jsonl` | 提供当前主模型逐样本 TEDS 表现 |
| `expC_horizontal_sample_teds.jsonl` | 提供 horizontal-only axis ablation 结果 |
| `expD_vertical_sample_teds.jsonl` | 提供 vertical-only axis ablation 结果 |

## 3. 实验操作流程

### 3.1 建立特殊数据集目录

首先在论文材料目录下建立特殊数据集目录：

```text
论文材料/complex_structure_special_dataset_20260625/
```

目录中包含 metadata、图片、代表性 figure 和数据集生成脚本。

### 3.2 读取 Phase10 逐样本实验结果

脚本读取 baseline、Phase5E、horizontal-only 和 vertical-only 四组逐样本 TEDS 结果，并按照 `sample_index` 对齐。

对每一个样本记录以下信息：

- 样本编号；
- 图片文件名；
- 原始图片路径；
- baseline TEDS；
- Phase5E TEDS；
- Phase5E 相对 baseline 的变化；
- horizontal-only TEDS；
- vertical-only TEDS；
- 是否 exact match；
- 复杂结构标签。

### 3.3 解析 GT HTML token

本次实验的关键操作是解析每个样本的 GT HTML token，统计表格结构复杂度。

统计指标包括：

| 指标 | 含义 |
|---|---|
| `row_count` | 表格总行数 |
| `header_rows` | 表头行数 |
| `body_rows` | 表体行数 |
| `max_width` | 最大列宽 |
| `span_tokens` | 含 rowspan 或 colspan 的单元格数量 |
| `rowspan_tokens` | 跨行单元格数量 |
| `colspan_tokens` | 跨列单元格数量 |
| `max_rowspan` | 最大跨行跨度 |
| `max_colspan` | 最大跨列跨度 |
| `span_area` | 跨行跨列结构面积 |
| `gt_len` | GT HTML token 序列长度 |

### 3.4 修正 rowspan / colspan 解析问题

实验过程中发现第一版解析脚本存在一个重要问题：PubTabNet 的 HTML 标注中，`rowspan` 和 `colspan` 不一定完整写在一个 `<td>` token 中，而是可能被拆成多个 token。例如：

```text
<td
 rowspan="2"
>
```

或：

```text
<td
 colspan="4"
>
```

因此，第一版脚本只在单个 `<td...>` token 中搜索 `rowspan/colspan`，会漏掉大量跨行跨列样本。

随后对脚本进行了修正：

1. 遇到 `<td` token 时，开始收集该单元格开头；
2. 向后继续读取 token，直到遇到 `>` 或新的结构 token；
3. 将多个 token 拼接成完整 `<td ...>` 开头；
4. 再从拼接后的字符串中解析 `rowspan` 和 `colspan`；
5. 重新统计跨行、跨列、横向依赖和纵向依赖标签。

修正后，跨行跨列样本统计明显增加，说明原先被漏掉的 span 结构已经被正确识别。

## 4. 复杂结构标签设计

本次实验为每个样本自动打上复杂结构标签。

| 标签 | 判定依据 | 论文含义 |
|---|---|---|
| `multi_level_header` | 表头行数大于等于 2，或表头存在 span | 多级表头 |
| `row_col_span` | 存在 rowspan 或 colspan | 跨行跨列结构 |
| `horizontal_dependency` | 存在 colspan | 横向结构依赖 |
| `vertical_dependency` | 存在 rowspan | 纵向结构依赖 |
| `hierarchical_span_header` | 多级表头同时存在 span | 层次化复杂表头 |
| `long_range_dependency` | 行数、列数、序列长度或跨度面积较大 | 长距离结构依赖 |
| `wide_table` | 最大列宽较大 | 宽表结构 |
| `tall_table` | 行数较多 | 长表结构 |

另外，结合 axis ablation 结果，还额外标记了分支贡献标签：

| 标签 | 含义 |
|---|---|
| `dual_branch_complementary` | 完整双分支模型高于 horizontal-only 和 vertical-only |
| `full_mamba_strong_gain_vs_single_axis` | 完整双分支相对两个单分支都有明显优势 |
| `horizontal_branch_better` | 横向分支显著优于纵向分支 |

这些标签可以用于说明：复杂表格不仅需要局部视觉特征，还需要显式建模横向、纵向和长距离单元格关系。

## 5. 最终数据集规模

修正 span 解析后，最终统计结果如下：

| 类别 | 数量 |
|---|---:|
| 复杂结构样本总数 | 4086 |
| 代表性图片子集 | 500 |
| 多级表头 `multi_level_header` | 1822 |
| 跨行/跨列 `row_col_span` | 2395 |
| 横向依赖 `horizontal_dependency` | 2053 |
| 纵向依赖 `vertical_dependency` | 1147 |
| 层次化跨行/跨列表头 `hierarchical_span_header` | 1500 |
| 长距离结构依赖 `long_range_dependency` | 3755 |
| 宽表 `wide_table` | 857 |
| 长表 `tall_table` | 1829 |

Axis ablation 相关统计如下：

| 类别 | 数量 |
|---|---:|
| `dual_branch_complementary` | 34 |
| `full_mamba_strong_gain_vs_single_axis` | 25 |
| `horizontal_branch_better` | 1 |

## 6. 输出文件

本次实验共输出以下文件：

```text
论文材料/complex_structure_special_dataset_20260625/
├── README.md
├── dataset_summary.json
├── build_complex_structure_dataset.py
├── complex_structure_full_5000_metadata.jsonl
├── complex_structure_curated_500_metadata.jsonl
├── complex_structure_curated_500_metadata.csv
├── complex_structure_curated_500_metadata_with_images.jsonl
├── images/
└── figures/
```

其中：

| 文件 | 说明 |
|---|---|
| `README.md` | 特殊数据集说明文档 |
| `dataset_summary.json` | 数据集统计摘要 |
| `build_complex_structure_dataset.py` | 可复现实验脚本 |
| `complex_structure_full_5000_metadata.jsonl` | 全部 4086 个复杂样本元数据 |
| `complex_structure_curated_500_metadata.jsonl` | 500 个代表性复杂样本元数据 |
| `complex_structure_curated_500_metadata.csv` | 便于人工筛选的 CSV 表格 |
| `complex_structure_curated_500_metadata_with_images.jsonl` | 带复制后图片路径的元数据 |
| `images/` | 500 张代表性复杂表格图片 |
| `figures/` | 自动导出的论文候选图 |

## 7. 代表性可视化结果

本次实验自动导出了多组 qualitative figure 候选图：

```text
figures/multi_level_header_contact_sheet.png
figures/hierarchical_span_header_contact_sheet.png
figures/horizontal_colspan_contact_sheet.png
figures/vertical_rowspan_contact_sheet.png
figures/long_range_dependency_contact_sheet.png
figures/exemplar_hierarchical_span_header_00045.png
figures/exemplar_horizontal_dependency_00652.png
figures/exemplar_vertical_dependency_00045.png
figures/exemplar_long_range_dependency_00045.png
```

其中，样本 `00045_PMC4262998_003_00.png` 是一个非常适合写进论文的案例。该样本同时包含：

- 多级表头；
- 大量 `rowspan`；
- 长距离结构依赖；
- 长表；
- 宽表；
- 完整双分支模型相对 baseline 有明显提升。

该样本可以作为“复杂结构关系建模有效性”的主案例图。

## 8. 对第二个创新点的支撑意义

本次特殊数据集能够为第二个创新点提供三类支撑。

第一，提供复杂结构样本来源。  
通过自动筛选得到 4086 个复杂结构样本，说明实验不是只在普通表格上验证，而是额外关注了更困难的表格结构。

第二，提供结构难点分类。  
数据集将复杂结构拆分为多级表头、跨行、跨列、横向依赖、纵向依赖和长距离依赖，有利于论文中更清楚地说明方法要解决的问题。

第三，提供可解释的案例图。  
导出的代表性图片可以直接用于 qualitative analysis，展示模型在复杂结构场景中的表现，也可以用于失败案例分析。

因此，该特殊数据集可以作为第二个创新点中的“复杂结构场景验证集”或“复杂表格专项评估子集”。

## 9. 可写入论文的表述

```text
为进一步验证模型在复杂表格结构场景下的鲁棒性，本文从验证集中构建了一个复杂结构专项评估子集。该子集基于 GT HTML token 自动筛选，覆盖多级表头、跨行跨列单元格、宽表、长表以及长距离结构依赖等典型困难场景。具体而言，我们跨 token 解析 `<td>` 中的 `rowspan` 和 `colspan` 属性，并结合表头层级、行列规模和 HTML 序列长度对样本进行复杂度标注。最终得到 4086 个复杂结构样本，并进一步选取 500 个代表性样本用于可视化分析。该专项评估子集为验证模型在复杂表格结构识别任务中的有效性提供了更有针对性的实验依据。
```

## 10. 后续建议

后续可以继续做三件事：

1. 在该特殊数据集上单独计算 baseline、Phase5E 和后续模型的整体 TEDS、TEDS-Struct、TEDS-Content；
2. 将 500 个代表样本中的成功案例和失败案例进一步人工筛选，形成论文最终 qualitative figure；
3. 如果论文强调金融场景，可以补充一批真实金融报表样本，将当前特殊数据集扩展为“复杂金融表格专项数据集”。
