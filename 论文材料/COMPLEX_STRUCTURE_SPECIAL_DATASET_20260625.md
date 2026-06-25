# 复杂结构特殊数据集说明（用于创新点三）

日期：2026-06-25

## 目的

为支撑论文创新点三“基于双分支 Mamba 的单元格结构关系建模方法”，从 Phase10 的 5000 样本验证集里自动筛选出一批复杂结构表格，作为专门分析多级表头、跨行跨列和长距离结构依赖的特殊数据集。

这个数据集不是人工随便挑图，而是根据 GT HTML token 和逐样本 TEDS 诊断结果生成。筛选依据包括：

- `<thead>` 中的多行表头；
- 跨 token 解析得到的 `rowspan` 和 `colspan`；
- 表格行数、最大列宽、HTML 序列长度和 span 面积；
- baseline、Phase5E / Dual-Branch Mamba、horizontal-only、vertical-only 的逐样本表现。

## 数据集位置

```text
论文材料/complex_structure_special_dataset_20260625/
```

核心文件：

| 文件 | 说明 |
|---|---|
| `README.md` | 数据集完整说明 |
| `dataset_summary.json` | 统计摘要 |
| `complex_structure_full_5000_metadata.jsonl` | 5000 验证集中命中复杂结构规则的全部样本元数据 |
| `complex_structure_curated_500_metadata.csv` | 便于人工筛选和写论文的 500 个代表样本表格 |
| `complex_structure_curated_500_metadata_with_images.jsonl` | 500 个代表样本元数据和复制后的图片路径 |
| `images/` | 500 张代表性复杂表格原图 |
| `figures/` | 自动导出的论文候选 qualitative figure |
| `build_complex_structure_dataset.py` | 数据集生成脚本 |

## 当前统计结果

| 指标 | 数量 |
|---|---:|
| 复杂结构样本总数 | 4086 |
| 已复制代表图片 | 500 |
| 多级表头 `multi_level_header` | 1822 |
| 跨行/跨列 `row_col_span` | 2395 |
| 横向依赖 `horizontal_dependency` | 2053 |
| 纵向依赖 `vertical_dependency` | 1147 |
| 层次化跨行/跨列表头 `hierarchical_span_header` | 1500 |
| 长距离结构依赖 `long_range_dependency` | 3755 |
| 宽表 `wide_table` | 857 |
| 长表 `tall_table` | 1829 |

Axis ablation 相关证据：

| 标签 | 数量 | 含义 |
|---|---:|---|
| `dual_branch_complementary` | 34 | 完整双分支高于 horizontal-only 和 vertical-only |
| `full_mamba_strong_gain_vs_single_axis` | 25 | 完整双分支相对两个单分支均有明显优势 |
| `horizontal_branch_better` | 1 | 横向分支显著优于纵向分支 |

## 代表性图像

自动导出的拼图和单图位于：

```text
论文材料/complex_structure_special_dataset_20260625/figures/
```

包括：

- `multi_level_header_contact_sheet.png`
- `hierarchical_span_header_contact_sheet.png`
- `horizontal_colspan_contact_sheet.png`
- `vertical_rowspan_contact_sheet.png`
- `long_range_dependency_contact_sheet.png`
- `exemplar_hierarchical_span_header_00045.png`
- `exemplar_horizontal_dependency_00652.png`
- `exemplar_vertical_dependency_00045.png`
- `exemplar_long_range_dependency_00045.png`

其中样本 `00045_PMC4262998_003_00.png` 是一个很强的论文案例：它同时命中多级表头、跨行结构、长距离依赖、长表、宽表，并且完整双分支相对 baseline 有明显提升。

## 可写入论文的表述

```text
为进一步验证模型对复杂结构关系的建模能力，本文从验证集中构建了一个复杂结构特殊子集，覆盖多级表头、跨行跨列单元格以及长距离结构依赖。该子集基于 GT HTML token 自动筛选：通过解析 thead 层级、跨 token 的 rowspan/colspan 属性、表格行列规模和 HTML 序列长度识别复杂结构类型，并结合 horizontal-only、vertical-only 与完整 Dual-Branch Mamba 的逐样本结果分析横向和纵向结构建模的贡献。实验结果表明，复杂结构样本大量包含跨行、跨列和长距离依赖，传统局部特征增强难以充分建模这些关系，而双分支 Mamba 能从横向和纵向两个方向补充单元格结构关系信息。
```

## 注意

第一版脚本曾漏掉部分 span，因为 PubTabNet 的 HTML 标注会把 `rowspan/colspan` 写成独立 token，例如 `<td`、` rowspan="2"`、`>`。当前版本已经改为跨 token 组装 `<td>` 开头后再解析属性，因此可以正确统计跨行跨列结构。
