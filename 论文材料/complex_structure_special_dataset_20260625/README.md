# 复杂结构特殊数据集 Complex Structure Special Dataset

Date: 2026-06-25

本数据集从 Phase10 的 5000 样本验证集和逐样本诊断结果中筛选，用于支撑“基于双分支 Mamba 的单元格结构关系建模方法”这一创新点。筛选不是人工凭感觉挑图，而是依据 GT HTML token、TEDS 逐样本结果以及 horizontal/vertical axis ablation 结果自动生成。

## 数据来源

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl
output/phase10_axis_ablation_20260624/expC_horizontal_sample_teds.jsonl
output/phase10_axis_ablation_20260624/expD_vertical_sample_teds.jsonl
```

## 数据规模

| 子集 | 数量 | 说明 |
|---|---:|---|
| full complex metadata | 4086 | 5000 验证集中所有命中复杂结构规则的样本 |
| curated image subset | 500 | 已复制图片的代表性复杂结构样本 |
| copied images | 500 | 位于 `images/` 目录 |

## 复杂结构标签统计 full set

| 标签 | 数量 | 论文含义 |
|---|---:|---|
| `long_range_dependency` | 3755 | 行数、列数、HTML 序列长度或跨度面积较大，适合分析长距离结构依赖。 |
| `row_col_span` | 2395 | 存在 rowspan 或 colspan。 |
| `horizontal_dependency` | 2053 | 存在 colspan，可作为横向关系建模证据。 |
| `tall_table` | 1829 | 行数较多。 |
| `multi_level_header` | 1822 | 多级表头，或表头区域出现跨行/跨列。 |
| `hierarchical_span_header` | 1500 | 多级表头同时含跨行/跨列，是复杂金融表头的核心类型。 |
| `vertical_dependency` | 1147 | 存在 rowspan，可作为纵向关系建模证据。 |
| `wide_table` | 857 | 列数较多。 |

## Axis Ablation 证据标签

| 标签 | 数量 | 含义 |
|---|---:|---|
| `dual_branch_complementary` | 34 | 双分支结果高于单独 horizontal 和 vertical 分支。 |
| `full_mamba_strong_gain_vs_single_axis` | 25 | 完整双分支相对两个单分支均有明显优势。 |
| `horizontal_branch_better` | 1 | 横向分支显著优于纵向分支，常与 colspan / 宽表有关。 |

## 与创新点三的对应关系

- `horizontal_dependency` / `colspan`：对应横向逻辑关系建模。
- `vertical_dependency` / `rowspan`：对应纵向逻辑关系建模。
- `multi_level_header` 与 `hierarchical_span_header`：对应复杂金融表格常见的层次化表头。
- `long_range_dependency`：对应跨多个行列的长距离结构依赖。
- `dual_branch_complementary` 等 axis ablation 标签：可用于说明单分支不足，双分支 Mamba 更适合复杂结构关系建模。

## 文件说明

| 文件 | 说明 |
|---|---|
| `complex_structure_full_5000_metadata.jsonl` | 全部复杂结构样本元数据 |
| `complex_structure_curated_500_metadata.jsonl` | 代表性 500 样本元数据 |
| `complex_structure_curated_500_metadata.csv` | 便于人工筛选的 CSV 表 |
| `complex_structure_curated_500_metadata_with_images.jsonl` | 含复制后图片路径的 500 样本元数据 |
| `images/` | 500 个代表性复杂表格图片 |
| `figures/` | 自动导出的代表性拼图和单图 |
| `dataset_summary.json` | 统计摘要 |

## 论文表述建议

```text
为验证模型对复杂结构关系的建模能力，我们从验证集中额外构建了一个复杂结构特殊子集，覆盖多级表头、跨行跨列单元格以及长距离结构依赖。该子集根据 HTML 结构 token 中的 thead 层级、跨 token 解析得到的 rowspan/colspan、行列规模和序列长度自动筛选，并结合 horizontal/vertical axis ablation 结果分析双分支 Mamba 的贡献。
```
