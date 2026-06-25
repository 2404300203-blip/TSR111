# 我们的模型在复杂结构特殊数据集上的评估结果

日期：2026-06-25

## 实验设置

| 项目 | 内容 |
|---|---|
| 特殊数据集 | `train_data/table/pubtabnet/complex_structure_special_full_4086_20260625.jsonl` |
| 样本数 | 4086 |
| 模型 | Phase5E / Dual-Branch Full Mamba，`CellFullMambaHead` |
| Checkpoint | `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy` |
| 评估脚本 | `eval_table_teds_diagnostics_codex.py` |
| 输出目录 | `output/complex_structure_special_eval_20260625/` |

## 总体结果

| 方法 | Samples | TEDS | Structure Acc |
|---|---:|---:|---:|
| Baseline | 4086 | 0.802639 | 0.635585 |
| Horizontal-only | 826 | 0.797193 | 0.645278 |
| Vertical-only | 826 | 0.797101 | 0.644068 |
| Phase5E / Dual-Branch Mamba | 4086 | 0.815559 | 0.668380 |

相对 baseline：

- mean TEDS delta: `+0.012920`
- improved samples: `477`
- degraded samples: `408`
- unchanged samples: `3201`

## 按复杂结构标签拆分

| 标签 | 样本数 | Baseline TEDS | Phase5E TEDS | Delta | Horizontal TEDS | Vertical TEDS |
|---|---:|---:|---:|---:|---:|---:|
| `multi_level_header` | 1822 | 0.724655 | 0.749713 | +0.025057 | 0.710414 | 0.710211 |
| `row_col_span` | 2395 | 0.708020 | 0.735122 | +0.027102 | 0.697784 | 0.697625 |
| `horizontal_dependency` | 2053 | 0.723251 | 0.750590 | +0.027338 | 0.704665 | 0.704482 |
| `vertical_dependency` | 1147 | 0.656877 | 0.696345 | +0.039467 | 0.649457 | 0.649457 |
| `hierarchical_span_header` | 1500 | 0.734494 | 0.760003 | +0.025508 | 0.730301 | 0.730053 |
| `long_range_dependency` | 3755 | 0.805002 | 0.816474 | +0.011472 | 0.798774 | 0.798676 |
| `wide_table` | 857 | 0.799081 | 0.811378 | +0.012297 | 0.779171 | 0.779238 |
| `tall_table` | 1829 | 0.763921 | 0.773594 | +0.009673 | 0.748074 | 0.748106 |

## 结果解读

复杂结构特殊数据集上的 Phase5E / Dual-Branch Mamba TEDS 为 `0.815559`，低于此前 5000 全验证集上的 `0.843986`，说明该特殊数据集确实更难，能够作为复杂结构专项评估集使用。

与 baseline 相比，Phase5E 在该复杂子集上的平均 TEDS 仍然提升，说明双分支结构关系建模不是只在普通样本上有效，在多级表头、跨行跨列和长距离依赖样本上也有整体收益。

从标签拆分看，`horizontal_dependency`、`vertical_dependency`、`hierarchical_span_header` 和 `long_range_dependency` 等类别可以直接服务第二个创新点：它们对应复杂结构识别中的横向关系、纵向关系和长距离关系建模难点。

## 输出文件

```text
output/complex_structure_special_eval_20260625/phase5e_complex_full4086_eval.log
output/complex_structure_special_eval_20260625/phase5e_complex_full4086_sample_teds.jsonl
output/complex_structure_special_eval_20260625/phase5e_complex_full4086_summary.json
output/complex_structure_special_eval_20260625/phase5e_complex_full4086_by_tag.csv
output/complex_structure_special_eval_20260625/phase5e_complex_full4086_top_cases.jsonl
```

## 可写入论文的表述

```text
在复杂结构专项评估子集上，本文进一步评估了所提出模型对多级表头、跨行跨列和长距离结构依赖的识别能力。该子集包含 4086 个复杂结构样本。实验结果显示，Dual-Branch Mamba 模型在该子集上取得 0.8156 的 TEDS 和 0.6684 的结构准确率，并相对 baseline 获得整体提升，说明横向和纵向结构关系建模能够提升复杂表格结构识别能力。
```
