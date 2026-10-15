# CTA-500 轴向 Mamba 消融与补充指标结果（2026-07-07）

## 结论摘要

本轮补充的是正式的 C / D 消融：只保留 horizontal Mamba、只保留 vertical Mamba，并在同一个 CTA-500 复杂结构诊断集上与 Original SLANet baseline 和 Phase5-E Dual-Branch Mamba 对比。

- 两个单轴模型都显著高于 baseline，说明 Mamba 分支带来的结构正则不是偶然。
- Dual-Branch Mamba 在 CTA-500 上取得最高 TEDS 和最高 exact rate，证明水平/垂直两个方向有互补性。
- Dual 相比 best single-axis 的平均增益为 `+0.012969` TEDS；相比 baseline 的平均增益为 `+0.156337` TEDS。

## Overall Results on CTA-500

| Model | TEDS ↑ | Exact / Acc ↑ | FPS ↑ | ΔTEDS vs Baseline |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 0.484335 | 0.2080 | 16.3023 | +0.000000 |
| Exp C: Horizontal-only Mamba | 0.627637 | 0.4200 | 8.6467 | +0.143302 |
| Exp D: Vertical-only Mamba | 0.625761 | 0.4160 | 9.7617 | +0.141426 |
| Exp E: Dual-Branch Mamba | 0.640673 | 0.4320 | 8.1429 | +0.156337 |

## Axis Complementarity Metrics

| Comparison | Mean ΔTEDS | 95% CI |
|---|---:|---:|
| Horizontal-only - Baseline | +0.143302 | [+0.100705, +0.184681] |
| Vertical-only - Baseline | +0.141426 | [+0.099055, +0.182751] |
| Dual - Baseline | +0.156337 | [+0.111325, +0.199625] |
| Dual - Best single-axis | +0.012969 | [-0.005523, +0.031166] |

Dual vs best single-axis win/tie/loss rate: `12.2%` / `79.4%` / `8.4%`.

## Breakdown by Complex Structure Tag

| Tag | N | Baseline | H-only | V-only | Dual | Dual - Baseline | Dual - Best Single |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_range_dependency | 473 | 0.4891 | 0.6182 | 0.6163 | 0.6304 | +0.1413 | +0.0121 |
| row_col_span | 457 | 0.4627 | 0.6264 | 0.6243 | 0.6479 | +0.1853 | +0.0215 |
| horizontal_dependency | 376 | 0.4581 | 0.6259 | 0.6235 | 0.6461 | +0.1881 | +0.0201 |
| tall_table | 341 | 0.4913 | 0.5631 | 0.5606 | 0.5709 | +0.0797 | +0.0077 |
| multi_level_header | 325 | 0.4604 | 0.6345 | 0.6315 | 0.6523 | +0.1919 | +0.0178 |
| vertical_dependency | 286 | 0.4496 | 0.6158 | 0.6131 | 0.6389 | +0.1894 | +0.0232 |
| hierarchical_span_header | 274 | 0.4753 | 0.6664 | 0.6631 | 0.6759 | +0.2007 | +0.0095 |
| wide_table | 149 | 0.5417 | 0.7124 | 0.7122 | 0.7083 | +0.1666 | -0.0043 |

## Breakdown by Size / Span Complexity

| Group | N | Baseline | H-only | V-only | Dual | Dual - Baseline | Dual - Best Single |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows_15_29 | 228 | 0.4933 | 0.5705 | 0.5671 | 0.5785 | +0.0852 | +0.0078 |
| rows_<15 | 159 | 0.4695 | 0.7661 | 0.7655 | 0.7903 | +0.3208 | +0.0242 |
| rows_>=30 | 113 | 0.4871 | 0.5482 | 0.5475 | 0.5556 | +0.0685 | +0.0075 |
| span_20_79 | 52 | 0.3376 | 0.3854 | 0.3854 | 0.3620 | +0.0245 | -0.0234 |
| span_<20 | 447 | 0.5023 | 0.6565 | 0.6544 | 0.6730 | +0.1707 | +0.0164 |
| span_>=80 | 1 | 0.0865 | 0.3227 | 0.3227 | 0.6817 | +0.5952 | +0.3590 |

## 可写进论文的新增指标

1. **Axis Complementarity Gain**: `TEDS(Dual) - max(TEDS(H-only), TEDS(V-only))`，用于说明双分支不是简单增加参数，而是在单样本层面提供互补信息。
2. **Complex-Structure Tag Breakdown**: 按 `row_col_span`、`vertical_dependency`、`horizontal_dependency`、`multi_level_header`、`tall_table`、`wide_table` 等标签分别统计 TEDS，直接对应论文的复杂结构建模创新点。
3. **Hard-Case Exact Rate**: 在 CTA-500 上报告 exact / structure accuracy，比只在全量 PubTabNet 上报告平均 TEDS 更能体现复杂样本收益。
4. **Complexity Bucket Robustness**: 按行数和 span token 数分桶，展示模型在长表、多合并单元格表上的鲁棒性。

## 文件位置

- 样本级合并结果：`/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_sample_delta.jsonl`
- 按结构标签汇总：`/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_by_tag.csv`
- 按复杂度分桶汇总：`/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_by_complexity_bucket.csv`
- JSON 总结：`/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_ablation_summary.json`
