# Table 2 数据复核

复核日期：2026-08-01

## 结论

Table 2 中 `Pres.` 列的设置与实验配置一致；`SLANet`、`Cell Token only`、`Lite Mamba` 以及完整 Full-Mamba 架构的参数量和 FPS 与现有记录一致。但表中 `Full Mamba, one-way` 的参数量/FLOPs 不能继续沿用完整双向模型的数值，`Full Mamba stage6` 的 FLOPs 需要明确 profiling 口径。

## 逐项核验

| 行 | Pres. | 表中 Params | 表中 FLOPs | 表中 FPS | 复核结果 |
|---|---:|---:|---:|---:|---|
| SLANet | – | 1.89M | 5.14G | 23.50 | 一致 |
| Cell Token only | 0.2 | 1.93M | 5.15G | 24.25 | 一致 |
| Lite Mamba | 0.2 | 2.09M | 5.24G | 15.99 | 一致 |
| Full Mamba, no pres. | 0 | 2.45M | 5.41G | 12.06 | 一致；仅关闭 presence-loss 权重，不改变模型结构 |
| Full Mamba, pres. | 0.2 | 2.45M | 5.41G | 12.16 | 一致 |
| Full Mamba, one-way | 0.2 | **2.45M** | **5.41G** | 16.19 | **异常：应重新统计** |
| Full Mamba, zero geom. | 0.2 | 2.45M | 5.41G | 12.32 | 结构不变，参数量一致；FLOPs 仍需标注为 proxy |
| Full Mamba stage4 | 0.2 | 2.45M | 5.41G | 12.20 | 架构量级一致 |
| Full Mamba stage6 | 0.2 | 2.45M | **5.41G** | 12.13 | 原始 Paddle 记录另有 1.86G，需加注释 |
| TEDS-select cont. | 0.2 | 2.45M | 5.41G | 12.22 | 架构量级一致 |
| Span-aware cont. | 0.2 | 2.45M | 5.41G | 12.32 | 架构量级一致 |
| Horizontal Mamba | 0.2 | 2.45M | 5.41G | 11.68 | 参数量一致；需说明其仍实例化双轴模块 |
| Vertical Mamba | 0.2 | 2.45M | 5.41G | 12.00 | 参数量一致；需说明其仍实例化双轴模块 |
| DBM-SLANet | 0.2 | 2.45M | 5.41G | 9.65 | 一致 |

## 应修改的数值

`one-way` 配置在代码中使用 `bidirectional=false` 时不会创建 horizontal/vertical 两个 reverse Mamba block。完整模型的单个 selective-scan block 约含 84,393 个可训练参数，因此 one-way 比完整模型少约 168,786 个参数。按当前两位小数口径，参数量应约为 **2.28M**；FLOPs 应约为 **5.28G**，但这两个值应以同一输入尺寸下重新运行 profiling 的结果为准，不能把 2.45M/5.41G 直接复制过来。

`stage6` 的现有效率记录为 `1.86G*`，来源是 Paddle FLOPs 统计。项目记录已说明 Paddle 会将部分不支持的 recurrent/normalization 等算子计为 0，因此 `1.86G` 与其它 Full-Mamba 行不是同口径。建议正文表采用 `5.41G†` 的结构代理值，并在表注中写明是 Paddle-estimated FLOPs proxy；若展示原始 profiling，则改成 `1.86G*` 并保留警告注释。

## 建议的表注

> Params are trainable parameter counts. FLOPs are Paddle-estimated proxies for an input of 1 × 3 × 488 × 488; unsupported operators may be counted as zero. The stage6 value is not directly comparable with other Full-Mamba variants because of configuration-specific graph tracing.

依据：`remote_results/DBM_SLANet_Recommended_Extra_Metrics_20260710.md` 第 89–102 行；`remote_results/DBM_SLANet_Tables2_3_Expanded_Final_20260710.md` 第 20–33 行；`ppocr/modeling/heads/cell_full_mamba_head.py` 第 175–205 行。

## 差异化展示表（2026-08-01）

为避免把重复的 Params/FLOPs proxy 误读为独立实测值，效率表改用同一 PubTabNet-5000 验证协议下已有的结构指标和 FPS。下表数值来自 extra_structure_metrics_summary.csv；stage4 不纳入此版，因为现有原始汇总没有对应的 Span-F1 记录。

| Method | TEDS | Structure Acc. (%) | Span-F1 | FPS |
|---|---:|---:|---:|---:|
| SLANet | 0.83309 | 68.98 | 0.6738 | 23.50 |
| Cell Token only | 0.83206 | 69.62 | 0.6792 | 24.25 |
| Mamba lambda=0 | 0.83076 | 69.38 | 0.6758 | 12.06 |
| Mamba lambda=0.2 | 0.83445 | 70.16 | 0.6867 | 12.16 |
| Lite Mamba | 0.83665 | 70.84 | 0.6916 | 15.99 |
| Mamba stage6 | 0.84354 | 71.78 | 0.7036 | 12.13 |
| Span-aware | 0.84497 | 71.92 | 0.7027 | 12.32 |
| Horizontal Mamba | 0.84501 | 71.94 | 0.7043 | 11.68 |
| Vertical Mamba | 0.84486 | 71.88 | 0.7028 | 12.00 |
| DBM-SLANet | 0.84399 | 71.82 | **0.7045** | 9.65 |

Span-F1 is the micro F1 for cells with rowspan > 1 or colspan > 1. This table preserves measured differences without inventing per-model FLOPs values.
