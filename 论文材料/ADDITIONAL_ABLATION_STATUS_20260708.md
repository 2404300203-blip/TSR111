# 补充对比实验与消融实验状态核验（2026-07-08）

## 当前结论

根据服务器 `/root/autodl-tmp/PaddleOCR` 的代码、配置和输出结果核验，目前已经补齐最关键的 C/D 轴向消融；另外补算了 `Empty-td` 诊断指标，并尝试计算 `Cell Presence F1`。需要注意：`Cell Presence F1` 当前结果存在标签泄漏风险，不能直接作为论文主证据。

## 图中缺口逐项状态

| 项目 | 当前状态 | 是否可写进论文 | 说明 |
|---|---|---:|---|
| Cell Token only vs Full Mamba | 未完成 | 暂不可 | 服务器上有 `DBM_SLANet_cell_token_mlp_codex.yml`，但未找到对应 `E2_cell_token_mlp` 权重。要做正式对比需要补训 CellTokenMLPHead。 |
| Geometry feature ablation | 未完成 | 暂不可 | 当前代码确认使用 8 维几何特征：`x_center, y_center, width, height, area, aspect, left, top`。但未跑去掉部分特征的 ablation。 |
| `lambda_pres=0` vs `lambda_pres=0.2` | 未完成 | 暂不可 | 当前主模型使用 `cell_presence_weight=0.2`。要证明 auxiliary loss 必要性，需要训练一版 `cell_presence_weight=0.0`。 |
| Cell Presence F1 | 已尝试，结果不可直接使用 | 不建议 | Phase5-E CTA-500 上算出 F1=1.0，但 eval 前向仍传入 GT targets，且 Mamba 分支使用 `bbox_mask` 作为 mask，指标有标签泄漏风险。该结果只能说明脚本链路跑通，不能作为论文证据。 |
| Bidirectional=false vs true | 部分已有 | 可作为弱证据，需谨慎 | 早期 lite/full 配置里有 `bidirectional=false/true` 差异，但同时改变了 `mamba_d_state`、`mamba_expand` 等容量参数，不是严格单变量消融。严格结论需要同容量重跑。 |
| Inference-off auxiliary branch | 未完成 | 暂不可 | 当前 fused cell branch 不直接写入 HTML logits，理论上可以测试推理关闭分支恢复速度；但代码还没加 `disable_aux_in_inference` 开关，也没跑速度对比。 |
| Empty-td rate / Empty-cell count | 已补算 | 可作为诊断，不适合作主线 | CTA-500 上 Phase5-E 的 TEDS/Exact 提升明显，但 empty `<td>` 数量没有下降。因此该指标更适合 OCR-aware matching 分析，不适合作 DBM 主创新指标。 |

## 已完成的关键补充实验

### CTA-500 轴向消融

| Model | TEDS | Exact / Acc | FPS |
|---|---:|---:|---:|
| Original SLANet baseline | 0.484335 | 0.2080 | 16.3023 |
| Exp C: Horizontal-only Mamba | 0.627637 | 0.4200 | 8.6467 |
| Exp D: Vertical-only Mamba | 0.625761 | 0.4160 | 9.7617 |
| Exp E: Dual-Branch Mamba | 0.640673 | 0.4320 | 8.1429 |

可用结论：单轴 Mamba 均明显强于 baseline，Dual-Branch 取得最高 TEDS 和 Exact，说明水平/垂直依赖建模存在互补性。

详细文件：

- `/Users/table/TSR/论文材料/CTA500_AXIS_ABLATION_RESULTS_20260707.md`
- `/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_by_tag.csv`
- `/Users/table/TSR/remote_results/cta500_axis_ablation_20260707/cta500_axis_sample_delta.jsonl`

### Empty-td 诊断指标

| Model | Pred empty td total | GT empty td total | Pred empty td / image | Empty td / GT ratio |
|---|---:|---:|---:|---:|
| baseline | 57685 | 55966 | 115.370 | 1.0307 |
| horizontal_only | 58330 | 55966 | 116.660 | 1.0422 |
| vertical_only | 58315 | 55966 | 116.630 | 1.0420 |
| dual_phase5e | 58148 | 55966 | 116.296 | 1.0390 |

解释：Phase5-E 的结构 TEDS 提升不是来自减少 empty `<td>`。这个指标不能支撑 DBM 主创新，但可用于说明 OCR-aware matching 与结构建模是两条不同问题线。

文件：

- `/Users/table/TSR/remote_results/cta500_aux_metrics_20260708/cta500_empty_td_summary.json`
- `/Users/table/TSR/remote_results/cta500_aux_metrics_20260708/cta500_empty_td_summary.csv`

## Cell Presence F1 结果处理建议

已运行：

```text
scripts/eval_cell_presence_codex.py
configs/table/DBM_SLANet_phase5e_eval_cta500_codex.yml
```

输出：

```text
positions: 131996
positives: 49682
positive_rate: 0.376390
dropped_positions_due_to_length_mismatch: 1964
threshold_0.5 F1: 1.0
```

该结果不应写成“auxiliary head 完美学习 cell 表示”。原因：

1. eval 前向调用 `model(images, data=batch[1:])`，仍把 GT `bbox_masks` 传入 head。
2. `CellFullMambaHead.forward()` 使用 `targets[2]` 生成 mask，参与 Mamba 编码。
3. F1 在 0.05 到 0.95 所有阈值均为 1.0，且 `score_mean` 与 positive rate 高度一致，表现不符合独立预测指标。

严谨做法：新增一个真正 inference-only presence 评估口径，即推理时不传 `targets` 或不让 auxiliary branch 访问 `bbox_mask`，再统计 predicted cell presence 与 GT cell presence 的对齐结果。

## 下一步优先级

1. **最高优先级：补 `lambda_pres=0` vs `0.2`**
   这是直接证明 auxiliary loss 是否必要的核心消融。需要训练一版同配置、仅 `cell_presence_weight=0.0` 的 Phase5-E。

2. **第二优先级：补 Cell Token only vs Full Mamba**
   用来拆分“几何 token”与“Mamba 关系建模”的贡献。需要训练或找回 `E2_cell_token_mlp` 权重。

3. **第三优先级：补严格 bidirectional false vs true**
   必须保持 `d_state/expand/dropout/train data/lr/checkpoint` 一致，只改 `bidirectional`。

4. **可选：inference-off auxiliary branch**
   如果论文要强调速度，可以加开关验证关闭 auxiliary 分支是否恢复推理速度；如果只强调精度，这项优先级低。

## 当前可直接写进论文的实验支撑

可直接使用：

- PubTabNet-5000 baseline vs Phase5-E。
- CTA-500 baseline vs Phase5-E。
- CTA-500 C/D/E 轴向消融。
- CTA-500 按复杂结构标签分解。
- CTA-500 complexity bucket robustness。

谨慎使用：

- Empty-td count：作为诊断，不作为主结论。
- Cell Presence F1：当前口径不使用。

