# DBM-SLANet Paper-Ready Experiment Tables

Date: 2026-06-24  
Project: `/root/autodl-tmp/PaddleOCR`  
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## 1. Executive Summary

当前实验已经形成一条可以写入论文的完整链路：

1. 以原始 SLANet 作为 baseline。
2. 引入 Mamba-enhanced table head 后，结构识别指标提升。
3. Full Mamba 比 Lite Mamba 更好，当前最佳纯模型为 Phase5-E。
4. `structure_acc` 与 TEDS 多次出现不一致，说明不能只按 exact-match accuracy 选择模型。
5. Span-aware loss、简单校准和 gating 没有带来稳定收益，可作为负向消融和误差分析。
6. Phase9-C 的 cell-pattern candidate reranking 在后处理条件下取得当前最高 TEDS。

需要注意：目前提升是真实且一致的，但幅度仍偏小。论文表述应强调“结构建模与候选重排带来的稳定增益和误差机制分析”，不要夸大为大幅性能突破。

## 2. Main Result Table

| ID | Method | Main Change | TEDS | structure_acc | FPS | Decision |
|---|---|---|---:|---:|---:|---|
| E0 | Original SLANet baseline | Original SLAHead | 0.8280000534 | 0.668 | 18.9382 | Baseline |
| E4 | Phase4 Stage6 Lite Mamba | Lite Mamba structural head | 0.8294241307 | 0.691 | 12.0009 | Improved |
| E5-D | Phase5-D Full Mamba | Full Mamba, lr=1e-5 | 0.8306299579 | 0.693 | N/A | Improved |
| E5-E | Phase5-E Full Mamba | Full Mamba, lr=5e-6 | 0.8308172190 | 0.693 | 9.5650 | Best pure model |
| E5-F | Phase5-F Full Mamba | Full Mamba, lr=2e-6 | 0.8285197073 | 0.694 | N/A | Rejected |
| E6-D | TEDS-selected continuation | Checkpoint selection by TEDS | 0.8287656930 | 0.689 | N/A | Rejected |
| E7-A | Span-aware loss | Span token weighting + false-positive penalty | 0.8280026826 | 0.697 | 8.6990 | Rejected |
| E7-B | Weak span-aware loss | Weaker span weighting | 0.8280026826 | 0.697 | 11.0159 | Rejected |
| E9-C | Cell-pattern reranking | Full cell-pattern candidate reranking | 0.8316667671 | 0.688 | N/A | Best TEDS with post-processing |
| E9-D | Gated reranking | Occupancy-badness gate for E9-C | 0.8312027446 | 0.688 | N/A | Ablation |

## 3. Best Model Comparison

| Setting | Method | TEDS | structure_acc | Compared Target | TEDS Gain | Acc Gain |
|---|---|---:|---:|---|---:|---:|
| Pure model | Phase5-E Full Mamba | 0.8308172190 | 0.693 | Baseline | +0.0028171655 | +0.025 |
| With post-processing | Phase9-C Reranking | 0.8316667671 | 0.688 | Baseline | +0.0036667136 | +0.020 |
| Post-processing only | Phase9-C Reranking | 0.8316667671 | 0.688 | Phase5-E | +0.0008495481 | -0.005 |

Recommended reporting:

- 如果只报告模型结构改进：使用 Phase5-E。
- 如果允许报告推理后处理：使用 Phase9-C 作为最高 TEDS。
- 论文主表可以同时列出两者，避免把后处理增益混进纯模型增益里。

## 4. Phase9-C Reranking Ablation

Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`

| top-k | shape_lambda | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---:|---|
| 3 | 0.0 | 0.8292348461 | 0.690 | 23 | Rejected |
| 3 | 0.5 | 0.8316667671 | 0.688 | 26 | Best |
| 3 | 1.0 | 0.8315895138 | 0.689 | 28 | Improved |
| 3 | 2.0 | 0.8312144700 | 0.690 | 36 | Improved |

Best setting:

```text
topk=3
shape_lambda=0.5
TEDS=0.8316667670595694
structure_acc=0.688
changed_samples=26
```

Diagnostic result:

| Category | Count |
|---|---:|
| Changed samples | 26 |
| Better TEDS | 10 |
| Worse TEDS | 4 |
| Same TEDS | 12 |
| Mean TEDS delta | +0.0008495481 |

Interpretation:

- Phase9-C 是当前第一个超过 Phase5-E 的后处理方法。
- 它提升了平均 TEDS，但牺牲了一部分 exact structure accuracy。
- 说明候选结构重排对 tree-edit similarity 有帮助，但还不够稳定。

## 5. Phase9-D Gating Ablation

Fixed setting:

```text
topk=3
shape_lambda=0.5
```

| gate_badness | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---|
| 0 | 0.8316667671 | 0.688 | 26 | Same as Phase9-C |
| 1 | 0.8316667671 | 0.688 | 26 | Same as Phase9-C |
| 2 | 0.8312027446 | 0.688 | 22 | Lower TEDS |
| 3 | 0.8309510115 | 0.690 | 18 | Lower TEDS, higher acc |
| 5 | 0.8309741571 | 0.691 | 12 | Lower TEDS, higher acc |

Conclusion:

- 简单 badness gate 不能超过 Phase9-C。
- gate 越强，修改样本越少，exact accuracy 略恢复，但 TEDS 下降。
- 这个实验可以作为负向消融：单阈值 gating 不足以区分有益修改和有害修改。

## 6. Negative Ablations Worth Reporting

| Experiment | Observation | Paper Value |
|---|---|---|
| Phase5-F lower LR | `structure_acc` 最高到 0.694，但 TEDS 下降到 0.8285197073 | 证明 exact accuracy 与 TEDS 不一致 |
| Phase6-D TEDS selection | 没有提升最终 TEDS，但验证了按 TEDS 保存 checkpoint 的机制 | 证明训练选择指标需要对齐评估指标 |
| Phase7-A/B span-aware loss | accuracy 到 0.697，但 TEDS 回到约 baseline | 说明简单 span 加权会增加结构树编辑风险 |
| Phase8 calibration | 简单 margin/confidence 抑制无效 | 说明错误 span 往往不是低置信度错误 |
| Phase9-D gating | TEDS 与 accuracy 存在 trade-off | 说明候选重排需要更精细的 scoring |

## 7. Paper Claims That Are Currently Supported

可以谨慎写的贡献点：

1. 提出一个 Mamba-enhanced table structure head，用于增强表格 HTML token 序列的结构依赖建模。
2. 在固定 PubTabNet 1000-sample validation split 上，Phase5-E 相比原始 SLANet baseline 提升 TEDS 和 structure accuracy。
3. 发现 exact structure accuracy 与 TEDS 存在明显不一致，说明表格结构识别不能只依赖 exact-match 指标。
4. 通过 span/cell-pattern diagnostics 证明 `rowspan/colspan` 幻觉是当前主要错误来源。
5. 提出 cell-pattern candidate reranking，在后处理条件下进一步提升 TEDS。

不建议现在写的过强表述：

1. 不要说“大幅提升”。
2. 不要说“解决了复杂跨行跨列识别问题”。
3. 不要说“速度更快”，因为当前最佳模型 FPS 低于 baseline。
4. 不要只报 Phase9-C，而不说明它是 post-processing result。

## 8. SCI-3-Oriented Gap Analysis

当前距离 SCI 3 区水平，主要差在三类证据：

| Gap | Current Status | Needed Evidence |
|---|---|---|
| Gain size | 有提升，但 TEDS 增益约 +0.0037 | 需要更大验证集或更多数据集证明稳定性 |
| Generalization | 当前主要是 PubTabNet 1000 split | 建议补 PubTabNet full val 或至少 3k/5k split |
| Baseline breadth | 目前主要对比 original SLANet | 建议补 SLANet+同训练设置、Phase4/5 消融、后处理开关对比 |
| Statistical credibility | 有样本级 diagnostics，但没有置信区间 | 建议补 bootstrap CI 或 paired significance test |
| Qualitative evidence | 有 worst/best sample id | 建议导出可视化案例，展示 span 修复和失败样例 |

## 9. Recommended Next Step

下一步建议进入 Phase10-B：补强论文证据，而不是继续堆小启发式。

优先级：

1. 在 3000 或 5000 样本验证集上复评 E0、Phase5-E、Phase9-C。
2. 对 E0 vs Phase5-E、Phase5-E vs Phase9-C 做 paired bootstrap，输出 TEDS 置信区间。
3. 导出 6-8 个 qualitative cases：
   - 2 个 baseline 错、Phase5-E 对；
   - 2 个 Phase5-E 错、Phase9-C 改善；
   - 2 个 Phase9-C regression；
   - 1-2 个 span hallucination failure。
4. 把最终实验表、消融表、案例图整理进论文草稿。

如果 3000/5000 样本上仍保持正向提升，这条线就更接近可投稿版本；如果增益消失，则应把论文重心转向“诊断与稳健后处理方法”，继续优化 candidate scoring。
