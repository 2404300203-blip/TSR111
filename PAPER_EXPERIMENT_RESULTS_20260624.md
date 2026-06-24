# DBM-SLANet Paper-Ready Experiment Results

Date: 2026-06-24
Project path: `/root/autodl-tmp/PaddleOCR`

## 1. Current Best Result

The current best model is Phase5-E:

```text
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams
```

External validation on the fixed 1000-sample PubTabNet split:

```text
Structure-TEDS: 0.8308172189543931
structure_acc: 0.693
fps: 9.565019614337581
```

This is the model that should currently be used as the best checkpoint for reporting and further experiments.

## 2. Main Comparison Table

All models below were evaluated on the same validation split:

```text
train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl
```

| Model | Key change | Structure-TEDS | structure_acc | fps | Decision |
|---|---|---:|---:|---:|---|
| Original SLANet baseline | Original SLAHead | 0.8280000534464453 | 0.668 | 18.938210592258457 | baseline |
| Phase4 Stage6 lite Mamba | Lite full-Mamba head, low-LR continuation | 0.8294241307122223 | 0.691 | 12.000899824853926 | improved |
| Phase5-D full Mamba | Full Mamba, lr=1e-5 | 0.8306299579222143 | 0.693 | not recorded | improved |
| Phase5-E full Mamba | Full Mamba, lr=5e-6 | 0.8308172189543931 | 0.693 | 9.565019614337581 | current best |
| Phase5-F full Mamba | Full Mamba, lr=2e-6 | 0.8285197072732923 | 0.694 | not recorded | rejected |
| Phase6-D TEDS-selected continuation | TEDS-based checkpoint selection, lr=2e-6 | 0.828765692993526 | 0.689 | not recorded | rejected |

## 3. Gains Over Baseline

Compared with the original SLANet baseline:

```text
Structure-TEDS gain: 0.8308172189543931 - 0.8280000534464453 = +0.0028171655079478
structure_acc gain: 0.693 - 0.668 = +0.025
```

Interpretation:

- The proposed Mamba-enhanced head improves exact structure prediction more clearly than TEDS.
- The TEDS gain is positive but still modest.
- The speed decreases from 18.94 fps to 9.57 fps, so the paper should present this as an accuracy-oriented structural enhancement, not a speed improvement.

## 4. Ablation Interpretation

### 4.1 Lite Mamba is useful

Phase4 Stage6 improves over the original SLANet baseline:

```text
0.8294241307122223 vs 0.8280000534464453
```

This supports the claim that adding a Mamba-style structural modeling component can improve table structure recognition.

### 4.2 Full Mamba is better than lite Mamba

Phase5-E improves over Phase4 Stage6:

```text
0.8308172189543931 vs 0.8294241307122223
```

This supports the claim that stronger sequence modeling capacity helps table structure decoding.

### 4.3 Lower learning rate is not always better

Phase5-F used a lower learning rate than Phase5-E and reached higher exact-match accuracy:

```text
Phase5-E structure_acc: 0.693
Phase5-F structure_acc: 0.694
```

But Phase5-F had worse TEDS:

```text
Phase5-E TEDS: 0.8308172189543931
Phase5-F TEDS: 0.8285197072732923
```

This proves that exact-match accuracy alone is not a reliable selection metric for this task.

### 4.4 TEDS-based checkpoint selection is necessary

Phase6-D did not improve the final score, but it validated the mechanism:

```text
step 6000: acc 0.689, TEDS 0.828766 -> saved as best
step 7500: acc 0.695, TEDS 0.828531 -> not saved
```

This is important because it shows that the training pipeline can now select checkpoints according to the real target metric.

## 5. Phase6 Error Analysis

Phase6 compared Phase5-E and Phase5-F on 1000 validation samples.

Summary:

```text
Same TEDS: 990 / 1000
Phase5-F worse: 7 samples
Phase5-F better: 3 samples
Overall mean delta: -0.002297511681
Exact matches: Phase5-E 693, Phase5-F 694
```

The main failure mode is span hallucination:

- Phase5-F sometimes adds wrong `rowspan` or `colspan`.
- A few catastrophic errors dominate the TEDS drop.
- The model can improve some hard span cases, but it also invents spans in originally simple tables.

Worst regressions:

| Sample | Phase5-E TEDS | Phase5-F TEDS | Main issue |
|---:|---:|---:|---|
| 192 | 1.000000 | 0.107914 | false `rowspan` |
| 778 | 1.000000 | 0.134454 | false `colspan` |
| 685 | 1.000000 | 0.277778 | false `rowspan` |

Best improvements:

| Sample | Phase5-E TEDS | Phase5-F TEDS | Main fix |
|---:|---:|---:|---|
| 178 | 0.445378 | 1.000000 | recovered correct `colspan` |
| 49 | 0.737864 | 1.000000 | recovered correct `rowspan` |
| 762 | 0.955752 | 1.000000 | corrected extra rows/spans |

## 6. Paper Story So Far

The current experimental story is:

1. Original SLANet is the baseline.
2. Adding a Mamba-enhanced table head improves structure recognition.
3. Increasing Mamba capacity from lite to full further improves TEDS.
4. Exact-match accuracy and TEDS can disagree.
5. TEDS-based checkpoint selection is implemented and should be used for future training.
6. The remaining bottleneck is span prediction stability, especially avoiding false `rowspan/colspan`.

This is already a coherent experimental chain, but the numerical gain is still modest for an SCI-level submission.

## 7. Recommended Phase7

Do not continue blind low-learning-rate training.

Recommended Phase7:

```text
Phase7: Span-aware training and decoding stabilization
```

Target:

```text
Reduce catastrophic false rowspan/colspan predictions without suppressing true spans.
```

Proposed changes:

1. Add span-token weighted loss:
   - Increase loss weight for structure tokens containing `rowspan` and `colspan`.
   - Keep normal `<td></td>` loss unchanged.
   - Goal: make span decisions more deliberate.

2. Add span false-positive penalty:
   - Penalize predicted span tokens when the ground truth has plain `<td></td>`.
   - This directly targets the Phase5-F failure mode.

3. Keep TEDS-based checkpoint selection:
   - Use `Metric.main_indicator: teds`.
   - Do not select by exact-match acc alone.

4. Run a conservative Phase7-A first:
   - Init from Phase5-E.
   - 10000 samples.
   - lr=2e-6.
   - Validate with TEDS.

Acceptance criteria:

```text
Phase7-A TEDS >= 0.8308172189543931
structure_acc >= 0.693
No increase in catastrophic span hallucination cases
```

If Phase7-A improves:

```text
Scale to Phase7-B with 30000 samples.
```

If Phase7-A drops:

```text
Keep Phase5-E as final model and write Phase7 as negative ablation/error analysis.
```

## 8. Immediate Next Action

Implement Phase7-A with a small, controlled span-aware loss modification.

The implementation should avoid changing the baseline path and should use a new config/output directory:

```text
configs/table/DBM_SLANet_phase7a_span_aware_10000_lr2e6_from_phase5e_codex.yml
output/E7a_span_aware_10000_lr2e6_from_phase5e_20260624
```

