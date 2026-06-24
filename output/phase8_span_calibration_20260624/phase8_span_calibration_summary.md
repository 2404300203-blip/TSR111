# Phase8 Span Calibration Experiment Summary

Date: 2026-06-24
Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## Goal

Phase8 tested whether post-hoc span calibration can reduce catastrophic false `rowspan` / `colspan` predictions without retraining the model.

The experiment keeps Phase5-E fixed and only changes decoded structure tokens after inference.

## Implementation

Script:

```text
eval_table_teds_phase8_calibrated_codex.py
```

Implemented calibration modes:

- `none`: exact reproduction of standard Phase5-E evaluation.
- `shape`: suppress span cells only when replacing them with `<td></td>` improves row-width consistency.
- `confidence`: suppress low-confidence span cells.
- `hybrid`: combine confidence and shape rules.
- `margin`: suppress span cells when their probability margin over plain `<td></td>` is small.

Important implementation detail:

The structure dictionary represents span cells as multi-token sequences, for example:

```text
<td +  rowspan="2" + > + </td>
<td +  colspan="2" + > + </td>
```

Therefore Phase8-B calibrates at the cell-sequence level, not at a single-token level.

## Sanity Check

`mode=none` exactly reproduces Phase5-E:

| Mode | Threshold | TEDS | structure_acc | span suppressions |
|---|---:|---:|---:|---:|
| none | 0.0 | 0.8308172189543931 | 0.693 | 0 |

This confirms the script is comparable with previous evaluations.

## Phase8-A: Shape / Confidence Calibration

| Mode | Threshold | TEDS | structure_acc | span suppressions | Decision |
|---|---:|---:|---:|---:|---|
| shape | 0.0 | 0.8308172189543931 | 0.693 | 0 | no effect |
| confidence | 0.90 | 0.8308172189543931 | 0.693 | 0 | no effect |
| confidence | 0.95 | 0.8308172189543931 | 0.693 | 0 | no effect |
| confidence | 0.98 | 0.8308172189543931 | 0.693 | 0 | no effect |
| hybrid | 0.95 | 0.8308172189543931 | 0.693 | 0 | no effect |

Interpretation:

- Simple row-width consistency did not detect useful span corrections.
- Single-token confidence was initially ineffective because span cells are split across multiple tokens.
- After inspecting decoded tokens, Phase8 moved to cell-sequence margin calibration.

## Span Confidence Distribution

From Phase5-E decoded outputs:

```text
span cells: 2410
score min / p01 / p05 / p10 / p50 / p90 / max:
0.1890 / 0.5008 / 0.6118 / 0.7355 / 0.9925 / 0.9996 / 0.99997

margin over plain <td></td> min / p01 / p05 / p10 / p50 / p90 / max:
0.0015 / 0.0975 / 0.3831 / 0.6002 / 0.9872 / 0.9993 / 0.99995
```

This shows that most span predictions are highly confident. Low-margin spans exist, but they are not necessarily wrong.

## Phase8-B: Cell-Sequence Margin Calibration

| Mode | Margin threshold | TEDS | structure_acc | span suppressions | Decision |
|---|---:|---:|---:|---:|---|
| margin | 0.01 | 0.8300003227003581 | 0.692 | 2 | rejected |
| margin | 0.03 | 0.8290709233605135 | 0.690 | 8 | rejected |
| margin | 0.05 | 0.8287830494806412 | 0.689 | 13 | rejected |
| margin | 0.10 | 0.8277324097628077 | 0.689 | 26 | rejected |

Interpretation:

- Increasing margin suppression monotonically reduces TEDS.
- Even the lowest threshold suppresses true span structures and hurts the current best model.
- False spans are not reliably identifiable by low confidence or low plain-cell margin.

## Conclusion

Phase8 post-hoc suppression does not improve Phase5-E.

Current best remains:

```text
Phase5-E full Mamba
TEDS: 0.8308172189543931
structure_acc: 0.693
```

The negative result is still useful: it shows the remaining span errors are not simple low-confidence mistakes. The model is often confident when it is wrong, so naive confidence calibration is insufficient.

## Recommended Next Step

Do not continue direct span suppression.

Recommended Phase9:

```text
Candidate-aware span decoding / constrained beam reranking
```

Instead of deleting predicted spans after decoding, Phase9 should compare multiple structural candidates:

1. Generate top-k alternatives for span-related token positions.
2. Reconstruct candidate table structures.
3. Score candidates with both model probability and table-shape consistency.
4. Select the best candidate by a weighted objective.
5. Evaluate only on Phase5-E first, without retraining.

This is more promising than hard thresholding because it can preserve true low-margin spans while rejecting structurally inconsistent false spans.
