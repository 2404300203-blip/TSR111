# Phase9-C Full Cell-Pattern Candidate Reranking Summary

Date: 2026-06-24
Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## Goal

Phase9-C extends Phase9-B from span-attribute reranking to full cell-pattern reranking.

It compares candidates such as:

```text
<td></td>
<td + span_attr + > + </td>
```

and scores candidate sequences with:

```text
model log probability - shape_lambda * occupancy_badness
```

The occupancy term approximates row/column validity by checking row width consistency, rowspan occupancy, overlap, and out-of-bound spans.

## Results

Current model-only best reference:

| Model | TEDS | structure_acc |
|---|---:|---:|
| Phase5-E full Mamba | 0.8308172189543931 | 0.693 |

Phase9-C results:

| top-k | shape_lambda | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---:|---|
| 3 | 0.0 | 0.8292348460505690 | 0.690 | 23 | rejected |
| 3 | 0.5 | 0.8316667670595694 | 0.688 | 26 | current best TEDS |
| 3 | 1.0 | 0.8315895138406852 | 0.689 | 28 | improved |
| 3 | 2.0 | 0.8312144700134876 | 0.690 | 36 | improved |

## Best Setting

```text
topk=3
shape_lambda=0.5
TEDS=0.8316667670595694
structure_acc=0.688
changed_samples=26
```

Compared with Phase5-E:

```text
TEDS gain: 0.8316667670595694 - 0.8308172189543931 = +0.0008495481051763
structure_acc change: 0.688 - 0.693 = -0.005
```

## Diagnostic Summary for topk=3, shape_lambda=0.5

Changed/delta samples: 26

```text
Better TEDS: 10
Worse TEDS: 4
Same TEDS: 12
Mean delta: +0.0008495481051764
```

Worst regressions:

| sample | Phase5-E TEDS | Phase9-C TEDS | delta |
|---:|---:|---:|---:|
| 673 | 0.3563218391 | 0.0114942529 | -0.3448275862 |
| 776 | 1.0000000000 | 0.8630136986 | -0.1369863014 |
| 309 | 0.0297619048 | 0.0000000000 | -0.0297619048 |
| 454 | 1.0000000000 | 0.9757785467 | -0.0242214533 |

Best improvements:

| sample | Phase5-E TEDS | Phase9-C TEDS | delta |
|---:|---:|---:|---:|
| 52 | 0.1698113208 | 0.4528301887 | +0.2830188679 |
| 168 | 0.5862068966 | 0.8148148148 | +0.2286079183 |
| 205 | 0.6607142857 | 0.8363636364 | +0.1756493506 |
| 220 | 0.4745762712 | 0.6379310345 | +0.1633547633 |
| 374 | 0.4160000000 | 0.5691056911 | +0.1531056911 |
| 803 | 0.1100917431 | 0.2616822430 | +0.1515904999 |

## Interpretation

- Phase9-C is the first post-processing experiment that improves the main TEDS metric over Phase5-E.
- The improvement is modest but real on the fixed 1000-sample validation split.
- The method trades exact-match accuracy for better tree-edit similarity.
- This is useful because previous experiments repeatedly showed exact accuracy and TEDS can disagree.
- The current implementation is slower than standard decoding because it evaluates extra candidate patterns in Python.

## Decision

Phase9-C is the current best TEDS result if post-processing is allowed.

Recommended reporting:

```text
DBM-SLANet + Phase9-C reranking
TEDS: 0.8316667670595694
structure_acc: 0.688
```

Keep Phase5-E as the best pure model checkpoint:

```text
Phase5-E full Mamba
TEDS: 0.8308172189543931
structure_acc: 0.693
```

## Next Step

Phase9-D should refine Phase9-C to reduce regressions while keeping improvements.

Recommended directions:

1. Add a confidence gate so high-confidence exact-looking structures are not changed.
2. Penalize candidates that modify samples with already low occupancy badness.
3. Restrict full cell-pattern expansion to samples with strong row-width inconsistency.
4. Optimize candidate enumeration for speed.
5. Re-evaluate whether TEDS can exceed 0.832 without further hurting exact accuracy.
