# Phase9-B Top-k Structure Token Reranking Summary

Date: 2026-06-24
Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## Goal

Phase9-B tested top-k structure token candidate reranking without retraining.

Unlike Phase8 and Phase9-A, this experiment does not only replace span cells with `<td></td>`. It uses top-k alternatives for span attribute tokens such as:

```text
rowspan="2", rowspan="3", colspan="2", colspan="5"
```

The reranker scores candidate sequences with:

```text
model log probability - shape_lambda * row_width_badness
```

## Sanity Check

| top-k | shape_lambda | TEDS | structure_acc | changed_samples |
|---:|---:|---:|---:|---:|
| 3 | 0.0 | 0.8308172189543931 | 0.693 | 0 |

This exactly reproduces Phase5-E, confirming that the reranker is comparable with previous evaluation.

## Results

Current best reference:

| Model | TEDS | structure_acc |
|---|---:|---:|
| Phase5-E full Mamba | 0.8308172189543931 | 0.693 |

Phase9-B top-k reranking:

| top-k | shape_lambda | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---:|---|
| 3 | 0.01 | 0.8308172189543931 | 0.693 | 0 | no effect |
| 3 | 0.05 | 0.8308172189543931 | 0.693 | 0 | no effect |
| 3 | 0.10 | 0.8308172189543931 | 0.693 | 2 | TEDS tie |
| 3 | 0.20 | 0.8308172189543931 | 0.693 | 3 | TEDS tie |
| 3 | 0.50 | 0.8308172189543931 | 0.693 | 5 | TEDS tie |
| 3 | 1.00 | 0.8308172189543931 | 0.694 | 6 | acc improves, TEDS tie |

## Changed Samples at shape_lambda=1.0

| sample | Original token | Reranked token | Base TEDS | Rerank TEDS | Base exact | Rerank exact |
|---:|---|---|---:|---:|---|---|
| 20 | `colspan="5"` | `colspan="4"` | 0.540000 | 0.540000 | false | false |
| 281 | `colspan="5"` | `colspan="6"` | 0.531532 | 0.531532 | false | false |
| 316 | `colspan="3"` | `colspan="4"` | 0.388889 | 0.388889 | false | false |
| 374 | `colspan="2"` | `rowspan="3"` | 0.416000 | 0.416000 | false | false |
| 894 | `colspan="6"` | `colspan="7"` | 1.000000 | 1.000000 | false | true |
| 981 | `colspan="9"` | `colspan="8"` | 0.506849 | 0.506849 | false | false |

## Interpretation

- Top-k reranking can make valid local span-attribute changes.
- The best observed setting, `topk=3, shape_lambda=1.0`, improves exact-match accuracy from 0.693 to 0.694.
- However, Structure-TEDS remains exactly tied with Phase5-E.
- The improvement on exact match is small and does not justify replacing Phase5-E as the main model.
- TEDS is insensitive to some attribute corrections when the tree-edit distance is unchanged, which explains why exact match can improve while TEDS stays flat.

## Decision

Phase9-B is not a new TEDS best, but it is a useful partial positive result.

Current best for main reporting remains:

```text
Phase5-E full Mamba
TEDS: 0.8308172189543931
structure_acc: 0.693
```

Optional reporting note:

```text
Phase9-B top-k reranking preserves TEDS while slightly improving exact structural accuracy to 0.694.
```

## Recommended Next Step

Phase9-C should not only rerank span attribute values. It should generate candidate sequences around full cell patterns:

```text
<td></td>
<td + span_attr + > + </td>
```

and compare candidates with a stronger structural objective, including:

1. row-width consistency,
2. row-span occupancy validity,
3. token log probability,
4. optional penalty for changing already high-confidence exact structures.

This may convert the Phase9-B exact-match improvement into a real TEDS improvement.
