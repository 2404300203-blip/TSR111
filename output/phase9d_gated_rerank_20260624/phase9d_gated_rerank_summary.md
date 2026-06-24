# Phase9-D Gated Cell-Pattern Reranking Summary

Date: 2026-06-24
Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## Goal

Phase9-D tested whether a simple occupancy-badness gate can reduce harmful Phase9-C reranking changes.

The gate only enables full cell-pattern reranking when the original decoded table has occupancy badness above a threshold.

Fixed reranking settings:

```text
topk=3
shape_lambda=0.5
```

## Results

Reference results:

| Method | TEDS | structure_acc | changed_samples |
|---|---:|---:|---:|
| Phase5-E pure model | 0.8308172189543931 | 0.693 | 0 |
| Phase9-C ungated | 0.8316667670595694 | 0.688 | 26 |

Phase9-D gated results:

| gate_badness | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---|
| 0 | 0.8316667670595694 | 0.688 | 26 | same as Phase9-C |
| 1 | 0.8316667670595694 | 0.688 | 26 | same as Phase9-C |
| 2 | 0.8312027446046724 | 0.688 | 22 | lower TEDS |
| 3 | 0.8309510114629975 | 0.690 | 18 | lower TEDS, better acc |
| 5 | 0.8309741571320005 | 0.691 | 12 | lower TEDS, better acc |

## Interpretation

- Simple occupancy-badness gating does not improve the best Phase9-C TEDS.
- Higher gate thresholds reduce the number of changed samples and recover some exact accuracy.
- However, filtering changes also removes useful TEDS-improving rerank decisions.
- The best main metric remains the ungated Phase9-C setting.

## Decision

Phase9-D is not a new best method.

Current best TEDS result remains:

```text
Phase9-C topk=3, shape_lambda=0.5
TEDS: 0.8316667670595694
structure_acc: 0.688
```

Best pure model remains:

```text
Phase5-E full Mamba
TEDS: 0.8308172189543931
structure_acc: 0.693
```

## Reporting Note

Phase9-D can be reported as an ablation showing that naive confidence/structure gating trades TEDS for exact accuracy. This supports the claim that span/cell-pattern errors require more careful candidate scoring than a single threshold gate.

## Recommended Next Step

Do not continue simple badness gating.

Recommended Phase10:

```text
Consolidate experiments and prepare paper tables/figures
```

At this point the experiment chain has:

1. a pure model improvement over baseline,
2. a stronger full-Mamba checkpoint,
3. negative span-loss ablations,
4. negative naive calibration ablations,
5. a positive Phase9-C reranking post-processing result.

The next productive step is to convert these into paper-ready tables, figures, and a narrative rather than keep adding small local heuristics.
