# Phase9-A Candidate Reranking Smoke Experiment

Date: 2026-06-24
Base checkpoint: `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams`
Validation split: `train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl`

## Goal

Phase9-A tested a conservative candidate-aware span reranking idea without retraining.

Instead of blindly suppressing all low-margin span cells, the `margin_shape` mode only suppresses a span cell when:

1. The span cell has low margin over plain `<td></td>`.
2. Replacing it with `<td></td>` does not make row-width consistency worse.

This is a first approximation of candidate-aware reranking using the existing Phase8 calibration script.

## Results

Current best reference:

| Model | TEDS | structure_acc |
|---|---:|---:|
| Phase5-E full Mamba | 0.8308172189543931 | 0.693 |

Phase9-A margin-shape results:

| Mode | Margin threshold | TEDS | structure_acc | span suppressions | Decision |
|---|---:|---:|---:|---:|---|
| margin_shape | 0.03 | 0.8298878196145485 | 0.691 | 6 | rejected |
| margin_shape | 0.05 | 0.8299534341067694 | 0.691 | 10 | rejected |
| margin_shape | 0.10 | 0.8295047777243713 | 0.691 | 18 | rejected |
| margin_shape | 0.30 | 0.8280996998208352 | 0.685 | 47 | rejected |
| margin_shape | 0.50 | 0.8205404204409094 | 0.675 | 102 | rejected |

## Interpretation

- Adding row-width protection reduces the damage compared with unconstrained margin suppression, but it still does not improve over Phase5-E.
- The best Phase9-A result is threshold 0.05 with TEDS 0.829953, still below 0.830817.
- As the number of suppressed span cells grows, both TEDS and structure accuracy drop.
- This confirms that replacing span cells with plain cells is too coarse, even when guarded by simple table-shape consistency.

## Decision

Phase9-A is rejected as a new best method.

Current best remains:

```text
Phase5-E full Mamba
TEDS: 0.8308172189543931
structure_acc: 0.693
```

## Next Direction

Do not continue single-action span deletion/suppression.

Recommended Phase9-B:

```text
Top-k structure token candidate decoding
```

Instead of only comparing `span cell` vs `<td></td>`, Phase9-B should use the model's top-k token alternatives around uncertain structural positions and reconstruct complete candidate sequences.

Suggested design:

1. Decode top-k token alternatives for each structural time step.
2. Identify local uncertain regions around `<td`, `rowspan`, `colspan`, `>`, and `</td>` tokens.
3. Generate a small beam of valid candidate token sequences.
4. Score candidates with model log probability plus structural consistency penalties.
5. Evaluate on Phase5-E first, without retraining.

This is more promising than hard span suppression because it can correct wrong span type/value while preserving true spans.
