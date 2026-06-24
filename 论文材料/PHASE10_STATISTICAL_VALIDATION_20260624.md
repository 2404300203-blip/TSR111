# Phase10 Statistical Validation Summary

Date: 2026-06-24

## Scope

This summary reports paired bootstrap validation on the fixed 1000-sample PubTabNet split:

```text
train_data/table/pubtabnet/phase4_stage5_val_1000.jsonl
```

The original full validation JSONL was not available on the current server, so larger-split validation is still pending.

## Paired Bootstrap Results

| Comparison | A TEDS | B TEDS | Delta | 95% CI of Delta | One-sided P(delta <= 0) | Better/Worse/Same |
|---|---:|---:|---:|---:|---:|---:|
| Phase5-E full Mamba - Original SLANet baseline | 0.8280000534 | 0.8308172190 | +0.0028171655 | [-0.0067156136, +0.0123883227] | 0.2821 | 89/88/823 |
| Phase9-C cell-pattern reranking - Phase5-E full Mamba | 0.8308172190 | 0.8316667671 | +0.0008495481 | [-0.0003619311, +0.0020864866] | 0.0794 | 10/4/986 |

## Interpretation

- Phase5-E shows a positive TEDS gain over the original SLANet baseline, but the 95% confidence interval includes zero on this 1000-sample split.
- Phase9-C also shows a positive mean gain over Phase5-E, with one-sided bootstrap p around 0.08.
- These results are useful as pilot/ablation evidence, but SCI-level claims still need full-val or at least 3000/5000-sample validation after restoring the full PubTabNet validation annotation file.

## Data Files on Server

```text
output/phase10_statistical_validation_20260624/baseline_sample_teds.jsonl
output/phase6_teds_diagnostics_20260624/phase5e_sample_teds.jsonl
output/phase9c_cellpattern_rerank_20260624/topk3_lambda05_diag.jsonl
output/phase10_statistical_validation_20260624/baseline_vs_phase5e_bootstrap.json
output/phase10_statistical_validation_20260624/phase5e_vs_phase9c_bootstrap.json
output/phase10_statistical_validation_20260624/phase10_statistical_validation_summary.md
```

## Next Step

Restore or regenerate the full PubTabNet validation annotation file, then run the same evaluation on a 3000/5000-sample split:

1. Original SLANet baseline.
2. Phase5-E full Mamba.
3. Phase9-C cell-pattern reranking.

If the positive trend remains on the larger split, the result becomes much stronger for paper reporting.
