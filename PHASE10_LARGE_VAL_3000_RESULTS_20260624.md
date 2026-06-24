# Phase10 Large Validation Results

Date: 2026-06-24

## Split

The validation split was regenerated from:

```text
/root/autodl-tmp/pubtabnet/pubtabnet_part_91150/pubtabnet_part_91150.jsonl
```

The source file contains:

| Split | Samples |
|---|---:|
| train | 91150 |
| val | 9115 |

The 3000-sample validation file is:

```text
train_data/table/pubtabnet/phase10_val_3000.jsonl
```

## Main Results on 3000 Validation Samples

| Method | TEDS | structure_acc | FPS | Samples | Decision |
|---|---:|---:|---:|---:|---|
| Original SLANet baseline | 0.8333671727 | 0.6873333333 | 22.1553 | 3000 | baseline |
| Phase5-E full Mamba | 0.8413762999 | 0.7146666667 | 9.6844 | 3000 | best main model |
| Phase9-C cell-pattern reranking | 0.8408456063 | 0.7120000000 | 9.7928 | 3000 | not better than Phase5-E |

## Phase5-E vs Baseline: Paired Bootstrap

| Comparison | Delta | 95% CI | One-sided P(delta <= 0) | Better/Worse/Same |
|---|---:|---:|---:|---:|
| Phase5-E - baseline | +0.0080091272 | [+0.0013414985, +0.0145446136] | 0.0081 | 290/267/2443 |

Interpretation:

Phase5-E shows a statistically supported positive gain over the original SLANet baseline on the 3000-sample validation split. This is much stronger evidence than the earlier 1000-sample pilot result.

## Phase9-C vs Phase5-E: Paired Bootstrap

| Comparison | Delta | 95% CI | One-sided P(delta <= 0) | Better/Worse/Same |
|---|---:|---:|---:|---:|
| Phase9-C - Phase5-E | -0.0005306936 | [-0.0016815485, +0.0004269174] | 0.8370 | 23/17/2960 |

Interpretation:

Phase9-C improved TEDS on the earlier 1000-sample split, but the gain does not hold on the 3000-sample validation split. Therefore, Phase9-C should be reported as an exploratory post-processing ablation, not as the main final method.

## Paper Positioning Update

Recommended main claim:

```text
The proposed full-Mamba table structure head improves the original SLANet baseline on a 3000-sample PubTabNet validation split, increasing TEDS from 0.83337 to 0.84138 and structure accuracy from 0.6873 to 0.7147.
```

Recommended secondary claim:

```text
Candidate reranking is useful as an error-analysis direction, but the current Phase9-C heuristic is not robust enough to be part of the final method.
```

## Data Files on Server

```text
train_data/table/pubtabnet/phase10_val_3000.jsonl
train_data/table/pubtabnet/phase10_val_5000.jsonl
output/phase10_large_val_20260624/baseline_val3000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val3000_sample_teds.jsonl
output/phase10_large_val_20260624/phase9c_val3000_topk3_lambda05_diag.jsonl
output/phase10_large_val_20260624/baseline_vs_phase5e_val3000_bootstrap.json
output/phase10_large_val_20260624/phase5e_vs_phase9c_val3000_bootstrap.json
```

## Next Step

Run Phase5-E on the 5000-sample validation split and, if the positive gain remains, use Phase5-E as the paper's primary reported model. Phase9-C should not be emphasized unless a stronger, more robust reranking strategy is developed.
