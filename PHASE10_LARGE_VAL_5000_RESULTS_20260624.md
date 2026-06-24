# Phase10 Large Validation Results on 5000 Samples

Date: 2026-06-24

## Split

The 5000-sample validation split was generated from the available PubTabNet subset:

```text
/root/autodl-tmp/pubtabnet/pubtabnet_part_91150/pubtabnet_part_91150.jsonl
```

Source split statistics:

| Split | Samples |
|---|---:|
| train | 91150 |
| val | 9115 |

Validation file:

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

## Main Results

| Method | TEDS | structure_acc | FPS | Samples | Decision |
|---|---:|---:|---:|---:|---|
| Original SLANet baseline | 0.8330900340 | 0.6898 | 23.4966 | 5000 | baseline |
| Phase5-E full Mamba | 0.8439856134 | 0.7182 | 9.6547 | 5000 | final main model |

## Gain

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.8330900340 | 0.8439856134 | +0.0108955794 |
| structure_acc | 0.6898 | 0.7182 | +0.0284 |

## Paired Bootstrap

| Comparison | Delta | 95% CI | One-sided P(delta <= 0) | Better/Worse/Same |
|---|---:|---:|---:|---:|
| Phase5-E - baseline | +0.0108955794 | [+0.0059055140, +0.0161684765] | 0.0000 | 497/424/4079 |

## Interpretation

Phase5-E full Mamba shows a stable and statistically supported improvement over the original SLANet baseline on the 5000-sample validation split.

This is now stronger than the earlier 1000-sample and 3000-sample evidence:

| Split | Baseline TEDS | Phase5-E TEDS | Gain | 95% CI |
|---|---:|---:|---:|---:|
| 1000 | 0.8280000534 | 0.8308172190 | +0.0028171655 | [-0.0067156136, +0.0123883227] |
| 3000 | 0.8333671727 | 0.8413762999 | +0.0080091272 | [+0.0013414985, +0.0145446136] |
| 5000 | 0.8330900340 | 0.8439856134 | +0.0108955794 | [+0.0059055140, +0.0161684765] |

The 5000-sample result supports using Phase5-E as the paper's primary reported method.

## Updated Paper Positioning

Recommended main claim:

```text
On a 5000-sample PubTabNet validation split, the proposed full-Mamba table structure head improves the original SLANet baseline from 0.83309 to 0.84399 TEDS and from 0.6898 to 0.7182 structure accuracy.
```

Recommended caveat:

```text
The improvement comes with lower inference speed, decreasing from 23.50 FPS to 9.65 FPS, so the method should be positioned as an accuracy-oriented structural enhancement rather than an efficiency-oriented model.
```

## Data Files on Server

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/baseline_vs_phase5e_val5000_bootstrap.json
output/phase10_large_val_20260624/baseline_val5000_eval.log
output/phase10_large_val_20260624/phase5e_val5000_eval.log
```

## Next Step

Use Phase5-E as the main model and move to paper preparation:

1. Update the paper-ready experiment table with the 3000/5000 validation results.
2. Generate qualitative examples for baseline vs Phase5-E.
3. Keep Phase9-C as an exploratory post-processing ablation, not the final method.
