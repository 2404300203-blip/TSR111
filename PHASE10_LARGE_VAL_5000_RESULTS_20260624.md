# Phase10 Large Validation Results on 5000 Samples

Date: 2026-06-24

## Split

Validation split:

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

The split was generated from:

```text
/root/autodl-tmp/pubtabnet/pubtabnet_part_91150/pubtabnet_part_91150.jsonl
```

## Main Results

| Method | TEDS | structure_acc | FPS | Samples | Decision |
|---|---:|---:|---:|---:|---|
| Original SLANet baseline | 0.8330900340 | 0.6898 | 23.4966 | 5000 | baseline |
| Phase5-E full Mamba | 0.8439856134 | 0.7182 | 9.6547 | 5000 | best main model |

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

Phase5-E full Mamba shows a stable and statistically supported improvement over the original SLANet baseline on the 5000-sample PubTabNet validation split.

This is currently the strongest evidence for the paper:

- The TEDS gain is larger than on the 3000-sample split.
- The 95% bootstrap confidence interval is fully positive.
- structure accuracy also improves by +0.0284.
- The trade-off is lower inference speed: 9.65 FPS vs 23.50 FPS.

## Recommended Paper Positioning

Use Phase5-E as the main reported method.

Recommended claim:

```text
On a 5000-sample PubTabNet validation split, the proposed full-Mamba table structure head improves SLANet from 0.8331 to 0.8440 TEDS and from 0.6898 to 0.7182 structure accuracy.
```

Recommended caution:

```text
The method improves structural recognition accuracy but reduces inference speed compared with the original SLANet baseline.
```

## Data Files on Server

```text
output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/baseline_vs_phase5e_val5000_bootstrap.json
output/phase10_large_val_20260624/baseline_val5000_eval.log
output/phase10_large_val_20260624/phase5e_val5000_eval.log
```

## Next Step

Prepare the final paper experiment table and select qualitative examples:

1. Baseline wrong, Phase5-E correct.
2. Baseline correct, Phase5-E wrong.
3. Both wrong on hard rowspan/colspan cases.
4. Representative complex tables where Phase5-E improves TEDS.
