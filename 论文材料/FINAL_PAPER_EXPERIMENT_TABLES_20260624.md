# DBM-SLANet Final Paper Experiment Tables

Date: 2026-06-24

## 1. Final Main Result

The final recommended paper method is **Phase5-E full Mamba**. Phase9-C is not used as the final method because its post-processing gain did not generalize from the 1000-sample split to the 3000-sample split.

### 5000-Sample PubTabNet Validation

| Method | TEDS | structure_acc | FPS | Samples | Role |
|---|---:|---:|---:|---:|---|
| Original SLANet baseline | 0.8330900340 | 0.6898 | 23.4966 | 5000 | baseline |
| DBM-SLANet / Phase5-E full Mamba | 0.8439856134 | 0.7182 | 9.6547 | 5000 | final main method |

### Main Gain

| Metric | Baseline | Phase5-E | Absolute Gain |
|---|---:|---:|---:|
| TEDS | 0.8330900340 | 0.8439856134 | +0.0108955794 |
| structure_acc | 0.6898 | 0.7182 | +0.0284 |
| FPS | 23.4966 | 9.6547 | -13.8419 |

### Paired Bootstrap on TEDS

| Comparison | Delta | 95% CI | One-sided P(delta <= 0) | Better/Worse/Same |
|---|---:|---:|---:|---:|
| Phase5-E - baseline | +0.0108955794 | [+0.0059055140, +0.0161684765] | 0.0000 | 497/424/4079 |

## 2. Validation-Scale Consistency

| Split | Baseline TEDS | Phase5-E TEDS | TEDS Gain | Baseline Acc | Phase5-E Acc | Acc Gain |
|---|---:|---:|---:|---:|---:|---:|
| 1000 samples | 0.8280000534 | 0.8308172190 | +0.0028171655 | 0.6680 | 0.6930 | +0.0250 |
| 3000 samples | 0.8333671727 | 0.8413762999 | +0.0080091272 | 0.6873 | 0.7147 | +0.0273 |
| 5000 samples | 0.8330900340 | 0.8439856134 | +0.0108955794 | 0.6898 | 0.7182 | +0.0284 |

Interpretation:

- The gain becomes clearer on larger validation splits.
- The 3000- and 5000-sample splits provide stronger evidence than the initial 1000-sample pilot.
- The method is accuracy-oriented and trades speed for better structure recognition.

## 3. Architecture and Training Ablation

| ID | Method | Key Change | TEDS | structure_acc | Split | Decision |
|---|---|---|---:|---:|---|---|
| E0 | Original SLANet baseline | Original SLAHead | 0.8280000534 | 0.668 | 1000 | baseline |
| E4 | Phase4 Stage6 lite Mamba | Lite Mamba structural head | 0.8294241307 | 0.691 | 1000 | improved |
| E5-D | Phase5-D full Mamba | Full Mamba, lr=1e-5 | 0.8306299579 | 0.693 | 1000 | improved |
| E5-E | Phase5-E full Mamba | Full Mamba, lr=5e-6 | 0.8308172190 | 0.693 | 1000 | selected main model |
| E5-F | Phase5-F full Mamba | Lower lr=2e-6 | 0.8285197073 | 0.694 | 1000 | rejected |
| E6-D | TEDS-selected continuation | Checkpoint selection by TEDS | 0.8287656930 | 0.689 | 1000 | rejected |
| E7-A/B | Span-aware loss | Span-token weighting and false-positive penalty | 0.8280026826 | 0.697 | 1000 | rejected |

Interpretation:

- Lite Mamba improves over SLANet, and full Mamba improves further.
- Exact structure accuracy and TEDS can disagree; Phase5-F and Phase7 improve or maintain accuracy but hurt TEDS.
- The final model is selected by TEDS-oriented evidence, not by exact accuracy alone.

## 4. Post-Processing Ablation

| Method | TEDS | structure_acc | Split | Decision |
|---|---:|---:|---|---|
| Phase5-E full Mamba | 0.8308172190 | 0.693 | 1000 | pure model reference |
| Phase9-C reranking | 0.8316667671 | 0.688 | 1000 | exploratory improvement |
| Phase5-E full Mamba | 0.8413762999 | 0.7147 | 3000 | pure model reference |
| Phase9-C reranking | 0.8408456063 | 0.7120 | 3000 | not robust |

Interpretation:

- Phase9-C improved the 1000-sample pilot split but failed to beat Phase5-E on 3000 samples.
- Therefore, post-processing reranking should be reported as an exploratory ablation and error-analysis direction, not as the final method.

## 5. Qualitative Case Summary

| Category | Count |
|---|---:|
| Total 5000-sample validation cases | 5000 |
| Phase5-E better than baseline | 497 |
| Phase5-E worse than baseline | 424 |
| Same TEDS | 4079 |
| Baseline wrong, Phase5-E exact | 310 |
| Baseline exact, Phase5-E wrong | 168 |
| Span-related improvements | 406 |
| Span-related regressions | 360 |

Representative figures are stored in:

```text
论文材料/qualitative_figures/
```

Recommended figure set:

| Role | Sample | Figure |
|---|---:|---|
| Strong improvement | 531 | case_531_success_PMC2898048_011_00.png |
| Strong improvement | 650 | case_650_success_PMC3496644_001_00.png |
| Strong improvement | 885 | case_885_success_PMC3126717_005_00.png |
| Honest failure | 48 | case_48_failure_PMC4225768_004_00.png |
| Honest failure | 414 | case_414_failure_PMC4652242_004_00.png |

## 6. Final Paper Claims

Supported claims:

1. A full-Mamba table structure head improves SLANet for table structure recognition.
2. The improvement is stable on a 5000-sample PubTabNet validation split.
3. The method improves both TEDS and exact structure accuracy.
4. Span-sensitive structures involving `rowspan` and `colspan` are a major source of both improvements and regressions.
5. Simple candidate reranking is not robust enough to serve as the final method.

Claims to avoid:

1. Do not claim speed improvement.
2. Do not claim that span prediction is solved.
3. Do not present Phase9-C as the final method.
4. Do not overstate the result as a large breakthrough; describe it as a statistically supported structural-recognition improvement.
