# Phase10 Qualitative Case Analysis

Date: 2026-06-24

## Source

Validation split:

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

Diagnostics:

```text
output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl
output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl
```

Full detailed case report on server:

```text
output/phase10_qualitative_cases_20260624/phase10_val5000_qualitative_case_analysis.md
```

## Overall Counts

| Category | Count |
|---|---:|
| Total samples | 5000 |
| Phase5-E better than baseline | 497 |
| Phase5-E worse than baseline | 424 |
| Same TEDS | 4079 |
| Baseline wrong, Phase5-E exact | 310 |
| Baseline exact, Phase5-E wrong | 168 |
| Span-related improvements | 406 |
| Span-related regressions | 360 |

## Representative Improvements

| sample | filename | baseline | Phase5-E | delta | Main observation |
|---:|---|---:|---:|---:|---|
| 531 | PMC2898048_011_00.png | 0.000000 | 1.000000 | +1.000000 | recovered `colspan="3"` rows |
| 650 | PMC3496644_001_00.png | 0.000000 | 1.000000 | +1.000000 | recovered mixed `rowspan`/`colspan` header |
| 885 | PMC3126717_005_00.png | 0.000000 | 1.000000 | +1.000000 | removed false spans and matched true colspan |
| 1455 | PMC5752601_003_01.png | 0.000000 | 1.000000 | +1.000000 | corrected header structure and body colspans |
| 1482 | PMC4709790_002_00.png | 0.000000 | 1.000000 | +1.000000 | recovered repeated `rowspan="2"` pattern |

## Representative Regressions

| sample | filename | baseline | Phase5-E | delta | Main observation |
|---:|---|---:|---:|---:|---|
| 48 | PMC4225768_004_00.png | 1.000000 | 0.000000 | -1.000000 | Phase5-E broke an originally exact span table |
| 414 | PMC4652242_004_00.png | 1.000000 | 0.000000 | -1.000000 | severe span-structure regression |
| 1231 | PMC4772352_003_00.png | 1.000000 | 0.000000 | -1.000000 | false structure change on an exact baseline case |
| 1515 | PMC3462120_006_00.png | 1.000000 | 0.000000 | -1.000000 | span-related exact-match break |
| 1818 | PMC6032105_005_00.png | 1.000000 | 0.000000 | -1.000000 | span-related exact-match break |

## Interpretation

The qualitative analysis supports the main quantitative finding:

1. Phase5-E improves many difficult span cases, especially tables involving `rowspan` and `colspan`.
2. The model fixes 310 samples that the baseline did not predict exactly.
3. The remaining failure mode is also span-related: Phase5-E sometimes introduces catastrophic structural changes on samples that baseline already predicted exactly.
4. This explains why the average TEDS gain is strong on 5000 samples, while regressions still exist and should be discussed honestly.

## Paper Use

Recommended qualitative figure set:

| Role | Suggested samples |
|---|---|
| Strong improvement | 531, 650, 885 |
| Header/span correction | 1455, 1482 |
| Honest failure case | 48 or 414 |

Suggested paper wording:

```text
Qualitative analysis shows that the full-Mamba head frequently corrects span-sensitive structural patterns missed by the original SLANet baseline, including repeated colspan rows and mixed rowspan/colspan headers. However, several regressions are also span-related, indicating that robust span prediction remains the primary bottleneck.
```

## Next Step

Export visual case figures for 3-5 representative samples:

1. Original table image.
2. Ground-truth structure tokens.
3. Baseline prediction.
4. Phase5-E prediction.
5. Highlighted structural difference.
