# CTA-500 Baseline vs Phase5-E Evaluation Results

Date: 2026-06-25  
Project: `/root/autodl-tmp/PaddleOCR`

## 1. Purpose

This experiment evaluates whether the complex-structure special dataset can support the dataset/diagnostic innovation point of the paper.

The CTA-500 split is constructed from the curated 500 representative samples in the complex-structure special dataset. These samples are selected from the Phase10 5000-sample validation split and emphasize multi-level headers, rowspan/colspan, horizontal and vertical dependencies, wide/tall tables, and long-range structural dependencies.

## 2. Dataset

Evaluation file:

```text
train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl
```

Source metadata:

```text
论文材料/complex_structure_special_dataset_20260625/complex_structure_curated_500_metadata.jsonl
```

The 500 samples are ordered according to the curated metadata file, and each sample is copied from:

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

## 3. Models

| Method | Checkpoint / Config | Role |
|---|---|---|
| Original SLANet baseline | `output/baseline_package_20260621/model/best_accuracy.pdparams` | baseline |
| Phase5-E Full Mamba | `output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy` | proposed main model |

Evaluation configs:

```text
configs/table/SLANet_pubtabnet_baseline_eval_cta500_codex.yml
configs/table/DBM_SLANet_phase5e_eval_cta500_codex.yml
```

Evaluation script:

```text
eval_table_teds_diagnostics_codex.py
```

## 4. Main Results

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 500 | 0.4843354130 | 0.2080 | 16.3023 |
| Phase5-E Full Mamba | 500 | 0.6406726544 | 0.4320 | 8.1429 |

## 5. Gains

| Metric | Baseline | Phase5-E | Absolute Gain |
|---|---:|---:|---:|
| TEDS | 0.4843354130 | 0.6406726544 | +0.1563372413 |
| Structure Acc | 0.2080 | 0.4320 | +0.2240 |
| FPS | 16.3023 | 8.1429 | -8.1594 |

Sample-level comparison:

| Category | Count |
|---|---:|
| Phase5-E better than baseline | 244 |
| Phase5-E worse than baseline | 114 |
| Same TEDS | 142 |

Paired bootstrap on TEDS:

| Comparison | Delta | 95% CI | P(delta <= 0) |
|---|---:|---:|---:|
| Phase5-E - baseline | +0.1563372413 | [+0.1118128127, +0.2007433510] | 0.0000 |

## 6. Results by Complex-Structure Tag

| Tag | Samples | Baseline TEDS | Phase5-E TEDS | Delta | Baseline Acc | Phase5-E Acc |
|---|---:|---:|---:|---:|---:|---:|
| hierarchical_span_header | 274 | 0.475252 | 0.675940 | +0.200688 | 0.175182 | 0.441606 |
| horizontal_dependency | 376 | 0.458076 | 0.646133 | +0.188058 | 0.172872 | 0.433511 |
| long_range_dependency | 473 | 0.489143 | 0.630415 | +0.141272 | 0.209302 | 0.416490 |
| multi_level_header | 325 | 0.460398 | 0.652324 | +0.191926 | 0.166154 | 0.424615 |
| row_col_span | 457 | 0.462662 | 0.647945 | +0.185283 | 0.188184 | 0.433260 |
| tall_table | 341 | 0.491260 | 0.570912 | +0.079652 | 0.222874 | 0.348974 |
| vertical_dependency | 286 | 0.449572 | 0.638929 | +0.189357 | 0.171329 | 0.405594 |
| wide_table | 149 | 0.541695 | 0.708302 | +0.166606 | 0.154362 | 0.422819 |

## 7. Interpretation

CTA-500 is substantially harder than the general PubTabNet validation split. The baseline TEDS drops to 0.4843 on this curated complex subset, confirming that the subset focuses on difficult structural cases.

Phase5-E improves TEDS by +0.1563 and structure accuracy by +0.2240 on CTA-500. The paired bootstrap confidence interval is fully positive, which provides strong evidence that the Full Mamba structure head is particularly useful for complex table structures.

The largest gains appear in:

- `hierarchical_span_header`: +0.2007 TEDS
- `multi_level_header`: +0.1919 TEDS
- `vertical_dependency`: +0.1894 TEDS
- `horizontal_dependency`: +0.1881 TEDS
- `row_col_span`: +0.1853 TEDS

This directly supports the dataset/diagnostic innovation point: the special dataset is not only constructed, but also used to demonstrate that the proposed model is more effective on multi-level headers, span-heavy tables, and long-range structural dependencies.

## 8. Paper-Ready Claim

Recommended claim:

```text
To evaluate complex structural reasoning, we construct CTA-500, a curated subset of 500 complex table samples selected by parsing multi-level headers, rowspan/colspan attributes, table scale, and long-range dependency indicators from the validation split. On CTA-500, the proposed Phase5-E Full Mamba model improves the original SLANet baseline from 0.4843 to 0.6407 TEDS and from 0.2080 to 0.4320 structure accuracy, with a paired-bootstrap 95% confidence interval of [+0.1118, +0.2007].
```

Caution:

```text
CTA-500 is a curated diagnostic subset derived from PubTabNet-style validation data, not a new independently collected dataset. It should be described as a complex-structure diagnostic benchmark or special evaluation subset.
```

## 9. Output Files

```text
train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl
configs/table/SLANet_pubtabnet_baseline_eval_cta500_codex.yml
configs/table/DBM_SLANet_phase5e_eval_cta500_codex.yml
output/cta500_baseline_eval_20260625/baseline_cta500_eval.log
output/cta500_baseline_eval_20260625/baseline_cta500_sample_teds.jsonl
output/cta500_phase5e_eval_20260625/phase5e_cta500_eval.log
output/cta500_phase5e_eval_20260625/phase5e_cta500_sample_teds.jsonl
output/cta500_baseline_vs_phase5e_20260625/cta500_baseline_vs_phase5e_summary.json
output/cta500_baseline_vs_phase5e_20260625/cta500_sample_delta.jsonl
output/cta500_baseline_vs_phase5e_20260625/cta500_by_tag.csv
output/cta500_baseline_vs_phase5e_20260625/cta500_top_improvements.jsonl
output/cta500_baseline_vs_phase5e_20260625/cta500_top_regressions.jsonl
```

