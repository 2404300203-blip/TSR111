# DBM-SLANet Paper Writing Blueprint

Date: 2026-06-24

## Recommended Title Options

1. DBM-SLANet: A Mamba-Enhanced Structural Decoder for Table Structure Recognition
2. Improving Table Structure Recognition with Mamba-Based Long-Range Structural Modeling
3. Mamba-Enhanced HTML Token Decoding for Table Structure Recognition

## Abstract Skeleton

Table structure recognition requires accurate decoding of long and span-sensitive HTML token sequences. Existing SLANet-style decoders are effective but still suffer from structural errors involving `rowspan` and `colspan`. This work introduces a full-Mamba table structure head to enhance long-range structural dependency modeling in table HTML token decoding. On a 5000-sample PubTabNet validation split, the proposed method improves the original SLANet baseline from 0.8331 to 0.8440 TEDS and from 0.6898 to 0.7182 structure accuracy. Paired bootstrap evaluation shows a statistically supported TEDS gain with a 95% confidence interval of [+0.0059, +0.0162]. Qualitative analysis further shows that the proposed head improves many span-sensitive cases, while remaining errors are also concentrated around complex rowspan/colspan patterns.

## 1. Introduction

Main points:

1. Tables are common in scientific and business documents.
2. Table structure recognition is difficult because table HTML tokens encode long-range row/column dependencies.
3. Span attributes such as `rowspan` and `colspan` make the sequence non-local.
4. SLANet is a strong baseline, but still has structural token errors.
5. Mamba-style sequence modeling is suitable for efficient long-context structural dependency modeling.
6. This paper proposes a full-Mamba structure head for SLANet-like table recognition.

Contribution bullets:

1. We introduce a Mamba-enhanced table structure decoding head for HTML token prediction.
2. We validate the method on PubTabNet subsets up to 5000 validation samples and show statistically supported gains.
3. We provide ablation and error analysis showing that span-sensitive structural errors are the main bottleneck.
4. We show that simple post-processing reranking is not robust enough, motivating structure-aware modeling inside the decoder.

## 2. Related Work

Suggested subsections:

1. Table structure recognition.
2. OCR-based table recognition and HTML reconstruction.
3. Sequence modeling for document understanding.
4. Mamba/state-space models for long-range dependency modeling.

## 3. Method

Suggested subsections:

1. Baseline SLANet structure decoder.
2. Full-Mamba structure head.
3. Cell token representation and structure-token prediction.
4. Training objective.
5. Inference and evaluation protocol.

Important wording:

- Present the method as an accuracy-oriented structural enhancement.
- Explain that the Mamba head aims to model long-range dependencies among table structure tokens.
- Avoid claiming that the method is faster than SLANet.

## 4. Experiments

### 4.1 Experimental Setup

Include:

- Dataset: PubTabNet partial/full subset from `pubtabnet_part_91150`.
- Validation splits: 1000, 3000, 5000.
- Metrics: TEDS, structure accuracy, FPS.
- Baseline: original SLANet.
- Main model: Phase5-E full Mamba.

### 4.2 Main Results

Use 5000-sample result as main table:

| Method | TEDS | structure_acc | FPS |
|---|---:|---:|---:|
| SLANet baseline | 0.8331 | 0.6898 | 23.50 |
| DBM-SLANet | 0.8440 | 0.7182 | 9.65 |

### 4.3 Ablation Study

Include:

- Lite Mamba vs full Mamba.
- Learning rate continuation variants.
- Span-aware loss negative result.
- TEDS-based checkpoint selection negative/diagnostic result.
- Post-processing reranking negative generalization result.

### 4.4 Statistical Validation

Use paired bootstrap:

```text
TEDS gain: +0.0108955794
95% CI: [+0.0059055140, +0.0161684765]
p(delta <= 0): 0.0000
```

### 4.5 Qualitative Analysis

Use five currently exported figures:

- 531, 650, 885 as improvements.
- 48, 414 as failure cases.

Main message:

- The model often repairs span-sensitive structures.
- Regressions are also span-related.
- Robust span prediction remains the key future direction.

## 5. Discussion

Discuss:

1. Why Mamba helps structural HTML token modeling.
2. Why speed decreases.
3. Why TEDS and exact accuracy can disagree.
4. Why post-processing reranking was not retained.
5. Limitations: only PubTabNet subset, speed trade-off, remaining span failures.

## 6. Conclusion

Suggested message:

The proposed full-Mamba structure head improves SLANet-based table structure recognition with statistically supported gains on PubTabNet validation subsets. Future work should focus on robust span modeling and improving inference efficiency.

## Remaining Work Before Submission

1. Draw a clean method architecture figure.
2. Add citations and related-work paragraph.
3. Convert qualitative case figures into a paper-ready multi-panel figure.
4. Optionally evaluate on another table dataset or the full 9115 validation split.
5. Write the manuscript in English and polish the abstract/introduction.
