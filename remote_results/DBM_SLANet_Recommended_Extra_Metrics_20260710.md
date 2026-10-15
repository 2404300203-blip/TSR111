# Recommended Extra Metrics for DBM-SLANet

Date: 2026-07-10  

This file organizes the three recommended extra metrics:

1. GriTS-Top-style topology similarity
2. Span-F1 for rowspan/colspan recovery
3. Params / FLOPs / latency

The goal is to strengthen the experimental evidence beyond TEDS and Structure Accuracy.

---

## 1. Topology Similarity and Span-F1

### Metric Definition

`Topology F1` is computed as a GriTS-Top-style proxy. Predicted and ground-truth HTML tokens are converted into grid-cell tuples:

```text
(row_index, col_index, rowspan, colspan)
```

Then micro precision / recall / F1 are computed by exact tuple matching.

`Span-F1` only evaluates cells with:

```text
rowspan > 1 or colspan > 1
```

This directly measures whether the model restores multi-row / multi-column cell structures.

Important note: this is an HTML-token-derived diagnostic metric, not the official GriTS package implementation. In the paper, describe it as **GriTS-Top-style topology F1** unless the official GriTS evaluator is integrated later.

---

## 2. PubTabNet-5000 Extra Metrics

| Method | TEDS | Structure Acc. | Topology F1 | Span-F1 |
|---|---:|---:|---:|---:|
| SLANet | 0.8331 | 0.6898 | 0.9601 | 0.6738 |
| Cell Token only | 0.8321 | 0.6962 | 0.9592 | 0.6792 |
| Full Mamba, $\lambda_{pres}=0$ | 0.8308 | 0.6938 | 0.9587 | 0.6758 |
| Full Mamba, $\lambda_{pres}=0.2$ | 0.8345 | 0.7016 | 0.9602 | 0.6867 |
| Lite Mamba | 0.8367 | 0.7084 | 0.9576 | 0.6916 |
| Full Mamba stage6 | 0.8435 | 0.7178 | 0.9603 | 0.7036 |
| Span-aware | **0.8450** | 0.7192 | **0.9618** | 0.7027 |
| Horizontal Mamba | **0.8450** | **0.7194** | **0.9618** | 0.7043 |
| Vertical Mamba | 0.8449 | 0.7188 | **0.9618** | 0.7028 |
| DBM-SLANet | 0.8440 | 0.7182 | 0.9610 | **0.7045** |

### PubTabNet-5000 Interpretation

On the general PubTabNet-5000 split, Mamba-enhanced models improve topology and span recovery over SLANet. Horizontal Mamba and Span-aware variants are slightly stronger in TEDS and topology F1, while DBM-SLANet obtains the best Span-F1. This means DBM-SLANet is competitive on generic validation data, but its strongest evidence should come from complex-structure subsets.

---

## 3. CTA-500 Extra Metrics

| Method | TEDS | Structure Acc. | Topology F1 | Span-F1 |
|---|---:|---:|---:|---:|
| SLANet | 0.4843 | 0.208 | 0.8916 | 0.5874 |
| Cell Token only | 0.5436 | 0.284 | 0.8990 | 0.6220 |
| Full Mamba, $\lambda_{pres}=0$ | 0.5392 | 0.282 | 0.8977 | 0.6171 |
| Horizontal Mamba | 0.6276 | 0.420 | 0.9038 | 0.6563 |
| Vertical Mamba | 0.6258 | 0.416 | 0.9038 | 0.6523 |
| DBM-SLANet | **0.6407** | **0.432** | **0.9044** | **0.6618** |

### CTA-500 Interpretation

CTA-500 is the strongest place to use these new metrics. DBM-SLANet is best in TEDS, Structure Accuracy, Topology F1, and Span-F1. This directly supports the claim that dual-branch row/column modeling helps recover complex topology and rowspan/colspan structures.

Recommended paper sentence:

> On the CTA-500 complex-structure subset, DBM-SLANet achieves the highest topology F1 and Span-F1, indicating that dual-axis Mamba modeling improves not only overall TEDS but also explicit row/column topology and rowspan/colspan recovery.

---

## 4. Params / FLOPs / Latency

Latency is reported as:

```text
Latency = 1000 / FPS
```

FLOPs are estimated by `paddle.flops` with input size `1 x 3 x 488 x 488`. Paddle reports several unsupported operators as zero FLOPs, including some activation, normalization, upsample, and recurrent operators. Therefore, FLOPs should be described as a **Paddle-estimated FLOPs proxy**, while Params and FPS/Latency are more reliable.

| Method | Params (M) | Paddle-est. FLOPs (G) | PubTabNet FPS | Latency (ms/img) |
|---|---:|---:|---:|---:|
| SLANet | 1.89 | 5.14 | 23.50 | 42.6 |
| Cell Token only | 1.93 | 5.15 | 24.25 | 41.2 |
| Lite Mamba | 2.09 | 5.24 | 15.99 | 62.5 |
| Full Mamba stage6 | 2.45 | 1.86* | 12.13 | 82.4 |
| Span-aware | 2.45 | 5.41 | 12.32 | 81.2 |
| Horizontal Mamba | 2.45 | 5.41 | 11.68 | 85.6 |
| Vertical Mamba | 2.45 | 5.41 | 12.00 | 83.3 |
| DBM-SLANet | 2.45 | 5.41 | 9.65 | 103.6 |

`*` The stage6 FLOPs estimate is likely affected by Paddle's unsupported-operator handling or config-specific graph tracing. Use with caution.

### Complexity Interpretation

DBM-SLANet increases parameters from 1.89M to 2.45M, an increase of about 0.56M parameters. The Paddle-estimated FLOPs increase from about 5.14G to 5.41G for the main comparable models, suggesting that the model remains lightweight in parameter and nominal compute cost. However, measured latency increases more clearly, mainly because the Mamba-style structural head introduces sequential row/column processing overhead.

Recommended paper sentence:

> Although DBM-SLANet increases the parameter count only moderately, from 1.89M to 2.45M, its measured latency is higher than SLANet due to the additional row/column structural modeling branch. This reflects an accuracy-efficiency trade-off rather than a large model-size increase.

---

## 5. Recommended Usage in Paper

Use the extra metrics like this:

| Evidence Need | Recommended Metric |
|---|---|
| General table quality | TEDS + Structure Acc. |
| Row/column topology recovery | Topology F1 |
| rowspan/colspan recovery | Span-F1 |
| Lightweight discussion | Params + FPS/Latency |
| Compute discussion | Paddle-estimated FLOPs, with caveat |

Best placement:

- Put TEDS / Structure Acc / FPS in the main comparison table.
- Add Topology F1 and Span-F1 either as extra columns in the complex-structure table or as a supplementary table.
- Put Params / FLOPs / Latency in an efficiency table or appendix.

