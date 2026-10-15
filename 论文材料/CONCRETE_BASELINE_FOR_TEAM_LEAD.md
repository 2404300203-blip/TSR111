# 具体 Baseline 说明

整理时间：2026-06-25  
项目：DBM-SLANet / 表格结构识别  

## 1. Baseline 到底是什么

本实验中的 baseline 不是泛指某篇旧论文，也不是一个概念性对照组，而是一个实际跑过评估的原始模型：

```text
Original SLANet baseline / original SLAHead
```

也就是说，baseline 是 PaddleOCR / PP-StructureV2 体系中的原始 SLANet 表格结构识别模型。它保留原始结构识别 head，不加入本文后续的 Mamba、dual-branch、cell token 或 OCR-aware matching 改动。

## 2. Baseline Checkpoint

远程服务器中的 baseline checkpoint：

```text
output/baseline_package_20260621/model/best_accuracy.pdparams
```

这个 checkpoint 是后续所有主模型对比的固定参照。

## 3. Baseline 模型结构

可写成：

```text
Backbone: PP-LCNet
Neck: CSP-PAN
Head: original SLAHead
Task: table image -> HTML-like table structure tokens
Framework: PaddleOCR / PP-StructureV2
```

中文解释：

> baseline 使用 PaddleOCR / PP-StructureV2 中的 SLANet-style 表格结构识别模型，以 PP-LCNet 作为轻量视觉骨干网络，结合 CSP-PAN 和原始 SLAHead 预测 HTML-like 表格结构 token。

## 4. Baseline 没有加什么

baseline 不包含以下改动：

- 不加 Mamba；
- 不加 Dual-Branch Mamba；
- 不加 horizontal / vertical selective scan；
- 不加新的 cell geometry token；
- 不加 Phase5-E 的 full-Mamba structure head；
- 不加 OCR-aware Hungarian matching；
- 不加后处理 reranking；
- 不使用 CTA-500 选择模型。

所以它是一个干净的原始 SLANet 对照组。

## 5. Baseline 评估设置

主评估数据：

```text
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

评价指标：

```text
TEDS
Structure Accuracy
FPS
```

评估原则：

```text
same validation split
same evaluation script
same dictionary
same post-processing protocol
```

这样才能说明 Phase5-E 的提升来自模型结构改动，而不是数据、脚本或评估方式不同。

## 6. Baseline 主结果

### 6.1 PubTabNet 5000 样本验证集

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 5000 | 0.8330900340 | 0.6898 | 23.4966 |

这是目前最适合写进论文主表的 baseline 数字。

### 6.2 PubTabNet 不同规模验证结果

| Split | Baseline TEDS | Baseline Structure Acc | FPS |
|---|---:|---:|---:|
| 1000 samples | 0.8280000534 | 0.6680 | 18.9382 |
| 3000 samples | 0.8333671727 | 0.6873 | 22.1553 |
| 5000 samples | 0.8330900340 | 0.6898 | 23.4966 |

### 6.3 CTA-500 复杂结构专项子集

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 500 | 0.4843354130 | 0.2080 | 16.3023 |

CTA-500 上 baseline 明显下降，说明复杂表格、多级表头、rowspan / colspan、横纵向依赖会显著增加识别难度。

## 7. Baseline 和最终模型对比

### 7.1 PubTabNet 5000 样本

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 5000 | 0.8330900340 | 0.6898 | 23.4966 |
| Phase5-E Full Mamba | 5000 | 0.8439856134 | 0.7182 | 9.6547 |

提升：

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.8330900340 | 0.8439856134 | +0.0108955794 |
| Structure Acc | 0.6898 | 0.7182 | +0.0284 |
| FPS | 23.4966 | 9.6547 | -13.8419 |

统计检验：

```text
TEDS delta: +0.0108955794
95% CI: [+0.0059055140, +0.0161684765]
p(delta <= 0): 0.0000
```

### 7.2 CTA-500 复杂结构子集

| Method | Samples | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 500 | 0.4843354130 | 0.2080 | 16.3023 |
| Phase5-E Full Mamba | 500 | 0.6406726544 | 0.4320 | 8.1429 |

提升：

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.4843354130 | 0.6406726544 | +0.1563372413 |
| Structure Acc | 0.2080 | 0.4320 | +0.2240 |
| FPS | 16.3023 | 8.1429 | -8.1594 |

统计检验：

```text
TEDS delta: +0.1563372413
95% CI: [+0.1118128127, +0.2007433510]
p(delta <= 0): 0.0000
```

## 8. 可以直接发给队长的版本

> 本实验的 baseline 是 PaddleOCR / PP-StructureV2 中的 Original SLANet table structure recognition model，使用原始 SLAHead，不加入 Mamba、Dual-Branch、cell token、OCR-aware matching 或后处理 reranking。固定 checkpoint 为 `output/baseline_package_20260621/model/best_accuracy.pdparams`。在 PubTabNet 5000-sample validation 上，baseline TEDS 为 0.8330900340，Structure Acc 为 0.6898，FPS 为 23.4966。最终 Phase5-E Full Mamba 在相同验证集和相同评估协议下达到 TEDS 0.8439856134、Structure Acc 0.7182，相比 baseline 分别提升 +0.0108955794 和 +0.0284。

## 9. 论文里可以这样写

英文：

> We adopt the original SLANet table structure recognition model from PaddleOCR / PP-StructureV2 as the baseline. The baseline uses a PP-LCNet backbone, CSP-PAN neck, and the original SLAHead for HTML-like table structure decoding. All comparisons are conducted under the same PubTabNet validation split, dictionary, post-processing protocol, and TEDS-based evaluation script.

中文：

> 本文采用 PaddleOCR / PP-StructureV2 中的原始 SLANet 表格结构识别模型作为 baseline。该 baseline 使用 PP-LCNet 作为视觉骨干网络，结合 CSP-PAN 与原始 SLAHead 完成 HTML-like 表格结构 token 解码。所有模型均在相同 PubTabNet 验证集、相同字典、相同后处理协议和相同 TEDS 评估脚本下进行对比。

## 10. 对 baseline 的一句话理解

```text
baseline = 没加本文方法之前的原始 SLANet 对照组
```

它的作用是证明：本文的 Phase5-E Full Mamba 到底有没有比原始 SLANet 更好。
