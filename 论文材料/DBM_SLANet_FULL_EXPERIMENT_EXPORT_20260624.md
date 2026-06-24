# DBM-SLANet 全实验总结与材料导出

Date: 2026-06-24  
Project path: `/root/autodl-tmp/PaddleOCR`  
Target: 表格结构识别 / Table Structure Recognition  
Main dataset: PubTabNet-style table recognition data  
Main final model: Phase5-E Full Mamba

---

## 0. 一句话总览

本轮实验以原始 SLANet 作为 baseline，在不重新设计完整 OCR 系统的前提下，围绕表格结构 HTML token 解码进行改进。核心路线包括：先审计和增强 OCR-cell matching，再尝试注意力/局部分支/轻量 Mamba，再推进到 Full Mamba 双轴结构建模，随后做 span-aware loss、TEDS 选择、候选重排、动态 OCR matching、大样本验证、统计检验和定性案例分析。

最终可作为论文主线的结论是：**Phase5-E Full Mamba table structure head 在 5000 样本 PubTabNet 验证集上，将原始 SLANet baseline 的 TEDS 从 0.8330900340 提升到 0.8439856134，structure accuracy 从 0.6898 提升到 0.7182；paired bootstrap 95% CI 为 [+0.0059055140, +0.0161684765]，p(delta <= 0)=0.0000。**

需要谨慎说明的是：最终模型的精度提升是稳定的，但推理速度下降；Phase9-C 后处理曾在 1000 样本上提升 TEDS，但在 3000 样本上未能超过 Phase5-E，因此不能作为最终主方法。

---

## 1. 实验目标与问题定义

### 1.1 任务

表格结构识别任务的目标是将表格图片转换为结构化 HTML token 序列，尤其要正确预测：

- `<thead>` / `<tbody>` / `<tr>` 等表格结构；
- `<td></td>` 普通单元格；
- `rowspan` / `colspan` 跨行跨列结构；
- 单元格位置与 OCR 内容匹配关系。

### 1.2 初始问题

原始 SLANet 在普通表格上已有较强能力，但在复杂科学表格中容易出现：

1. 空 `<td></td>` 过多；
2. OCR 文本和结构单元格匹配不稳定；
3. `rowspan` / `colspan` 预测错误；
4. exact structure accuracy 与 TEDS 不一致；
5. 少量 span 错误会造成 TEDS 大幅下降。

### 1.3 总体策略

实验没有一开始就大改模型，而是按风险逐步推进：

1. 先保持 baseline 行为不变，审计匹配、HTML reconstruction、推理入口和评估链路；
2. 尝试后处理 OCR-aware matching，验证空 `<td></td>` 是否能减少；
3. 尝试 cell token、局部分支、注意力/Transformer-like 方向；
4. 进入 Mamba-based structural head；
5. 通过大样本验证和 bootstrap 证明主模型增益；
6. 最后整理论文表格、图、定性案例和参考文献。

---

## 2. Baseline 设置

### 2.1 Baseline 模型

Baseline 使用原始 SLANet-style 表格结构识别模型：

```text
Original SLANet baseline / original SLAHead
```

关键 baseline checkpoint：

```text
output/baseline_package_20260621/model/best_accuracy.pdparams
```

1000 样本 baseline eval config：

```text
configs/table/SLANet_pubtabnet_baseline_eval_1000_codex.yml
```

### 2.2 Baseline 在不同验证规模上的表现

| Split | Baseline TEDS | Baseline structure_acc | FPS |
|---|---:|---:|---:|
| 1000 samples | 0.8280000534 | 0.6680 | 18.9382 |
| 3000 samples | 0.8333671727 | 0.6873 | 22.1553 |
| 5000 samples | 0.8330900340 | 0.6898 | 23.4966 |

### 2.3 Baseline 分层诊断

早期做过一个 100-sample stratified baseline/current 对比，用 GT 表格属性分层，包括 cell count、平均 cell area、structure token count、span presence。

结果提示：

- no-span 表格更容易提升；
- span_present 表格仍然困难；
- 复杂/小单元格/中等面积单元格上局部有提升，但不稳定；
- end-to-end 结果仍受 OCR matching 与 HTML reconstruction 影响。

对应材料：

```text
output/baseline_package_20260621/eval/stratified_report.md
output/baseline_package_20260621/eval/stratified_metrics.json
```

---

## 3. 特殊数据集与验证集构建

### 3.1 分阶段训练集

为了控制训练风险，构建了从小到大的训练/验证子集：

| File | Size | Role |
|---|---:|---|
| `phase4_smoke_train_100.jsonl` | 100 train | smoke test |
| `phase4_smoke_val_20.jsonl` | 20 val | smoke validation |
| `phase4_stage1_train_500.jsonl` | 500 train | first stability run |
| `phase4_stage1_val_50.jsonl` | 50 val | quick validation |
| `phase4_stage2_train_2000.jsonl` | 2000 train | warmup scale |
| `phase4_stage2_val_200.jsonl` | 200 val | small validation |
| `phase4_stage4_train_10000.jsonl` | 10000 train | medium training |
| `phase4_stage4_val_500.jsonl` | 500 val | medium validation |
| `phase4_stage5_train_30000.jsonl` | 30000 train | main training subset |
| `phase4_stage5_val_1000.jsonl` | 1000 val | early main validation |

这些文件位于：

```text
train_data/table/pubtabnet/
```

### 3.2 大样本验证集

为避免 1000 样本偶然性，后续构建了 3000 和 5000 验证集：

```text
train_data/table/pubtabnet/phase10_val_3000.jsonl
train_data/table/pubtabnet/phase10_val_5000.jsonl
```

来源：

```text
/root/autodl-tmp/pubtabnet/pubtabnet_part_91150/pubtabnet_part_91150.jsonl
```

原始切分统计：

| Split | Samples |
|---|---:|
| train | 91150 |
| val | 9115 |

### 3.3 CTA / benchmark 特殊数据

项目中还保留了面向 CTA benchmark 的特殊数据：

```text
train_data/table/pubtabnet_cta_benchmark_v1/cta_benchmark_v1_500.jsonl
train_data/table/pubtabnet_cta_benchmark_v1/cta_benchmark_pilot_100.jsonl
```

这部分可作为后续第二数据集或鲁棒性验证材料，但当前主论文证据仍以 PubTabNet-style 3000/5000 validation 为核心。

---

## 4. 模型改进路线

### 4.1 Phase0 / Phase1：代码审计与 OCR-aware Cell Matching

目标：在不改训练模型的情况下，先检查 OCR-cell matching 是否能减少空 `<td></td>`。

实现要点：

- 保留原始 `TableMatch.match_result` legacy 行为；
- 新增 `match_mode="legacy" | "ocr_aware_hungarian"`；
- 引入 bbox normalization、IoU、OCR coverage、cell coverage、center similarity、inside score、direction prior、OCR confidence 等打分；
- 使用 Hungarian matching 做主匹配；
- 主匹配后支持未分配 OCR 的二次归属；
- 支持 one-cell-to-many-OCR；
- OCR token 按阅读顺序拼接；
- 可选 diagnostics JSON。

重要结果：

| Split | Method | TEDS | empty_td_total | Comment |
|---|---|---:|---:|---|
| val100 | legacy | 0.4510377129 | 4079 | reference |
| val100 | ocr_aware_hungarian | 0.4442198104 | 4505 | worse |
| val1000 | legacy | 0.4510299445 | 37063 | reference |
| val1000 | ocr_aware_hungarian | 0.4457551663 | 40381 | worse |
| val1000 | ocr_aware_relaxed_secondary | 0.4477778055 | 39209 | still worse than legacy |
| val1000 | F3 no-GT gate best-empty variant | 0.4543432762 | 36431 | exploratory positive |

结论：

- 直接替换 Hungarian OCR-aware matcher 不可靠，TEDS 下降且空 `<td></td>` 增加；
- 但结合 no-GT gate 选择 legacy / OCR-aware variant，可以在 1000 样本上略高于 legacy，并减少空 `<td></td>`；
- 这条线适合作为“后处理和 OCR matching 探索”，暂不作为主模型贡献。

对应材料：

```text
output/phaseF_dynamic_ocr_aware_20260624/
eval_phaseF_dynamic_ocr_aware.py
eval_phaseF2_dynamic_ocr_relaxed.py
```

### 4.2 Phase2：Cell Token / 几何辅助分支

核心代码：

```text
ppocr/modeling/heads/cell_token_mlp_head.py
```

核心思想：

- 继承 `SLAHead`，保留原始 SLANet 输出；
- 根据 `loc_preds` 构造 cell geometry token；
- 几何特征包括中心点、宽高、面积、aspect ratio、left/top；
- 增加 `cell_has_text_logits`，预测每个 decoded cell step 是否有 supervised bbox/text target。

代码结构：

```text
GeometryTokenEncoder
CellTokenMLPHead
```

意义：

- 这一阶段不是最终最强模型，但为后续 horizontal/vertical token sequence modeling 提供了 cell token 表示。

### 4.3 注意力机制 / Transformer-like 尝试

配置文件中保留了多种注意力或 Transformer-like 尝试：

```text
configs/table/SLANet_pubtabnet_attn_transformer_codex.yml
configs/table/SLANet_pubtabnet_attn_transformer_quick_codex.yml
configs/table/SLANet_pubtabnet_transformer_only_codex.yml
configs/table/SLANet_pubtabnet_transformer_only_quick_codex.yml
configs/table/SLANet_pubtabnet_lsnet_attn_codex.yml
```

这条线的实验意图是：

- 用 attention / Transformer-like 结构增强结构 token 的上下文依赖；
- 评估全局注意力是否能处理跨行跨列结构；
- 与后续 Mamba/SSM 方向形成方法选择对照。

当前 westb 服务器上没有找到这些 attention 方向对应的完整 eval output 目录，因此不能写具体 TEDS 数字。报告中应谨慎表述为：**注意力机制方向做过配置与早期尝试，但没有形成最终可报告的最优结果；最终主线转向 Mamba-based structural head。**

### 4.4 Phase3：Dual-Branch / Lightweight Axis Mamba

核心代码：

```text
ppocr/modeling/heads/cell_dual_branch_mamba_head.py
```

核心结构：

- 基于 `CellTokenMLPHead`；
- 增加 horizontal order 和 vertical order；
- 用 lightweight gated recurrent scan 模拟 Mamba-style sequence interface；
- 对 cell tokens 做横向/纵向建模；
- 最后将 original / horizontal / vertical token concat 后 fusion。

关键类：

```text
LightweightAxisMamba
CellDualBranchMambaHead
```

意义：

- Phase3 是从几何 MLP 到双轴结构建模的过渡；
- 验证了“按表格空间顺序建模 cell token”的方向可行；
- 后续 Full Mamba 沿用 horizontal / vertical 双轴思路。

### 4.5 Phase4：Lite Full-Mamba Head

目标：将 lightweight scan 推进为更完整的 Mamba-style selective scan block。

核心结果：

| Experiment | Init | Train samples | LR | Internal best acc | External TEDS | Decision |
|---|---|---:|---:|---:|---:|---|
| Stage4 baseline | previous lite setup | 10000 | baseline | 0.684 | 0.8187506322 | baseline |
| Stage5 lr=1e-4 | Stage4 best | 30000 | 1e-4 | ~0.673 | 0.8173615266 | rejected |
| Stage5 lr=2e-5 | Stage4 best | 30000 | 2e-5 | 0.6870 | 0.8287605834 | improved |
| Stage6 lr=1e-5 | Stage5 best | 30000 | 1e-5 | 0.6910 | 0.8294241307 | best Phase4 |
| Stage7 lr=5e-6 | Stage6 best | 30000 | 5e-6 | 0.6860 | not run | stopped early |

Phase4 best checkpoint：

```text
output/E4_lite_stage6_lr1e5_from_stage5_ft_30000_20260623/best_accuracy.pdparams
```

Phase4 best external validation：

```text
TEDS: 0.8294241307
structure_acc: 0.691
FPS: 12.0009
```

结论：

- 低学习率 continuation 对 lite Mamba 有帮助；
- Phase4 已超过 1000-sample baseline；
- 但提升幅度仍小，因此进入更强 Full Mamba。

### 4.6 Phase5：Full Mamba / 最终主模型

核心代码：

```text
ppocr/modeling/heads/cell_full_mamba_head.py
```

核心结构：

```text
SelectiveScanMambaBlock
CellFullMambaHead
```

主要机制：

- input/gate projection；
- depthwise Conv1D local convolution；
- input-dependent delta/B/C parameters；
- diagonal A state transition；
- D skip；
- selective scan；
- gated output projection；
- horizontal / vertical 双轴编码；
- optional bidirectional reverse scan；
- fusion original / horizontal / vertical tokens。

关键配置：

```yaml
Head:
  name: CellFullMambaHead
  mamba_d_state: 16
  mamba_expand: 2
  bidirectional: true
```

Phase5 训练路径：

| Experiment | Init | Train samples | LR | Internal best acc | External TEDS | Decision |
|---|---|---:|---:|---:|---:|---|
| Phase5-A smoke | Phase4 Stage6 best | 500 | smoke | 0.5800 | not run | stable |
| Phase5-B warmup | Phase5-A best | 2000 | warmup | 0.6500 | not run | stable |
| Phase5-C scale-up | Phase5-B best | 10000 | scale-up | 0.6820 | not run | stable |
| Phase5-D | Phase5-C best | 30000 | 1e-5 | 0.6930 | 0.8306299579 | improved |
| Phase5-E | Phase5-D best | 30000 | 5e-6 | 0.6930 | 0.8308172190 | selected |
| Phase5-F | Phase5-E best | 30000 | 2e-6 | 0.6940 | 0.8285197073 | rejected |

最终主模型 checkpoint：

```text
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams
```

1000-sample 结果：

```text
TEDS: 0.8308172190
structure_acc: 0.693
FPS: 9.5650
```

结论：

- Full Mamba 比 Lite Mamba 更强；
- Phase5-E 是最终主模型；
- Phase5-F 说明更低 LR / 更高 exact acc 不一定带来更高 TEDS。

---

## 5. Phase6-Phase9：诊断与消融实验

### 5.1 Phase6：TEDS-based checkpoint selection

目标：验证 checkpoint selection 是否应该按 TEDS 而不是 exact structure accuracy。

结果：

```text
Phase6-D TEDS: 0.8287656930
structure_acc: 0.689
```

没有超过 Phase5-E，但证明了：

```text
step 6000: acc 0.689, TEDS 0.828766 -> saved as best
step 7500: acc 0.695, TEDS 0.828531 -> not saved
```

结论：exact acc 与 TEDS 不一致，后续模型选择应该优先对齐 TEDS。

### 5.2 Phase6 误差诊断：Phase5-E vs Phase5-F

| Metric | Value |
|---|---:|
| Samples | 1000 |
| Same TEDS | 990 |
| Phase5-F worse | 7 |
| Phase5-F better | 3 |
| Overall mean delta | -0.002297511681 |
| Exact matches Phase5-E | 693 |
| Exact matches Phase5-F | 694 |

结论：

- Phase5-F exact acc 更高，但 TEDS 更低；
- 少数 catastrophic span errors 主导整体 TEDS 下降；
- 主要错误是错误添加或删除 `rowspan/colspan`。

### 5.3 Phase7：Span-aware loss

目标：通过 span-token weighted loss 和 false-positive penalty 改善 span 预测。

结果：

| Method | TEDS | structure_acc | FPS | Decision |
|---|---:|---:|---:|---|
| Phase5-E | 0.8308172190 | 0.693 | 9.5650 | reference |
| Phase7-A span-aware | 0.8280026826 | 0.697 | 8.6990 | rejected |
| Phase7-B weak span-aware | 0.8280026826 | 0.697 | 11.0159 | rejected |

诊断：

| Metric | Phase5-E | Phase7-B |
|---|---:|---:|
| Samples with extra predicted spans | 94 | 91 |
| Total extra predicted spans | 456 | 475 |
| Samples with missing spans | 127 | 131 |
| Total missing spans | 446 | 448 |

结论：

- span-aware loss 提高 exact acc，但降低 TEDS；
- 说明问题不是“让模型更重视 span token”这么简单，而是要判断 span 是否结构安全；
- 可作为负向消融写入论文。

### 5.4 Phase8：Span calibration

目标：不训练模型，只在推理后抑制可疑 span。

尝试模式：

- shape；
- confidence；
- hybrid；
- margin；
- cell-sequence margin。

关键结果：

| Mode | Threshold | TEDS | structure_acc | span suppressions | Decision |
|---|---:|---:|---:|---:|---|
| none | 0.0 | 0.8308172190 | 0.693 | 0 | reference |
| confidence 0.90/0.95/0.98 | various | 0.8308172190 | 0.693 | 0 | no effect |
| margin | 0.01 | 0.8300003227 | 0.692 | 2 | rejected |
| margin | 0.10 | 0.8277324098 | 0.689 | 26 | rejected |

span confidence 观察：

```text
span cells: 2410
score p50: 0.9925
margin p50: 0.9872
```

结论：错误 span 往往也是高置信度，简单阈值抑制不可行。

### 5.5 Phase9：Candidate reranking

#### Phase9-A：margin_shape

结论：带 row-width guard 的 span suppression 仍低于 Phase5-E。

#### Phase9-B：Top-k span attribute reranking

| top-k | shape_lambda | TEDS | structure_acc | changed_samples |
|---:|---:|---:|---:|---:|
| 3 | 0.0 | 0.8308172190 | 0.693 | 0 |
| 3 | 1.0 | 0.8308172190 | 0.694 | 6 |

结论：能略提升 exact acc，但 TEDS 持平。

#### Phase9-C：Full cell-pattern reranking

1000-sample 结果：

| top-k | shape_lambda | TEDS | structure_acc | changed_samples | Decision |
|---:|---:|---:|---:|---:|---|
| 3 | 0.0 | 0.8292348461 | 0.690 | 23 | rejected |
| 3 | 0.5 | 0.8316667671 | 0.688 | 26 | best on 1000 |
| 3 | 1.0 | 0.8315895138 | 0.689 | 28 | improved |
| 3 | 2.0 | 0.8312144700 | 0.690 | 36 | improved |

但在 3000-sample 复评：

| Method | TEDS | structure_acc | Samples | Decision |
|---|---:|---:|---:|---|
| Phase5-E | 0.8413762999 | 0.7147 | 3000 | main |
| Phase9-C | 0.8408456063 | 0.7120 | 3000 | not robust |

结论：

- Phase9-C 是有价值的后处理探索；
- 1000 样本上提升，但 3000 样本上不稳；
- 最终论文主方法不能用 Phase9-C，只能作为 exploratory/negative ablation。

#### Phase9-D：Gated reranking

| gate_badness | TEDS | structure_acc | changed_samples |
|---:|---:|---:|---:|
| 0 | 0.8316667671 | 0.688 | 26 |
| 2 | 0.8312027446 | 0.688 | 22 |
| 3 | 0.8309510115 | 0.690 | 18 |
| 5 | 0.8309741571 | 0.691 | 12 |

结论：简单 gating 会减少修改并恢复部分 exact acc，但 TEDS 下降。

---

## 6. Phase10：大样本验证与统计检验

### 6.1 1000 / 3000 / 5000 一致性

| Split | Baseline TEDS | Phase5-E TEDS | TEDS Gain | Baseline Acc | Phase5-E Acc | Acc Gain |
|---|---:|---:|---:|---:|---:|---:|
| 1000 samples | 0.8280000534 | 0.8308172190 | +0.0028171655 | 0.6680 | 0.6930 | +0.0250 |
| 3000 samples | 0.8333671727 | 0.8413762999 | +0.0080091272 | 0.6873 | 0.7147 | +0.0273 |
| 5000 samples | 0.8330900340 | 0.8439856134 | +0.0108955794 | 0.6898 | 0.7182 | +0.0284 |

解释：

- 1000 样本提升较小；
- 3000/5000 样本提升更明显；
- 这说明 Phase5-E 的增益不是只靠少数样本偶然出现。

### 6.2 5000 样本最终主结果

| Method | TEDS | structure_acc | FPS | Samples | Role |
|---|---:|---:|---:|---:|---|
| Original SLANet baseline | 0.8330900340 | 0.6898 | 23.4966 | 5000 | baseline |
| DBM-SLANet / Phase5-E Full Mamba | 0.8439856134 | 0.7182 | 9.6547 | 5000 | final main method |

增益：

| Metric | Baseline | Phase5-E | Gain |
|---|---:|---:|---:|
| TEDS | 0.8330900340 | 0.8439856134 | +0.0108955794 |
| structure_acc | 0.6898 | 0.7182 | +0.0284 |
| FPS | 23.4966 | 9.6547 | -13.8419 |

### 6.3 Paired bootstrap

| Split | Comparison | Delta | 95% CI | P(delta <= 0) | Better/Worse/Same |
|---|---|---:|---:|---:|---:|
| 3000 | Phase5-E - baseline | +0.0080091272 | [+0.0013414985, +0.0145446136] | 0.0081 | 290/267/2443 |
| 5000 | Phase5-E - baseline | +0.0108955794 | [+0.0059055140, +0.0161684765] | 0.0000 | 497/424/4079 |

结论：5000 样本上置信区间完全为正，可以作为当前最强论文证据。

---

## 7. 定性案例分析

5000 样本 case mining：

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

代表性提升案例：

| sample | filename | baseline | Phase5-E | delta | Observation |
|---:|---|---:|---:|---:|---|
| 531 | PMC2898048_011_00.png | 0.000000 | 1.000000 | +1.000000 | recovered `colspan="3"` rows |
| 650 | PMC3496644_001_00.png | 0.000000 | 1.000000 | +1.000000 | recovered mixed `rowspan`/`colspan` header |
| 885 | PMC3126717_005_00.png | 0.000000 | 1.000000 | +1.000000 | removed false spans and matched true colspan |

代表性失败案例：

| sample | filename | baseline | Phase5-E | delta | Observation |
|---:|---|---:|---:|---:|---|
| 48 | PMC4225768_004_00.png | 1.000000 | 0.000000 | -1.000000 | broke originally exact span table |
| 414 | PMC4652242_004_00.png | 1.000000 | 0.000000 | -1.000000 | severe span-structure regression |

论文图文件：

```text
论文材料/figures/dbm_slanet_architecture.png
论文材料/figures/dbm_slanet_architecture.pdf
论文材料/figures/qualitative_cases_multipanel.png
论文材料/qualitative_figures/*.png
```

---

## 8. 当前可以写入论文的内容

### 8.1 可以作为主贡献写入

1. 提出 DBM-SLANet / Full-Mamba table structure head；
2. 将 cell geometry token 与 horizontal/vertical 双轴 Mamba-style selective scan 结合；
3. 在 5000 样本 PubTabNet validation 上显著提升 TEDS 和 structure_acc；
4. 用 paired bootstrap 给出统计置信区间；
5. 证明 exact acc 与 TEDS 可能不一致，模型选择应对齐 TEDS；
6. 通过定性案例说明收益和失败都集中在 span-sensitive 结构。

### 8.2 可以作为消融/负结果写入

1. Lite Mamba vs Full Mamba；
2. Lower LR 不一定更好；
3. TEDS-based checkpoint selection 机制有效但 continuation 未提升；
4. Span-aware loss 提升 exact acc 但损害 TEDS；
5. Span calibration / margin suppression 无效；
6. Phase9-C candidate reranking 在小样本上有用但大样本不稳；
7. OCR-aware matching 直接替换不稳定，F3 gate 有探索价值。

### 8.3 不建议写成强结论

不要写：

- 达到 SOTA；
- 彻底解决 rowspan/colspan；
- 速度提升；
- Phase9-C 是最终方法；
- OCR-aware matcher 全面优于 legacy；
- 注意力机制实验有明确数值优势。

---

## 9. 距离 SCI 3 区目标的评估

当前实验已经具备论文雏形，强项是：

- baseline 明确；
- 主模型改进清晰；
- 大样本验证完成；
- bootstrap 统计检验完成；
- 消融链条丰富；
- 有图、有定性案例、有参考文献材料。

当前短板是：

1. 只在 PubTabNet-style 验证上形成强证据；
2. 没有与更多公开 SOTA 方法做直接同环境复现对比；
3. 注意力机制方向没有完整可报告结果；
4. 速度明显低于 baseline；
5. span regression 仍存在，失败案例需要诚实讨论。

综合判断：

```text
当前不是“已经稳投 SCI 3 区”的状态，
但已经达到“可以开始写 SCI 3 区导向论文初稿”的状态。
```

更准确的 readiness：约 82%。

---

## 10. 文件与材料索引

### 10.1 总结材料

```text
论文材料/README.md
论文材料/FINAL_PAPER_EXPERIMENT_TABLES_20260624.md
论文材料/PAPER_READY_EXPERIMENT_TABLES_20260624.md
论文材料/PAPER_EXPERIMENT_RESULTS_20260624.md
论文材料/PHASE4_PHASE5_EXPERIMENT_SUMMARY_20260623.md
论文材料/PHASE10_LARGE_VAL_3000_RESULTS_20260624.md
论文材料/PHASE10_LARGE_VAL_5000_RESULTS_20260624.md
论文材料/PHASE10_STATISTICAL_VALIDATION_20260624.md
论文材料/PHASE10_QUALITATIVE_CASE_ANALYSIS_20260624.md
论文材料/PAPER_WRITING_BLUEPRINT_20260624.md
论文材料/RELATED_WORK_REFERENCES_20260624.md
论文材料/SUBMISSION_READINESS_CHECKLIST_20260624.md
```

### 10.2 关键代码

```text
ppocr/modeling/heads/cell_token_mlp_head.py
ppocr/modeling/heads/cell_dual_branch_mamba_head.py
ppocr/modeling/heads/cell_full_mamba_head.py
ppocr/modeling/heads/__init__.py
ppocr/metrics/table_metric.py
```

### 10.3 关键配置

```text
configs/table/SLANet_pubtabnet_baseline_eval_1000_codex.yml
configs/table/DBM_SLANet_full_mamba_lite_stage6_30000_lr1e5_from_stage5_codex.yml
configs/table/DBM_SLANet_phase5_full_mamba_stage5_30000_lr5e6_from_stage4_codex.yml
configs/table/DBM_SLANet_phase6d_teds_select_30000_lr2e6_from_phase5e_codex.yml
configs/table/DBM_SLANet_phase7a_span_aware_10000_lr2e6_from_phase5e_codex.yml
configs/table/DBM_SLANet_phase7b_span_aware_weak_10000_lr2e6_from_phase5e_codex.yml
```

### 10.4 关键输出目录

```text
output/E4_lite_stage6_lr1e5_from_stage5_ft_30000_20260623/
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/
output/E6d_teds_select_30000_lr2e6_from_phase5e_20260624/
output/E7a_span_aware_10000_lr2e6_from_phase5e_20260624/
output/phase8_span_calibration_20260624/
output/phase9c_cellpattern_rerank_20260624/
output/phase10_large_val_20260624/
output/phase10_statistical_validation_20260624/
output/phase10_qualitative_cases_20260624/
output/phaseF_dynamic_ocr_aware_20260624/
```

### 10.5 最终模型路径

```text
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams
```

注意：导出包默认不包含大模型权重和完整训练数据，以免体积过大；报告中保留路径。

---

## 11. 下一步建议

如果目标是继续冲 SCI 3 区，下一步优先级应该是：

1. 写英文论文初稿；
2. 把 5000 样本主结果、bootstrap、消融表和定性图放入正文；
3. 补 Related Work 的正式引用格式；
4. 如算力允许，补 9115 full validation；
5. 尝试第二数据集或更强 external comparison；
6. 对速度下降给出明确 limitation。

