# 主论文全部表格更新总结报告

日期：2026-08-11  
对应稿件：`第九版.pdf`  
实验对象：SLANet、DBM-SLANet 及其 Cell Token、Mamba、结构尺度、方向和单轴消融变体

## 一、总体结论

本轮共完成 14 个模型在固定 PubTabNet-5000 上的统一推理，累计获得 70,000 条样本级预测；CTA-500 由同一批预测按文件名筛选，因此 Table 3、Table 4 和 Table 5 共用一次推理结果。Table 2 的参数量、Paddle FLOPs proxy、FPS 和 Latency 已按统一实现重新核验。Table 10 和 Table 11 已纳入精确 one-way、真实 Horizontal-only、真实 Vertical-only 以及 `d=64`、`d_s=8`、`d=192`、`e=1` 等新增结构变体。

后处理复核修正了两个实质性错误：

1. PubTabNet 的 `<td rowspan="2">` 会拆成多个 token。旧脚本没有重建标签，错误地产生 `Span-F1=1.0` 和 `gt_span_samples=0`。修正后，PubTabNet-5000 恢复 2,395 个含 span 的 GT 样本，CTA-500 恢复 457 个，所有模型 `parse_bad=0`。
2. `d_s=8`、`d=192` 和 zero-geometry 的旧 FLOPs 被不完整图追踪污染，现分别修正为 5.401138 G、5.736981 G 和 5.409346 G。

同时，扩展 company-disjoint 实验改变了第九版中 Table 9 的结论。三种子、每种子 249 个未见公司测试样本的结果显示，DBM-SLANet 平均 TEDS 比 SLANet 低 0.000666，不能继续声称 DBM 提升未见公司金融泛化。

## 二、Table 1–11 更新状态总览

| 表格 | 第九版内容 | 最终处理 | 状态 |
|---|---|---|---|
| Table 1 | 数据集与评估子集统计 | 数据划分未改变，原表保留 | 可直接保留 |
| Table 2 | 架构与效率 | 用 14 模型统一 profile 替换；新增尺度、one-way、真实单轴行 | 已更新 |
| Table 3 | PubTabNet-5000 准确率与效率 | 14 模型统一 TEDS、Acc.、FPS、Latency | 已更新 |
| Table 4 | PubTabNet-5000 综合结构指标 | 重算 Topology F1、Span-F1 | 已更新 |
| Table 5 | CTA-500 综合比较 | 从同一 PubTabNet-5000 预测筛选并重算结构指标 | 已更新 |
| Table 6 | FinTabNet.c 领域适配汇总 | pilot 与 train2000 结果保留；旧 company-disjoint100 只能标为历史小划分 | 有条件保留 |
| Table 7 | CTA-500 分类 TEDS | 最终 SLANet/DBM 预测未变，分类统计保留 | 可直接保留 |
| Table 8 | independent train2000 | 独立实验未受本轮架构扫描影响 | 可直接保留 |
| Table 9 | FinTabNet.c company-disjoint | 必须用扩展三种子结果替换旧 100 样本单划分 | 已更新，结论反转 |
| Table 10 | Phase 11 消融 | 扩为 12 个可核验变体；不纳入 incomplete aux-off 完整指标 | 已更新 |
| Table 11 | CTA-500 轴向消融 | 真实单轴结果已获得；完整指标版与严格匹配版需区分 | 已更新但须注明协议 |

## 三、Table 1：数据集与子集统计

Table 1 不涉及模型变化，保留第九版数据：

| Split | Samples | Source | Purpose |
|---|---:|---|---|
| Training subset | 30,000 | PubTabNet train | Model optimization |
| Main validation set | 5,000 | PubTabNet val | Overall comparison |
| Complex subset | 4,086 | Main validation | Complex-structure analysis |
| CTA-500 | 500 | Complex subset | Difficult-case analysis |

需要保留的限定是：Complex-4086 与 CTA-500 均由 PubTabNet validation 派生，只能解释为复杂结构诊断集，不能写成独立跨数据集泛化基准。

## 四、Table 2：架构与效率最终更新

所有数值采用 `1 x 3 x 488 x 488` 输入；Params 为可训练参数精确计数，FLOPs 为 Paddle profile proxy。

| Model | d | d_s | e | Axis | Dir. | Params (M) | FLOPs (G) | FPS | Latency (ms) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| SLANet | – | – | – | none | none | 1.892610 | 5.135768 | 24.636 | 40.591 |
| Cell Token only | 128 | – | – | none | none | 1.927427 | 5.152762 | 24.038 | 41.602 |
| Lite Mamba | 128 | 8 | 1 | H+V | uni | 2.088373 | 5.239529 | 16.423 | 60.889 |
| Full Mamba, no presence | 128 | 16 | 2 | H+V | bi | 2.445607 | 5.409346 | 12.459 | 80.265 |
| Full Mamba, presence | 128 | 16 | 2 | H+V | bi | 2.445607 | 5.409346 | 12.428 | 80.464 |
| Full Mamba, one-way | 128 | 16 | 2 | H+V | uni | 2.211285 | 5.297503 | 17.579 | 56.886 |
| Full DBM, `d=64` | 64 | 16 | 2 | H+V | bi | 2.046295 | 5.208942 | 13.643 | 73.297 |
| Full DBM, `d_s=8` | 128 | 8 | 2 | H+V | bi | 2.420967 | 5.401138 | 13.535 | 73.884 |
| Full DBM, `d=192` | 192 | 16 | 2 | H+V | bi | 3.090679 | 5.736981 | 13.384 | 74.714 |
| Full DBM, `e=1` | 128 | 16 | 1 | H+V | bi | 2.212135 | 5.297503 | 13.485 | 74.156 |
| Full Mamba, zero geometry | 128 | 16 | 2 | H+V | bi | 2.445607 | 5.409346 | 12.626 | 79.200 |
| True Horizontal-only | 128 | 16 | 2 | H | bi | 2.194901 | 5.289295 | 17.284 | 57.857 |
| True Vertical-only | 128 | 16 | 2 | V | bi | 2.194901 | 5.289295 | 17.247 | 57.982 |
| DBM-SLANet | 128 | 16 | 2 | H+V | bi | 2.445607 | 5.409346 | 12.455 | 80.292 |

主要变化：

- one-way 不再使用 `≈2.28M/≈5.28G`，而是精确的 2.211285 M/5.297503 G。
- 真实单轴不实例化未使用轴，参数量由旧表误导性的 2.445607 M 降为 2.194901 M。
- Horizontal-only 与 Vertical-only 参数量/FLOPs 相同是正确结果，因为二者是对称结构；模型不同体现在轴方向和实测性能，而不是参数数量。
- `d=192` 明显更大但没有带来性能提升，可作为“性能不是参数堆叠所得”的负控制。

## 五、Table 3 和 Table 4：PubTabNet-5000 最终结果

Table 3 使用 TEDS、Acc.、FPS、Latency；Table 4 使用同一次预测的 Topology F1 和 Span-F1。下表合并展示最终数据，排版时再按论文列拆分。

| Model | TEDS | Acc. | Topology F1 | Span-F1 | FPS | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SLANet | 0.833090 | 0.6898 | 0.960107 | 0.673797 | 24.636 | 40.591 |
| Cell Token only | 0.832060 | 0.6962 | 0.959172 | 0.679183 | 24.038 | 41.602 |
| Lite Mamba | 0.836651 | 0.7084 | 0.957574 | 0.691496 | 16.423 | 60.889 |
| Full Mamba, no presence | 0.830761 | 0.6938 | 0.958653 | 0.675810 | 12.459 | 80.265 |
| Full Mamba, presence | 0.834451 | 0.7016 | 0.960240 | 0.686676 | 12.428 | 80.464 |
| Full Mamba, one-way | 0.835282 | 0.7018 | 0.959785 | 0.685627 | 17.579 | 56.886 |
| Full DBM, `d=64` | 0.835356 | 0.7010 | 0.959754 | 0.683495 | 13.643 | 73.297 |
| Full DBM, `d_s=8` | 0.835802 | 0.7018 | 0.959076 | 0.684840 | 13.535 | 73.884 |
| Full DBM, `d=192` | 0.835805 | 0.7026 | 0.959319 | 0.683691 | 13.384 | 74.714 |
| Full DBM, `e=1` | 0.835875 | 0.7018 | 0.959107 | 0.685622 | 13.485 | 74.156 |
| Full Mamba, zero geometry | 0.830772 | 0.6936 | 0.958653 | 0.675841 | 12.626 | 79.200 |
| True Horizontal-only | 0.835092 | 0.7022 | 0.959811 | 0.684579 | 17.284 | 57.857 |
| True Vertical-only | 0.835514 | 0.7036 | 0.959799 | 0.684681 | 17.247 | 57.982 |
| DBM-SLANet | **0.843986** | **0.7182** | **0.961003** | **0.704468** | 12.455 | 80.292 |

本轮 14 模型中，DBM-SLANet 在 PubTabNet-5000 的 TEDS、Acc.、Topology F1 和 Span-F1 均最高，但速度明显低于 SLANet。该结论只适用于表中这些最终检查点；不能把它替换成“相同初始化与相同步数下 DBM 必然优于调参后的 SLANet”，因为公平控制实验不支持后一种强表述。

## 六、Table 5：CTA-500 最终结果

| Model | TEDS | Acc. | Topology F1 | Span-F1 | FPS |
|---|---:|---:|---:|---:|---:|
| SLANet | 0.484335 | 0.208 | 0.891642 | 0.587424 | 24.636 |
| Cell Token only | 0.543644 | 0.284 | 0.898967 | 0.621918 | 24.038 |
| Lite Mamba | 0.605711 | 0.386 | 0.895811 | 0.642716 | 16.423 |
| Full Mamba, no presence | 0.539207 | 0.282 | 0.897682 | 0.617009 | 12.459 |
| Full Mamba, presence | 0.542149 | 0.302 | 0.897894 | 0.613752 | 12.428 |
| Full Mamba, one-way | 0.546673 | 0.300 | 0.891796 | 0.615024 | 17.579 |
| Full DBM, `d=64` | 0.543784 | 0.290 | 0.891696 | 0.608184 | 13.643 |
| Full DBM, `d_s=8` | 0.549311 | 0.300 | 0.892709 | 0.619550 | 13.535 |
| Full DBM, `d=192` | 0.546811 | 0.302 | 0.891302 | 0.615023 | 13.384 |
| Full DBM, `e=1` | 0.549363 | 0.302 | 0.892554 | 0.620471 | 13.485 |
| Full Mamba, zero geometry | 0.539207 | 0.282 | 0.897682 | 0.617009 | 12.626 |
| True Horizontal-only | 0.546040 | 0.294 | 0.893229 | 0.615583 | 17.284 |
| True Vertical-only | 0.545614 | 0.296 | 0.893110 | 0.616069 | 17.247 |
| DBM-SLANet | **0.640673** | **0.432** | **0.904405** | **0.661715** | 12.455 |

与旧 Table 5 相比，最重要的替换是：旧 `Horizontal Mamba=0.6276`、`Vertical Mamba=0.6258` 属于旧 continuation/屏蔽输出实现，不能改名后继续作为真实单轴结果。真实单轴 TEDS 分别为 0.546040 和 0.545614。

## 七、Table 6：金融域适配汇总

Table 6 中 pilot 与 independent train2000 结果可以保留：

| Setting | Test set | Model | TEDS | Acc. | FPS | N |
|---|---|---|---:|---:|---:|---:|
| Pilot split | heldout75 | SLANet-ft | 0.4165 | 0.1467 | 33.74 | 75 |
| Pilot split | heldout75 | DBM-SLANet | 0.5520 | 0.2667 | 16.59 | 75 |
| Independent train2000 | cached-test500 | SLANet | 0.4396 | 0.0540 | 32.76 | 500 |
| Independent train2000 | cached-test500 | DBM-SLANet | 0.4877 | 0.0660 | 18.85 | 500 |
| Independent train2000 | company-heldout100 | SLANet | 0.4211 | 0.0200 | 22.24 | 100 |
| Independent train2000 | company-heldout100 | DBM-SLANet | 0.4558 | 0.0700 | 17.69 | 100 |

第九版中 `unseen-company100` 的 0.4739/0.6127 是单次 100 样本历史划分。它可以作为 pilot 结果保留在附录，但不能继续作为主要 company-disjoint 泛化证据；Table 9 的扩展三种子结果应当具有更高证据优先级。

## 八、Table 7：CTA-500 分类结果

该表使用最终 SLANet 与最终 DBM-SLANet，预测未变，保留以下结果：

| Category | N | SLANet | DBM-SLANet | Gain |
|---|---:|---:|---:|---:|
| Hierarchical span header | 274 | 0.4753 | 0.6759 | +0.2007 |
| Multi-level header | 325 | 0.4604 | 0.6523 | +0.1919 |
| Vertical dependency | 286 | 0.4496 | 0.6389 | +0.1894 |
| Horizontal dependency | 376 | 0.4581 | 0.6461 | +0.1881 |
| Row/column span | 457 | 0.4627 | 0.6479 | +0.1853 |
| Long-range dependency | 473 | 0.4891 | 0.6304 | +0.1413 |

各类别相互重叠，Gain 不能相加。该表证明最终模型在 CTA-500 的特定复杂结构类别上表现较好，但仍属于同源诊断子集分析。

## 九、Table 8：Independent train2000

| Model | Test set | TEDS | Acc. | FPS | N |
|---|---|---:|---:|---:|---:|
| SLANet | cached-test500 | 0.4396 | 0.0540 | 32.76 | 500 |
| DBM-R2 | cached-test500 | 0.4877 | 0.0660 | 18.85 | 500 |
| SLANet | company-heldout100 | 0.4211 | 0.0200 | 22.24 | 100 |
| DBM-R2 | company-heldout100 | 0.4558 | 0.0700 | 17.69 | 100 |

该实验与本轮 PubTabNet 架构扫描相互独立，数值不变。正文可表述为独立 train2000 设置下的正向结果，但不应覆盖 Table 9 更严格的多种子 company-disjoint 结论。

## 十、Table 9：扩展 Company-Disjoint 最终结果

每个 seed 使用 25 家公司、251 张训练表，以及互斥的 25 家公司、249 张测试表。

| Seed | Model | TEDS | Acc. | FPS | N |
|---:|---|---:|---:|---:|---:|
| 20260806 | SLANet | 0.494855 | 0.220884 | 37.353 | 249 |
| 20260806 | DBM-SLANet | 0.494946 | 0.220884 | 21.530 | 249 |
| 20260807 | SLANet | 0.449169 | 0.204819 | 38.403 | 249 |
| 20260807 | DBM-SLANet | 0.446978 | 0.204819 | 21.829 | 249 |
| 20260808 | SLANet | 0.427055 | 0.188755 | 37.999 | 249 |
| 20260808 | DBM-SLANet | 0.427158 | 0.188755 | 21.773 | 249 |
| Mean ± SD | SLANet | 0.457026 ± 0.034577 | 0.204819 ± 0.016064 | 37.918 ± 0.530 | 3×249 |
| Mean ± SD | DBM-SLANet | 0.456361 ± 0.034855 | 0.204819 ± 0.016064 | 21.711 ± 0.159 | 3×249 |

逐 seed 的 DBM-SLANet TEDS 差值为 +0.000091、−0.002191、+0.000103，平均为 −0.000666。最终结论必须写为：在该扩展协议下，没有证据支持 DBM 改善未见公司金融泛化；模型速度成本在三种子上稳定存在。

因此，第九版正文中“TEDS 从 0.4739 提升至 0.6127，证明未见公司泛化”的表述必须删除或降级为历史 pilot 观察。

## 十一、Table 10：Phase 11 与结构边界消融

| Variant | Params (M) | FLOPs (G) | TEDS | Acc. | Topology F1 | Span-F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cell Token only | 1.927427 | 5.152762 | 0.543644 | 0.284 | 0.898967 | 0.621918 | 24.038 |
| Lite Mamba | 2.088373 | 5.239529 | 0.605711 | 0.386 | 0.895811 | 0.642716 | 16.423 |
| Full Mamba, no presence | 2.445607 | 5.409346 | 0.539207 | 0.282 | 0.897682 | 0.617009 | 12.459 |
| Full Mamba, presence | 2.445607 | 5.409346 | 0.542149 | 0.302 | 0.897894 | 0.613752 | 12.428 |
| Full Mamba, one-way | 2.211285 | 5.297503 | 0.546673 | 0.300 | 0.891796 | 0.615024 | 17.579 |
| Full DBM, `d=64` | 2.046295 | 5.208942 | 0.543784 | 0.290 | 0.891696 | 0.608184 | 13.643 |
| Full DBM, `d_s=8` | 2.420967 | 5.401138 | 0.549311 | 0.300 | 0.892709 | 0.619550 | 13.535 |
| Full DBM, `d=192` | 3.090679 | 5.736981 | 0.546811 | 0.302 | 0.891302 | 0.615023 | 13.384 |
| Full DBM, `e=1` | 2.212135 | 5.297503 | 0.549363 | 0.302 | 0.892554 | 0.620471 | 13.485 |
| Full Mamba, zero geometry | 2.445607 | 5.409346 | 0.539207 | 0.282 | 0.897682 | 0.617009 | 12.626 |
| True Horizontal-only | 2.194901 | 5.289295 | 0.546040 | 0.294 | 0.893229 | 0.615583 | 17.284 |
| True Vertical-only | 2.194901 | 5.289295 | 0.545614 | 0.296 | 0.893110 | 0.616069 | 17.247 |

历史 `Aux. off at inference` 只有 TEDS=0.54215、Acc.=0.302、FPS=18.68，其日志审计状态为 `incomplete`，且没有同协议样本级预测。可以在文字中作为历史速度观察说明，但不能伪造 Topology F1、Span-F1 后填入完整表格。

Table 10 混合了历史 Phase 11 检查点和后来补充的匹配架构扫描检查点，因此适合做结果总览；如果正文要声称单一组件的因果贡献，应优先引用 matched-protocol 架构扫描，而不是跨训练阶段直接比较差值。

## 十二、Table 11：真实轴向消融

### 12.1 与当前主表一致的完整指标版

| Model | TEDS | Acc. | Topology F1 | Span-F1 | FPS | ΔTEDS vs SLANet |
|---|---:|---:|---:|---:|---:|---:|
| SLANet | 0.484335 | 0.208 | 0.891642 | 0.587424 | 24.636 | – |
| True Horizontal-only | 0.546040 | 0.294 | 0.893229 | 0.615583 | 17.284 | +0.061704 |
| True Vertical-only | 0.545614 | 0.296 | 0.893110 | 0.616069 | 17.247 | +0.061279 |
| Final DBM-SLANet | 0.640673 | 0.432 | 0.904405 | 0.661715 | 12.455 | +0.156337 |

这四行都经过相同样本和相同后处理评估，可以作为描述性结果表。但 Final DBM-SLANet 使用渐进训练检查点，真实单轴使用匹配架构扫描检查点，因此 Final DBM 与单轴之间的差值不能全部归因于“增加第二个轴”。

### 12.2 严格匹配训练协议的轴向结论

| Model | Params | FLOPs | TEDS | Acc. | FPS |
|---|---:|---:|---:|---:|---:|
| Full dual-axis bidirectional | 2,445,607 | 5.409346 G | 0.549239 | 0.302 | 10.278 |
| True Horizontal-only | 2,194,901 | 5.289295 G | 0.546040 | 0.294 | 13.894 |
| True Vertical-only | 2,194,901 | 5.289295 G | 0.545614 | 0.296 | 13.809 |

在相同初始化、相同 30,000 样本、相同 7,500 steps、`loc_weight=10`、GT mask off 条件下，完整双轴比 Horizontal-only 高 0.003199 TEDS，比 Vertical-only 高 0.003624；Exact 分别高 0.008 和 0.006，但速度慢约 26%。该结果支持“双轴有小幅互补收益并伴随效率代价”，不支持把 0.0946 的 Final DBM–Horizontal 差值全部解释为轴向贡献。

匹配 full dual-axis 评估只保存了 TEDS、Acc. 和 FPS，没有保存可重算 Topology/Span 的样本级 diagnostics。因此，严格因果版 Table 11 应使用上述五列；若必须保留 Topology F1/Span-F1，需要对 matched full dual-axis checkpoint 再运行一次 diagnostics evaluator。不能用 Final DBM 的 Topology/Span 代替 matched full dual-axis。

## 十三、必须从第九版删除或改写的旧数值

1. 删除所有 `Span-F1=1.0`、`gt_span_samples=0` 的中间表。
2. Table 2 中 one-way 的 `≈2.28M/≈5.28G` 替换为 2.211285 M/5.297503 G。
3. Table 2 中旧 Horizontal/Vertical 的 2.45 M/5.41 G 不能继续标作真实单轴。
4. Table 5、Table 11 中旧 Horizontal=0.6276、Vertical=0.6258 属于旧 continuation/屏蔽输出模型，不能作为真实单轴结果。
5. 删除第九版关于“Dual-branch 相对 stronger single-axis 平均仅 +0.0130”的旧解释；它基于旧单轴实现。
6. Table 9 删除“DBM 将 unseen-company TEDS 从 0.4739 提升至 0.6127，因此提高跨公司泛化”的主结论；扩展三种子平均差为 −0.000666。
7. Table 10 不应给 incomplete `aux-off` 行补造 Topology/Span 指标。
8. 所有正文 FPS 必须与对应表使用同一统一评估批次，不能混用旧 CTA 专项评估的 8.14/9.65 FPS 和本轮 PubTabNet-5000 统一评估的 12.455 FPS。

## 十四、更新后能够支持的论文结论

1. 最终 DBM-SLANet 在当前 14 个最终检查点中取得最高 PubTabNet-5000 和 CTA-500 核心结构指标，尤其在 CTA-500 的复杂 span 结构上表现最好。
2. 真实单轴模型确实更小、更快；匹配协议下完整双轴取得约 0.0032–0.0036 的小幅 TEDS 优势，说明两轴存在有限互补，而不是数量级很大的单独贡献。
3. `d=192` 没有带来性能提升，说明结果不能简单归因于增加参数；`e=1`、one-way 和单轴模型给出了明确的精度–效率边界。
4. presence、geometry 与 directionality 的结果支持这些设计参与性能形成，但跨训练阶段的消融只能作为关联证据，严格因果解释应使用 matched protocol。
5. DBM 的速度成本稳定存在；任何精度结论都应同时报告 FPS/Latency。
6. 扩展 company-disjoint 实验不支持 DBM 改善未见公司泛化。因此论文的金融域贡献应聚焦“领域适配后的特定设置表现”和“存在明显域差异”，而不是宣称稳定跨公司优势。

## 十五、最终文件与证据索引

| 内容 | 本地路径 |
|---|---|
| 修正复核详细报告 | `remote_results/main_tables_rebuild_20260811/corrected/RECHECK_REPORT_20260811.md` |
| Table 2 修正版 | `remote_results/main_tables_rebuild_20260811/corrected/table2_profile_corrected.csv` |
| Table 3 修正版 | `remote_results/main_tables_rebuild_20260811/corrected/table3_pubtabnet5000_corrected.csv` |
| Table 4 修正版 | `remote_results/main_tables_rebuild_20260811/corrected/table4_pubtabnet5000_comprehensive_corrected.csv` |
| Table 5 修正版 | `remote_results/main_tables_rebuild_20260811/corrected/table5_cta500_from_pub5000_corrected.csv` |
| Table 10 修正版 | `remote_results/main_tables_rebuild_20260811/corrected/table10_phase11_cta500_corrected.csv` |
| Table 11 完整指标版 | `remote_results/main_tables_rebuild_20260811/corrected/table11_true_axis_rerun_corrected.csv` |
| 架构扫描与严格匹配轴向结果 | `remote_results/architecture_sweep_true_axis_20260810/results.csv` |
| 扩展 company-disjoint 结果 | `remote_results/fair_controls_and_company_disjoint_20260807/RESULTS.md` |
| 可复现指标重算脚本 | `scripts/recompute_main_table_structure_metrics_20260811.py` |

## 十六、最终审核结论

从“实验是否已产生结果”看，本轮主表更新所需的训练、统一推理、结构指标重算和架构 profile 已经完成。从“表格能否不加限定地直接替换第九版”看，需要遵守三项最终规则：

1. Table 9 必须换成扩展三种子结果，并同步删除跨公司泛化提升的强结论。
2. Table 11 必须明确区分完整指标描述版和 matched-protocol 因果消融版；不能把 Final DBM 与真实单轴的全部差值归因于第二个轴。
3. incomplete aux-off 行不得补造缺失指标。

满足以上规则后，Table 1–11 的数据来源、模型实现和结论边界才能保持一致。
