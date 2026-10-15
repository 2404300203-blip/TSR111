# DBM-SLANet 论文图表组织与曲线图规划

日期：2026-07-10  
用途：用于论文写作、组会汇报和后续绘图任务分配。  
项目：DBM-SLANet / Table Structure Recognition  

## 1. 总体建议

本项目已经有较完整的模型设计、消融实验、金融表格实验和指标分析。论文图表不宜堆太多，而应该围绕以下主线组织：

> 方法整体框架 → 核心模块结构 → 主结果对比 → 可视化案例 → 精度-效率权衡 → 附录补充消融和训练曲线

建议正文放 **5 张主图**，附录放 **3 到 4 张补充图**。  
表格部分则放 **1 张主结果表 + 1 张消融主表 + 若干附表**。

## 2. 正文主图规划

| 图号 | 图名 | 类型 | 主要内容 | 作用 |
|---|---|---|---|---|
| Fig. 1 | Overall Framework of DBM-SLANet | 模型框架图 | 输入图像、SLANet backbone、cell token、dual-branch Mamba、HTML/table 输出 | 说明整体方法流程 |
| Fig. 2 | Dual-Branch Mamba Cell Modeling Module | 模块结构图 | Geometry Token Encoder、Horizontal Mamba、Vertical Mamba、Fusion、Cell Presence auxiliary branch | 突出核心创新模块 |
| Fig. 3 | Quantitative Comparison on Table Recognition Benchmarks | 柱状图 / 表格图 | SLANet baseline vs DBM-SLANet / DBM-R2，在 CTA500、FinTabNet.c、company-disjoint 上的结果 | 展示主要性能提升 |
| Fig. 4 | Qualitative Visualization of Table Structure Recovery | 可视化案例图 | 原图、GT、SLANet baseline、DBM-SLANet 预测结果 | 直观展示结构恢复效果 |
| Fig. 5 | Accuracy-Efficiency Trade-off | 曲线图 / 散点图 | TEDS 或 Structure Acc vs FPS | 展示精度与速度权衡 |

## 3. 附录图规划

| 图号 | 图名 | 类型 | 主要内容 | 作用 |
|---|---|---|---|---|
| Fig. S1 | Dataset Construction and Evaluation Splits | 流程图 | FinTabNet / FinTabNet.c 数据处理、pilot split、company-disjoint、train2000 | 说明数据来源和划分方式 |
| Fig. S2 | Complete Ablation Study | 柱状图 | Cell Token only、lambda=0、lambda=0.2、bidirectional=false、geometry zero、aux-off | 补充展示完整消融结果 |
| Fig. S3 | Additional Qualitative Results | 可视化案例图 | 更多复杂表格案例 | 支撑正文可视化结论 |
| Fig. S4 | Training Loss Curves | 曲线图 | total loss、structure loss、loc loss、cell presence loss | 展示训练收敛情况 |
| Fig. S5 | Validation Performance Curves | 曲线图 | validation acc / TEDS 随 step 变化 | 展示训练过程中的性能变化 |
| Fig. S6 | Performance under Different Training Data Scales | 曲线图，可选 | 不同训练数据规模下的 TEDS | 展示数据规模影响 |

## 4. 推荐曲线图

### 4.1 Accuracy-Efficiency Trade-off Curve

优先级：最高  
建议位置：正文 Fig. 5  

#### 画法

- x 轴：FPS
- y 轴：TEDS 或 Structure Acc
- 每个点表示一个模型或消融设置

建议包含以下点：

| Model | TEDS | Structure Acc | FPS |
|---|---:|---:|---:|
| Cell Token only | 0.54364 | 0.284 | 20.33 |
| Full Mamba, lambda=0 | 0.53921 | 0.282 | 9.40 |
| Full Mamba, lambda=0.2 | 0.54215 | 0.302 | 9.54 |
| Bidirectional=false | 0.53040 | 0.272 | 14.14 |
| Geometry zero | 0.53921 | 0.282 | 11.12 |
| Inference auxiliary off | 0.54215 | 0.302 | 18.68 |

#### 重点解释

- `Inference auxiliary off` 是关键点：精度不变，但 FPS 大幅提升；
- `Bidirectional=false` 速度更快，但精度下降；
- Full Mamba 提升结构准确率，但速度低于 Cell Token only；
- 该图可以体现 DBM-SLANet 的精度-效率权衡。

#### 推荐标题

> Fig. 5. Accuracy-efficiency trade-off of DBM-SLANet variants.

### 4.2 Training Loss Curves

优先级：高  
建议位置：附录 Fig. S4  

#### 画法

- x 轴：global step
- y 轴：loss
- 多条曲线：
  - total loss
  - structure loss
  - loc loss
  - cell presence loss

#### 作用

- 证明训练过程稳定；
- 展示 `cell_presence_loss` 随训练下降；
- 支撑辅助分支确实参与训练。

#### 推荐标题

> Fig. S4. Training loss curves of DBM-SLANet.

### 4.3 Validation Performance Curves

优先级：中高  
建议位置：附录 Fig. S5  

#### 画法

- x 轴：global step 或 epoch
- y 轴：validation Structure Acc 或 TEDS
- 对比：
  - SLANet baseline
  - DBM-R2
  - lambda=0
  - lambda=0.2

#### 作用

- 展示模型不是最后一个点偶然变好；
- 说明 DBM-R2 在训练过程中持续获得结构性能提升；
- 适合作为附录补充证据。

#### 注意

如果日志中只有少量 eval 点，则不建议强行画过于平滑的曲线，可以改成折线点图。

#### 推荐标题

> Fig. S5. Validation performance curves during fine-tuning.

### 4.4 TEDS vs Training Data Size Curve

优先级：可选  
建议位置：附录 Fig. S6  

#### 画法

- x 轴：training samples
- y 轴：TEDS
- 两条线：
  - SLANet baseline
  - DBM-R2

可选数据点：

- pilot train350
- train2000
- 若有更多规模，可加入 500、1000、30000

#### 作用

- 说明数据规模变化下 DBM-R2 是否稳定优于 baseline；
- 适合金融表格实验分析。

#### 注意

如果数据点少于 3 个，不建议作为正文主图，可以作为补充分析或不画。

#### 推荐标题

> Fig. S6. Performance under different training data scales.

### 4.5 Cell Presence PR / Threshold Curve

优先级：不推荐  
建议位置：暂不使用  

#### 原因

当前 `Cell Presence F1=1.0` 已经过代码核验，存在标签泄漏：

- 原评估脚本使用 `model(images, data=batch[1:])`；
- GT `bbox_masks` 被传入 `CellFullMambaHead.forward()`；
- 评估 target 也来自同一个 `bbox_masks`；
- 因此 `F1=1.0` 不是可靠的独立预测能力。

#### 建议

论文中不要画 Cell Presence PR 曲线，也不要把 `Cell Presence F1=1.0` 作为核心指标。

更稳妥的写法是：

> cell presence 分支主要作为训练期辅助监督使用，其贡献通过 `lambda_pres=0` vs `lambda_pres=0.2`、推理阶段关闭辅助分支、TEDS、Structure Acc 和 FPS 的变化来验证。

## 5. 正文图表与实验结论对应关系

| 论文论点 | 推荐图表 | 支撑内容 |
|---|---|---|
| DBM-SLANet 的整体设计 | Fig. 1 | 展示完整模型流程 |
| 双分支 Mamba 是核心创新 | Fig. 2 | 展示 horizontal / vertical branch 和 fusion |
| 最终模型优于 baseline | Fig. 3 + 主结果表 | CTA500 / FinTabNet.c / company-disjoint 对比 |
| 模型能改善复杂表格结构恢复 | Fig. 4 | qualitative visualization |
| 模型存在精度-速度权衡 | Fig. 5 | TEDS / Structure Acc vs FPS |
| 消融实验支持模块有效性 | 表 1 + Fig. S2 | lambda、geometry、bidirectional、aux-off |
| 训练稳定 | Fig. S4 | loss curve |
| 金融域适配有效 | FinTabNet 主表 + Fig. S6 可选 | baseline vs DBM-R2 |

## 6. 推荐最终图表清单

### 正文

| 编号 | 内容 | 是否必须 |
|---|---|---|
| Fig. 1 Overall Framework | 必须 |
| Fig. 2 DBM Module Structure | 必须 |
| Fig. 3 Main Quantitative Comparison | 必须 |
| Fig. 4 Qualitative Visualization | 必须 |
| Fig. 5 Accuracy-Efficiency Trade-off | 推荐必须 |
| Table 1 Main Results on Financial Tables | 必须 |
| Table 2 Main Ablation Study | 必须 |

### 附录

| 编号 | 内容 | 是否必须 |
|---|---|---|
| Fig. S1 Dataset Construction and Splits | 推荐 |
| Fig. S2 Complete Ablation Bar Chart | 推荐 |
| Fig. S3 More Qualitative Results | 推荐 |
| Fig. S4 Training Loss Curves | 推荐 |
| Fig. S5 Validation Performance Curves | 可选 |
| Fig. S6 Data Scale Curve | 可选 |
| Table S1 Complete Phase 11 Ablation | 推荐 |
| Table S2 Empty-td Auxiliary Metrics | 可选 |
| Table S3 Zero-shot FinTabNet Results | 推荐 |

## 7. 绘图优先级

如果时间有限，建议按以下顺序完成：

1. Fig. 1 Overall Framework
2. Fig. 2 DBM Module Structure
3. Fig. 5 Accuracy-Efficiency Trade-off
4. Fig. 3 Main Quantitative Comparison
5. Fig. 4 Qualitative Visualization
6. Fig. S4 Training Loss Curves
7. Fig. S2 Complete Ablation Bar Chart
8. Fig. S1 Dataset Construction and Splits

## 8. 最终建议

论文正文不要放太多曲线。最合理的组合是：

- 正文放 1 张曲线图：`Accuracy-Efficiency Trade-off`
- 附录放 1 到 2 张曲线图：`Training Loss Curves`、`Validation Performance Curves`
- 不画 `Cell Presence PR Curve`

这样图表组织会更稳，不会显得为了凑图而凑图，也能清楚支撑论文创新点。

