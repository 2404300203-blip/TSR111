# FinTabNet.c 金融域实验进展记录（2026-07-09）

## 结论摘要

当前最有价值的结果是：在 FinTabNet.c cached-test500 的公司级隔离切分上，DBM-R2 明显超过 SLANet baseline。

公司级隔离切分意味着 train、val、heldout 的公司互不重叠，比随机切分更能说明跨公司泛化能力。

| 模型/策略 | Heldout TEDS | Structure Acc | FPS | 样本数 |
|---|---:|---:|---:|---:|
| SLANet baseline 微调前 | 0.1626 | 0.0000 | 26.72 | 100 |
| DBM-SLANet 微调前 | 0.1590 | 0.0000 | 11.66 | 100 |
| SLANet baseline 金融微调 | 0.4739 | 0.2000 | 36.37 | 100 |
| **DBM-R2 金融微调** | **0.6127** | **0.3500** | 12.11 | 100 |

DBM-R2 相比 baseline 金融微调：

- TEDS 提升：+0.1389
- Structure Acc 提升：0.20 -> 0.35
- 代价：推理速度下降，FPS 约为 baseline 的 1/3

## 实验设置

数据来源：

- 远程服务器：`/root/autodl-tmp/PaddleOCR`
- 数据集：FinTabNet.c cached-test500
- 切分目录：`train_data/table/fintabnet_processed/fintabnet_c_company_disjoint_500/`
- 输出目录：`output/fintabnet_c_company_disjoint_r2_20260709/`

公司级隔离切分：

| Split | 样本数 | 公司数 |
|---|---:|---:|
| train | 349 | 35 |
| val | 51 | 5 |
| heldout | 100 | 10 |

DBM-R2 策略：

1. 先用金融域数据微调 SLANet baseline。
2. 再用该 baseline 金融权重部分初始化 DBM-SLANet。
3. DBM 微调时使用 `lr=1e-5`、`loc_weight=2`、`cell_presence_weight=0`。

## 对比实验脉络

早期零样本跨域测试中，DBM 没有超过 baseline：

| 数据集 | 模型 | TEDS | Structure Acc | FPS | 样本数 |
|---|---|---:|---:|---:|---:|
| FinTabNet original test500 | SLANet baseline | 0.2270 | 0.0000 | 28.16 | 500 |
| FinTabNet original test500 | DBM-SLANet | 0.2049 | 0.0000 | 11.86 | 500 |
| FinTabNet.c cached-test500 | SLANet baseline | 0.1928 | 0.0000 | 36.20 | 500 |
| FinTabNet.c cached-test500 | DBM-SLANet | 0.1783 | 0.0000 | 12.75 | 500 |

随机 pilot split 中，R2 已经显示出明显收益：

| 模型/策略 | Heldout TEDS | Structure Acc | FPS | 样本数 |
|---|---:|---:|---:|---:|
| SLANet baseline 微调前 | 0.2091 | 0.0000 | 30.87 | 75 |
| SLANet baseline 金融微调 | 0.4165 | 0.1467 | 33.74 | 75 |
| 原 DBM 微调 | 0.1894 | 0.0000 | 11.78 | 75 |
| DBM-R1 | 0.4442 | 0.2000 | 13.11 | 75 |
| DBM-R2 | 0.5520 | 0.2667 | 16.59 | 75 |

公司级隔离结果进一步说明：R2 的提升不是简单依赖同公司版式记忆。

## Heldout 复杂度概况

公司级隔离 heldout 共 100 张表：

| 指标 | 最小值 | 中位数 | 均值 | 最大值 |
|---|---:|---:|---:|---:|
| structure tokens | 20 | 80 | 98.55 | 364 |
| cells | 6 | 26 | 36.34 | 157 |
| rows | 3 | 8 | 9.71 | 36 |
| span 数 | 0 | 2 | 2.23 | 8 |
| empty cells | 0 | 1 | 1.23 | 21 |

补充统计：

- 含 rowspan/colspan 的表：66/100
- 无 span 的表：34/100
- structure tokens > 150：14/100
- cells > 50：22/100
- cells > 100：5/100

这说明 heldout 不是纯简单表，包含一定比例的跨行/跨列和大表。

## 当前证据能写什么

可以写入论文实验部分的方向：

1. 零样本跨域泛化不足：PubTabNet 训练的模型直接迁移到金融表格时 TEDS 很低。
2. 金融域微调有效：SLANet baseline 经金融域微调后显著提升。
3. DBM 需要正确初始化和训练策略：原始 DBM 微调不稳定，R2 策略显著优于原始 DBM。
4. 在公司级隔离 heldout 上，DBM-R2 相比 SLANet baseline 仍有明显提升，说明其对跨公司金融表格结构有更强建模能力。

目前还需要谨慎表述：

- 这些结果仍基于 cached-test500 的再切分，属于强诊断实验，不应直接包装成完整官方 benchmark。
- 最终论文主表仍应补全官方 FinTabNet.c train/val/test 子集。

## 正式数据补全计划

服务器目前没有 Kaggle CLI，也没有 `~/.kaggle/kaggle.json`，暂时不能直接下载官方 PDF。

已生成正式子集规划清单：

- 远程目录：`output/fintabnet_c_official_subset_plan_20260709/`
- 目标规模：train 2000、val 300、test 500
- 结构长度过滤：`max_structure_len=500`

规划结果：

| Split | 目标表数 | 唯一 PDF | 当前已缓存表 | 缺失表 | 公司数 |
|---|---:|---:|---:|---:|---:|
| train | 2000 | 1696 | 0 | 2000 | 328 |
| val | 300 | 262 | 0 | 300 | 50 |
| test | 500 | 433 | 256 | 244 | 50 |

总计：

- 目标唯一 PDF：2391
- 当前缺失唯一 PDF：2173

关键清单文件：

- `output/fintabnet_c_official_subset_plan_20260709/train_target_2000_missing_pdfs.txt`
- `output/fintabnet_c_official_subset_plan_20260709/val_target_300_missing_pdfs.txt`
- `output/fintabnet_c_official_subset_plan_20260709/test_target_500_missing_pdfs.txt`
- `output/fintabnet_c_official_subset_plan_20260709/manifest.json`

## 下一步建议

优先级 1：补齐官方子集 PDF。

可选路径：

- 在服务器配置 Kaggle CLI 和 `kaggle.json`，直接下载缺失 PDF。
- 或者在本地下载缺失 PDF 后上传到服务器 `train_data/table/fintabnet_processed/pdf_cache/`。

优先级 2：用官方子集重跑主实验。

主实验最少需要：

| 实验 | 训练集 | 测试集 | 模型 |
|---|---|---|---|
| B0 | FinTabNet.c train2000 | FinTabNet.c test500 | SLANet baseline |
| B1 | FinTabNet.c train2000 | FinTabNet.c test500 | DBM-R2 |
| B2 | FinTabNet.c train2000 | FinTabNet.c company-heldout | DBM-R2 |

优先级 3：补充分析指标。

建议增加：

- span 表 vs non-span 表 TEDS
- 大表 vs 小表 TEDS
- structure token 长度分桶 TEDS
- 推理速度与参数量对比
- 错例可视化：baseline 错、DBM-R2 对的复杂金融表

