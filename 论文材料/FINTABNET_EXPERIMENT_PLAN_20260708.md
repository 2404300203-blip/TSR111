# FinTabNet / FinTabNet.c 金融表格实验计划（2026-07-08）

## 目标

补充公开金融专用数据集实验，回答两个问题：

1. **跨域直接测试**：不使用金融数据训练时，DBM-SLANet 是否比原始 SLANet 具有更好的金融领域迁移能力？
2. **金融领域微调**：在相同 FinTabNet 训练条件下，DBM-SLANet 是否仍优于 SLANet？

按要求同时做两个版本：

- 原始 **FinTabNet**
- 清洗修正版 **FinTabNet.c**

两者结果不能混在同一张主表里，应分别成表，避免标注规则差异影响结论。

## 数据集版本

| 版本 | 用途 | 说明 |
|---|---|---|
| FinTabNet original | 便于和历史论文直接比较 | 原始自动匹配标注，存在一定过切分和跨数据集标注不一致问题 |
| FinTabNet.c | 更适合作为干净结构识别评估 | 经过规范化和自动修正，结构标注更一致 |

## 实验组一：跨域直接测试

设置：

```text
Train on PubTabNet -> Test on FinTabNet
Train on PubTabNet -> Test on FinTabNet.c
```

分别测试：

- SLANet baseline
- DBM-SLANet / Phase5-E Full Dual-Branch Mamba

推荐指标：

- TEDS
- Structure Acc / Exact Match
- FPS
- 按复杂结构标签或 span 数分桶的 TEDS

核心回答：

> 在没有使用金融数据训练的情况下，双轴几何关系建模是否具有更好的金融领域迁移能力？

## 实验组二：金融领域微调

设置：

```text
Fine-tune on FinTabNet -> Test on FinTabNet
Fine-tune on FinTabNet.c -> Test on FinTabNet.c
```

公平性约束：

- 相同训练样本
- 相同初始化
- 相同训练轮数 / step 数
- 相同学习率
- 相同解码与评价脚本
- FinTabNet 原始版和 FinTabNet.c 分开报告

推荐先跑小规模版本：

| 阶段 | Train | Val/Test | 目的 |
|---|---:|---:|---|
| smoke | 100 | 50 | 验证转换和训练流程 |
| pilot | 2,000 | 500 | 观察趋势 |
| formal | 10,000 或全量 | company-disjoint test | 论文主结果 |

## 公司维度划分

如果标注中含公司字段，按公司划分：

- train：50 家公司
- test：50 家公司

如果没有显式公司字段，先用文件名 / 文档 ID 的前缀作为弱替代，并在报告中说明这是 fallback。

## 已新增脚本

本地脚本：

```text
/Users/table/TSR/scripts/prepare_fintabnet_experiments.py
```

用途：

- 读取 FinTabNet / FinTabNet.c JSONL
- 转成 PaddleOCR table JSONL 风格
- 生成 company-disjoint train/test
- 输出 manifest

示例：

```bash
python scripts/prepare_fintabnet_experiments.py \
  --version-name fintabnet_c \
  --train-jsonl train_data/table/fintabnet_c/raw/train.jsonl \
  --val-jsonl train_data/table/fintabnet_c/raw/val.jsonl \
  --test-jsonl train_data/table/fintabnet_c/raw/test.jsonl \
  --output-dir train_data/table/fintabnet_c/processed \
  --image-prefix train_data/table/fintabnet_c/images \
  --train-limit 10000 \
  --val-limit 1000
```

## 当前执行状态

- 已确认服务器 `/root/autodl-tmp` 可用空间约 35GB。
- 已确认服务器缺少 `datasets` 和 `pyarrow`，但有 `huggingface_hub`、`PIL`、`pandas`。
- FinTabNet.c 优先尝试 Hugging Face 下载。
- FinTabNet original 优先尝试 Kaggle / 公开镜像下载。
- 服务器访问 Hugging Face / Kaggle 当前响应较慢，已准备转换脚本，等待数据源可下载后继续。

## 预期论文表格

### Cross-Domain Generalization

| Dataset Version | Train | Test | Model | TEDS | Acc | FPS |
|---|---|---|---|---:|---:|---:|
| FinTabNet original | PubTabNet | FinTabNet | SLANet | TBD | TBD | TBD |
| FinTabNet original | PubTabNet | FinTabNet | DBM-SLANet | TBD | TBD | TBD |
| FinTabNet.c | PubTabNet | FinTabNet.c | SLANet | TBD | TBD | TBD |
| FinTabNet.c | PubTabNet | FinTabNet.c | DBM-SLANet | TBD | TBD | TBD |

### Financial Fine-Tuning

| Dataset Version | Train | Test | Model | TEDS | Acc | FPS |
|---|---|---|---|---:|---:|---:|
| FinTabNet original | FinTabNet | FinTabNet | SLANet | TBD | TBD | TBD |
| FinTabNet original | FinTabNet | FinTabNet | DBM-SLANet | TBD | TBD | TBD |
| FinTabNet.c | FinTabNet.c | FinTabNet.c | SLANet | TBD | TBD | TBD |
| FinTabNet.c | FinTabNet.c | FinTabNet.c | DBM-SLANet | TBD | TBD | TBD |

