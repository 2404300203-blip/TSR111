# TSR：表格结构识别实验项目

这个仓库是基于 PaddleOCR 改造的表格结构识别（Table Structure Recognition, TSR）实验工作区。项目保留 PaddleOCR 原始代码作为基础，在此之上加入了围绕 SLANet、OCR-cell 匹配、局部/全局特征融合、cell-level 辅助监督等方向的实验代码。

当前上传的是远程服务器中的代码快照：

```text
/root/autodl-tmp/PaddleOCR
```

为了保证 GitHub 仓库可正常使用，数据集、模型权重、训练输出等大文件没有直接上传。

## 项目目标

本项目主要研究如何提升表格结构识别效果，尤其关注金融表格和科研论文表格中常见的几个问题：

- 空 `<td></td>` 过多
- OCR 文本与 cell bbox 匹配错误
- 局部结构建模不足
- HTML 重建时内容顺序和跨行跨列结构不稳定

整体目标是在 SLANet 系列表格识别模型的基础上，验证后处理匹配、辅助监督和结构建模模块是否能带来更稳定的表格 HTML 还原效果。

## 实验路线

### Phase 0：代码审计

这一阶段主要定位表格识别链路中的关键模块，包括：

- table matcher
- HTML reconstruction
- 表格推理入口
- 评估脚本
- `dt_boxes`、`rec_res`、`structure_res`、`cell_bbox`、HTML token 等真实字段结构

目的是先搞清楚 PaddleOCR/SLANet 的表格推理结果是如何从模型输出转换成最终 HTML 的。

### Phase 1：OCR-aware Cell Matching

这一阶段新增了一个可选的 OCR 感知 cell 匹配策略，同时保留 legacy matcher 作为默认行为。

关键文件：

- `ppstructure/table/matcher.py`
- `ppstructure/table/predict_table.py`
- `ppstructure/utility.py`
- `tests/test_ocr_aware_table_matcher.py`
- `scripts/eval_table_teds_codex.py`
- `eval_table_teds_codex.py`

核心思路：

- 默认仍然使用 `match_mode=legacy`
- 新增 `match_mode=ocr_aware_hungarian`
- 对 OCR 和 cell 之间的候选匹配进行综合打分
- 打分因素包括 bbox 归一化、IoU、OCR coverage、cell coverage、中心点相似度、inside score、方向先验和 OCR confidence
- 主匹配阶段使用 Hungarian matching
- 主匹配后支持一个 cell 聚合多个 OCR token

第一轮实验验证了代码链路可运行，但 OCR-aware matcher 并没有稳定优于 legacy matcher，因此后续仍需继续分析 HTML 拼接、阅读顺序和 span 结构问题。

### Phase 2：Cell Token MLP Head

这一阶段加入了轻量级 cell token 辅助预测头，用于给表格结构识别增加 cell presence/content 相关监督。

关键文件：

- `ppocr/modeling/heads/cell_token_mlp_head.py`
- `ppocr/losses/table_att_loss.py`
- `configs/table/DBM_SLANet_cell_token_mlp_codex.yml`
- `tests/test_cell_token_mlp_head.py`

该阶段主要验证在不大幅改动主干模型的情况下，能否通过 cell-level 辅助任务改善模型对表格单元格内容存在性的感知。

### Phase 3：Dual-Branch Mamba-Style Head

这一阶段加入了一个轻量级双分支序列建模 head，用于探索更强的表格 token 序列建模能力。

关键文件：

- `ppocr/modeling/heads/cell_dual_branch_mamba_head.py`
- `configs/table/DBM_SLANet_dual_branch_mamba_codex.yml`
- `tests/test_cell_dual_branch_mamba_head.py`

说明：当前实现是一个无额外依赖的 gated recurrent/scan-style 模块，接口上接近 Mamba 风格，但不是完整 selective-scan Mamba 实现。

## 其他实验配置

表格相关实验配置主要放在：

```text
configs/table/
```

其中包括：

- `SLANet_pubtabnet_loc_codex.yml`
- `SLANet_pubtabnet_lsnet_codex.yml`
- `SLANet_pubtabnet_fsam_codex.yml`
- `SLANet_plus_E0_baseline_codex.yml`
- `SLANet_plus_E1_local_only_codex.yml`
- `SLANet_plus_E2_mamba_only_codex.yml`
- `SLANet_plus_E3_mamba_local_parallel_codex.yml`
- `SLANet_plus_E5_fsam_stable_codex.yml`
- `DBM_SLANet_cell_token_mlp_codex.yml`
- `DBM_SLANet_dual_branch_mamba_codex.yml`

这些配置记录了不同阶段对 SLANet、LSNet、FSAM、Mamba-style head 和辅助 cell token 任务的尝试。

## 没有上传的内容

以下内容在远程服务器上可能存在，但没有进入 GitHub 仓库：

- `train_data/`
- `output/`
- `inference/`
- `inference_results/`
- `tmp/`
- `*.pdparams`
- `*.pdopt`
- `*.pdstates`
- `*.states`
- `*.ckpt`
- `*.jsonl`
- `*.zip`
- `*.tar.gz`

原因是这些文件通常是数据集、模型权重、训练日志或实验输出，体积很大，不适合直接放入普通 GitHub 仓库。

如果后续需要保存模型权重或数据集，建议使用：

- GitHub Release
- Git LFS
- 网盘/对象存储
- 单独的数据集仓库

## 常用命令

运行 Phase 1 matcher 单元测试：

```bash
python -m pytest tests/test_ocr_aware_table_matcher.py
```

运行 Phase 2 cell token head 单元测试：

```bash
python -m pytest tests/test_cell_token_mlp_head.py
```

运行 Phase 3 dual-branch head 单元测试：

```bash
python -m pytest tests/test_cell_dual_branch_mamba_head.py
```

检查当前 Git 跟踪文件体积：

```bash
git ls-files -z | xargs -0 du -ch | tail -1
git ls-files -z | xargs -0 du -h | sort -hr | head -30
```

从远程服务器推送到 GitHub：

```bash
git remote set-url tsr ssh://git@ssh.github.com:443/2404300203-blip/TSR.git
git push -u tsr tsr-code-only:main --force
```

## 仓库结构提示

主要实验代码集中在：

```text
ppstructure/table/
ppocr/modeling/heads/
ppocr/losses/
configs/table/
scripts/
tests/
```

其中 `ppstructure/table/` 主要对应后处理和 HTML 重建链路，`ppocr/modeling/heads/` 和 `ppocr/losses/` 主要对应模型结构与训练损失改动。

## 上游项目

本项目基于 PaddleOCR：

```text
https://github.com/PaddlePaddle/PaddleOCR
```

原始 PaddleOCR 的许可证保留在 `LICENSE` 文件中。
