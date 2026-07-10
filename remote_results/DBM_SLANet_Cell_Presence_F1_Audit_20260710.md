# DBM-SLANet Cell Presence F1 指标代码核验报告

日期：2026-07-10  
核验目标：确认 `Cell Presence F1=1.0` 是否可以作为论文核心实验指标。  
结论：**不建议使用该指标作为核心证据。当前 `F1=1.0` 存在标签泄漏，指标不可信。**

## 1. 核验对象

远程服务器代码路径：

- `/root/autodl-tmp/PaddleOCR/scripts/eval_cell_presence_codex.py`
- `/root/autodl-tmp/PaddleOCR/ppocr/modeling/heads/cell_full_mamba_head.py`
- `/root/autodl-tmp/PaddleOCR/ppocr/losses/table_att_loss.py`
- `/root/autodl-tmp/PaddleOCR/ppocr/data/imaug/label_ops.py`

原始指标文件：

- `/root/autodl-tmp/PaddleOCR/output/cta500_aux_metrics_20260708/phase5e_cell_presence_cta500.json`

## 2. 原始指标结果

原始 JSON 中报告：

| Metric | Value |
|---|---:|
| samples | 500 |
| positions | 131996 |
| positives | 49682 |
| negatives | 82314 |
| positive rate | 0.37639 |
| threshold 0.5 precision | 1.0 |
| threshold 0.5 recall | 1.0 |
| threshold 0.5 F1 | 1.0 |
| threshold 0.5 accuracy | 1.0 |

异常点：

- 从 threshold `0.05` 到 `0.95`，F1 全部为 `1.0`；
- 每个 batch 的 `score_mean` 几乎等于 `positive / positions`；
- 这说明预测分数接近二值化标签，而不像真实模型概率输出。

## 3. 标签定义

在 `label_ops.py` 中，`bbox_masks` 的构造逻辑为：

```python
if self.idx2char[token] in self.td_token:
    if "bbox" in cells[bbox_idx] and len(cells[bbox_idx]["tokens"]) > 0:
        bbox_masks[i] = 1.0
```

因此当前所谓 `cell presence` 标签并不是泛化意义上的“是否存在 cell”，而是：

> 当前解码位置是否对应一个带有 bbox 且含文本 token 的 `<td>` 单元格。

它更准确的名字应是：`cell_has_supervised_text_bbox`。

## 4. 评估脚本问题

`eval_cell_presence_codex.py` 中存在如下逻辑：

```python
preds = model(images, data=batch[1:])
logits = preds["cell_has_text_logits"]
bbox_mask = batch[3].astype("float32")[:, : max_len + 1]
target = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)
```

问题在于：

1. `target` 来自 `batch[3]`，即 GT `bbox_masks`；
2. 同时，评估时又把 `batch[1:]` 作为 `data` 传入模型；
3. 对于 `CellFullMambaHead`，`targets` 中的 `bbox_mask` 会进入 forward 过程，作为 Mamba 分支的 mask。

也就是说，评估时模型并不是在纯推理条件下预测 presence，而是间接看到了 GT presence mask。

## 5. 模型 forward 中的标签泄漏位置

`cell_full_mamba_head.py` 中：

```python
if targets is not None and len(targets) >= 3:
    bbox_mask = targets[2].astype("float32")
    bbox_mask = bbox_mask[:, : tokens.shape[1]]
    mask = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)
```

由于评估脚本调用：

```python
model(images, data=batch[1:])
```

所以 `targets[2]` 实际对应原始 batch 中的 `batch[3]`，也就是 GT `bbox_masks`。

这会让辅助分支在评估时直接使用 GT mask，导致 F1 虚高。

## 6. 对照实验

我在远程服务器上做了一个小对照实验，使用同一 checkpoint 和同一批 CTA500 数据，比较两种调用方式：

1. **with targets**：`model(images, data=batch[1:])`
2. **without targets**：`model(images)`

输出文件：

- `/root/autodl-tmp/PaddleOCR/output/cta500_aux_metrics_20260708/cell_presence_leak_check_phase5e.json`

前 20 个 batch 的结果如下：

| 调用方式 | Precision | Recall | F1 | Accuracy | 现象 |
|---|---:|---:|---:|---:|---|
| with targets | 1.000 | 1.000 | 1.000 | 1.000 | 使用 GT mask，结果完美 |
| without targets | 0.349 | 1.000 | 0.517 | 0.349 | 分数全为 1.0，退化为全正预测 |

其中 without targets 的分数统计为：

| score min | score max | score mean | unique rounded |
|---:|---:|---:|---:|
| 1.0 | 1.0 | 1.0 | 1 |

这进一步说明：当前 `Cell Presence F1=1.0` 不是可靠的独立预测能力，而是由评估方式引入 GT mask 后产生的虚高结果。

## 7. 结论

`Cell Presence F1=1.0` **不能作为论文核心指标**。原因如下：

- 评估时 `model(images, data=batch[1:])` 把 GT 信息传入模型；
- `CellFullMambaHead.forward()` 使用 `targets[2]` 的 `bbox_mask` 作为分支 mask；
- `target` 本身也来自同一个 `bbox_mask`；
- 因此该指标存在标签泄漏；
- 去掉 targets 后，辅助分支 presence 输出退化为全正预测，F1 明显下降。

## 8. 论文中应如何处理

建议在论文中：

1. **不要报告 `Cell Presence F1=1.0` 作为核心指标**；
2. 可以删除该指标，或只在内部实验记录中保留并注明“不作为最终评价”；
3. 证明 `cell presence` 辅助监督有效时，应使用更可信的消融结果：
   - `lambda_pres=0` vs `lambda_pres=0.2`
   - inference auxiliary off
   - TEDS / Structure Acc / FPS

推荐替代表述：

> The cell-presence branch is used as a training-time auxiliary objective rather than an independent inference target. Therefore, we evaluate its contribution through controlled ablations on TEDS, structure accuracy, and inference speed, instead of reporting standalone cell-presence F1.

中文表述：

> cell presence 分支主要作为训练期辅助监督使用，而不是独立推理目标。因此本文不将单独的 cell-presence F1 作为核心指标，而是通过 `lambda_pres` 消融、推理阶段关闭辅助分支，以及 TEDS、Structure Acc 和 FPS 的变化来验证其贡献。

## 9. 如果后续仍想保留该指标

如果确实需要一个可信的 presence 指标，需要修改评估逻辑：

- 评估时调用 `model(images)`，不能传入 `data=batch[1:]`；
- 或者修改 `CellFullMambaHead.forward()`，使其在 `not self.training` 时不使用 GT `bbox_mask`；
- 重新训练或校准 presence head，使其在无 GT mask 的情况下能输出有效概率；
- 重新报告 threshold sweep、PR 曲线或 F1。

在当前实现下，不建议继续使用该指标。

