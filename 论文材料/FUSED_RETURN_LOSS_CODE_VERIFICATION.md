# Fused Token、Forward 返回值与 Loss 代码核验

核验时间：2026-07-05  
远程代码路径：

```text
/root/autodl-tmp/PaddleOCR
```

重点核验文件：

```text
ppocr/modeling/heads/cell_full_mamba_head.py
ppocr/modeling/heads/cell_token_mlp_head.py
ppocr/modeling/heads/table_att_head.py
ppocr/losses/table_att_loss.py
configs/table/DBM_SLANet_phase5_full_mamba_stage5_30000_lr5e6_from_stage4_codex.yml
```

---

## 1. `fused` 是否与原始 logits 相加

结论：**没有。**

当前代码中，`fused` 的生成位置是：

```python
fused = self.fusion(paddle.concat([tokens, h_encoded, v_encoded], axis=-1))
out["cell_tokens"] = fused
out["horizontal_cell_tokens"] = h_encoded
out["vertical_cell_tokens"] = v_encoded
out["cell_has_text_logits"] = self.cell_presence_head(fused).squeeze(-1)
return out
```

但 `structure_probs` / structure logits 是在更早的 `SLAHead.forward()` 中生成的：

```python
hidden, structure_step, loc_step = self._decode(...)
structure_preds[:, i, :] = structure_step
loc_preds[:, i, :] = loc_step
...
return {"structure_probs": structure_preds, "loc_preds": loc_preds}
```

其中 `structure_step` 来自：

```python
structure_step = self.structure_generator(output)
```

代码中没有出现：

```python
structure_probs = structure_probs + f(fused)
```

也没有出现：

```python
structure_logits = structure_logits + fused_logits
```

因此，当前实现中：

```text
fused 没有直接与原始 structure logits 相加。
```

---

## 2. `fused` 是否先映射到完整结构序列

结论：**没有映射到完整结构词表 logits。**

`fused` 的形状与 cell/token step 序列对齐，但它没有经过类似：

```python
nn.Linear(cell_token_dim, out_channels)
```

来映射到完整结构 token vocabulary。

当前 `fused` 只进入了：

```python
self.cell_presence_head(fused)
```

对应输出：

```python
out["cell_has_text_logits"]
```

也就是说，`fused` 当前被用于预测每个 cell step 是否存在 supervised bbox/text target，而不是用于直接预测 HTML structure token。

更准确的描述是：

```text
fused is aligned with decoded cell steps and used by an auxiliary cell-presence branch.
```

不应写成：

```text
fused is projected to structure logits and added to the original decoder logits.
```

---

## 3. 定位损失是 L1、Smooth L1 还是其他形式

结论：**Phase5-E 配置使用 Smooth L1。**

配置文件中：

```yaml
Loss:
  name: SLACellTokenLoss
  structure_weight: 1.0
  loc_weight: 10.0
  loc_loss: smooth_l1
  cell_presence_weight: 0.2
```

实际 `SLALoss.forward()` 中使用：

```python
loc_loss = (
    F.smooth_l1_loss(
        loc_preds * loc_targets_mask,
        loc_targets * loc_targets_mask,
        reduction="sum",
    )
    * self.loc_weight
)

loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
```

因此当前定位损失为：

```text
masked Smooth L1 loss
```

计算方式：

```text
loc_loss = SmoothL1(loc_preds * mask, loc_targets * mask)
loc_loss = loc_loss * loc_weight
loc_loss = loc_loss / sum(mask)
```

注意：虽然 `SLALoss.__init__` 接收 `loc_loss="smooth_l1"` 参数，但当前 forward 中实际固定调用的是：

```python
F.smooth_l1_loss(...)
```

没有根据 `self.loc_loss` 分支切换到 L1 或 MSE。

---

## 4. 最终 `loc_weight` 是否确实为 10

结论：**Phase5-E 配置中确实是 10.0。**

配置：

```yaml
loc_weight: 10.0
```

loss 中使用：

```python
self.loc_weight = loc_weight
...
loc_loss = F.smooth_l1_loss(...) * self.loc_weight
```

因此 Phase5-E 的定位损失权重为：

```text
λ_loc = 10.0
```

---

## 5. 是否还有其他辅助损失

结论：**有。当前 Phase5-E 使用 `cell_presence_loss`。**

损失类为：

```yaml
Loss:
  name: SLACellTokenLoss
```

`SLACellTokenLoss` 继承自 `SLALoss`，除了结构损失和定位损失，还会在存在 `cell_has_text_logits` 时加入 cell presence 辅助损失。

代码：

```python
if "cell_has_text_logits" not in predicts:
    return losses

logits = predicts["cell_has_text_logits"]
bbox_mask = batch[3].astype("float32")[:, : max_len + 1]
target = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)

cell_presence_loss = F.binary_cross_entropy_with_logits(
    logits, target, reduction="none"
)
cell_presence_loss = (cell_presence_loss * valid).sum() / (valid.sum() + self.eps)
cell_presence_loss = cell_presence_loss * self.cell_presence_weight
losses["cell_presence_loss"] = cell_presence_loss
losses["loss"] = losses["loss"] + cell_presence_loss
```

配置：

```yaml
cell_presence_weight: 0.2
```

因此总损失为：

```text
loss = structure_loss + loc_loss + cell_presence_loss
```

其中：

```text
structure_loss       = CrossEntropy structure loss
loc_loss             = masked Smooth L1 bbox loss * 10.0
cell_presence_loss   = BCEWithLogits auxiliary loss * 0.2
```

---

## 6. Span 相关辅助损失是否启用

结论：**Phase5-E 主配置中没有显式启用 span-aware loss。**

`SLALoss` 代码中确实保留了 span 相关参数：

```python
span_token_indices=None
span_token_weight=1.0
span_fp_weight=0.0
plain_cell_token_index=None
```

并返回：

```python
"span_token_loss": span_token_loss,
"span_fp_loss": span_fp_loss,
```

但 Phase5-E 配置中没有设置：

```yaml
span_token_indices
span_token_weight
span_fp_weight
plain_cell_token_index
```

所以在 Phase5-E 中：

```text
span_token_loss = 0
span_fp_loss = 0
```

它们只是代码预留/其他 Phase7 实验使用的接口，不是 Phase5-E 的有效主损失项。

---

## 7. 当前实现的关键风险点

当前代码中，`fused` 并没有直接参与结构 logits 的生成。

真实路径是：

```text
SLAHead decoder
  → structure_probs
  → loc_preds
  → GeometryTokenEncoder(loc_preds)
  → cell_tokens
  → horizontal / vertical Mamba
  → fused
  → cell_presence_head
  → cell_presence_loss
```

因此，Mamba 分支对主结构识别的影响更准确地说是：

```text
通过 cell presence auxiliary loss 对 loc_preds / shared decoder 表示产生训练期约束，
而不是在推理阶段直接修正 structure logits。
```

论文中不宜写：

```text
fused features are added to the original structure logits
```

也不宜写：

```text
fused tokens are projected to the full HTML vocabulary for structure decoding
```

更稳妥写法：

```text
The fused cell representation is used by an auxiliary cell-presence prediction branch, which regularizes the cell-level structural representation during training.
```

---

## 8. 建议论文修改点

如果原文里有如下公式：

```text
D_str(F, Z_f)
ℓ_loc
λ_loc
```

建议修改为：

### 8.1 结构解码部分

不要暗示 `Z_f` 直接进入 structure decoder logits。

更准确：

```text
The original SLA decoder produces structure logits and location predictions. The predicted locations are further encoded into geometry tokens, which are processed by horizontal and vertical Mamba branches to form fused cell representations for auxiliary cell-presence supervision.
```

### 8.2 定位损失

明确写：

```text
ℓ_loc = SmoothL1(B_pred ⊙ M, B_gt ⊙ M)
```

并说明：

```text
λ_loc = 10.0
```

### 8.3 辅助损失

加入：

```text
ℓ_cell = BCEWithLogits(p_cell, y_cell)
```

总损失更准确为：

```text
ℓ = ℓ_str + 10.0 · ℓ_loc + 0.2 · ℓ_cell
```

其中：

```text
ℓ_str: structure cross-entropy loss
ℓ_loc: masked Smooth L1 location loss
ℓ_cell: cell-presence binary classification loss
```

注意代码中 `loc_loss` 是先乘 `loc_weight=10.0` 后除以 mask sum，因此论文公式可以等价写成加权定位损失。

---

## 9. 总结表

| 核验项 | 结论 |
|---|---|
| `fused` 是否与原始 logits 相加 | 否 |
| `fused` 是否映射到完整结构词表 | 否 |
| `fused` 当前用途 | 输入 `cell_presence_head` 产生辅助监督 |
| 定位损失形式 | masked Smooth L1 |
| `loc_weight` | 10.0 |
| 其他辅助损失 | `cell_presence_loss`，权重 0.2 |
| span loss 是否在 Phase5-E 启用 | 否，主配置未启用 |
| 总损失 | `structure_loss + loc_loss + cell_presence_loss` |

