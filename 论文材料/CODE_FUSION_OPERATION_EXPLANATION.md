# 代码中的 Fusion 操作说明

整理时间：2026-07-04  
项目：DBM-SLANet / Phase5-E Full Mamba  
代码位置：

```text
/root/autodl-tmp/PaddleOCR/ppocr/modeling/heads/cell_full_mamba_head.py
```

## 1. Fusion 的核心代码

当前模型中的 fusion 操作位于 `CellFullMambaHead.forward()` 中，核心代码为：

```python
fused = self.fusion(paddle.concat([tokens, h_encoded, v_encoded], axis=-1))
```

其中：

```text
tokens     = 原始 cell geometry token
h_encoded  = horizontal Mamba 编码后的 cell token
v_encoded  = vertical Mamba 编码后的 cell token
```

## 2. Fusion 层定义

代码中的 fusion 层定义为：

```python
fusion_inputs = cell_token_dim * 3
self.fusion = nn.Sequential(
    nn.Linear(fusion_inputs, cell_token_dim),
    nn.LayerNorm(cell_token_dim),
    nn.GELU(),
    nn.Dropout(cell_token_dropout),
)
```

当前配置中：

```text
cell_token_dim = 128
```

因此输入输出维度为：

```text
tokens:     128 dim
h_encoded:  128 dim
v_encoded:  128 dim
concat:     384 dim
fusion out: 128 dim
```

## 3. Fusion 的具体流程

整体流程可以表示为：

```text
cell bbox prediction
        ↓
GeometryTokenEncoder
        ↓
tokens
        ↓
┌──────────────────────────────┬──────────────────────────────┐
│ horizontal order              │ vertical order                │
│ sort by y_center, then x      │ sort by x_center, then y      │
│          ↓                    │          ↓                    │
│ horizontal Mamba              │ vertical Mamba                │
│          ↓                    │          ↓                    │
│ scatter back                  │ scatter back                  │
│          ↓                    │          ↓                    │
│ h_encoded                     │ v_encoded                     │
└──────────────────────────────┴──────────────────────────────┘
        ↓
concat([tokens, h_encoded, v_encoded])
        ↓
Linear(384 → 128)
        ↓
LayerNorm
        ↓
GELU
        ↓
Dropout
        ↓
fused cell token
```

## 4. Horizontal / Vertical 编码来源

模型先根据预测 cell bbox 的中心点计算排序：

```python
x_center = (loc_preds[:, :, 0] + loc_preds[:, :, 2]) * 0.5
y_center = (loc_preds[:, :, 1] + loc_preds[:, :, 3]) * 0.5

horizontal_order = paddle.argsort(y_center * 1000.0 + x_center, axis=1)
vertical_order = paddle.argsort(x_center * 1000.0 + y_center, axis=1)
```

含义：

```text
horizontal_order:
  先按 y_center 排序，再按 x_center 排序
  近似模拟从上到下、从左到右的行方向阅读顺序

vertical_order:
  先按 x_center 排序，再按 y_center 排序
  近似模拟从左到右、从上到下的列方向结构顺序
```

随后分别经过 horizontal Mamba 和 vertical Mamba：

```python
h_encoded = self.encode_axis(
    tokens, horizontal_order, self.horizontal_mamba, h_reverse, mask=mask
)
v_encoded = self.encode_axis(
    tokens, vertical_order, self.vertical_mamba, v_reverse, mask=mask
)
```

## 5. Bidirectional 处理

如果 `bidirectional=True`，每个方向会额外做反向 Mamba 编码：

```python
reversed_tokens = paddle.flip(ordered, axis=[1])
reversed_encoded = paddle.flip(
    reverse_block(reversed_tokens, reversed_mask), axis=[1]
)
encoded = 0.5 * (encoded + reversed_encoded)
```

也就是说，每个方向的编码结果是：

```text
axis_encoded = 0.5 * (forward_encoded + backward_encoded)
```

这样可以同时捕获正向和反向的结构依赖。

## 6. Axis Mode 消融设置

代码中支持三种 `axis_mode`：

```python
if self.axis_mode == "horizontal":
    v_encoded = tokens
elif self.axis_mode == "vertical":
    h_encoded = tokens
```

含义：

| axis_mode | Fusion 输入 | 用途 |
|---|---|---|
| `dual` | `[tokens, h_encoded, v_encoded]` | 完整双轴模型 |
| `horizontal` | `[tokens, h_encoded, tokens]` | 只保留横向 Mamba 的消融实验 |
| `vertical` | `[tokens, tokens, v_encoded]` | 只保留纵向 Mamba 的消融实验 |

因此，完整 Phase5-E 使用的是：

```text
axis_mode = dual
```

## 7. Fusion 的本质

当前代码中的 fusion 不是 attention fusion，也不是简单加权求和，而是：

```text
feature concatenation + linear projection
```

更具体地说：

```text
原始 cell token
+ 横向结构关系 token
+ 纵向结构关系 token
→ 拼接成 384 维 token
→ 通过 Linear + LayerNorm + GELU + Dropout 压回 128 维
→ 得到最终 fused cell token
```

可以写成公式：

```text
T_fused = Fusion([T_original ; T_horizontal ; T_vertical])
```

其中：

```text
Fusion(x) = Dropout(GELU(LayerNorm(Wx + b)))
```

## 8. 论文中可用的表述

中文：

> 为融合原始单元格几何表示与横纵两个方向的结构依赖信息，本文将原始 cell token、horizontal Mamba 编码后的 token 以及 vertical Mamba 编码后的 token 在特征维度上进行拼接，并通过一个轻量 fusion MLP 将三路特征从 `3d` 投影回 `d` 维。该设计保留了原始几何信息，同时引入行方向和列方向的上下文结构表示。

英文：

> To fuse the original cell-level geometric representation with horizontal and vertical structural dependencies, we concatenate the original cell token, the horizontal Mamba-encoded token, and the vertical Mamba-encoded token along the feature dimension. A lightweight fusion MLP then projects the concatenated `3d`-dimensional representation back to `d` dimensions. This design preserves the original geometric information while incorporating row-wise and column-wise contextual structure.

## 9. 一句话总结

```text
代码中的 fusion = concat(original token, horizontal Mamba token, vertical Mamba token) → Linear(3d, d) → LayerNorm → GELU → Dropout
```

在当前配置中：

```text
concat(128, 128, 128) → 384 → 128
```
