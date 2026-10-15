# Full Mamba Block 代码核验说明

核验时间：2026-07-05  
远程代码路径：

```text
/root/autodl-tmp/PaddleOCR/ppocr/modeling/heads/cell_full_mamba_head.py
```

本文件针对正式定稿前需要确认的 `FullMambaBlock` 细节进行逐项核验。代码中的实际类名不是 `FullMambaBlock`，而是：

```python
class SelectiveScanMambaBlock(nn.Layer):
```

它被 `CellFullMambaHead` 用作 horizontal / vertical Mamba 编码块。

---

## 1. 输入投影是否一次性拆分为 content 和 gate 两部分

结论：**是。**

代码：

```python
self.in_proj = nn.Linear(dim, self.d_inner * 2)
...
projected = self.in_proj(tokens)
x, z = paddle.chunk(projected, chunks=2, axis=-1)
```

解释：

- `tokens` 先经过 `in_proj`；
- 输出维度是 `2 * d_inner`；
- 然后通过 `paddle.chunk(..., chunks=2, axis=-1)` 拆成两部分：
  - `x`：content / state-space 输入分支；
  - `z`：gate / output gating 分支。

可写表述：

> The input token is projected into two branches using a single linear layer, which are then split into a content branch `x` and a gating branch `z`.

---

## 2. 局部卷积是否为 depthwise Conv1D，以及卷积核大小

结论：**是 depthwise Conv1D。默认卷积核大小为 3。**

代码：

```python
self.conv1d = nn.Conv1D(
    in_channels=self.d_inner,
    out_channels=self.d_inner,
    kernel_size=d_conv,
    padding=d_conv - 1,
    groups=self.d_inner,
)
```

默认参数：

```python
d_conv=3
```

配置中也使用：

```yaml
mamba_d_conv: 3
```

解释：

- `groups=self.d_inner`；
- `in_channels=self.d_inner`；
- `out_channels=self.d_inner`；
- 因此这是 depthwise 1D convolution；
- 代码中使用 `padding=d_conv - 1`，卷积后再裁剪回原序列长度：

```python
x_conv = self.conv1d(x_conv)
x_conv = x_conv[:, :, : x.shape[1]]
```

---

## 3. 卷积后的激活函数是否为 SiLU

结论：**是。**

代码中自定义了 SiLU：

```python
def silu(x):
    return x * F.sigmoid(x)
```

卷积后使用：

```python
x = silu(paddle.transpose(x_conv, [0, 2, 1]))
```

因此卷积后激活为：

```text
SiLU(x) = x * sigmoid(x)
```

---

## 4. Δt、Bt、Ct 的具体投影方式及维度

结论：**由 `x_proj` 从卷积后的 `x` 一次性投影得到，再切分为 `dt_raw`、`B`、`C`。**

代码：

```python
self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2)
self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
```

forward 中：

```python
params = self.x_proj(x)
dt_raw = params[:, :, : self.dt_rank]
b_param = params[:, :, self.dt_rank : self.dt_rank + self.d_state]
c_param = params[:, :, self.dt_rank + self.d_state :]
delta = F.softplus(self.dt_proj(dt_raw))
```

维度：

```text
x:       [B, L, d_inner]
params:  [B, L, dt_rank + 2 * d_state]
dt_raw:  [B, L, dt_rank]
B_t:     [B, L, d_state]
C_t:     [B, L, d_state]
delta:   [B, L, d_inner]
```

默认参数：

```python
d_state = 16
expand = 2
d_inner = expand * dim
dt_rank = ceil(dim / 16)  # when dt_rank="auto"
```

如果当前 `cell_token_dim=128`：

```text
dim = 128
d_inner = 256
d_state = 16
dt_rank = ceil(128 / 16) = 8
params dim = 8 + 16 * 2 = 40
```

---

## 5. A = -exp(A_log) 是否与代码一致

结论：**一致。**

初始化：

```python
a = paddle.arange(1, d_state + 1, dtype="float32").unsqueeze(0)
a = paddle.tile(a, [self.d_inner, 1])
self.A_log = self.create_parameter(
    shape=[self.d_inner, d_state],
    default_initializer=nn.initializer.Assign(paddle.log(a)),
)
```

使用时：

```python
a = -paddle.exp(self.A_log.astype(x.dtype))
```

因此：

```text
A = -exp(A_log)
```

维度：

```text
A_log: [d_inner, d_state]
A:     [d_inner, d_state]
```

---

## 6. 离散化采用精确零阶保持、近似离散化，还是自定义形式

结论：**代码中是自定义的逐步 selective scan 形式，不是完整标准 ZOH 实现。**

状态更新代码：

```python
transition = paddle.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
update = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) * x_t.unsqueeze(-1)
state = transition * state + update
y_t = (state * c_t.unsqueeze(1)).sum(axis=-1) + d * x_t
```

对应形式：

```text
state_t = exp(Δt * A) * state_{t-1} + Δt * B_t * x_t
y_t     = C_t * state_t + D * x_t
```

注意：

- transition 使用了 `exp(Δt * A)`；
- update 使用的是 `Δt * B_t * x_t`；
- 没有实现完整标准 ZOH 中常见的 `(exp(ΔA)-I) A^{-1} B x` 项；
- 因此论文中不宜写“exact zero-order hold discretization”。

更稳妥表述：

> We implement a lightweight selective-scan update with an exponential diagonal state transition and an input-dependent update term, rather than relying on a fused CUDA selective-scan kernel.

---

## 7. 是否包含可学习的 D 跳跃项

结论：**包含。**

代码：

```python
self.D = self.create_parameter(
    shape=[self.d_inner], default_initializer=nn.initializer.Constant(1.0)
)
```

使用：

```python
d = self.D.astype(x.dtype)
...
y_t = (state * c_t.unsqueeze(1)).sum(axis=-1) + d * x_t
```

因此输出包含：

```text
D * x_t
```

这是可学习 skip / residual-like 项。

---

## 8. 输出门控是否使用 SiLU

结论：**是。**

代码：

```python
y = self.selective_scan(x, delta, b_param, c_param, mask=mask)
y = y * silu(z)
```

其中：

```python
def silu(x):
    return x * F.sigmoid(x)
```

因此输出门控为：

```text
y = y * SiLU(z)
```

---

## 9. Full-Mamba Block 内部是否还有残差连接、归一化或额外 Dropout

结论：**有残差连接、有 LayerNorm、有 Dropout，还有一个可学习缩放系数 gamma。**

代码：

```python
residual = tokens
...
y = self.dropout(self.out_proj(y))
...
return self.norm(residual + self.gamma * y)
```

初始化：

```python
self.dropout = nn.Dropout(dropout)
self.norm = nn.LayerNorm(dim)
self.gamma = self.create_parameter(
    shape=[1], default_initializer=nn.initializer.Constant(0.0)
)
```

说明：

- `out_proj` 后接 Dropout；
- 最终输出为 `LayerNorm(residual + gamma * y)`；
- `gamma` 是可学习标量，初始为 0；
- 这使 block 初始状态更接近 identity mapping。

---

## 10. mask 是在输入、状态扫描还是输出阶段实施

结论：**三个阶段都有使用。**

### 10.1 输入阶段

在进入 selective scan 前，对 `x` 做 mask：

```python
if mask is not None:
    x = x * mask.unsqueeze(-1).astype(x.dtype)
```

### 10.2 状态扫描输出阶段

在每个时间步的 `y_t` 上做 mask：

```python
if mask is not None:
    y_t = y_t * mask[:, idx : idx + 1].astype(y_t.dtype)
```

注意：

- mask 没有直接阻止 `state = transition * state + update` 的计算；
- 但是因为输入 `x` 已经被 mask，padding token 的 update 会被压到 0；
- 同时该步输出 `y_t` 也会被 mask。

### 10.3 输出阶段

在 `out_proj + dropout` 后再次 mask：

```python
if mask is not None:
    y = y * mask.unsqueeze(-1).astype(y.dtype)
```

因此更准确的说法是：

```text
mask is applied to the scan input, per-step scan output, and final projected output.
```

---

## 总结表

| 核验项 | 结论 |
|---|---|
| 输入投影是否拆为 content / gate | 是，`in_proj` 后 chunk 为 `x` 和 `z` |
| 局部卷积 | depthwise Conv1D |
| 卷积核大小 | 默认 `d_conv=3` |
| 卷积后激活 | SiLU |
| Δt / B / C 投影 | `x_proj` 得到 `[dt_raw, B, C]`，`dt_proj + softplus` 得到 `delta` |
| A 参数化 | `A = -exp(A_log)` |
| 离散化 | 自定义 selective-scan update，非完整 ZOH |
| D 跳跃项 | 有，可学习，初始为 1 |
| 输出门控 | `y * SiLU(z)` |
| 残差 / 归一化 / Dropout | 有 residual、LayerNorm、Dropout、learnable gamma |
| mask 位置 | 输入、逐步 scan 输出、最终输出阶段均使用 |

---

## 建议论文表述

中文：

> 本文实现了一个轻量化 selective-scan Mamba-style block。输入 token 首先通过线性投影拆分为内容分支和门控分支，内容分支经过 depthwise Conv1D 与 SiLU 激活后，用于生成输入依赖的 `Δt`、`B_t` 和 `C_t` 参数。状态转移采用对角参数化 `A=-exp(A_log)`，并通过逐步 scan 执行 `exp(ΔtA)` 形式的状态更新，同时保留可学习的 `D` skip 项。输出经过 `SiLU` 门控、线性投影、Dropout、残差连接和 LayerNorm。该实现采用 Paddle 原生循环以保证可移植性，而不是依赖 fused CUDA selective-scan kernel。

英文：

> We implement a lightweight Mamba-style selective-scan block. The input token is first projected and split into a content branch and a gating branch. The content branch is processed by a depthwise 1D convolution followed by SiLU activation, and is then used to generate input-dependent `Δt`, `B_t`, and `C_t` parameters. The state transition uses a diagonal parameterization `A=-exp(A_log)` and performs a step-wise scan with an exponential transition term, while retaining a learnable `D` skip connection. The output is modulated by a SiLU gate, followed by output projection, dropout, residual connection, and LayerNorm. The implementation uses a native Paddle loop for portability instead of a fused CUDA selective-scan kernel.
