# copyright (c) 2026 Authors. All Rights Reserved.
#
# Full selective-scan cell-token sequence head for DBM-SLANet Phase 4.

from __future__ import absolute_import, division, print_function

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .cell_token_mlp_head import CellTokenMLPHead
from .table_att_head import SLAHead


def silu(x):
    return x * F.sigmoid(x)


class SelectiveScanMambaBlock(nn.Layer):
    """Dependency-free Mamba-style selective state-space block.

    This block follows the core Mamba ingredients closely enough for Phase 4
    feasibility tests: input/gate projection, depthwise local convolution,
    input-dependent delta/B/C parameters, diagonal A state transition, D skip,
    selective scan, and gated output projection. It uses a Paddle Python scan
    loop for portability instead of a fused CUDA selective-scan kernel.
    """

    def __init__(
        self,
        dim,
        d_state=16,
        expand=2,
        d_conv=3,
        dt_rank="auto",
        dropout=0.1,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        a = paddle.arange(1, d_state + 1, dtype="float32").unsqueeze(0)
        a = paddle.tile(a, [self.d_inner, 1])
        self.A_log = self.create_parameter(
            shape=[self.d_inner, d_state],
            default_initializer=nn.initializer.Assign(paddle.log(a)),
        )
        self.D = self.create_parameter(
            shape=[self.d_inner], default_initializer=nn.initializer.Constant(1.0)
        )
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

        dt_init = math.log(math.exp((dt_min + dt_max) * 0.5) - 1.0)
        self.dt_proj.bias.set_value(
            paddle.full(shape=self.dt_proj.bias.shape, fill_value=dt_init, dtype="float32")
        )

    def selective_scan(self, x, delta, b_param, c_param, mask=None):
        batch_size, seq_len, _ = x.shape
        a = -paddle.exp(self.A_log.astype(x.dtype))
        d = self.D.astype(x.dtype)
        state = paddle.zeros([batch_size, self.d_inner, self.d_state], dtype=x.dtype)
        outputs = []

        for idx in range(seq_len):
            x_t = x[:, idx, :]
            dt_t = delta[:, idx, :]
            b_t = b_param[:, idx, :]
            c_t = c_param[:, idx, :]

            transition = paddle.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            update = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) * x_t.unsqueeze(-1)
            state = transition * state + update
            y_t = (state * c_t.unsqueeze(1)).sum(axis=-1) + d * x_t
            if mask is not None:
                y_t = y_t * mask[:, idx : idx + 1].astype(y_t.dtype)
            outputs.append(y_t)

        return paddle.stack(outputs, axis=1)

    def forward(self, tokens, mask=None):
        residual = tokens
        projected = self.in_proj(tokens)
        x, z = paddle.chunk(projected, chunks=2, axis=-1)

        x_conv = paddle.transpose(x, [0, 2, 1])
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[:, :, : x.shape[1]]
        x = silu(paddle.transpose(x_conv, [0, 2, 1]))

        params = self.x_proj(x)
        dt_raw = params[:, :, : self.dt_rank]
        b_param = params[:, :, self.dt_rank : self.dt_rank + self.d_state]
        c_param = params[:, :, self.dt_rank + self.d_state :]
        delta = F.softplus(self.dt_proj(dt_raw))

        if mask is not None:
            x = x * mask.unsqueeze(-1).astype(x.dtype)

        y = self.selective_scan(x, delta, b_param, c_param, mask=mask)
        y = y * silu(z)
        y = self.dropout(self.out_proj(y))
        if mask is not None:
            y = y * mask.unsqueeze(-1).astype(y.dtype)
        return self.norm(residual + self.gamma * y)


def scatter_ordered(encoded, order):
    restored = paddle.zeros_like(encoded)
    index = order.unsqueeze(-1).expand([-1, -1, encoded.shape[-1]])
    return paddle.put_along_axis(restored, index, encoded, axis=1)


class CellFullMambaHead(CellTokenMLPHead):
    """Cell token head with horizontal/vertical full selective-scan blocks."""

    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        use_attn=False,
        cell_token_dim=128,
        cell_token_dropout=0.1,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_d_conv=3,
        mamba_dt_rank="auto",
        bidirectional=True,
        horizontal_bidirectional=None,
        vertical_bidirectional=None,
        share_directional_mamba=False,
        axis_mode="dual",
        use_gt_bbox_mask=True,
        prune_aux_at_inference=False,
        skip_redundant_presence_at_inference=False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            out_channels=out_channels,
            max_text_length=max_text_length,
            loc_reg_num=loc_reg_num,
            fc_decay=fc_decay,
            use_attn=use_attn,
            cell_token_dim=cell_token_dim,
            cell_token_dropout=cell_token_dropout,
            **kwargs,
        )
        self.bidirectional = bidirectional
        self.horizontal_bidirectional = (
            bidirectional
            if horizontal_bidirectional is None
            else horizontal_bidirectional
        )
        self.vertical_bidirectional = (
            bidirectional if vertical_bidirectional is None else vertical_bidirectional
        )
        self.share_directional_mamba = share_directional_mamba
        self.use_gt_bbox_mask = use_gt_bbox_mask
        self.prune_aux_at_inference = prune_aux_at_inference
        self.skip_redundant_presence_at_inference = (
            skip_redundant_presence_at_inference
        )
        if axis_mode not in ("dual", "horizontal", "vertical"):
            raise ValueError("axis_mode must be one of 'dual', 'horizontal', or 'vertical'")
        if share_directional_mamba and (
            axis_mode != "dual"
            or not self.horizontal_bidirectional
            or not self.vertical_bidirectional
        ):
            raise ValueError(
                "share_directional_mamba requires dual-axis bidirectional mode"
            )
        self.axis_mode = axis_mode
        if share_directional_mamba:
            self.shared_mamba = SelectiveScanMambaBlock(
                cell_token_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
                d_conv=mamba_d_conv,
                dt_rank=mamba_dt_rank,
                dropout=cell_token_dropout,
            )
            self.direction_embedding = self.create_parameter(
                shape=[4, cell_token_dim],
                default_initializer=nn.initializer.Normal(std=0.02),
            )
        elif axis_mode in ("dual", "horizontal"):
            self.horizontal_mamba = SelectiveScanMambaBlock(
                cell_token_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
                d_conv=mamba_d_conv,
                dt_rank=mamba_dt_rank,
                dropout=cell_token_dropout,
            )
        if not share_directional_mamba and axis_mode in ("dual", "vertical"):
            self.vertical_mamba = SelectiveScanMambaBlock(
                cell_token_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
                d_conv=mamba_d_conv,
                dt_rank=mamba_dt_rank,
                dropout=cell_token_dropout,
            )
        if (
            not share_directional_mamba
            and self.horizontal_bidirectional
            and axis_mode in ("dual", "horizontal")
        ):
            self.horizontal_reverse_mamba = SelectiveScanMambaBlock(
                cell_token_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
                d_conv=mamba_d_conv,
                dt_rank=mamba_dt_rank,
                dropout=cell_token_dropout,
            )
        if (
            not share_directional_mamba
            and self.vertical_bidirectional
            and axis_mode in ("dual", "vertical")
        ):
            self.vertical_reverse_mamba = SelectiveScanMambaBlock(
                cell_token_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
                d_conv=mamba_d_conv,
                dt_rank=mamba_dt_rank,
                dropout=cell_token_dropout,
            )

        fusion_inputs = cell_token_dim * (3 if axis_mode == "dual" else 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_inputs, cell_token_dim),
            nn.LayerNorm(cell_token_dim),
            nn.GELU(),
            nn.Dropout(cell_token_dropout),
        )
        self.cell_presence_head = nn.Sequential(
            nn.Linear(cell_token_dim, cell_token_dim),
            nn.GELU(),
            nn.Dropout(cell_token_dropout),
            nn.Linear(cell_token_dim, 1),
        )

    def encode_axis(self, tokens, order, block, reverse_block=None, mask=None):
        ordered = paddle.take_along_axis(
            tokens, order.unsqueeze(-1).expand([-1, -1, tokens.shape[-1]]), axis=1
        )
        ordered_mask = paddle.take_along_axis(mask, order, axis=1) if mask is not None else None
        encoded = block(ordered, ordered_mask)
        if reverse_block is not None:
            reversed_tokens = paddle.flip(ordered, axis=[1])
            reversed_mask = paddle.flip(ordered_mask, axis=[1]) if ordered_mask is not None else None
            reversed_encoded = paddle.flip(reverse_block(reversed_tokens, reversed_mask), axis=[1])
            encoded = 0.5 * (encoded + reversed_encoded)
        return scatter_ordered(encoded, order)

    def encode_shared_dual(self, tokens, horizontal_order, vertical_order, mask=None):
        token_index = lambda order: order.unsqueeze(-1).expand(
            [-1, -1, tokens.shape[-1]]
        )
        horizontal = paddle.take_along_axis(
            tokens, token_index(horizontal_order), axis=1
        )
        vertical = paddle.take_along_axis(tokens, token_index(vertical_order), axis=1)
        sequences = [
            horizontal,
            paddle.flip(horizontal, axis=[1]),
            vertical,
            paddle.flip(vertical, axis=[1]),
        ]
        sequences = [
            sequence + self.direction_embedding[idx].reshape([1, 1, -1])
            for idx, sequence in enumerate(sequences)
        ]
        shared_tokens = paddle.concat(sequences, axis=0)

        shared_mask = None
        if mask is not None:
            horizontal_mask = paddle.take_along_axis(mask, horizontal_order, axis=1)
            vertical_mask = paddle.take_along_axis(mask, vertical_order, axis=1)
            shared_mask = paddle.concat(
                [
                    horizontal_mask,
                    paddle.flip(horizontal_mask, axis=[1]),
                    vertical_mask,
                    paddle.flip(vertical_mask, axis=[1]),
                ],
                axis=0,
            )

        encoded = self.shared_mamba(shared_tokens, shared_mask)
        h_forward, h_reverse, v_forward, v_reverse = paddle.split(
            encoded, num_or_sections=4, axis=0
        )
        h_encoded = 0.5 * (h_forward + paddle.flip(h_reverse, axis=[1]))
        v_encoded = 0.5 * (v_forward + paddle.flip(v_reverse, axis=[1]))
        return (
            scatter_ordered(h_encoded, horizontal_order),
            scatter_ordered(v_encoded, vertical_order),
        )

    def forward(self, inputs, targets=None):
        if self.prune_aux_at_inference and not self.training:
            return SLAHead.forward(self, inputs, targets=targets)

        if self.skip_redundant_presence_at_inference and not self.training:
            out = SLAHead.forward(self, inputs, targets=targets)
            out["cell_tokens"] = self.cell_token_encoder(out["loc_preds"])
        else:
            out = super().forward(inputs, targets=targets)
        loc_preds = out["loc_preds"]
        tokens = out["cell_tokens"]
        mask = None
        if self.use_gt_bbox_mask and targets is not None and len(targets) >= 3:
            bbox_mask = targets[2].astype("float32")
            bbox_mask = bbox_mask[:, : tokens.shape[1]]
            mask = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)

        x_center = (loc_preds[:, :, 0] + loc_preds[:, :, 2]) * 0.5
        y_center = (loc_preds[:, :, 1] + loc_preds[:, :, 3]) * 0.5
        horizontal_order = paddle.argsort(y_center * 1000.0 + x_center, axis=1)
        vertical_order = paddle.argsort(x_center * 1000.0 + y_center, axis=1)

        h_encoded = None
        v_encoded = None
        fusion_tokens = [tokens]
        if self.share_directional_mamba:
            h_encoded, v_encoded = self.encode_shared_dual(
                tokens, horizontal_order, vertical_order, mask=mask
            )
            fusion_tokens.extend([h_encoded, v_encoded])
        elif self.axis_mode in ("dual", "horizontal"):
            h_reverse = getattr(self, "horizontal_reverse_mamba", None)
            h_encoded = self.encode_axis(
                tokens, horizontal_order, self.horizontal_mamba, h_reverse, mask=mask
            )
            fusion_tokens.append(h_encoded)
        if not self.share_directional_mamba and self.axis_mode in ("dual", "vertical"):
            v_reverse = getattr(self, "vertical_reverse_mamba", None)
            v_encoded = self.encode_axis(
                tokens, vertical_order, self.vertical_mamba, v_reverse, mask=mask
            )
            fusion_tokens.append(v_encoded)
        fused = self.fusion(paddle.concat(fusion_tokens, axis=-1))
        out["cell_tokens"] = fused
        out["horizontal_cell_tokens"] = h_encoded if h_encoded is not None else tokens
        out["vertical_cell_tokens"] = v_encoded if v_encoded is not None else tokens
        out["cell_has_text_logits"] = self.cell_presence_head(fused).squeeze(-1)
        return out
