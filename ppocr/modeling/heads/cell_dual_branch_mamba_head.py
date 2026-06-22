# copyright (c) 2026 Authors. All Rights Reserved.
#
# Dual-branch cell-token sequence head for DBM-SLANet Phase 3 experiments.

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .cell_token_mlp_head import CellTokenMLPHead


class LightweightAxisMamba(nn.Layer):
    """A dependency-free gated recurrent scan for ordered cell tokens.

    This is intentionally lightweight for Phase 3 smoke tests: it gives the
    same sequence interface expected by a future Mamba block without adding an
    external selective-scan dependency.
    """

    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, direction="bidirectional")
        self.out_proj = nn.Linear(hidden_dim * 2, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, tokens, mask=None):
        u, gate = paddle.chunk(self.in_proj(tokens), chunks=2, axis=-1)
        u = F.gelu(u) * F.sigmoid(gate)
        out, _ = self.gru(u)
        out = self.out_proj(out)
        out = self.dropout(out)
        gated = F.sigmoid(self.gate_proj(tokens)) * out
        if mask is not None:
            gated = gated * mask.unsqueeze(-1).astype(gated.dtype)
        return self.norm(tokens + self.gamma * gated)


def scatter_ordered(encoded, order):
    restored = paddle.zeros_like(encoded)
    index = order.unsqueeze(-1).expand([-1, -1, encoded.shape[-1]])
    return paddle.put_along_axis(restored, index, encoded, axis=1)


class CellDualBranchMambaHead(CellTokenMLPHead):
    """Cell token MLP head plus horizontal/vertical sequence modeling."""

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
        axis_hidden_dim=None,
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
        self.horizontal_mamba = LightweightAxisMamba(
            cell_token_dim, hidden_dim=axis_hidden_dim, dropout=cell_token_dropout
        )
        self.vertical_mamba = LightweightAxisMamba(
            cell_token_dim, hidden_dim=axis_hidden_dim, dropout=cell_token_dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(cell_token_dim * 3, cell_token_dim),
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

    def forward(self, inputs, targets=None):
        out = super().forward(inputs, targets=targets)
        loc_preds = out["loc_preds"]
        tokens = out["cell_tokens"]
        mask = None
        if targets is not None and len(targets) >= 3:
            bbox_mask = targets[2].astype("float32")
            bbox_mask = bbox_mask[:, : tokens.shape[1]]
            mask = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)

        x_center = (loc_preds[:, :, 0] + loc_preds[:, :, 2]) * 0.5
        y_center = (loc_preds[:, :, 1] + loc_preds[:, :, 3]) * 0.5
        horizontal_order = paddle.argsort(y_center * 1000.0 + x_center, axis=1)
        vertical_order = paddle.argsort(x_center * 1000.0 + y_center, axis=1)

        h_tokens = paddle.take_along_axis(
            tokens,
            horizontal_order.unsqueeze(-1).expand([-1, -1, tokens.shape[-1]]),
            axis=1,
        )
        v_tokens = paddle.take_along_axis(
            tokens,
            vertical_order.unsqueeze(-1).expand([-1, -1, tokens.shape[-1]]),
            axis=1,
        )
        h_mask = paddle.take_along_axis(mask, horizontal_order, axis=1) if mask is not None else None
        v_mask = paddle.take_along_axis(mask, vertical_order, axis=1) if mask is not None else None
        h_encoded = scatter_ordered(self.horizontal_mamba(h_tokens, h_mask), horizontal_order)
        v_encoded = scatter_ordered(self.vertical_mamba(v_tokens, v_mask), vertical_order)
        fused = self.fusion(paddle.concat([tokens, h_encoded, v_encoded], axis=-1))
        out["cell_tokens"] = fused
        out["horizontal_cell_tokens"] = h_encoded
        out["vertical_cell_tokens"] = v_encoded
        out["cell_has_text_logits"] = self.cell_presence_head(fused).squeeze(-1)
        return out
