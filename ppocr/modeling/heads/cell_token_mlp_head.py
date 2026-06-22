# copyright (c) 2026 Authors. All Rights Reserved.
#
# Cell-token auxiliary head for DBM-SLANet Phase 2 experiments.

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn

from .table_att_head import SLAHead


class GeometryTokenEncoder(nn.Layer):
    """Encode normalized cell geometry into a lightweight token."""

    def __init__(self, input_dim=8, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, loc_preds):
        x1 = loc_preds[:, :, 0:1]
        y1 = loc_preds[:, :, 1:2]
        x2 = loc_preds[:, :, 2:3]
        y2 = loc_preds[:, :, 3:4]
        left = paddle.minimum(x1, x2)
        top = paddle.minimum(y1, y2)
        right = paddle.maximum(x1, x2)
        bottom = paddle.maximum(y1, y2)
        width = paddle.clip(right - left, min=0.0)
        height = paddle.clip(bottom - top, min=0.0)
        area = width * height
        aspect = width / (height + 1e-6)
        x_center = (left + right) * 0.5
        y_center = (top + bottom) * 0.5
        geom = paddle.concat(
            [x_center, y_center, width, height, area, aspect, left, top], axis=-1
        )
        return self.net(geom)


class CellTokenMLPHead(SLAHead):
    """SLANet-compatible head with an auxiliary cell-token MLP branch.

    The original SLANet outputs are preserved. The auxiliary branch predicts
    whether each decoded cell step has a supervised bbox/text target, using
    geometry tokens derived from `loc_preds`.
    """

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
            **kwargs,
        )
        self.cell_token_encoder = GeometryTokenEncoder(
            input_dim=8, hidden_dim=cell_token_dim, dropout=cell_token_dropout
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
        cell_tokens = self.cell_token_encoder(loc_preds)
        out["cell_tokens"] = cell_tokens
        out["cell_has_text_logits"] = self.cell_presence_head(cell_tokens).squeeze(-1)
        return out
