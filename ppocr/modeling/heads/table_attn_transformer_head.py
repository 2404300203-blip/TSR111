# copyright (c) 2026 Authors. All Rights Reserved.
#
# Attention-enhanced table head for the Codex SLANet experiment.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn

from .table_master_head import FeedForward, MultiHeadAttention, TableMasterHead


class VisualSelfAttentionBlock(nn.Layer):
    def __init__(self, hidden_size, headers=8, d_ff=384, dropout=0.1):
        super(VisualSelfAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(headers, hidden_size, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.dropout1(self.attn(normed, normed, normed))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class SLAAttnTransformerHead(TableMasterHead):
    """SLANet-compatible table head with visual self-attention refinement.

    The head keeps PaddleOCR table outputs unchanged:
    `structure_probs` for table tokens and `loc_preds` for cell boxes.
    Compared with `SLAHead`, the recurrent AttentionGRUCell decoder is replaced
    by the Transformer decoder inherited from `TableMasterHead`; before decoding,
    the last CSPPAN feature map is refined by visual token self-attention blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels=30,
        headers=8,
        d_ff=384,
        dropout=0.1,
        max_text_length=500,
        loc_reg_num=4,
        use_refine=True,
        refine_layers=2,
        refine_heads=8,
        refine_mlp_ratio=4.0,
        refine_drop_path=0.0,
        **kwargs,
    ):
        super(SLAAttnTransformerHead, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            headers=headers,
            d_ff=d_ff,
            dropout=dropout,
            max_text_length=max_text_length,
            loc_reg_num=loc_reg_num,
            **kwargs,
        )
        hidden_size = in_channels[-1] if isinstance(in_channels, (list, tuple)) else in_channels
        self.use_refine = use_refine
        refine_d_ff = int(hidden_size * refine_mlp_ratio)
        if self.use_refine and refine_layers > 0:
            self.refine = nn.Sequential(
                *[
                    VisualSelfAttentionBlock(
                        hidden_size=hidden_size,
                        headers=refine_heads,
                        d_ff=refine_d_ff,
                        dropout=dropout,
                    )
                    for _ in range(refine_layers)
                ]
            )
        else:
            self.refine = None

    def forward_train(self, out_enc, targets):
        padded_targets = targets[0]
        max_len = int(targets[-2].max().numpy().item())
        max_len = max(1, max_len)
        decoder_input = padded_targets[:, : max_len + 1]
        tgt_mask = self.make_mask(decoder_input)
        output, bbox_output = self.decode(decoder_input, out_enc, None, tgt_mask)
        return {"structure_probs": output, "loc_preds": bbox_output}

    def forward(self, feat, targets=None):
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        b, c, h, w = feat.shape
        feat = feat.reshape([b, c, h * w]).transpose((0, 2, 1))
        if self.refine is not None:
            feat = self.refine(feat)
        out_enc = self.positional_encoding(feat)
        if self.training:
            return self.forward_train(out_enc, targets)
        return self.forward_test(out_enc)
