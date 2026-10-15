import os
import sys

import paddle


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppocr.losses.table_att_loss import SLACellTokenLoss
from ppocr.modeling.heads.cell_full_mamba_head import (
    CellFullMambaHead,
    SelectiveScanMambaBlock,
)


def test_selective_scan_mamba_block_forward_backward():
    paddle.seed(123)
    block = SelectiveScanMambaBlock(
        dim=12, d_state=4, expand=2, d_conv=3, dt_rank=2, dropout=0.0
    )
    tokens = paddle.randn([2, 5, 12])
    mask = paddle.to_tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype="float32")
    out = block(tokens, mask=mask)
    assert out.shape == tokens.shape
    loss = out.mean()
    loss.backward()


def test_full_mamba_head_forward_backward():
    paddle.seed(123)
    head = CellFullMambaHead(
        in_channels=[24, 48, 96, 96],
        hidden_size=32,
        out_channels=12,
        max_text_length=6,
        loc_reg_num=4,
        cell_token_dim=16,
        cell_token_dropout=0.0,
        mamba_d_state=4,
        mamba_expand=2,
        mamba_d_conv=3,
        mamba_dt_rank=2,
        bidirectional=True,
    )
    head.train()
    feat = paddle.randn([2, 96, 4, 4])
    structure = paddle.randint(0, 12, shape=[2, 8], dtype="int64")
    bboxes = paddle.rand([2, 8, 4])
    bbox_masks = paddle.ones([2, 8, 4])
    length = paddle.to_tensor([6, 6], dtype="int64")
    shape = paddle.ones([2, 6])
    head_targets = [structure, bboxes, bbox_masks, length, shape]
    loss_batch = [None, structure, bboxes, bbox_masks, length, shape]
    out = head([feat], targets=head_targets)
    assert "horizontal_cell_tokens" in out
    assert "vertical_cell_tokens" in out
    assert out["cell_tokens"].shape == out["horizontal_cell_tokens"].shape
    loss_fn = SLACellTokenLoss(
        structure_weight=1.0,
        loc_weight=1.0,
        loc_loss="smooth_l1",
        cell_presence_weight=0.2,
    )
    losses = loss_fn(out, loss_batch)
    losses["loss"].backward()
