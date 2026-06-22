import os
import sys

import paddle


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppocr.modeling.heads.cell_token_mlp_head import CellTokenMLPHead
from ppocr.losses.table_att_loss import SLACellTokenLoss


def test_cell_token_head_and_loss_forward_backward():
    paddle.seed(123)
    head = CellTokenMLPHead(
        in_channels=[24, 48, 96, 96],
        hidden_size=32,
        out_channels=12,
        max_text_length=6,
        loc_reg_num=4,
        cell_token_dim=16,
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
    assert "structure_probs" in out
    assert "loc_preds" in out
    assert "cell_tokens" in out
    assert "cell_has_text_logits" in out
    loss_fn = SLACellTokenLoss(
        structure_weight=1.0,
        loc_weight=1.0,
        loc_loss="smooth_l1",
        cell_presence_weight=0.2,
    )
    losses = loss_fn(out, loss_batch)
    assert "cell_presence_loss" in losses
    losses["loss"].backward()
