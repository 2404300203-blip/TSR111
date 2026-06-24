# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
from paddle.nn import functional as F


class TableAttentionLoss(nn.Layer):
    def __init__(self, structure_weight=1.0, loc_weight=0.0, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction="none")
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight

    def forward(self, predicts, batch):
        structure_probs = predicts["structure_probs"]
        structure_targets = batch[1].astype("int64")
        structure_targets = structure_targets[:, 1:]
        structure_probs = paddle.reshape(
            structure_probs, [-1, structure_probs.shape[-1]]
        )
        structure_targets = paddle.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)

        structure_loss = paddle.mean(structure_loss) * self.structure_weight

        loc_preds = predicts["loc_preds"]
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[3].astype("float32")
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = (
            F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight
        )

        total_loss = structure_loss + loc_loss
        return {
            "loss": total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss,
        }


class SLALoss(nn.Layer):
    def __init__(
        self,
        structure_weight=1.0,
        loc_weight=0.0,
        loc_loss="mse",
        span_token_indices=None,
        span_token_weight=1.0,
        span_fp_weight=0.0,
        plain_cell_token_index=None,
        **kwargs,
    ):
        super(SLALoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction="none")
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.loc_loss = loc_loss
        self.span_token_indices = span_token_indices or []
        self.span_token_weight = float(span_token_weight)
        self.span_fp_weight = float(span_fp_weight)
        self.plain_cell_token_index = plain_cell_token_index
        self.eps = 1e-12

    def _is_in_indices(self, values, indices):
        mask = paddle.zeros_like(values, dtype="bool")
        for idx in indices:
            idx_tensor = paddle.full(values.shape, int(idx), dtype=values.dtype)
            mask = paddle.logical_or(mask, values == idx_tensor)
        return mask

    def _structure_loss(self, structure_probs, structure_targets):
        ce_loss = self.loss_func(structure_probs, structure_targets)
        valid_mask = paddle.ones_like(ce_loss, dtype="float32")
        loss_weight = paddle.ones_like(ce_loss, dtype="float32")
        span_target_mask = None
        span_fp_loss = paddle.to_tensor(0.0, dtype=ce_loss.dtype)

        if self.span_token_indices and self.span_token_weight != 1.0:
            span_target_mask = self._is_in_indices(
                structure_targets, self.span_token_indices
            )
            span_target_mask_f = span_target_mask.astype("float32")
            loss_weight = loss_weight + span_target_mask_f * (
                self.span_token_weight - 1.0
            )

        weighted_loss = (ce_loss * loss_weight * valid_mask).sum() / (
            (loss_weight * valid_mask).sum() + self.eps
        )

        if (
            self.span_fp_weight > 0
            and self.span_token_indices
            and self.plain_cell_token_index is not None
        ):
            plain_idx = paddle.full(
                structure_targets.shape,
                int(self.plain_cell_token_index),
                dtype=structure_targets.dtype,
            )
            plain_cell_mask = (structure_targets == plain_idx).astype("float32")
            probs = F.softmax(structure_probs, axis=-1)
            span_prob = paddle.zeros_like(plain_cell_mask, dtype=probs.dtype)
            for idx in self.span_token_indices:
                span_prob = span_prob + probs[:, :, int(idx)]
            span_fp_loss = (span_prob * plain_cell_mask).sum() / (
                plain_cell_mask.sum() + self.eps
            )
            weighted_loss = weighted_loss + span_fp_loss * self.span_fp_weight

        span_token_loss = paddle.to_tensor(0.0, dtype=ce_loss.dtype)
        if span_target_mask is not None:
            span_mask_f = span_target_mask.astype("float32")
            span_token_loss = (ce_loss * span_mask_f).sum() / (
                span_mask_f.sum() + self.eps
            )

        return weighted_loss, span_token_loss, span_fp_loss

    def forward(self, predicts, batch):
        structure_probs = predicts["structure_probs"]
        structure_targets = batch[1].astype("int64")
        max_len = batch[-2].max().astype("int32")
        structure_targets = structure_targets[:, 1 : max_len + 2]

        structure_loss, span_token_loss, span_fp_loss = self._structure_loss(
            structure_probs, structure_targets
        )
        structure_loss = structure_loss * self.structure_weight

        loc_preds = predicts["loc_preds"]
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[3].astype("float32")
        loc_targets = loc_targets[:, 1 : max_len + 2]
        loc_targets_mask = loc_targets_mask[:, 1 : max_len + 2]

        loc_loss = (
            F.smooth_l1_loss(
                loc_preds * loc_targets_mask,
                loc_targets * loc_targets_mask,
                reduction="sum",
            )
            * self.loc_weight
        )

        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
        total_loss = structure_loss + loc_loss
        return {
            "loss": total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss,
            "span_token_loss": span_token_loss,
            "span_fp_loss": span_fp_loss,
        }


class SLACellTokenLoss(SLALoss):
    def __init__(
        self,
        structure_weight=1.0,
        loc_weight=0.0,
        loc_loss="smooth_l1",
        cell_presence_weight=0.2,
        **kwargs,
    ):
        super().__init__(
            structure_weight=structure_weight,
            loc_weight=loc_weight,
            loc_loss=loc_loss,
            **kwargs,
        )
        self.cell_presence_weight = cell_presence_weight

    def forward(self, predicts, batch):
        losses = super().forward(predicts, batch)
        if "cell_has_text_logits" not in predicts:
            return losses
        max_len = batch[-2].max().astype("int32")
        logits = predicts["cell_has_text_logits"]
        logits = logits[:, : max_len + 1]
        bbox_mask = batch[3].astype("float32")[:, : max_len + 1]
        target = paddle.clip(bbox_mask.sum(axis=-1), min=0.0, max=1.0)
        valid = paddle.ones_like(target)
        cell_presence_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        cell_presence_loss = (cell_presence_loss * valid).sum() / (valid.sum() + self.eps)
        cell_presence_loss = cell_presence_loss * self.cell_presence_weight
        losses["cell_presence_loss"] = cell_presence_loss
        losses["loss"] = losses["loss"] + cell_presence_loss
        return losses
