# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


__all__ = ["GlobalLocalMambaFusion"]


class LightweightVisionMamba(nn.Layer):
    """Dependency-free vision state-space approximation for table features."""

    def __init__(self, channels, expand=2, drop_path=0.0):
        super().__init__()
        inner_channels = channels * expand
        self.in_proj = nn.Conv2D(channels, inner_channels * 2, kernel_size=1)
        self.dw_conv = nn.Conv2D(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
            groups=inner_channels,
        )
        self.out_proj = nn.Conv2D(inner_channels, channels, kernel_size=1)
        self.drop = nn.Dropout(drop_path) if drop_path > 0 else None
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def _normalized_cumsum(self, x, axis):
        length = x.shape[axis]
        denom = paddle.arange(1, length + 1, dtype=x.dtype)
        view_shape = [1] * len(x.shape)
        view_shape[axis] = length
        denom = denom.reshape(view_shape)
        return paddle.cumsum(x, axis=axis) / denom

    def _scan_axis(self, x, axis):
        forward = self._normalized_cumsum(x, axis)
        backward = paddle.flip(
            self._normalized_cumsum(paddle.flip(x, axis=[axis]), axis), axis=[axis]
        )
        return 0.5 * (forward + backward)

    def forward(self, x):
        u, gate = paddle.chunk(self.in_proj(x), chunks=2, axis=1)
        u = F.hardswish(self.dw_conv(u))
        u = u * F.sigmoid(gate)
        row_context = self._scan_axis(u, axis=3)
        col_context = self._scan_axis(u, axis=2)
        out = self.out_proj(0.5 * (row_context + col_context))
        if self.drop is not None:
            out = self.drop(out)
        return x + self.gamma * out


class LocalDetailAttention(nn.Layer):
    def __init__(self, channels, kernel_size=3, drop_path=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2D(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.pw_conv = nn.Conv2D(channels, channels, kernel_size=1)
        self.mask_conv = nn.Conv2D(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(drop_path) if drop_path > 0 else None
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x):
        local = F.hardswish(self.pw_conv(self.dw_conv(x)))
        mask = F.sigmoid(self.mask_conv(local))
        out = local * mask
        if self.drop is not None:
            out = self.drop(out)
        return x + self.gamma * out


class GlobalLocalMambaFusion(nn.Layer):
    def __init__(
        self,
        in_channels,
        target_index=-1,
        enable_mamba=True,
        enable_local=True,
        fusion_mode="gated_sum",
        mamba_expand=2,
        local_kernel=3,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(in_channels, int):
            channels = in_channels
            self.is_sequence = False
        else:
            channels = in_channels[target_index]
            self.is_sequence = True

        self.target_index = target_index
        self.enable_mamba = enable_mamba
        self.enable_local = enable_local
        self.fusion_mode = fusion_mode

        if enable_mamba:
            self.mamba_branch = LightweightVisionMamba(
                channels, expand=mamba_expand, drop_path=drop_path
            )
        if enable_local:
            self.local_branch = LocalDetailAttention(
                channels, kernel_size=local_kernel, drop_path=drop_path
            )
        if enable_mamba and enable_local and fusion_mode == "gated_sum":
            self.fusion_gate = nn.Conv2D(channels * 2, channels, kernel_size=1)

        self.fusion_gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )
        self.out_channels = in_channels

    def _enhance(self, x):
        if self.enable_mamba and self.enable_local:
            global_feat = self.mamba_branch(x)
            local_feat = self.local_branch(x)
            if self.fusion_mode == "gated_sum":
                gate = F.sigmoid(
                    self.fusion_gate(paddle.concat([global_feat, local_feat], axis=1))
                )
                fused = gate * global_feat + (1.0 - gate) * local_feat
            elif self.fusion_mode == "sum":
                fused = 0.5 * (global_feat + local_feat)
            else:
                raise ValueError("Unsupported fusion_mode: {}".format(self.fusion_mode))
            return x + self.fusion_gamma * (fused - x)

        if self.enable_mamba:
            return self.mamba_branch(x)
        if self.enable_local:
            return self.local_branch(x)
        return x

    def forward(self, inputs):
        if self.is_sequence:
            outs = list(inputs)
            outs[self.target_index] = self._enhance(outs[self.target_index])
            return tuple(outs)
        return self._enhance(inputs)
