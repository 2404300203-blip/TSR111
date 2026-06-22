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


__all__ = ["FinancialStructureAttention"]


class HeaderAwareSpatialAttention(nn.Layer):
    def __init__(self, channels, reduction=4, act="hard_swish"):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.conv1 = nn.Conv2D(channels, hidden_channels, kernel_size=1, bias_attr=True)
        self.conv2 = nn.Conv2D(hidden_channels, 1, kernel_size=1, bias_attr=True)
        self.act = act
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x):
        mask = self.conv1(x)
        if self.act == "hard_swish":
            mask = F.hardswish(mask)
        else:
            mask = F.relu(mask)
        mask = F.sigmoid(self.conv2(mask))
        return x + self.gamma * x * mask


class SpanAwareAxialAttention(nn.Layer):
    def __init__(self, channels, num_heads=2, attn_drop=0.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2D(channels, channels * 3, kernel_size=1, bias_attr=True)
        self.proj = nn.Conv2D(channels, channels, kernel_size=1, bias_attr=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def _axis_attention(self, q, k, v):
        b_axis, length, channels = q.shape
        q = q.reshape([b_axis, length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        k = k.reshape([b_axis, length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        v = v.reshape([b_axis, length, self.num_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        attn = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3]).reshape([b_axis, length, channels])
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).transpose([0, 2, 3, 1])
        q, k, v = paddle.split(qkv, num_or_sections=3, axis=-1)

        row_q = q.reshape([b * h, w, c])
        row_k = k.reshape([b * h, w, c])
        row_v = v.reshape([b * h, w, c])
        row_out = self._axis_attention(row_q, row_k, row_v).reshape([b, h, w, c])

        col_q = q.transpose([0, 2, 1, 3]).reshape([b * w, h, c])
        col_k = k.transpose([0, 2, 1, 3]).reshape([b * w, h, c])
        col_v = v.transpose([0, 2, 1, 3]).reshape([b * w, h, c])
        col_out = self._axis_attention(col_q, col_k, col_v).reshape([b, w, h, c])
        col_out = col_out.transpose([0, 2, 1, 3])

        out = (row_out + col_out).transpose([0, 3, 1, 2]) * 0.5
        out = self.proj(out)
        return x + self.gamma * out


class FinancialStructureAttention(nn.Layer):
    def __init__(
        self,
        in_channels,
        target_index=-1,
        enable_header=True,
        enable_span=True,
        header_reduction=4,
        span_num_heads=2,
        attn_drop=0.0,
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
        self.enable_header = enable_header
        self.enable_span = enable_span

        if enable_header:
            self.header_attn = HeaderAwareSpatialAttention(
                channels, reduction=header_reduction
            )
        if enable_span:
            self.span_attn = SpanAwareAxialAttention(
                channels, num_heads=span_num_heads, attn_drop=attn_drop
            )
        self.out_channels = in_channels

    def forward(self, inputs):
        if self.is_sequence:
            outs = list(inputs)
            x = outs[self.target_index]
            if self.enable_header:
                x = self.header_attn(x)
            if self.enable_span:
                x = self.span_attn(x)
            outs[self.target_index] = x
            return tuple(outs)

        x = inputs
        if self.enable_header:
            x = self.header_attn(x)
        if self.enable_span:
            x = self.span_attn(x)
        return x
