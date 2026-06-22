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

from .global_local_mamba_fusion import GlobalLocalMambaFusion


__all__ = ["YOLOMabeFusion"]


class TableCoordinateAttention(nn.Layer):
    """Lightweight row/column coordinate attention for table features."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.conv1 = nn.Conv2D(channels, hidden_channels, kernel_size=1, bias_attr=True)
        self.conv_h = nn.Conv2D(hidden_channels, channels, kernel_size=1, bias_attr=True)
        self.conv_w = nn.Conv2D(hidden_channels, channels, kernel_size=1, bias_attr=True)
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        row_context = F.adaptive_avg_pool2d(x, output_size=[h, 1])
        col_context = F.adaptive_avg_pool2d(x, output_size=[1, w]).transpose([0, 1, 3, 2])
        context = paddle.concat([row_context, col_context], axis=2)
        context = F.hardswish(self.conv1(context))
        row_context, col_context = paddle.split(context, num_or_sections=[h, w], axis=2)
        col_context = col_context.transpose([0, 1, 3, 2])
        row_attn = F.sigmoid(self.conv_h(row_context))
        col_attn = F.sigmoid(self.conv_w(col_context))
        return x + self.gamma * x * row_attn * col_attn


class GatedMultiScaleResidualFusion(nn.Layer):
    """Gentle YOLO-style multi-scale fusion without replacing CSPPAN outputs."""

    def __init__(self, in_channels, target_index=-1, reduction=4, source_indices=None):
        super().__init__()
        assert not isinstance(in_channels, int), "multi-scale fusion needs sequence inputs"
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.target_index = target_index if target_index >= 0 else self.num_levels + target_index
        self.source_indices = source_indices
        if self.source_indices is None:
            self.source_indices = [i for i in range(self.num_levels) if i != self.target_index]
        self.source_indices = [i if i >= 0 else self.num_levels + i for i in self.source_indices]
        target_channels = in_channels[self.target_index]
        hidden_channels = max(target_channels // reduction, 8)

        self.proj_layers = nn.LayerList()
        for index in self.source_indices:
            self.proj_layers.append(
                nn.Sequential(
                    nn.Conv2D(in_channels[index], target_channels, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(target_channels),
                    nn.Hardswish(),
                )
            )
        fusion_channels = target_channels * (len(self.source_indices) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2D(fusion_channels, hidden_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hidden_channels),
            nn.Hardswish(),
            nn.Conv2D(hidden_channels, target_channels, kernel_size=1, bias_attr=True),
        )
        self.gate = nn.Conv2D(fusion_channels, target_channels, kernel_size=1, bias_attr=True)
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, inputs):
        target = inputs[self.target_index]
        target_size = target.shape[2:]
        feats = [target]
        for proj, index in zip(self.proj_layers, self.source_indices):
            feat = proj(inputs[index])
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feats.append(feat)
        fused_input = paddle.concat(feats, axis=1)
        residual = self.fuse(fused_input)
        gate = F.sigmoid(self.gate(fused_input))
        return target + self.gamma * gate * residual


class YOLOMabeFusion(nn.Layer):
    """Mabe/Mamba global-local modeling plus gentle YOLO-style table cues.

    The module is intended as a PostNeck after CSPPAN. It keeps the original
    dataset/loss/head contract unchanged and initializes all new residual paths
    at zero contribution for stable fine-tuning from an existing SLANet model.
    """

    def __init__(
        self,
        in_channels,
        target_index=-1,
        enable_mamba=True,
        enable_local=True,
        fusion_mode="gated_sum",
        mamba_expand=1,
        local_kernel=3,
        drop_path=0.0,
        enable_coord=True,
        coord_reduction=8,
        enable_multiscale=True,
        multiscale_reduction=4,
        multiscale_source_indices=None,
        **kwargs,
    ):
        super().__init__()
        self.is_sequence = not isinstance(in_channels, int)
        if self.is_sequence:
            num_levels = len(in_channels)
            self.target_index = target_index if target_index >= 0 else num_levels + target_index
            channels = in_channels[self.target_index]
        else:
            self.target_index = target_index
            channels = in_channels
        self.enable_coord = enable_coord
        self.enable_multiscale = enable_multiscale and self.is_sequence

        self.mabe = GlobalLocalMambaFusion(
            in_channels=in_channels,
            target_index=target_index,
            enable_mamba=enable_mamba,
            enable_local=enable_local,
            fusion_mode=fusion_mode,
            mamba_expand=mamba_expand,
            local_kernel=local_kernel,
            drop_path=drop_path,
        )
        if enable_coord:
            self.coord_attn = TableCoordinateAttention(channels, reduction=coord_reduction)
        if self.enable_multiscale:
            self.ms_fusion = GatedMultiScaleResidualFusion(
                in_channels,
                target_index=target_index,
                reduction=multiscale_reduction,
                source_indices=multiscale_source_indices,
            )
        self.out_channels = in_channels

    def forward(self, inputs):
        outputs = self.mabe(inputs)
        if self.is_sequence:
            outs = list(outputs)
            x = outs[self.target_index]
            if self.enable_coord:
                x = self.coord_attn(x)
            outs[self.target_index] = x
            if self.enable_multiscale:
                outs[self.target_index] = self.ms_fusion(tuple(outs))
            return tuple(outs)

        x = outputs
        if self.enable_coord:
            x = self.coord_attn(x)
        return x
