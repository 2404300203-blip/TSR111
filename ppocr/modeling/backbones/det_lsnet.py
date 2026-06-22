# Copyright (c) 2026. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBNAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LSBlock(nn.Layer):
    """Large-small convolution block for table visual feature extraction.

    The block first mixes a wider spatial neighborhood with a large-kernel
    depthwise convolution, then refines local responses with a small-kernel
    depthwise convolution. Pointwise layers provide channel mixing while the
    residual path keeps optimization stable for SLANet fine-tuning.
    """

    def __init__(self, channels, large_kernel=7, mlp_ratio=2.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.large_dw = ConvBNAct(
            channels, channels, kernel_size=large_kernel, stride=1, groups=channels
        )
        self.small_dw = ConvBNAct(
            channels, channels, kernel_size=3, stride=1, groups=channels
        )
        self.pw1 = ConvBNAct(channels, hidden_channels, kernel_size=1, stride=1)
        self.pw2 = ConvBNAct(hidden_channels, channels, kernel_size=1, stride=1, act=False)
        self.act = nn.Hardswish()

    def forward(self, x):
        identity = x
        y = self.large_dw(x)
        y = self.small_dw(y)
        y = self.pw1(y)
        y = self.pw2(y)
        return self.act(identity + y)


class LSStage(nn.Layer):
    def __init__(self, in_channels, out_channels, depth, large_kernel, mlp_ratio):
        super().__init__()
        self.downsample = ConvBNAct(
            in_channels, out_channels, kernel_size=3, stride=2
        )
        self.blocks = nn.Sequential(
            *[
                LSBlock(
                    out_channels,
                    large_kernel=large_kernel,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class LSNet(nn.Layer):
    """LSNet-style multi-scale backbone for PaddleOCR table models.

    This implementation keeps the SLANet contract rather than mirroring the
    classification-only LSNet head: forward returns four feature maps for
    CSPPAN, and out_channels reports the corresponding channel dimensions.
    """

    VARIANTS = {
        "tiny": {
            "channels": [64, 128, 256, 512],
            "depths": [2, 2, 4, 2],
            "large_kernels": [7, 7, 9, 9],
            "mlp_ratio": 2.0,
        },
        "small": {
            "channels": [64, 128, 256, 512],
            "depths": [2, 3, 6, 3],
            "large_kernels": [7, 7, 9, 9],
            "mlp_ratio": 2.0,
        },
    }

    def __init__(self, in_channels=3, variant="tiny", pretrained=False, **kwargs):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(
                "Unsupported LSNet variant {}. Supported variants: {}".format(
                    variant, list(self.VARIANTS.keys())
                )
            )
        cfg = self.VARIANTS[variant]
        channels = cfg["channels"]
        depths = cfg["depths"]
        large_kernels = cfg["large_kernels"]
        mlp_ratio = cfg["mlp_ratio"]

        self.out_channels = channels
        self.stem = ConvBNAct(in_channels, 32, kernel_size=3, stride=2)
        self.stages = nn.LayerList()
        stage_in_channels = [32] + channels[:-1]
        for in_c, out_c, depth, large_kernel in zip(
            stage_in_channels, channels, depths, large_kernels
        ):
            self.stages.append(
                LSStage(
                    in_c,
                    out_c,
                    depth=depth,
                    large_kernel=large_kernel,
                    mlp_ratio=mlp_ratio,
                )
            )

        if pretrained:
            print(
                "LSNet pretrained=True was requested, but no Paddle pretrained "
                "weights are configured. Training this backbone from scratch."
            )

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs
