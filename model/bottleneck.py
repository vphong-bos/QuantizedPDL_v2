# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    PyTorch implementation of BottleneckBlock for ResNet.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        has_shortcut: bool = False,
        shortcut_stride: int = 1,
    ):
        super().__init__()
        self.has_shortcut = has_shortcut

        self.can_add_identity = (in_channels == out_channels and stride == 1)

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.conv1.norm = nn.SyncBatchNorm(bottleneck_channels, eps=1e-05, momentum=0.1)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.conv2.norm = nn.SyncBatchNorm(bottleneck_channels, eps=1e-05, momentum=0.1)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv3.norm = nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1)

        if has_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=shortcut_stride, bias=False)
            self.shortcut.norm = nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.has_shortcut:
            identity = self.shortcut(identity)
            if getattr(self.shortcut, "norm", None) is not None:
                identity = self.shortcut.norm(identity)

        out = self.conv1(x)
        if getattr(self.conv1, "norm", None) is not None:
            out = self.conv1.norm(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if getattr(self.conv2, "norm", None) is not None:
            out = self.conv2.norm(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if getattr(self.conv3, "norm", None) is not None:
            out = self.conv3.norm(out)

        if self.has_shortcut or self.can_add_identity:
            out = out + identity

        out = self.relu3(out)
        return out