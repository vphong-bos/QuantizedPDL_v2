# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

def _check_if_dynamo_compiling() -> bool:
    """Check if running under torch.compile dynamo compilation"""
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False

import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# class Conv2d(nn.Conv2d):
#     """
#     Conv2d compatible with existing checkpoints.

#     - Keeps weight/bias at the top level, so old state_dict keys still match
#     - Supports optional norm and activation
#     - Uses nn.Conv2d.forward() for the convolution itself
#     """

#     def __init__(self, *args, **kwargs):
#         norm = kwargs.pop("norm", None)
#         activation = kwargs.pop("activation", None)

#         super().__init__(*args, **kwargs)

#         self.norm = norm
#         self.activation = activation

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Optional empty-input handling, preserving old behavior
#         if not torch.jit.is_scripting():
#             is_dynamo_compiling = _check_if_dynamo_compiling()
#             if not is_dynamo_compiling:
#                 with warnings.catch_warnings(record=True):
#                     if x.numel() == 0 and self.training:
#                         assert not isinstance(
#                             self.norm, nn.SyncBatchNorm
#                         ), "SyncBatchNorm does not support empty inputs!"

#         # Important: use parent forward, not raw F.conv2d
#         x = super().forward(x)

#         if self.norm is not None:
#             x = self.norm(x)

#         if self.activation is not None:
#             x = self.activation(x)

#         return x

# class Conv2d(torch.nn.Conv2d):
#     """
#     PyTorch Conv2d wrapper with normalization and activation support.

#     This is a cleaner implementation of a conv2d wrapper that supports:
#     - Optional normalization layers
#     - Optional activation functions
#     - Empty input handling (for compatibility)

#     Similar in structure to the TTNN wrapper but maintains backward compatibility
#     with the existing torch.nn.Conv2d interface.

#     Usage examples:

#     # Basic convolution with norm and activation
#     conv = Conv2d(in_channels, out_channels, kernel_size=3,
#                   norm=nn.BatchNorm2d(out_channels), activation=F.relu)

#     # Simple convolution
#     conv = Conv2d(in_channels, out_channels, kernel_size=1)
#     """

#     def __init__(self, *args, **kwargs):
#         """
#         Initialize Conv2d with optional normalization and activation.

#         Args:
#             norm (nn.Module, optional): normalization layer to apply after convolution
#             activation (callable, optional): activation function to apply after normalization

#         All other arguments are passed to torch.nn.Conv2d.

#         Note: Normalization is applied before activation.
#         """
#         # Extract our custom arguments before calling parent init
#         norm = kwargs.pop("norm", None)
#         activation = kwargs.pop("activation", None)

#         # Initialize the base Conv2d first
#         super().__init__(*args, **kwargs)

#         # Now assign the custom attributes after parent init
#         self.norm = norm
#         self.activation = activation

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#     x = F.conv2d(
#         x, self.weight, self.bias,
#         self.stride, self.padding, self.dilation, self.groups
#     )

#     if self.norm is not None:
#         x = self.norm(x)

#     if self.activation is not None:
#         x = self.activation(x)

#     return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Forward pass with conv2d, optional norm, and optional activation.

    #     Args:
    #         x: Input tensor

    #     Returns:
    #         Output tensor after convolution, normalization, and activation
    #     """
    #     # Handle empty inputs for compatibility (from original implementation)
    #     if not torch.jit.is_scripting():
    #         is_dynamo_compiling = _check_if_dynamo_compiling()
    #         if not is_dynamo_compiling:
    #             with warnings.catch_warnings(record=True):
    #                 if x.numel() == 0 and self.training:
    #                     # https://github.com/pytorch/pytorch/issues/12013
    #                     assert not isinstance(
    #                         self.norm, torch.nn.SyncBatchNorm
    #                     ), "SyncBatchNorm does not support empty inputs!"

    #     # Standard conv2d operation
    #     x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    #     # Apply normalization if provided
    #     if self.norm is not None:
    #         x = self.norm(x)

    #     # Apply activation if provided
    #     if self.activation is not None:
    #         x = self.activation(x)

    #     return x