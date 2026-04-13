# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS, SHARED_HEADS)

MODELS = Registry('models', parent=MMCV_MODELS)

VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

