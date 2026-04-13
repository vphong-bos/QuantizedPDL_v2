import os

from mmcv import Config
from mmdet.datasets import replace_ImageToTensor

def prepare_cfg(cfg):
    """Normalize cfg.data.test for dataloader creation.

    Returns:
        tuple: (cfg, samples_per_gpu)
    """
    samples_per_gpu = 1

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

        samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)

        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    return cfg, samples_per_gpu
