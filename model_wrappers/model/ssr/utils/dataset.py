import os

from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS

from model_wrappers.model.ssr.utils.config import prepare_cfg
from model_wrappers.model.ssr.mmdet3d_plugin.datasets.builder import build_dataloader

def build_dataset(cfg, default_args=None):
    """Build a dataset from an MMDetection-style dataset config."""
    return build_from_cfg(cfg, DATASETS, default_args)


def extract_data(data):
    """Extract tensor payloads from MMCV DataContainer fields."""
    data["img_metas"] = data["img_metas"][0].data
    data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data
    data["gt_labels_3d"] = data["gt_labels_3d"][0].data
    data["img"] = data["img"][0].data
    data["ego_his_trajs"] = data["ego_his_trajs"][0].data
    data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data
    data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data
    data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data
    data["gt_attr_labels"] = data["gt_attr_labels"][0].data
    data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
    data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
    return data

def build_eval_loader(
    config_path,
    cfg_options=None,
    distributed=False,
    shuffle=False,
    is_calib=False,
):
    """Create dataset and dataloader for test/inference from a config file.

    Args:
        config_path (str): Path to the MMCV config file.
        cfg_options (dict | None): Optional config overrides.
        distributed (bool): Whether to build a distributed dataloader.
        shuffle (bool): Whether to shuffle the dataloader.

    Returns:
        tuple: (cfg, dataset, data_loader)
    """
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    cfg.model.pretrained = None
    cfg, samples_per_gpu = prepare_cfg(cfg)

    dataset = None
    if is_calib:
        dataset = build_dataset(cfg.data.val)
    else:
        dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=shuffle,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return cfg, dataset, data_loader
