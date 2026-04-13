from mmcv.cnn import fuse_conv_bn as fuse_conv_bn_model
from mmcv.runner import load_checkpoint
from model_wrappers.model.ssr.mmdet3d_plugin.SSR.utils.builder import build_model

def load_default_model(
    cfg,
    checkpoint_path,
    dataset=None,
    fuse_conv_bn=False,
    map_location="cpu",
):
    """
    Build the original torch model and load checkpoint in the same style as MMDet/MMDet3D.

    Args:
        cfg: mmcv Config
        checkpoint_path (str): path to checkpoint
        build_model (callable): function like build_model(cfg.model, test_cfg=...)
        dataset: optional dataset, used for PALETTE fallback
        fuse_conv_bn_flag (bool): whether to fuse conv and bn
        map_location (str): checkpoint load location

    Returns:
        tuple: (model, checkpoint)
    """
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    checkpoint = load_checkpoint(model, checkpoint_path, map_location=map_location)

    if fuse_conv_bn:
        model = fuse_conv_bn_model(model)

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]

    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif dataset is not None and hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    return model, checkpoint