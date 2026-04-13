#!/usr/bin/env python3
"""
Run PCC comparison between FP32 SSR model and custom quantized SSR wrapper.

Example:
    python ssr.py \
        --fp32_weights model_wrappers/model/ssr/data/ckpts/ssr_pt.pth \
        --encodings_path qparams/ssr.encodings \
        --config model_wrappers/model/ssr/configs/SSR_e2e.py \
        --max_samples 1 --eval bbox
"""

import argparse
import os
from typing import Dict, Optional, Tuple
import mmcv
print("mmcv loaded from:", mmcv.__file__)
from mmcv.runner import get_dist_info

import torch

from model_wrappers.model.ssr.mmdet3d_plugin.SSR.model import load_default_model
from model_wrappers.model.ssr.utils.dataset import build_eval_loader
from model_wrappers.model.ssr.utils.metrics import evaluate_model
from model_wrappers.ssr import QuantSSRWrapper
from utils.metrics import compute_pcc

import platform
from mmcv.utils import Registry

if platform.system() != "Windows":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP32 SSR with custom wrapped quantized SSR using PCC."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--fp32_weights",
        type=str,
        required=True,
        help="Path to FP32 .pkl weights",
    )

    parser.add_argument(
        "--encodings_path",
        type=str,
        required=True,
        help="Path to activation/parameter encodings JSON",
    )

    parser.add_argument(
        "--config", type=str,
        required=True,
        help="Config file for model creation",
    )

    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase the inference speed",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Number of random test iterations to run",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    cfg, dataset, data_loader = build_eval_loader(args.config)

    rank, _ = get_dist_info()

    fp32_results = None
    quant_results = None

    print("Loading FP32 SSR model...")
    fp32_model, _ = load_default_model(
        cfg=cfg,
        checkpoint_path=args.fp32_weights,
        dataset=dataset,
        fuse_conv_bn=args.fuse_conv_bn,
        map_location=args.device,
    )
    fp32_model.eval()

    fp32_results = evaluate_model(
        model_obj={
            "backend": "torch",
            "model": fp32_model,
            "session": None,
            "input_name": None,
            "output_names": None,
            "is_quant": False,
        },
        data_loader=data_loader,
        max_samples=args.max_samples,
    )

    if rank == 0:
        print("======================================================")
        print(dataset.evaluate(fp32_results, metric=args.eval))

    print("Loading custom quantized SSR wrapper...")
    quant_model = QuantSSRWrapper(
        model=load_default_model(
            cfg=cfg,
            checkpoint_path=args.fp32_weights,
            dataset=dataset,
            fuse_conv_bn=args.fuse_conv_bn,
            map_location=args.device,
        )[0],
        encodings_path=args.encodings_path,
    )
    quant_model.eval()

    quant_results = evaluate_model(
        model_obj={
            "backend": "torch",
            "model": quant_model,
            "session": None,
            "input_name": None,
            "output_names": None,
            "is_quant": True,
        },
        data_loader=data_loader,
        max_samples=args.max_samples,
    )

    if rank == 0:
        print("======================================================")
        print(dataset.evaluate(quant_results, metric=args.eval))

    if rank == 0 and fp32_results is not None and quant_results is not None:
        pcc, num_values = compute_pcc(fp32_results, quant_results)
        print("======================================================")
        if pcc is None:
            print(f"PCC could not be computed (num_values={num_values})")
        else:
            print(f"FP32 vs Quant PCC: {pcc:.8f} (num_values={num_values})")


if __name__ == "__main__":
    main()