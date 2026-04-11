#!/usr/bin/env python3
"""
Run PCC comparison between FP32 PDL model and custom quantized PDL wrapper.

Example:
    python pdl.py \
        --fp32_weights /path/to/model.pkl \
        --encodings_path /path/to/encodings.json \
        --image_height 512 \
        --image_width 1024 \
        --max_samples 10 \
        --debug
"""

import argparse
import os
from typing import Dict, Optional, Tuple

import torch

from model_wrappers.model.pdl.pdl import build_model
from model_wrappers.pdl import QuantPDLWrapper


def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient between two tensors.
    """
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)

    if x.numel() == 0 or y.numel() == 0:
        return float("nan")

    if x.numel() != y.numel():
        raise ValueError(f"PCC shape mismatch after flatten: {x.numel()} vs {y.numel()}")

    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = torch.sqrt((x_centered ** 2).sum()) * torch.sqrt((y_centered ** 2).sum())
    if denom.item() == 0:
        return float("nan")

    pcc = (x_centered * y_centered).sum() / denom
    return float(pcc.item())


def compare_outputs(
    fp32_out: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]],
    quant_out: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]],
):
    names = ["semantic_logits", "center_heatmap", "offset_map"]
    pcc_results = {}

    for idx, name in enumerate(names):
        fp32_tensor = fp32_out[idx]
        quant_tensor = quant_out[idx]

        if fp32_tensor is None or quant_tensor is None:
            pcc_results[name] = None
            continue

        pcc_results[name] = compute_pcc(fp32_tensor, quant_tensor)

    # Optional features comparison
    fp32_features = fp32_out[3]
    quant_features = quant_out[3]

    feature_pcc = {}
    if isinstance(fp32_features, dict) and isinstance(quant_features, dict):
        common_keys = sorted(set(fp32_features.keys()) & set(quant_features.keys()))
        for key in common_keys:
            if torch.is_tensor(fp32_features[key]) and torch.is_tensor(quant_features[key]):
                feature_pcc[key] = compute_pcc(fp32_features[key], quant_features[key])

    return pcc_results, feature_pcc


def make_input(batch_size: int, image_height: int, image_width: int, device: str) -> torch.Tensor:
    return torch.randn(
        batch_size,
        3,
        image_height,
        image_width,
        device=device,
        dtype=torch.float32,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP32 PDL with custom wrapped quantized PDL using PCC."
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
        "--model_category",
        type=str,
        default="PANOPTIC_DEEPLAB",
        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
    )

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Number of random test iterations to run",
    )

    parser.add_argument(
        "--return_features",
        action="store_true",
        help="Compare backbone feature dict too",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable activation quant debug in QuantPDLWrapper",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading FP32 PDL model...")
    fp32_model, model_category_const = build_model(
        weights_path=args.fp32_weights,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    fp32_model.eval()

    print("Loading custom quantized PDL wrapper...")
    quant_model = QuantPDLWrapper(
        model=build_model(
            weights_path=args.fp32_weights,
            model_category=args.model_category,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device,
        )[0],
        encodings_path=args.encodings_path,
        debug_activation_quant=args.debug,
    )
    quant_model.eval()

    all_pcc = {
        "semantic_logits": [],
        "center_heatmap": [],
        "offset_map": [],
    }
    all_feature_pcc = {}

    with torch.no_grad():
        for i in range(args.max_samples):
            x = make_input(
                batch_size=args.batch_size,
                image_height=args.image_height,
                image_width=args.image_width,
                device=args.device,
            )

            fp32_out = fp32_model(x, return_features=args.return_features)
            quant_out = quant_model(x, return_features=args.return_features)

            pcc_results, feature_pcc = compare_outputs(fp32_out, quant_out)

            print(f"\n===== Sample {i + 1}/{args.max_samples} =====")
            for name, value in pcc_results.items():
                if value is None:
                    print(f"{name:20s}: None")
                else:
                    print(f"{name:20s}: {value:.6f}")
                    all_pcc[name].append(value)

            if args.return_features and feature_pcc:
                print("Feature PCC:")
                for key, value in feature_pcc.items():
                    print(f"  {key:16s}: {value:.6f}")
                    all_feature_pcc.setdefault(key, []).append(value)

    print("\n================ Final PCC Summary ================")
    for name, values in all_pcc.items():
        if not values:
            print(f"{name:20s}: None")
        else:
            avg = sum(values) / len(values)
            print(f"{name:20s}: avg={avg:.6f}  min={min(values):.6f}  max={max(values):.6f}")

    if args.return_features and all_feature_pcc:
        print("\nFeature PCC Summary:")
        for key, values in sorted(all_feature_pcc.items()):
            avg = sum(values) / len(values)
            print(f"{key:20s}: avg={avg:.6f}  min={min(values):.6f}  max={max(values):.6f}")

    print("===================================================")


if __name__ == "__main__":
    main()