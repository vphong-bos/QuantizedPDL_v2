#!/usr/bin/env python3
import argparse
import os

import torch

from model.pdl import (
    build_model,
)

from quantization.quantize_function import load_aimet_quantized_model

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

from utils.pcc_metric import evaluate_pcc
from utils.export_onnx import export_optimized_onnx_model

import torch
from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d
from model.quantized_conv2d import QuantizedConv2d

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_EXPORT_PATH = os.path.join(pdl_home_path, "quantized_export")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cityscapes_root", type=str, required=True)

    parser.add_argument("--fp32_weights", type=str, required=True,
                        help="Path to FP32 .pkl weights")
    parser.add_argument("--quant_weights", type=str, required=True,
                        help="Path to quantized model: AIMET checkpoint (.pt/.pth/.pkl) or ONNX (.onnx)")

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Use only first N val images, -1 for full val")
    parser.add_argument("--split", type=str, default="val",
                        choices=["test", "val"])

    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")

    parser.add_argument("--onnx_provider", type=str, default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
                        help="ONNX Runtime execution provider when quant_weights is .onnx")

    parser.add_argument(
        "--export_optimized_onnx",
        action="store_true",
        help="Export ORT-optimized ONNX model from --quant_weights if it is .onnx",
    )
    parser.add_argument(
        "--optimized_onnx_name",
        type=str,
        default="optimized_pdl",
        help="Output path for optimized ONNX model",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.export_optimized_onnx:
        optimized_onnx_path = os.path.join(DEFAULT_EXPORT_PATH, f"{args.optimized_onnx_name}.onnx")

        export_optimized_onnx_model(
            quant_weights=args.quant_weights,
            output_path=optimized_onnx_path,
            provider=args.onnx_provider,
        )

    loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        split=args.split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.fp32_weights:
        print("Loading FP32 model...")
        fp32_model, fp32_category = build_model(
            weights_path=args.fp32_weights,
            model_category=args.model_category,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device,
        )

        print("Evaluating FP32...")
        fp32_results = evaluate_model(
            model_obj={
                "backend": "torch",
                "model": fp32_model,
                "session": None,
                "input_name": None,
                "output_names": None,
            },
            model_category_const=fp32_category,
            loader=loader,
            device=args.device,
            max_samples=args.max_samples,
        )

    print("Loading quantized model...")
    quant_obj = load_aimet_quantized_model(
        quant_weights=args.quant_weights,
        model_category=args.model_category,
        device=args.device,
        provider=args.onnx_provider,
    )

    print("Evaluating quantized...")
    quant_results = evaluate_model(
        model_obj=quant_obj,
        model_category_const=quant_obj["model_category_const"],
        loader=loader,
        device=args.device,
        max_samples=args.max_samples,
    )

    if not args.fp32_weights:
        print(quant_results)

    if args.fp32_weights:
        print("Evaluating PCC between FP32 and quantized outputs...")
        if quant_obj["backend"] == "torch":
            pcc_results = evaluate_pcc(
                fp32_model=fp32_model,
                quant_model=quant_obj["model"],
                loader=loader,
                device=args.device,
                max_samples=args.max_samples,
            )
            pcc_value = pcc_results["PCC"]
        else:
            print("Skipping PCC: current evaluate_pcc expects a torch model, but quant model is ONNX Runtime.")
            pcc_value = float("nan")

        print("\n================ Compare FP32 vs Quantized ================")

        print("---- Accuracy ----")
        print(f"FP32  mIoU : {fp32_results['mIoU']:.4f}")
        print(f"INT8  mIoU : {quant_results['mIoU']:.4f}")
        print(f"Drop       : {quant_results['mIoU'] - fp32_results['mIoU']:.4f}")

        print("\n---- Correlation ----")
        print(f"PCC        : {pcc_value:.6f}")

        print("\n---- Performance ----")
        print(f"FP32  FPS  : {fp32_results['FPS']:.2f}")
        print(f"INT8  FPS  : {quant_results['FPS']:.2f}")
        print(f"Speedup    : {quant_results['FPS'] / fp32_results['FPS']:.2f}x")

        print(f"FP32  Latency (ms): {fp32_results['Avg_Inference_Time_ms']:.2f}")
        print(f"INT8  Latency (ms): {quant_results['Avg_Inference_Time_ms']:.2f}")
        print(f"Latency reduction : {fp32_results['Avg_Inference_Time_ms'] - quant_results['Avg_Inference_Time_ms']:.2f} ms")

        print("===========================================================")

if __name__ == "__main__":
    main()