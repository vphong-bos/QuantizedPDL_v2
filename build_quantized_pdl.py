#!/usr/bin/env python3
import argparse
import os
import copy
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

from model.pdl import build_model

from quantization.calibration_dataset import (
    create_calibration_loader,
    sample_calibration_images,
)
from quantization.quantize_function import (
    AimetTraceWrapper,
    create_quant_sim,
    calibration_forward_pass,
)
from quantization.bias_correction import apply_bias_correction, copy_biases
from utils.image_loader import load_images

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model
from secret_incrediants.fold_conv_bn import (
    count_custom_conv_with_bn,
    fold_custom_conv_bn_inplace,
    debug_remaining_custom_conv_with_bn,
)

from aimet_torch.batch_norm_fold import fold_all_batch_norms, fold_all_batch_norms_to_scale
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams
from aimet_torch.bn_reestimation import reestimate_bn_stats
from aimet_common.utils import CallbackFunc
from aimet_torch import quantsim, onnx as aimet_onnx

from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d
from model.quantized_conv2d import QuantizedConv2d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--cityscapes_root", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="val", choices=["test", "val"])

    parser.add_argument("--calib_images", type=str, required=True)
    parser.add_argument("--num_calib", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--export_path", type=str, default="quantized_export")
    parser.add_argument("--enable_custom_conv_bn_fold", action="store_true")

    parser.add_argument("--export_qoperator", action="store_true")
    parser.add_argument("--qop_name", type=str, default="model_qop.onnx")
    parser.add_argument("--calib_max_samples", type=int, default=-1)

    return parser.parse_args()


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _is_numeric_tensor_or_array(x) -> bool:
    if torch.is_tensor(x):
        return x.dtype in {
            torch.float16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool,
        }
    if isinstance(x, np.ndarray):
        return x.dtype.kind in ("b", "i", "u", "f")
    return False


def _find_first_numeric_tensor(obj):
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj if _is_numeric_tensor_or_array(obj) else None

    if isinstance(obj, dict):
        preferred_keys = ["image", "images", "input", "inputs", "pixel_values", "img"]
        for key in preferred_keys:
            if key in obj:
                found = _find_first_numeric_tensor(obj[key])
                if found is not None:
                    return found

        for _, value in obj.items():
            found = _find_first_numeric_tensor(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = _find_first_numeric_tensor(item)
            if found is not None:
                return found
        return None

    return None


class LoaderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, loader, input_name, max_samples=-1):
        self.loader = loader
        self.input_name = input_name
        self.max_samples = max_samples
        self._iter = None
        self._count = 0
        self.rewind()

    def get_next(self):
        if self.max_samples >= 0 and self._count >= self.max_samples:
            return None

        try:
            batch = next(self._iter)
        except StopIteration:
            return None

        image = _find_first_numeric_tensor(batch)
        if image is None:
            raise ValueError(f"Could not find numeric tensor in calibration batch type: {type(batch)}")

        image = _to_numpy(image)

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        self._count += image.shape[0] if image.ndim > 0 else 1
        return {self.input_name: image}

    def rewind(self):
        self._iter = iter(self.loader)
        self._count = 0


def collect_onnx_op_counts(model_path: str):
    model = onnx.load(model_path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"[INFO] ONNX stats for: {model_path}")
    print(f"  Conv            : {op_counts.get('Conv', 0)}")
    print(f"  QuantizeLinear  : {op_counts.get('QuantizeLinear', 0)}")
    print(f"  DequantizeLinear: {op_counts.get('DequantizeLinear', 0)}")
    print(f"  QLinearConv     : {op_counts.get('QLinearConv', 0)}")
    print(f"  QLinearMatMul   : {op_counts.get('QLinearMatMul', 0)}")
    return op_counts

def export_qoperator_onnx_model(
    fp32_onnx_path: str,
    output_path: str,
    calib_loader,
    provider: str = "CPUExecutionProvider",
    calib_samples: int = -1,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    print("[INFO] Creating ORT session for QOperator export...")
    sess = ort.InferenceSession(fp32_onnx_path, providers=providers)

    input_names = [x.name for x in sess.get_inputs()]
    if len(input_names) != 1:
        raise ValueError(f"Expected exactly 1 ONNX input, got {input_names}")

    input_name = input_names[0]
    print(f"[INFO] Using ONNX input name: {input_name}")

    data_reader = LoaderCalibrationDataReader(
        loader=calib_loader,
        input_name=input_name,
        max_samples=calib_samples,
    )

    print("[INFO] Running ORT static quantization to QOperator (fully symmetric)...")
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    print(f"[INFO] Saved QOperator ONNX to: {output_path}")

def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    os.makedirs(args.export_path, exist_ok=True)

    print("Loading FP32 model...")
    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    model = model.to(args.device).eval()
    dummy_input = torch.randn(1, 3, args.image_height, args.image_width, device=args.device)

    if args.enable_custom_conv_bn_fold:
        before_count, before_names = count_custom_conv_with_bn(model)
        print(f"[INFO] Custom Conv+BN before folding: {before_count}")

        folded, skipped = fold_custom_conv_bn_inplace(model)
        print(f"[INFO] Folded count : {folded}")
        print(f"[INFO] Skipped count: {skipped}")

        after_count, after_names = count_custom_conv_with_bn(model)
        print(f"[INFO] Custom Conv+BN after folding: {after_count}")

        if after_count > 0:
            print("[INFO] Remaining modules with BN:")
            for n in after_names[:50]:
                print("  ", n)
            debug_remaining_custom_conv_with_bn(model, max_items=20)

    fp32_onnx_path = os.path.join(args.export_path, "model_fp32.onnx")
    qop_onnx_path = os.path.join(args.export_path, args.qop_name)

    print("[INFO] Exporting plain FP32 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        do_constant_folding=True,
        dynamo=False,
    )

    fp32_stats = collect_onnx_op_counts(fp32_onnx_path)
    if fp32_stats.get("QuantizeLinear", 0) > 0 or fp32_stats.get("DequantizeLinear", 0) > 0:
        raise RuntimeError(
            "Exported FP32 ONNX still contains QuantizeLinear/DequantizeLinear. "
            "This is not a plain FP32 ONNX."
        )

    print("Collecting calibration images...")
    all_calib_images = load_images(args.calib_images, num_iters=-1, recursive=True)
    calib_images = sample_calibration_images(all_calib_images, args.num_calib, args.seed)
    print(f"Found {len(all_calib_images)} candidate calibration images")
    print(f"Using {len(calib_images)} images for calibration")

    calib_loader = create_calibration_loader(
        calib_image_paths=calib_images,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.export_qoperator:
        export_qoperator_onnx_model(
            fp32_onnx_path=fp32_onnx_path,
            output_path=qop_onnx_path,
            calib_loader=calib_loader,
            provider="CPUExecutionProvider",
            calib_samples=args.calib_max_samples,
        )

        qop_stats = collect_onnx_op_counts(qop_onnx_path)
        if qop_stats.get("QLinearConv", 0) == 0 and qop_stats.get("QLinearMatMul", 0) == 0:
            print("[WARN] QOperator export produced no QLinearConv/QLinearMatMul.")
        else:
            print("[INFO] QOperator export looks valid.")

if __name__ == "__main__":
    args = parse_args()
    main(args)