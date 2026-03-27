#!/usr/bin/env python3
import argparse
import os
import random
from typing import Any, Dict, Iterable, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from model.pdl import build_model
from quantization.calibration_dataset import (
    create_calibration_loader,
    sample_calibration_images,
)
from secret_incrediants.fold_conv_bn import (
    count_custom_conv_with_bn,
    debug_remaining_custom_conv_with_bn,
    fold_custom_conv_bn_inplace,
)
from utils.image_loader import load_images

from aimet_torch.cross_layer_equalization import equalize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FP32 ONNX and quantized ONNX with improved PTQ preprocessing."
    )

    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
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
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--calib_images", type=str, required=True)
    parser.add_argument("--num_calib", type=int, default=200)
    parser.add_argument("--calib_max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--export_path", type=str, default="quantized_export")
    parser.add_argument("--fp32_name", type=str, default="model_fp32.onnx")
    parser.add_argument("--quant_name", type=str, default="model_int8.onnx")
    parser.add_argument(
        "--quant_format",
        type=str,
        default="qoperator",
        choices=["qdq", "qoperator"],
        help="Actual ONNX quantization format to export.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        choices=["minmax", "entropy", "percentile"],
    )

    parser.add_argument("--enable_custom_conv_bn_fold", action="store_true")
    parser.add_argument(
        "--run_cle",
        action="store_true",
        help="Run AIMET cross-layer equalization before ONNX export.",
    )

    parser.add_argument(
        "--activation_type",
        type=str,
        default="qint8",
        choices=["qint8", "quint8"],
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="qint8",
        choices=["qint8"],
    )
    parser.add_argument("--per_channel", action="store_true", default=True)
    parser.add_argument("--disable_per_channel", action="store_true")
    parser.add_argument("--activation_symmetric", action="store_true")
    parser.add_argument("--weight_symmetric", action="store_true", default=True)
    parser.add_argument("--disable_weight_symmetric", action="store_true")

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def is_numeric_tensor_or_array(x: Any) -> bool:
    if torch.is_tensor(x):
        return x.dtype in {
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        }
    if isinstance(x, np.ndarray):
        return x.dtype.kind in ("b", "i", "u", "f")
    return False


def find_first_numeric_tensor(obj: Any) -> Optional[Any]:
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj if is_numeric_tensor_or_array(obj) else None

    if isinstance(obj, dict):
        preferred_keys = ["image", "images", "input", "inputs", "pixel_values", "img"]
        for key in preferred_keys:
            if key in obj:
                found = find_first_numeric_tensor(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = find_first_numeric_tensor(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = find_first_numeric_tensor(item)
            if found is not None:
                return found
        return None

    return None


class LoaderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, loader: Iterable[Any], input_name: str, max_samples: int = -1):
        self.loader = loader
        self.input_name = input_name
        self.max_samples = max_samples
        self._iter = None
        self._count = 0
        self.rewind()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.max_samples >= 0 and self._count >= self.max_samples:
            return None

        try:
            batch = next(self._iter)
        except StopIteration:
            return None

        image = find_first_numeric_tensor(batch)
        if image is None:
            raise ValueError(
                f"Could not find numeric tensor in calibration batch type: {type(batch)}"
            )

        image = to_numpy(image)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        batch_count = int(image.shape[0]) if image.ndim > 0 else 1
        self._count += batch_count
        return {self.input_name: image}

    def rewind(self) -> None:
        self._iter = iter(self.loader)
        self._count = 0


def collect_onnx_op_counts(model_path: str) -> Dict[str, int]:
    model = onnx.load(model_path)
    op_counts: Dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"[INFO] ONNX stats for: {model_path}")
    for op_name in [
        "Conv",
        "QuantizeLinear",
        "DequantizeLinear",
        "QLinearConv",
        "QLinearMatMul",
    ]:
        print(f"  {op_name:<16}: {op_counts.get(op_name, 0)}")
    return op_counts


def get_quant_format(name: str) -> QuantFormat:
    mapping = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator,
    }
    return mapping[name]


def get_calibration_method(name: str) -> CalibrationMethod:
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return mapping[name]


def get_quant_type(name: str) -> QuantType:
    mapping = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }
    return mapping[name]


def maybe_fold_custom_conv_bn(model: torch.nn.Module, enabled: bool) -> None:
    if not enabled:
        return

    before_count, _ = count_custom_conv_with_bn(model)
    print(f"[INFO] Custom Conv+BN before folding: {before_count}")

    folded, skipped = fold_custom_conv_bn_inplace(model)
    print(f"[INFO] Folded count : {folded}")
    print(f"[INFO] Skipped count: {skipped}")

    after_count, after_names = count_custom_conv_with_bn(model)
    print(f"[INFO] Custom Conv+BN after folding: {after_count}")

    if after_count > 0:
        print("[INFO] Remaining modules with BN:")
        for name in after_names[:50]:
            print(f"  {name}")
        debug_remaining_custom_conv_with_bn(model, max_items=20)


def maybe_run_cle(model: torch.nn.Module, dummy_input: torch.Tensor, enabled: bool) -> None:
    if not enabled:
        return

    print("[INFO] Running AIMET Cross-Layer Equalization...")
    equalize_model(model, dummy_input)
    print("[INFO] Cross-Layer Equalization completed.")


def export_fp32_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[INFO] Exporting plain FP32 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
        do_constant_folding=True,
        dynamo=False,
    )

    fp32_stats = collect_onnx_op_counts(output_path)
    if fp32_stats.get("QuantizeLinear", 0) > 0 or fp32_stats.get("DequantizeLinear", 0) > 0:
        raise RuntimeError(
            "Exported FP32 ONNX still contains QuantizeLinear/DequantizeLinear. "
            "This is not a plain FP32 ONNX."
        )


def build_calibration_loader(args: argparse.Namespace):
    print("[INFO] Collecting calibration images...")
    all_calib_images = load_images(args.calib_images, num_iters=-1, recursive=True)
    calib_images = sample_calibration_images(all_calib_images, args.num_calib, args.seed)

    if not calib_images:
        raise RuntimeError("No calibration images were selected.")

    print(f"[INFO] Found {len(all_calib_images)} candidate calibration images")
    print(f"[INFO] Using {len(calib_images)} images for calibration")

    return create_calibration_loader(
        calib_image_paths=calib_images,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def export_quantized_onnx(
    fp32_onnx_path: str,
    output_path: str,
    calib_loader,
    quant_format_name: str,
    activation_type_name: str,
    weight_type_name: str,
    calibration_method_name: str,
    per_channel: bool,
    activation_symmetric: bool,
    weight_symmetric: bool,
    calib_samples: int,
    provider: str = "CPUExecutionProvider",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    providers = ["CPUExecutionProvider"]
    if provider == "CUDAExecutionProvider":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("[INFO] Creating ORT session for static quantization...")
    sess = ort.InferenceSession(fp32_onnx_path, providers=providers)

    input_names = [x.name for x in sess.get_inputs()]
    if len(input_names) != 1:
        raise ValueError(f"Expected exactly 1 ONNX input, got {input_names}")

    input_name = input_names[0]
    print(f"[INFO] Using ONNX input name: {input_name}")
    print(f"[INFO] Quant format       : {quant_format_name}")
    print(f"[INFO] Calibration method: {calibration_method_name}")
    print(f"[INFO] Activation type   : {activation_type_name}")
    print(f"[INFO] Weight type       : {weight_type_name}")
    print(f"[INFO] Per-channel       : {per_channel}")
    print(f"[INFO] Act symmetric     : {activation_symmetric}")
    print(f"[INFO] Weight symmetric  : {weight_symmetric}")

    data_reader = LoaderCalibrationDataReader(
        loader=calib_loader,
        input_name=input_name,
        max_samples=calib_samples,
    )

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=get_quant_format(quant_format_name),
        activation_type=get_quant_type(activation_type_name),
        weight_type=get_quant_type(weight_type_name),
        calibrate_method=get_calibration_method(calibration_method_name),
        per_channel=per_channel,
        extra_options={
            "ActivationSymmetric": activation_symmetric,
            "WeightSymmetric": weight_symmetric,
        },
    )

    print(f"[INFO] Saved quantized ONNX to: {output_path}")

    quant_stats = collect_onnx_op_counts(output_path)
    if quant_format_name == "qoperator":
        if quant_stats.get("QLinearConv", 0) == 0 and quant_stats.get("QLinearMatMul", 0) == 0:
            print("[WARN] QOperator export produced no QLinearConv/QLinearMatMul.")
        else:
            print("[INFO] QOperator export looks valid.")
    else:
        if quant_stats.get("QuantizeLinear", 0) == 0 and quant_stats.get("DequantizeLinear", 0) == 0:
            print("[WARN] QDQ export produced no QuantizeLinear/DequantizeLinear nodes.")
        else:
            print("[INFO] QDQ export looks valid.")


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.num_calib < 1:
        raise ValueError("num_calib must be >= 1")

    set_random_seed(args.seed)
    os.makedirs(args.export_path, exist_ok=True)

    per_channel = False if args.disable_per_channel else args.per_channel
    weight_symmetric = False if args.disable_weight_symmetric else args.weight_symmetric

    print("[INFO] Loading FP32 model...")
    model, _ = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    model = model.to(args.device).eval()

    dummy_input = torch.randn(
        1,
        3,
        args.image_height,
        args.image_width,
        device=args.device,
    )

    maybe_fold_custom_conv_bn(model, args.enable_custom_conv_bn_fold)
    maybe_run_cle(model, dummy_input, args.run_cle)

    fp32_onnx_path = os.path.join(args.export_path, args.fp32_name)
    quant_onnx_path = os.path.join(args.export_path, args.quant_name)

    export_fp32_onnx(model, dummy_input, fp32_onnx_path)

    calib_loader = build_calibration_loader(args)
    export_quantized_onnx(
        fp32_onnx_path=fp32_onnx_path,
        output_path=quant_onnx_path,
        calib_loader=calib_loader,
        quant_format_name=args.quant_format,
        activation_type_name=args.activation_type,
        weight_type_name=args.weight_type,
        calibration_method_name=args.calibration_method,
        per_channel=per_channel,
        activation_symmetric=args.activation_symmetric,
        weight_symmetric=weight_symmetric,
        calib_samples=args.calib_max_samples,
        provider="CPUExecutionProvider",
    )

    print("[INFO] Done.")
    

if __name__ == "__main__":
    main(parse_args())
