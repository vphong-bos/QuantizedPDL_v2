#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from typing import Dict, List

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

from evaluation.eval_dataset import build_eval_loader


# -----------------------------
# JSON helpers
# -----------------------------
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


# -----------------------------
# ORT helpers
# -----------------------------
def make_session_from_model_path(
    model_path: str,
    provider: str = "CPUExecutionProvider",
):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=providers,
    )


def list_input_names(session: ort.InferenceSession) -> List[str]:
    return [x.name for x in session.get_inputs()]


# -----------------------------
# Model stats
# -----------------------------
def collect_model_stats(model_path: str) -> Dict:
    model = onnx.load(model_path)
    op_counts = Counter(node.op_type for node in model.graph.node)

    return {
        "model_path": model_path,
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "model_size_bytes": os.path.getsize(model_path),
        "op_counts": dict(sorted(op_counts.items())),
        "num_quantizelinear": op_counts.get("QuantizeLinear", 0),
        "num_dequantizelinear": op_counts.get("DequantizeLinear", 0),
        "num_qlinearconv": op_counts.get("QLinearConv", 0),
        "num_qlinearmatmul": op_counts.get("QLinearMatMul", 0),
        "num_conv": op_counts.get("Conv", 0),
    }


# -----------------------------
# Loader helpers
# -----------------------------
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


def _debug_print_batch(batch, prefix="[DEBUG]"):
    print(f"{prefix} batch type: {type(batch)}")

    if torch.is_tensor(batch):
        print(f"{prefix} tensor shape={tuple(batch.shape)} dtype={batch.dtype}")
        return

    if isinstance(batch, np.ndarray):
        print(f"{prefix} ndarray shape={batch.shape} dtype={batch.dtype}")
        return

    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"{prefix} dict[{k}] -> tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"{prefix} dict[{k}] -> ndarray shape={v.shape} dtype={v.dtype}")
            else:
                print(f"{prefix} dict[{k}] -> type={type(v)}")
        return

    if isinstance(batch, (tuple, list)):
        for i, v in enumerate(batch):
            if torch.is_tensor(v):
                print(f"{prefix} batch[{i}] -> tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"{prefix} batch[{i}] -> ndarray shape={v.shape} dtype={v.dtype}")
            else:
                print(f"{prefix} batch[{i}] -> type={type(v)}")


# -----------------------------
# Quantization helpers
# -----------------------------
def quant_type_from_name(name: str) -> QuantType:
    if name == "QInt8":
        return QuantType.QInt8
    if name == "QUInt8":
        return QuantType.QUInt8
    raise ValueError(f"Unknown quant type: {name}")


def calibration_method_from_name(name: str) -> CalibrationMethod:
    if name == "MinMax":
        return CalibrationMethod.MinMax
    if name == "Entropy":
        return CalibrationMethod.Entropy
    if name == "Percentile":
        return CalibrationMethod.Percentile
    raise ValueError(f"Unknown calibration method: {name}")


class LoaderCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        loader,
        input_names: List[str],
        max_samples: int = 50,
        debug_batch: bool = False,
    ):
        if len(input_names) != 1:
            raise ValueError(f"Expected exactly 1 input, got {input_names}")

        self.loader = loader
        self.input_name = input_names[0]
        self.max_samples = max_samples
        self.debug_batch = debug_batch
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

        if self.debug_batch and self._count == 0:
            _debug_print_batch(batch, prefix="[DEBUG][CALIB]")

        image = _find_first_numeric_tensor(batch)
        if image is None:
            raise ValueError(f"Could not find numeric tensor in batch type: {type(batch)}")

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


def export_qoperator_static(
    input_model_path: str,
    output_model_path: str,
    cityscapes_root: str,
    split: str,
    image_width: int,
    image_height: int,
    batch_size: int,
    num_workers: int,
    provider: str,
    calib_samples: int,
    activation_type: str,
    weight_type: str,
    calibration_method: str,
    debug_batch: bool,
):
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    sess = make_session_from_model_path(
        input_model_path,
        provider=provider,
    )
    input_names = list_input_names(sess)

    loader = build_eval_loader(
        cityscapes_root=cityscapes_root,
        split=split,
        image_width=image_width,
        image_height=image_height,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_reader = LoaderCalibrationDataReader(
        loader=loader,
        input_names=input_names,
        max_samples=calib_samples,
        debug_batch=debug_batch,
    )

    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=quant_type_from_name(activation_type),
        weight_type=quant_type_from_name(weight_type),
        calibrate_method=calibration_method_from_name(calibration_method),
    )

    if not os.path.exists(output_model_path):
        raise RuntimeError(f"QOperator export failed: {output_model_path}")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--qdq_model", type=str, required=True)
    parser.add_argument(
        "--fp32_model",
        type=str,
        default=None,
        help="FP32 ONNX model path. Required for actual QOperator export.",
    )

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"])

    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    parser.add_argument("--debug_batch", action="store_true")
    parser.add_argument("--output_dir", type=str, default="onnx_qop_convert")
    parser.add_argument("--report_json", type=str, default="qop_convert_report.json")

    parser.add_argument("--calib_samples", type=int, default=50)
    parser.add_argument(
        "--qoperator_activation_type",
        type=str,
        default="QInt8",
        choices=["QInt8", "QUInt8"],
    )
    parser.add_argument(
        "--qoperator_weight_type",
        type=str,
        default="QInt8",
        choices=["QInt8", "QUInt8"],
    )
    parser.add_argument(
        "--qoperator_calibration_method",
        type=str,
        default="MinMax",
        choices=["MinMax", "Entropy", "Percentile"],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_model_path = os.path.join(args.output_dir, "model_qoperator.onnx")
    report_path = os.path.join(args.output_dir, args.report_json)

    input_qdq_stats = collect_model_stats(args.qdq_model)

    result = {
        "input_qdq_model": args.qdq_model,
        "input_qdq_model_stats": input_qdq_stats,
        "fp32_model": args.fp32_model,
        "output_qoperator_model": None,
        "output_qoperator_stats": None,
        "status": "failed",
        "error": None,
        "config": {
            "split": args.split,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "provider": args.provider,
            "calib_samples": args.calib_samples,
            "qoperator_activation_type": args.qoperator_activation_type,
            "qoperator_weight_type": args.qoperator_weight_type,
            "qoperator_calibration_method": args.qoperator_calibration_method,
        },
    }

    try:
        if args.fp32_model is None:
            raise ValueError(
                "Direct QDQ -> QOperator conversion is not implemented here. "
                "Pass --fp32_model and this tool will generate a real QOperator model from FP32."
            )

        print("[INFO] Exporting QOperator model from FP32 ONNX...")
        export_qoperator_static(
            input_model_path=args.fp32_model,
            output_model_path=output_model_path,
            cityscapes_root=args.cityscapes_root,
            split=args.split,
            image_width=args.image_width,
            image_height=args.image_height,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            provider=args.provider,
            calib_samples=args.calib_samples,
            activation_type=args.qoperator_activation_type,
            weight_type=args.qoperator_weight_type,
            calibration_method=args.qoperator_calibration_method,
            debug_batch=args.debug_batch,
        )

        output_stats = collect_model_stats(output_model_path)

        result["output_qoperator_model"] = output_model_path
        result["output_qoperator_stats"] = output_stats
        result["status"] = "ok"

        print("[INFO] Input QDQ stats:")
        print(json.dumps(to_jsonable(input_qdq_stats), indent=2))

        print("[INFO] Output QOperator stats:")
        print(json.dumps(to_jsonable(output_stats), indent=2))

        if output_stats["num_qlinearconv"] == 0 and output_stats["num_qlinearmatmul"] == 0:
            print("[WARN] Output model has no QLinearConv/QLinearMatMul. Check calibration path and source model.")
        else:
            print("[INFO] QOperator conversion looks valid.")

    except Exception as e:
        result["error"] = str(e)
        print(f"[ERROR] {e}")

    with open(report_path, "w") as f:
        json.dump(to_jsonable(result), f, indent=2)

    print(f"[INFO] Wrote report to: {report_path}")


if __name__ == "__main__":
    main()