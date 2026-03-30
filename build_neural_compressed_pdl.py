#!/usr/bin/env python3
import argparse
import os
import random
from typing import Any, Dict, Iterable, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from aimet_common.utils import CallbackFunc
from aimet_torch.quant_analyzer import QuantAnalyzer
from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams
from aimet_torch.cross_layer_equalization import equalize_model

from onnxruntime.quantization import quant_pre_process
from onnx_neural_compressor import data_reader
from onnx_neural_compressor.quantization import config, quantize

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

from collections import defaultdict


def should_track_module(name: str, module: torch.nn.Module) -> bool:
    track_types = (
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.SiLU,
        torch.nn.Hardswish,
    )
    return isinstance(module, track_types)


def tensor_stat_dict(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {}

    x_flat = x.reshape(-1)
    x_min = float(x_flat.min().item())
    x_max = float(x_flat.max().item())
    absmax = max(abs(x_min), abs(x_max))
    neg_ratio = float((x_flat < 0).float().mean().item())
    zero_ratio = float((x_flat == 0).float().mean().item())

    sym_scale = absmax / 127.0 if absmax > 0 else 1.0
    asym_range = x_max - x_min
    asym_scale = asym_range / 255.0 if asym_range > 0 else 1.0
    used_fraction_in_sym = (x_max - x_min) / (2.0 * absmax) if absmax > 0 else 1.0
    wasted_fraction_in_sym = 1.0 - used_fraction_in_sym

    return {
        "min": x_min,
        "max": x_max,
        "absmax": absmax,
        "neg_ratio": neg_ratio,
        "zero_ratio": zero_ratio,
        "sym_scale": sym_scale,
        "asym_scale": asym_scale,
        "sym_over_asym_scale": sym_scale / asym_scale if asym_scale > 0 else float("inf"),
        "used_fraction_in_sym": used_fraction_in_sym,
        "wasted_fraction_in_sym": wasted_fraction_in_sym,
    }


def collect_activation_stats(
    model: torch.nn.Module,
    data_loader,
    max_batches: int = 8,
) -> Dict[str, Dict[str, float]]:
    device = next(model.parameters()).device
    stats_accum = defaultdict(list)
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            tensor = None

            if torch.is_tensor(output):
                tensor = output
            elif isinstance(output, (tuple, list)):
                for v in output:
                    if torch.is_tensor(v):
                        tensor = v
                        break
            elif isinstance(output, dict):
                for v in output.values():
                    if torch.is_tensor(v):
                        tensor = v
                        break

            if tensor is None:
                return

            s = tensor_stat_dict(tensor)
            if s:
                stats_accum[name].append(s)

        return hook

    for name, module in model.named_modules():
        if should_track_module(name, module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()

    with torch.no_grad():
        batch_count = 0
        for batch in data_loader:
            image = find_first_numeric_tensor(batch)
            if image is None:
                raise ValueError(f"Could not find numeric tensor in batch type: {type(batch)}")

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            image = image.to(device)
            if image.ndim == 3:
                image = image.unsqueeze(0)

            _ = model(image)
            batch_count += 1
            if batch_count >= max_batches:
                break

    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    summary = {}
    for name, items in stats_accum.items():
        if not items:
            continue

        merged = {}
        for key in items[0].keys():
            merged[key] = float(np.mean([x[key] for x in items]))
        summary[name] = merged

    return summary


def print_top_activation_problems(stats: Dict[str, Dict[str, float]], top_k: int = 25) -> None:
    rows = []
    for name, s in stats.items():
        rows.append((
            name,
            s["wasted_fraction_in_sym"],
            s["sym_over_asym_scale"],
            s["neg_ratio"],
            s["zero_ratio"],
            s["min"],
            s["max"],
        ))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)

    print("\n[INFO] Top activation tensors likely to be hurt by symmetric quantization:")
    print(
        f"{'name':60s} {'wasted_sym':>12s} {'sym/asym':>10s} "
        f"{'neg%':>8s} {'zero%':>8s} {'min':>12s} {'max':>12s}"
    )
    for row in rows[:top_k]:
        name, wasted_sym, sym_over_asym, neg_ratio, zero_ratio, x_min, x_max = row
        print(
            f"{name[:60]:60s} "
            f"{wasted_sym:12.4f} {sym_over_asym:10.4f} "
            f"{100.0 * neg_ratio:8.2f} {100.0 * zero_ratio:8.2f} "
            f"{x_min:12.5f} {x_max:12.5f}"
        )


class CleTraceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        if torch.is_tensor(out):
            return out

        if isinstance(out, dict):
            tensor_items = {k: v for k, v in out.items() if torch.is_tensor(v)}
            if len(tensor_items) == 0:
                raise RuntimeError("Model output dict contains no tensor values for CLE tracing.")
            return tensor_items

        if isinstance(out, (tuple, list)):
            tensor_items = tuple(v for v in out if torch.is_tensor(v))
            if len(tensor_items) == 0:
                raise RuntimeError("Model output tuple/list contains no tensor values for CLE tracing.")
            if len(tensor_items) == 1:
                return tensor_items[0]
            return tensor_items

        for attr in ["logits", "pred", "prediction", "out", "output"]:
            if hasattr(out, attr):
                value = getattr(out, attr)
                if torch.is_tensor(value):
                    return value

        raise RuntimeError(
            f"Unsupported model output type for CLE tracing: {type(out)}. "
            "Return a tensor, tuple/list of tensors, or dict of tensors."
        )


class TraceOnlyTensorOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        if torch.is_tensor(out):
            return out

        if isinstance(out, dict):
            for key in ["out", "logits", "pred", "prediction", "output"]:
                if key in out and torch.is_tensor(out[key]):
                    return out[key]

            tensor_dict = {k: v for k, v in out.items() if torch.is_tensor(v)}
            if len(tensor_dict) == 0:
                raise RuntimeError("Model output dict contains no tensor values.")
            if len(tensor_dict) == 1:
                return next(iter(tensor_dict.values()))
            return tensor_dict

        if isinstance(out, (tuple, list)):
            tensor_items = tuple(v for v in out if torch.is_tensor(v))
            if len(tensor_items) == 0:
                raise RuntimeError("Model output tuple/list contains no tensor values.")
            if len(tensor_items) == 1:
                return tensor_items[0]
            return tensor_items

        for attr in ["logits", "pred", "prediction", "out", "output"]:
            if hasattr(out, attr):
                value = getattr(out, attr)
                if torch.is_tensor(value):
                    return value

        raise RuntimeError(f"Unsupported output type for tracing: {type(out)}")


def load_excluded_node_names(path: str) -> list[str]:
    if not path:
        return []

    if not os.path.isfile(path):
        raise FileNotFoundError(f"exclude_nodes_file not found: {path}")

    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def get_last_conv_node_names(model_path: str, count: int) -> list[str]:
    if count <= 0:
        return []

    model = onnx.load(model_path)
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    selected = conv_nodes[-count:] if count < len(conv_nodes) else conv_nodes

    names = []
    unnamed_idx = 0
    for node in selected:
        if node.name:
            names.append(node.name)
        else:
            fallback_name = f"__unnamed_conv_{unnamed_idx}"
            unnamed_idx += 1
            print(
                f"[WARN] Found Conv node without name. "
                f"Cannot exclude it reliably by name: outputs={list(node.output)}"
            )
    return names


def summarize_onnx_nodes(model_path: str, op_types: Optional[set[str]] = None, limit: int = 200) -> None:
    model = onnx.load(model_path)
    print(f"[INFO] ONNX node summary for: {model_path}")
    shown = 0
    for idx, node in enumerate(model.graph.node):
        if op_types is not None and node.op_type not in op_types:
            continue
        print(
            f"[{idx:04d}] op={node.op_type:<12} "
            f"name={node.name!r} "
            f"inputs={list(node.input)} "
            f"outputs={list(node.output)}"
        )
        shown += 1
        if shown >= limit:
            print(f"[INFO] Reached display limit={limit}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FP32 ONNX and quantized ONNX using onnx-neural-compressor while keeping the existing PDL flow."
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
    parser.add_argument(
        "--preprocessed_name",
        type=str,
        default="model_fp32_preprocessed.onnx",
        help="Intermediate ONNX after ONNX Runtime quantization preprocessing.",
    )
    parser.add_argument("--quant_name", type=str, default="model_int8.onnx")
    parser.add_argument(
        "--quant_format",
        type=str,
        default="qoperator",
        choices=["qoperator"],
        help="Kept fixed to QOperator as requested.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="Kept for CLI compatibility. INC static config here uses MinMax-style PTQ flow.",
    )

    parser.add_argument("--enable_custom_conv_bn_fold", action="store_true")
    parser.add_argument(
        "--run_cle",
        action="store_true",
        help="Run AIMET cross-layer equalization before ONNX export.",
    )

    parser.add_argument(
        "--skip_onnx_preprocess",
        action="store_true",
        help="Skip ONNX Runtime quantization pre-processing step.",
    )
    parser.add_argument(
        "--skip_preprocess_optimization",
        action="store_true",
        help="Disable ONNX Runtime graph optimization during quantization pre-processing.",
    )
    parser.add_argument(
        "--skip_symbolic_shape_inference",
        action="store_true",
        help="Disable symbolic shape inference during quantization pre-processing.",
    )
    parser.add_argument(
        "--disable_auto_merge",
        action="store_true",
        help="Disable auto_merge in ONNX Runtime quantization pre-processing.",
    )

    parser.add_argument(
        "--activation_type",
        type=str,
        default="qint8",
        choices=["qint8"],
        help="Kept fixed to symmetric int8 activations as requested.",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="qint8",
        choices=["qint8"],
        help="Kept fixed to symmetric int8 weights as requested.",
    )
    parser.add_argument("--per_channel", action="store_true", default=True)
    parser.add_argument("--disable_per_channel", action="store_true")
    parser.add_argument("--activation_symmetric", action="store_true", default=True)
    parser.add_argument("--weight_symmetric", action="store_true", default=True)
    parser.add_argument("--disable_weight_symmetric", action="store_true")
    parser.add_argument(
        "--force_qoperator",
        action="store_true",
        default=True,
        help="Kept for CLI compatibility. Export stays QOperator.",
    )

    parser.add_argument(
        "--run_bn_fold",
        action="store_true",
        help="Run AIMET generic batch norm folding before CLE/export.",
    )

    parser.add_argument(
        "--run_seq_mse",
        action="store_true",
        help="Run AIMET SeqMSE before ONNX export.",
    )

    parser.add_argument(
        "--seq_mse_num_batches",
        type=int,
        default=4,
        help="Number of calibration batches to use for SeqMSE.",
    )

    parser.add_argument(
        "--execution_provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="Execution provider for checking the model input / export path.",
    )

    parser.add_argument(
        "--exclude_nodes",
        type=str,
        nargs="*",
        default=[],
        help="Exact ONNX node names to exclude from quantization.",
    )

    parser.add_argument(
        "--exclude_nodes_file",
        type=str,
        default="",
        help="Optional text file containing ONNX node names to exclude, one per line.",
    )

    parser.add_argument(
        "--auto_exclude_last_conv_count",
        type=int,
        default=0,
        help="Automatically exclude the last N Conv nodes in the ONNX graph.",
    )

    parser.add_argument(
        "--run_quant_analyzer",
        action="store_true",
        help="Run AIMET QuantAnalyzer before ONNX export.",
    )

    parser.add_argument(
        "--quant_analyzer_dir",
        type=str,
        default="quant_analyzer_results",
        help="Directory to save QuantAnalyzer outputs.",
    )

    parser.add_argument(
        "--analyzer_num_batches",
        type=int,
        default=None,
        help="Number of calibration batches for analyzer forward pass; default uses all.",
    )

    parser.add_argument(
        "--cityscapes_root",
        type=str,
        default=None,
        help="Cityscapes root, required when --run_quant_analyzer is set or evaluation is used externally.",
    )

    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split for QuantAnalyzer.",
    )

    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=-1,
        help="Max eval samples for QuantAnalyzer, -1 means full split.",
    )

    parser.add_argument(
        "--inc_quant_format",
        type=str,
        default="QOperator",
        choices=["QOperator"],
        help="INC export format. Kept fixed to QOperator.",
    )
    parser.add_argument(
        "--op_types_to_quantize",
        type=str,
        nargs="*",
        default=["Conv", "MatMul", "Add", "Mul"],
        help="Op types passed into INC static quant config.",
    )

    return parser.parse_args()


class LoaderCalibrationDataReader(data_reader.CalibrationDataReader):
    def __init__(self, model_path: str, loader: Iterable[Any], max_samples: int = -1):
        self.loader = loader
        self.max_samples = max_samples
        self._iter = None
        self._count = 0

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_names = [x.name for x in sess.get_inputs()]
        if len(input_names) != 1:
            raise ValueError(f"Expected exactly 1 ONNX input, got {input_names}")
        self.input_name = input_names[0]
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


class INCExcludeOutputQuantConfig(config.StaticQuantConfig):
    def __init__(
        self,
        calibration_data_reader,
        nodes_to_exclude: Optional[list[str]] = None,
        op_types_to_quantize: Optional[list[str]] = None,
        per_channel: bool = True,
    ):
        super().__init__(
            calibration_data_reader=calibration_data_reader,
            quant_format="QOperator",
            activation_type="QInt8",
            weight_type="QInt8",
            per_channel=per_channel,
            op_types_to_quantize=op_types_to_quantize or ["Conv", "MatMul", "Add", "Mul"],
            nodes_to_exclude=nodes_to_exclude or [],
        )



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



def maybe_fold_all_batch_norms(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    enabled: bool,
) -> torch.nn.Module:
    if not enabled:
        return model

    print("[INFO] Running AIMET fold_all_batch_norms...")
    was_training = model.training
    model.eval()

    try:
        wrapped_model = TraceOnlyTensorOutputWrapper(model)
        with torch.no_grad():
            folded_pairs = fold_all_batch_norms(
                wrapped_model,
                input_shapes=tuple(dummy_input.shape),
            )
        num_folded = len(folded_pairs) if folded_pairs is not None else 0
        print(f"[INFO] AIMET BN fold completed. Folded pairs: {num_folded}")
    except Exception as e:
        print(f"[WARN] AIMET BN fold failed and will be skipped: {e}")

    if was_training:
        model.train()
    return model



def maybe_run_cle(model: torch.nn.Module, dummy_input: torch.Tensor, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model

    print("[INFO] Running AIMET Cross-Layer Equalization...")
    was_training = model.training
    model.eval()

    wrapped_model = CleTraceWrapper(model)
    with torch.no_grad():
        equalize_model(wrapped_model, dummy_input=dummy_input)

    if was_training:
        model.train()

    print("[INFO] Cross-Layer Equalization completed.")
    return model



def maybe_run_seq_mse(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    calib_loader,
    enabled: bool,
    num_batches: int,
) -> torch.nn.Module:
    if not enabled:
        return model

    print("[INFO] Running AIMET SeqMSE...")
    was_training = model.training
    model.eval()

    try:
        params = SeqMseParams(num_batches=num_batches)
        with torch.no_grad():
            apply_seq_mse(
                model=model,
                dummy_input=dummy_input,
                data_loader=calib_loader,
                params=params,
                forward_fn=run_seq_mse_forward_pass,
            )
        print("[INFO] SeqMSE completed.")
    except TypeError:
        try:
            params = SeqMseParams(num_batches=num_batches)
            with torch.no_grad():
                apply_seq_mse(model, dummy_input, calib_loader, params)
            print("[INFO] SeqMSE completed.")
        except Exception as e:
            print(f"[WARN] SeqMSE failed and will be skipped: {e}")
    except Exception as e:
        print(f"[WARN] SeqMSE failed and will be skipped: {e}")

    if was_training:
        model.train()
    return model



def maybe_run_quant_analyzer(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    calib_loader,
    model_category_const,
    device: str,
    enabled: bool,
    quant_analyzer_dir: str,
    analyzer_num_batches: Optional[int],
    cityscapes_root: Optional[str],
    eval_split: str,
    eval_max_samples: int,
) -> None:
    if not enabled:
        return

    if cityscapes_root is None:
        raise ValueError("--cityscapes_root is required when --run_quant_analyzer is set")

    print("[INFO] Building evaluation loader for QuantAnalyzer...")
    eval_loader = build_eval_loader(
        cityscapes_root=cityscapes_root,
        split=eval_split,
        image_width=dummy_input.shape[-1],
        image_height=dummy_input.shape[-2],
        batch_size=1,
        num_workers=2,
    )

    print("[INFO] Running AIMET QuantAnalyzer...")
    os.makedirs(quant_analyzer_dir, exist_ok=True)

    forward_pass_callback = CallbackFunc(
        analyzer_forward_pass,
        func_callback_args=(calib_loader, device, analyzer_num_batches),
    )

    eval_callback = CallbackFunc(
        analyzer_eval_callback,
        func_callback_args=(
            eval_loader,
            model_category_const,
            device,
            eval_max_samples,
        ),
    )

    analyzer = QuantAnalyzer(
        model=model,
        dummy_input=dummy_input,
        forward_pass_callback=forward_pass_callback,
        eval_callback=eval_callback,
        modules_to_ignore=None,
    )

    analyzer.analyze(
        quant_scheme="tf_enhanced",
        default_param_bw=8,
        default_output_bw=8,
        config_file=None,
        results_dir=quant_analyzer_dir,
    )

    print(f"[INFO] QuantAnalyzer results saved to: {quant_analyzer_dir}")



def run_seq_mse_forward_pass(model: torch.nn.Module, data_loader, max_batches: int) -> None:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            image = find_first_numeric_tensor(batch)
            if image is None:
                raise ValueError(
                    f"Could not find numeric tensor in SeqMSE batch type: {type(batch)}"
                )

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            image = image.to(device)
            if image.ndim == 3:
                image = image.unsqueeze(0)

            _ = model(image)
            num_batches += 1
            if num_batches >= max_batches:
                break

    if was_training:
        model.train()



def analyzer_forward_pass(model, callback_args):
    calib_loader, device, max_batches = callback_args
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            image = find_first_numeric_tensor(batch)
            if image is None:
                raise ValueError(f"Could not find numeric tensor in analyzer batch type: {type(batch)}")

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            image = image.to(device)
            if image.ndim == 3:
                image = image.unsqueeze(0)

            _ = model(image)

    if was_training:
        model.train()



def analyzer_eval_callback(model, callback_args):
    eval_loader, model_category_const, device, max_samples = callback_args
    results = evaluate_model(
        model_obj=model,
        model_category_const=model_category_const,
        loader=eval_loader,
        device=device,
        max_samples=max_samples,
    )
    return float(results["mIoU"])



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



def run_onnx_preprocessing(
    input_onnx_path: str,
    output_onnx_path: str,
    skip_optimization: bool = False,
    skip_symbolic_shape: bool = False,
    auto_merge: bool = True,
) -> None:
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    print("[INFO] Running ONNX Runtime quantization pre-processing...")
    print(f"[INFO]   Input model : {input_onnx_path}")
    print(f"[INFO]   Output model: {output_onnx_path}")
    print(f"[INFO]   skip_optimization   : {skip_optimization}")
    print(f"[INFO]   skip_symbolic_shape : {skip_symbolic_shape}")
    print(f"[INFO]   auto_merge          : {auto_merge}")

    quant_pre_process(
        input_model_path=input_onnx_path,
        output_model_path=output_onnx_path,
        skip_optimization=skip_optimization,
        skip_symbolic_shape=skip_symbolic_shape,
        auto_merge=auto_merge,
    )

    print("[INFO] ONNX Runtime quantization pre-processing completed.")
    collect_onnx_op_counts(output_onnx_path)



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



def export_quantized_onnx_with_inc(
    fp32_onnx_path: str,
    output_path: str,
    calib_loader,
    per_channel: bool,
    nodes_to_exclude: Optional[list[str]] = None,
    op_types_to_quantize: Optional[list[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nodes_to_exclude = nodes_to_exclude or []

    if nodes_to_exclude:
        print("[INFO] Excluding ONNX nodes from quantization:")
        for name in nodes_to_exclude:
            print(f"  - {name}")

    data_reader_obj = LoaderCalibrationDataReader(
        model_path=fp32_onnx_path,
        loader=calib_loader,
        max_samples=-1,
    )

    cfg = INCExcludeOutputQuantConfig(
        calibration_data_reader=data_reader_obj,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=per_channel,
    )

    print("[INFO] INC quantization config summary:")
    print("[INFO]   quant_format      : QOperator")
    print("[INFO]   activation_type   : QInt8")
    print("[INFO]   weight_type       : QInt8")
    print("[INFO]   activation_sym    : True")
    print("[INFO]   weight_sym        : True")
    print(f"[INFO]   per_channel       : {per_channel}")
    print(f"[INFO]   op_types_to_quant : {op_types_to_quantize}")

    qmodel = quantize(
        model_input=fp32_onnx_path,
        quant_config=cfg,
        calibration_data_reader=data_reader_obj,
    )
    onnx.save_model(qmodel, output_path)

    print(f"[INFO] Saved quantized ONNX to: {output_path}")

    quant_stats = collect_onnx_op_counts(output_path)
    qlinear_conv = quant_stats.get("QLinearConv", 0)
    qlinear_matmul = quant_stats.get("QLinearMatMul", 0)

    if qlinear_conv == 0 and qlinear_matmul == 0:
        raise RuntimeError(
            "Requested QOperator export, but the output model does not contain "
            "QLinearConv or QLinearMatMul nodes."
        )

    print(
        f"[INFO] Verified QOperator export: "
        f"QLinearConv={qlinear_conv}, QLinearMatMul={qlinear_matmul}"
    )



def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.num_calib < 1:
        raise ValueError("num_calib must be >= 1")
    if not args.activation_symmetric:
        raise ValueError("This INC path is pinned to symmetric activations. Keep --activation_symmetric enabled.")

    set_random_seed(args.seed)
    os.makedirs(args.export_path, exist_ok=True)

    per_channel = False if args.disable_per_channel else args.per_channel
    weight_symmetric = False if args.disable_weight_symmetric else args.weight_symmetric
    if not weight_symmetric:
        raise ValueError("This INC path is pinned to symmetric weights. Do not disable weight symmetry.")

    print("[INFO] Loading FP32 model...")
    model, model_category_const = build_model(
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
    model = maybe_fold_all_batch_norms(model, dummy_input, args.run_bn_fold)
    model = maybe_run_cle(model, dummy_input, args.run_cle)

    calib_loader = build_calibration_loader(args)

    print("[INFO] Collecting activation statistics on calibration data...")
    act_stats = collect_activation_stats(model, calib_loader, max_batches=8)
    print_top_activation_problems(act_stats, top_k=30)

    maybe_run_quant_analyzer(
        model=model,
        dummy_input=dummy_input,
        calib_loader=calib_loader,
        model_category_const=model_category_const,
        device=args.device,
        enabled=args.run_quant_analyzer,
        quant_analyzer_dir=os.path.join(args.export_path, args.quant_analyzer_dir),
        analyzer_num_batches=args.analyzer_num_batches,
        cityscapes_root=args.cityscapes_root,
        eval_split=args.eval_split,
        eval_max_samples=args.eval_max_samples,
    )

    model = maybe_run_seq_mse(
        model=model,
        dummy_input=dummy_input,
        calib_loader=calib_loader,
        enabled=args.run_seq_mse,
        num_batches=args.seq_mse_num_batches,
    )

    fp32_onnx_path = os.path.join(args.export_path, args.fp32_name)
    preprocessed_onnx_path = os.path.join(args.export_path, args.preprocessed_name)
    quant_onnx_path = os.path.join(args.export_path, args.quant_name)

    export_fp32_onnx(model, dummy_input, fp32_onnx_path)

    quant_input_path = fp32_onnx_path
    if args.skip_onnx_preprocess:
        print("[INFO] Skipping ONNX Runtime quantization pre-processing.")
    else:
        run_onnx_preprocessing(
            input_onnx_path=fp32_onnx_path,
            output_onnx_path=preprocessed_onnx_path,
            skip_optimization=args.skip_preprocess_optimization,
            skip_symbolic_shape=args.skip_symbolic_shape_inference,
            auto_merge=not args.disable_auto_merge,
        )
        quant_input_path = preprocessed_onnx_path

    calib_loader = build_calibration_loader(args)

    file_excluded_nodes = load_excluded_node_names(args.exclude_nodes_file)
    auto_excluded_nodes = get_last_conv_node_names(
        quant_input_path,
        args.auto_exclude_last_conv_count,
    )

    nodes_to_exclude = []
    nodes_to_exclude.extend(args.exclude_nodes)
    nodes_to_exclude.extend(file_excluded_nodes)
    nodes_to_exclude.extend(auto_excluded_nodes)

    seen = set()
    nodes_to_exclude = [n for n in nodes_to_exclude if not (n in seen or seen.add(n))]

    if nodes_to_exclude:
        print("[INFO] Final selective quantization config:")
        print(f"[INFO]   excluded nodes: {nodes_to_exclude}")

    export_quantized_onnx_with_inc(
        fp32_onnx_path=quant_input_path,
        output_path=quant_onnx_path,
        calib_loader=calib_loader,
        per_channel=per_channel,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=args.op_types_to_quantize,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main(parse_args())
