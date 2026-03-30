#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnx_neural_compressor import data_reader, quantization
from onnx_neural_compressor.quantization import config, tuning

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model
from model.pdl import build_model
from quantization.calibration_dataset import create_calibration_loader, sample_calibration_images
from secret_incrediants.fold_conv_bn import fold_custom_conv_bn_inplace
from utils.image_loader import load_images

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.seq_mse import apply_seq_mse, SeqMseParams


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
            if len(tensor_dict) == 1:
                return next(iter(tensor_dict.values()))
            if len(tensor_dict) > 1:
                return tensor_dict

        if isinstance(out, (tuple, list)):
            tensor_items = tuple(v for v in out if torch.is_tensor(v))
            if len(tensor_items) == 1:
                return tensor_items[0]
            if len(tensor_items) > 1:
                return tensor_items

        for attr in ["logits", "pred", "prediction", "out", "output"]:
            if hasattr(out, attr):
                value = getattr(out, attr)
                if torch.is_tensor(value):
                    return value

        raise RuntimeError(f"Unsupported output type for tracing: {type(out)}")


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
            if tensor_items:
                return tensor_items
        if isinstance(out, (tuple, list)):
            tensor_items = tuple(v for v in out if torch.is_tensor(v))
            if len(tensor_items) == 1:
                return tensor_items[0]
            if tensor_items:
                return tensor_items
        for attr in ["logits", "pred", "prediction", "out", "output"]:
            if hasattr(out, attr):
                value = getattr(out, attr)
                if torch.is_tensor(value):
                    return value
        raise RuntimeError(f"Unsupported model output type for CLE tracing: {type(out)}")


class PDLDataReader(data_reader.CalibrationDataReader):
    def __init__(self, model_path: str, loader: Iterable[Any], max_samples: int = -1):
        self.loader = loader
        self.max_samples = max_samples
        self._iter = None
        self._count = 0
        model = onnx.load(model_path, load_external_data=False)
        self.input_names = [inp.name for inp in model.graph.input]
        if len(self.input_names) != 1:
            raise ValueError(f"Expected exactly 1 model input, got {self.input_names}")
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
            raise ValueError(f"Could not find numeric tensor in calibration batch type: {type(batch)}")

        image = to_numpy(image)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        batch_count = int(image.shape[0]) if image.ndim > 0 else 1
        self._count += batch_count
        return {self.input_names[0]: image}

    def rewind(self) -> None:
        self._iter = iter(self.loader)
        self._count = 0


class BenchmarkDataReader(PDLDataReader):
    pass


def maybe_fold_custom_conv_bn(model: torch.nn.Module, enabled: bool) -> None:
    if enabled:
        fold_custom_conv_bn_inplace(model)



def maybe_fold_all_batch_norms(model: torch.nn.Module, dummy_input: torch.Tensor, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    was_training = model.training
    model.eval()
    wrapped = TraceOnlyTensorOutputWrapper(model)
    try:
        with torch.no_grad():
            fold_all_batch_norms(wrapped, input_shapes=tuple(dummy_input.shape))
    finally:
        if was_training:
            model.train()
    return model



def maybe_run_cle(model: torch.nn.Module, dummy_input: torch.Tensor, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    was_training = model.training
    model.eval()
    wrapped = CleTraceWrapper(model)
    try:
        with torch.no_grad():
            equalize_model(wrapped, dummy_input=dummy_input)
    finally:
        if was_training:
            model.train()
    return model



def run_seq_mse_forward_pass(model: torch.nn.Module, data_loader, max_batches: int) -> None:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            image = find_first_numeric_tensor(batch)
            if image is None:
                raise ValueError(f"Could not find numeric tensor in SeqMSE batch type: {type(batch)}")
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            image = image.to(device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            _ = model(image)
            count += 1
            if count >= max_batches:
                break
    if was_training:
        model.train()



def maybe_run_seq_mse(model: torch.nn.Module, dummy_input: torch.Tensor, calib_loader, enabled: bool, num_batches: int):
    if not enabled:
        return model
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
    except TypeError:
        with torch.no_grad():
            apply_seq_mse(model, dummy_input, calib_loader, SeqMseParams(num_batches=num_batches))
    finally:
        if was_training:
            model.train()
    return model



def build_calibration_loader(args: argparse.Namespace):
    all_calib_images = load_images(args.calib_images, num_iters=-1, recursive=True)
    calib_images = sample_calibration_images(all_calib_images, args.num_calib, args.seed)
    if not calib_images:
        raise RuntimeError("No calibration images were selected.")
    return create_calibration_loader(
        calib_image_paths=calib_images,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )



def export_fp32_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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



def eval_func(onnx_model_path: str, args: argparse.Namespace) -> float:
    providers = [args.onnx_provider]
    if args.onnx_provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_model_path, providers=providers)
    eval_loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        split=args.eval_split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=1,
        num_workers=args.num_workers,
    )

    class ORTWrapper:
        def __init__(self, session: ort.InferenceSession):
            self.session = session
            self.input_name = self.session.get_inputs()[0].name

        def __call__(self, image_tensor: torch.Tensor):
            if torch.is_tensor(image_tensor):
                image_np = image_tensor.detach().cpu().numpy().astype(np.float32)
            else:
                image_np = np.asarray(image_tensor, dtype=np.float32)
            outputs = self.session.run(None, {self.input_name: image_np})
            return outputs

    wrapper = ORTWrapper(sess)
    results = evaluate_model(
        model_obj=wrapper,
        model_category_const=args.model_category,
        loader=eval_loader,
        device=args.device,
        max_samples=args.eval_max_samples,
    )
    return float(results["mIoU"])



def benchmark(onnx_model_path: str, args: argparse.Namespace) -> None:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra_op_num_threads
    providers = [args.onnx_provider]
    if args.onnx_provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)
    reader = BenchmarkDataReader(onnx_model_path, build_calibration_loader(args), max_samples=args.benchmark_iters)
    total_time = 0.0
    num_warmup = args.benchmark_warmup
    for idx, batch in enumerate(reader):
        tic = time.time()
        _ = session.run(None, batch)
        toc = time.time()
        if idx >= num_warmup:
            total_time += toc - tic
    effective = max(args.benchmark_iters - num_warmup, 1)
    print(f"Throughput: {effective / max(total_time, 1e-12):.4f} samples/s")



def quantize_with_inc(fp32_model_path: str, output_model_path: str, args: argparse.Namespace) -> None:
    calibration_reader = PDLDataReader(fp32_model_path, build_calibration_loader(args), max_samples=args.calibration_sampling_size)

    quant_format = (
        quantization.QuantFormat.QOperator
        if args.quant_format == "QOperator"
        else quantization.QuantFormat.QDQ
    )

    cfg_set = config.StaticQuantConfig.get_config_set_for_tuning(
        quant_format=quant_format,
        calibration_data_reader=calibration_reader,
    )

    tuning_config = tuning.TuningConfig(
        config_set=cfg_set,
        tolerable_loss=args.tolerable_loss,
        max_trials=args.max_trials,
    )

    def eval_callback(model_or_path):
        candidate_path = model_or_path if isinstance(model_or_path, str) else output_model_path + ".tmp.onnx"
        if not isinstance(model_or_path, str):
            onnx.save(model_or_path, candidate_path)
        score = eval_func(candidate_path, args)
        if not isinstance(model_or_path, str) and os.path.exists(candidate_path):
            os.remove(candidate_path)
        return score

    best_model = tuning.autotune(
        model_input=fp32_model_path,
        tune_config=tuning_config,
        eval_fn=eval_callback,
        calibration_data_reader=calibration_reader,
    )
    onnx.save(best_model, output_model_path)
    print(f"[INFO] Saved INC quantized model to: {output_model_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Panoptic-DeepLab PTQ static quantization using onnx-neural-compressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--calib_images", type=str, required=True)
    parser.add_argument("--dataset_location", type=str, default="", help="Kept for example-style CLI compatibility.")
    parser.add_argument("--label_path", type=str, default="", help="Kept for example-style CLI compatibility.")
    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB", choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onnx_provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_calib", type=int, default=200)
    parser.add_argument("--calibration_sampling_size", type=int, default=100)
    parser.add_argument("--export_path", type=str, default="quantized_export")
    parser.add_argument("--fp32_name", type=str, default="model_fp32.onnx")
    parser.add_argument("--output_model", type=str, default="model_int8_inc.onnx")
    parser.add_argument("--quant_format", type=str, default="QOperator", choices=["QDQ", "QOperator"])
    parser.add_argument("--tolerable_loss", type=float, default=0.01)
    parser.add_argument("--max_trials", type=int, default=50)
    parser.add_argument("--tune", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="accuracy", choices=["accuracy", "performance"])
    parser.add_argument("--intra_op_num_threads", type=int, default=4)
    parser.add_argument("--benchmark_iters", type=int, default=100)
    parser.add_argument("--benchmark_warmup", type=int, default=10)
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--enable_custom_conv_bn_fold", action="store_true")
    parser.add_argument("--run_bn_fold", action="store_true")
    parser.add_argument("--run_cle", action="store_true")
    parser.add_argument("--run_seq_mse", action="store_true")
    parser.add_argument("--seq_mse_num_batches", type=int, default=4)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.export_path, exist_ok=True)

    model, _ = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    model = model.to(args.device).eval()

    dummy_input = torch.randn(1, 3, args.image_height, args.image_width, device=args.device)
    maybe_fold_custom_conv_bn(model, args.enable_custom_conv_bn_fold)
    model = maybe_fold_all_batch_norms(model, dummy_input, args.run_bn_fold)
    model = maybe_run_cle(model, dummy_input, args.run_cle)

    calib_loader = build_calibration_loader(args)
    model = maybe_run_seq_mse(model, dummy_input, calib_loader, args.run_seq_mse, args.seq_mse_num_batches)

    fp32_model_path = os.path.join(args.export_path, args.fp32_name)
    output_model_path = os.path.join(args.export_path, args.output_model)
    export_fp32_onnx(model, dummy_input, fp32_model_path)

    if args.tune:
        quantize_with_inc(fp32_model_path, output_model_path, args)

    if args.benchmark:
        target_model = output_model_path if args.tune else fp32_model_path
        if args.mode == "performance":
            benchmark(target_model, args)
        else:
            score = eval_func(target_model, args)
            print(f"mIoU: {score:.5f}")


if __name__ == "__main__":
    main()
