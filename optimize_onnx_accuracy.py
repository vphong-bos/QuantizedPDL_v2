#!/usr/bin/env python3
import argparse
import copy
import json
import os
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, shape_inference

from evaluation.eval_dataset import build_eval_loader

EPS = 1e-12


def make_session_from_model_path(
    model_path: str,
    provider: str = "CPUExecutionProvider",
    enable_all_optimizations: bool = False,
):
    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if enable_all_optimizations
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

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


def make_session_from_onnx_model(
    model: onnx.ModelProto,
    provider: str = "CPUExecutionProvider",
    enable_all_optimizations: bool = False,
):
    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if enable_all_optimizations
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    model_bytes = model.SerializeToString()
    return ort.InferenceSession(
        model_bytes,
        sess_options=so,
        providers=providers,
    )


def list_input_names(session: ort.InferenceSession) -> List[str]:
    return [x.name for x in session.get_inputs()]


def list_output_names(session: ort.InferenceSession) -> List[str]:
    return [x.name for x in session.get_outputs()]


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
        preferred_keys = [
            "image", "images", "input", "inputs", "pixel_values", "img"
        ]
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
        return


def build_sample_from_loader(
    cityscapes_root: str,
    split: str,
    image_width: int,
    image_height: int,
    batch_size: int,
    num_workers: int,
    input_names: List[str],
    debug_batch: bool = False,
):
    loader = build_eval_loader(
        cityscapes_root=cityscapes_root,
        split=split,
        image_width=image_width,
        image_height=image_height,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    batch = next(iter(loader))

    if debug_batch:
        _debug_print_batch(batch)

    image = _find_first_numeric_tensor(batch)
    if image is None:
        raise ValueError(
            f"Could not find numeric tensor/array in loader batch. "
            f"Batch type: {type(batch)}"
        )

    image = _to_numpy(image)

    if image.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(
            f"Selected loader field is not numeric. dtype={image.dtype}, type={type(image)}"
        )

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # ORT model usually expects NCHW. If loader returns CHW for a single image,
    # add batch dimension -> 1CHW.
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    if len(input_names) != 1:
        raise ValueError(
            f"Expected 1 ONNX input, but got {len(input_names)} inputs: {input_names}"
        )

    print(
        f"[INFO] Using input '{input_names[0]}' "
        f"with shape={image.shape}, dtype={image.dtype}"
    )
    return {input_names[0]: image}


def flatten_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32).reshape(-1)
    return arr.astype(np.float32).reshape(-1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = flatten_float(a)
    b = flatten_float(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = flatten_float(a)
    b = flatten_float(b)
    if a.size == 0 or b.size == 0:
        return 1.0
    if np.std(a) < EPS and np.std(b) < EPS:
        return 1.0 if np.allclose(a, b, atol=1e-6, rtol=1e-6) else 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def compare_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape != b.shape:
        return {
            "shape_match": 0.0,
            "cosine": -1.0,
            "pcc": -1.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
        }

    diff = flatten_float(a) - flatten_float(b)
    return {
        "shape_match": 1.0,
        "cosine": cosine_similarity(a, b),
        "pcc": pcc(a, b),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
    }


def run_model(session: ort.InferenceSession, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    output_names = list_output_names(session)
    values = session.run(output_names, sample)
    return dict(zip(output_names, values))


def compare_model_outputs(
    original_model_path: str,
    optimized_model_path: str,
    sample: Dict[str, np.ndarray],
    provider: str = "CPUExecutionProvider",
):
    orig_sess = make_session_from_model_path(
        original_model_path,
        provider=provider,
        enable_all_optimizations=False,
    )
    opt_sess = make_session_from_model_path(
        optimized_model_path,
        provider=provider,
        enable_all_optimizations=False,
    )

    orig_out = run_model(orig_sess, sample)
    opt_out = run_model(opt_sess, sample)

    report = {}
    for name in orig_out:
        if name in opt_out:
            report[name] = compare_arrays(orig_out[name], opt_out[name])

    return report


def collect_all_value_names(model: onnx.ModelProto) -> List[str]:
    names = []
    existing_outputs = set(o.name for o in model.graph.output)
    initializers = set(i.name for i in model.graph.initializer)

    for vi in model.graph.value_info:
        if vi.name not in existing_outputs and vi.name not in initializers:
            names.append(vi.name)

    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing_outputs and out not in initializers:
                names.append(out)

    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def collect_real_tensor_names(model: onnx.ModelProto) -> List[str]:
    """
    Collect only tensor names that are guaranteed to exist at runtime:
    - graph inputs
    - node outputs
    Exclude initializers and existing graph outputs.
    """
    existing_outputs = set(o.name for o in model.graph.output)
    initializer_names = set(i.name for i in model.graph.initializer)
    graph_input_names = set(i.name for i in model.graph.input)

    names = []

    # Real graph inputs
    for name in model.graph.input:
        if name.name not in existing_outputs and name.name not in initializer_names:
            names.append(name.name)

    # Real node outputs
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing_outputs and out not in initializer_names:
                names.append(out)

    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _is_float_tensor_type(value_info: onnx.ValueInfoProto) -> bool:
    if not value_info.type.HasField("tensor_type"):
        return False
    elem_type = value_info.type.tensor_type.elem_type
    return elem_type in (
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.DOUBLE,
    )


def add_all_intermediate_float_outputs_to_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] shape inference failed: {e}")

    existing_output_names = {o.name for o in model.graph.output}
    initializer_names = {i.name for i in model.graph.initializer}

    known_value_infos = {}
    for vi in model.graph.value_info:
        known_value_infos[vi.name] = vi
    for vi in model.graph.input:
        known_value_infos[vi.name] = vi
    for vi in model.graph.output:
        known_value_infos[vi.name] = vi

    added = 0
    skipped = 0

    for node in model.graph.node:
        for out_name in node.output:
            if not out_name:
                continue
            if out_name in existing_output_names:
                continue
            if out_name in initializer_names:
                continue

            vi = known_value_infos.get(out_name)
            if vi is None or not _is_float_tensor_type(vi):
                skipped += 1
                continue

            model.graph.output.append(copy.deepcopy(vi))
            existing_output_names.add(out_name)
            added += 1

    print(f"[INFO] Added {added} float intermediate outputs, skipped {skipped} non-float/unknown outputs")
    return model

def run_all_outputs(
    model_path: str,
    sample: Dict[str, np.ndarray],
    provider: str = "CPUExecutionProvider",
):
    model = onnx.load(model_path)
    model_with_all_outputs = add_all_intermediate_float_outputs_to_model(model)

    try:
        sess = make_session_from_onnx_model(
            model_with_all_outputs,
            provider=provider,
            enable_all_optimizations=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create debug session for {model_path}: {e}")

    output_names = list_output_names(sess)
    values = sess.run(output_names, sample)
    return dict(zip(output_names, values))

def compare_all_tensors(
    original_model_path: str,
    optimized_model_path: str,
    sample: Dict[str, np.ndarray],
    provider: str = "CPUExecutionProvider",
    pcc_threshold: float = 0.99,
    cosine_threshold: float = 0.99,
):
    orig_vals = run_all_outputs(original_model_path, sample, provider)
    opt_vals = run_all_outputs(optimized_model_path, sample, provider)

    common_names = [n for n in orig_vals if n in opt_vals]
    rows = []

    for name in common_names:
        try:
            m = compare_arrays(orig_vals[name], opt_vals[name])
            rows.append({"name": name, **m})
        except Exception:
            pass

    rows.sort(key=lambda x: (x["pcc"], x["cosine"], -x["max_abs"]))

    first_bad = None
    for row in rows:
        if row["shape_match"] < 1.0:
            first_bad = row
            break
        if row["pcc"] < pcc_threshold or row["cosine"] < cosine_threshold:
            first_bad = row
            break

    return {
        "first_bad_tensor": first_bad,
        "worst_50_tensors": rows[:50],
        "total_compared_tensors": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--original_model", type=str, required=True)
    parser.add_argument("--optimized_model", type=str, required=True)

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
    parser.add_argument("--output_json", type=str, default="onnx_compare_report.json")
    parser.add_argument("--pcc_threshold", type=float, default=0.99)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)
    parser.add_argument("--debug_batch", action="store_true")

    args = parser.parse_args()

    ref_sess = make_session_from_model_path(
        args.original_model,
        provider=args.provider,
        enable_all_optimizations=False,
    )

    sample = build_sample_from_loader(
        cityscapes_root=args.cityscapes_root,
        split=args.split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_names=list_input_names(ref_sess),
        debug_batch=args.debug_batch,
    )

    final_output_report = compare_model_outputs(
        original_model_path=args.original_model,
        optimized_model_path=args.optimized_model,
        sample=sample,
        provider=args.provider,
    )

    tensor_report = compare_all_tensors(
        original_model_path=args.original_model,
        optimized_model_path=args.optimized_model,
        sample=sample,
        provider=args.provider,
        pcc_threshold=args.pcc_threshold,
        cosine_threshold=args.cosine_threshold,
    )

    report = {
        "final_outputs": final_output_report,
        "first_bad_tensor": tensor_report["first_bad_tensor"],
        "worst_50_tensors": tensor_report["worst_50_tensors"],
        "total_compared_tensors": tensor_report["total_compared_tensors"],
    }

    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Wrote report to: {args.output_json}")
    print("[INFO] Final output summary:")
    for name, metrics in final_output_report.items():
        print(
            f"  {name}: "
            f"pcc={metrics['pcc']:.6f}, "
            f"cosine={metrics['cosine']:.6f}, "
            f"max_abs={metrics['max_abs']:.6e}, "
            f"mean_abs={metrics['mean_abs']:.6e}"
        )

    if report["first_bad_tensor"] is None:
        print("[INFO] No clearly bad tensor found with current thresholds.")
    else:
        bad = report["first_bad_tensor"]
        print("[INFO] First bad tensor:")
        print(
            f"  name={bad['name']}, "
            f"pcc={bad['pcc']:.6f}, "
            f"cosine={bad['cosine']:.6f}, "
            f"max_abs={bad['max_abs']:.6e}, "
            f"mean_abs={bad['mean_abs']:.6e}"
        )


if __name__ == "__main__":
    main()