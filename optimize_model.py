#!/usr/bin/env python3
import argparse
import copy
import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

EPS = 1e-12


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
def ort_opt_level_from_name(name: str):
    name = name.lower()
    if name == "disable":
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if name == "basic":
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if name == "extended":
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if name == "all":
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    raise ValueError(f"Unknown ORT optimization level: {name}")


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

    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=providers,
    )


def list_input_names(session: ort.InferenceSession) -> List[str]:
    return [x.name for x in session.get_inputs()]


def list_output_names(session: ort.InferenceSession) -> List[str]:
    return [x.name for x in session.get_outputs()]


def save_ort_optimized_model(
    input_model_path: str,
    output_model_path: str,
    provider: str,
    opt_level_name: str,
):
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort_opt_level_from_name(opt_level_name)
    so.optimized_model_filepath = output_model_path

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    _ = ort.InferenceSession(
        input_model_path,
        sess_options=so,
        providers=providers,
    )

    if not os.path.exists(output_model_path):
        raise RuntimeError(f"ORT did not write optimized model: {output_model_path}")


# -----------------------------
# Model helpers
# -----------------------------
def fix_model_input_shape(
    model: onnx.ModelProto,
    fixed_batch: int,
    image_height: int,
    image_width: int,
) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    if len(model.graph.input) != 1:
        print("[WARN] fix_input_shape supports only single-input models. Skipping.")
        return model

    input_value = model.graph.input[0]
    tensor_type = input_value.type.tensor_type
    dims = tensor_type.shape.dim

    if len(dims) != 4:
        print("[WARN] fix_input_shape expects 4D input [N,C,H,W]. Skipping.")
        return model

    dims[0].dim_value = fixed_batch
    dims[0].dim_param = ""

    if not dims[1].HasField("dim_value"):
        dims[1].dim_param = ""

    dims[2].dim_value = image_height
    dims[2].dim_param = ""
    dims[3].dim_value = image_width
    dims[3].dim_param = ""

    return model


def maybe_prepare_model_for_rewrite(
    input_model_path: str,
    fix_input_shape_flag: bool,
    fixed_batch: int,
    image_height: int,
    image_width: int,
) -> onnx.ModelProto:
    model = onnx.load(input_model_path)
    if fix_input_shape_flag:
        model = fix_model_input_shape(
            model=model,
            fixed_batch=fixed_batch,
            image_height=image_height,
            image_width=image_width,
        )
    return model


def save_model(model: onnx.ModelProto, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)


def get_onnxoptimizer_passes(preset: str) -> List[str]:
    if preset == "conservative":
        return [
            "extract_constant_to_initializer",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_cast",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
        ]
    if preset == "transpose":
        return [
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
        ]
    if preset == "extended":
        return [
            "extract_constant_to_initializer",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_cast",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
        ]
    if preset == "basic":
        return [
            "extract_constant_to_initializer",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_cast",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_transpose_into_gemm",
        ]
    raise ValueError(f"Unknown onnxoptimizer preset: {preset}")


def collect_model_stats(model_path: str) -> Dict:
    model = onnx.load(model_path)
    op_counts = Counter(node.op_type for node in model.graph.node)

    return {
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "model_size_bytes": os.path.getsize(model_path),
        "op_counts": dict(sorted(op_counts.items())),
        "num_quantizelinear": op_counts.get("QuantizeLinear", 0),
        "num_dequantizelinear": op_counts.get("DequantizeLinear", 0),
        "num_qlinearconv": op_counts.get("QLinearConv", 0),
        "num_qlinearmatmul": op_counts.get("QLinearMatMul", 0),
    }


def get_attribute(node: onnx.NodeProto, name: str):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def build_output_to_consumers(model: onnx.ModelProto):
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def find_tensor_producer(model: onnx.ModelProto, tensor_name: str):
    for node in model.graph.node:
        if tensor_name in node.output:
            return node
    return None


def find_tensor_consumers(model: onnx.ModelProto, tensor_name: str):
    consumers = []
    for node in model.graph.node:
        if tensor_name in node.input:
            consumers.append(node)
    return consumers


def node_to_dict(node: Optional[onnx.NodeProto]):
    if node is None:
        return None
    return {
        "name": node.name,
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
    }


def inspect_tensor_context(model_path: str, tensor_name: str):
    model = onnx.load(model_path)
    producer = find_tensor_producer(model, tensor_name)
    consumers = find_tensor_consumers(model, tensor_name)
    return {
        "tensor_name": tensor_name,
        "producer": node_to_dict(producer),
        "consumers": [node_to_dict(n) for n in consumers],
    }


# -----------------------------
# Manual optimization passes
# -----------------------------
def remove_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    used_names = set()
    for node in model.graph.node:
        for x in node.input:
            if x:
                used_names.add(x)
    for out in model.graph.output:
        used_names.add(out.name)

    kept = []
    removed = []
    for init in model.graph.initializer:
        if init.name in used_names:
            kept.append(init)
        else:
            removed.append(init.name)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    if removed:
        print(f"[INFO] Removed {len(removed)} unused initializers.")
    return model


def remove_identity_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    nodes = []
    removed = 0

    for node in model.graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            nodes.append(node)
            continue

        src = node.input[0]
        dst = node.output[0]

        for other in model.graph.node:
            for i, name in enumerate(other.input):
                if name == dst:
                    other.input[i] = src

        for out in model.graph.output:
            if out.name == dst:
                out.name = src

        removed += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    if removed:
        print(f"[INFO] Removed {removed} Identity nodes.")
    return model


def remove_nop_transposes(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    nodes = []
    removed = 0

    for node in model.graph.node:
        if node.op_type != "Transpose" or len(node.input) != 1 or len(node.output) != 1:
            nodes.append(node)
            continue

        attr = get_attribute(node, "perm")
        if attr is None:
            nodes.append(node)
            continue

        perm = list(attr.ints)
        if perm != list(range(len(perm))):
            nodes.append(node)
            continue

        src = node.input[0]
        dst = node.output[0]

        for other in model.graph.node:
            for i, name in enumerate(other.input):
                if name == dst:
                    other.input[i] = src

        for out in model.graph.output:
            if out.name == dst:
                out.name = src

        removed += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    if removed:
        print(f"[INFO] Removed {removed} no-op Transpose nodes.")
    return model

def manual_optimize_model(
    input_model_path: str,
    output_model_path: str,
    blocked_tensor_names: Optional[Set[str]] = None,
    blocked_node_names: Optional[Set[str]] = None,
    fix_input_shape_flag: bool = False,
    fixed_batch: int = 1,
    image_height: int = 512,
    image_width: int = 1024,
):
    model = maybe_prepare_model_for_rewrite(
        input_model_path=input_model_path,
        fix_input_shape_flag=fix_input_shape_flag,
        fixed_batch=fixed_batch,
        image_height=image_height,
        image_width=image_width,
    )

    model = remove_unused_initializers(model)
    model = remove_identity_nodes(model)
    model = remove_nop_transposes(model)
    model = remove_selected_qdq_pairs(
        model,
        blocked_tensor_names=blocked_tensor_names,
        blocked_node_names=blocked_node_names,
    )

    save_model(model, output_model_path)


# -----------------------------
# Sample building
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


# -----------------------------
# Numeric compare
# -----------------------------
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

    std_a = float(np.std(a))
    std_b = float(np.std(b))

    if std_a < EPS and std_b < EPS:
        return 1.0 if np.allclose(a, b, atol=1e-6, rtol=1e-6) else 0.0

    if std_a < EPS or std_b < EPS:
        return 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(a, b)[0, 1]

    if np.isnan(corr) or np.isinf(corr):
        return 0.0

    return float(corr)


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


def collect_real_tensor_names(model: onnx.ModelProto) -> List[str]:
    existing_outputs = set(o.name for o in model.graph.output)
    initializer_names = set(i.name for i in model.graph.initializer)

    names = []

    for value in model.graph.input:
        if value.name not in existing_outputs and value.name not in initializer_names:
            names.append(value.name)

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


def add_all_intermediate_outputs_to_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    names = collect_real_tensor_names(model)
    existing_output_names = set(o.name for o in model.graph.output)

    known_value_infos = {vi.name: vi for vi in model.graph.value_info}
    known_value_infos.update({vi.name: vi for vi in model.graph.input})
    known_value_infos.update({vi.name: vi for vi in model.graph.output})

    skipped = []

    for name in names:
        if name in existing_output_names:
            continue

        vi = known_value_infos.get(name)
        if vi is None:
            skipped.append(name)
            continue

        if not vi.type.HasField("tensor_type"):
            skipped.append(name)
            continue

        elem_type = vi.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.UNDEFINED:
            skipped.append(name)
            continue

        model.graph.output.append(copy.deepcopy(vi))

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} intermediate tensors with unknown type info during debug export.")

    return model


def run_all_outputs(
    model_path: str,
    sample: Dict[str, np.ndarray],
    provider: str = "CPUExecutionProvider",
):
    model = onnx.load(model_path)
    model_with_all_outputs = add_all_intermediate_outputs_to_model(model)

    sess = make_session_from_onnx_model(
        model_with_all_outputs,
        provider=provider,
        enable_all_optimizations=False,
    )

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
    try:
        orig_vals = run_all_outputs(original_model_path, sample, provider)
        opt_vals = run_all_outputs(optimized_model_path, sample, provider)
    except Exception as e:
        return {
            "first_bad_tensor": None,
            "worst_50_tensors": [],
            "total_compared_tensors": 0,
            "debug_error": str(e),
        }

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
        "debug_error": None,
    }


def is_good_final_output(report: Dict[str, Dict[str, float]], pcc_threshold: float, cosine_threshold: float) -> bool:
    if not report:
        return False

    for _, metrics in report.items():
        if metrics["shape_match"] < 1.0:
            return False
        if metrics["pcc"] < pcc_threshold:
            return False
        if metrics["cosine"] < cosine_threshold:
            return False
    return True


# -----------------------------
# Optimization variants
# -----------------------------
def remove_selected_qdq_pairs(
    model: onnx.ModelProto,
    blocked_tensor_names: Optional[Set[str]] = None,
    blocked_node_names: Optional[Set[str]] = None,
    allow_substrings: Optional[List[str]] = None,
) -> onnx.ModelProto:
    blocked_tensor_names = set(blocked_tensor_names or [])
    blocked_node_names = set(blocked_node_names or [])
    allow_substrings = list(allow_substrings or [])

    model = copy.deepcopy(model)
    consumers = build_output_to_consumers(model)

    nodes_to_remove = set()
    rewires = []
    removed_pairs = []

    for q_node in model.graph.node:
        if q_node.op_type != "QuantizeLinear":
            continue
        if q_node.name in blocked_node_names:
            continue
        if len(q_node.output) != 1 or len(q_node.input) < 1:
            continue

        q_out = q_node.output[0]
        if q_out in blocked_tensor_names:
            continue

        if allow_substrings:
            haystack = " ".join([q_node.name, q_out, q_node.input[0]])
            if not any(s in haystack for s in allow_substrings):
                continue

        q_consumers = consumers.get(q_out, [])
        if len(q_consumers) != 1:
            continue

        dq_node = q_consumers[0]
        if dq_node.op_type != "DequantizeLinear":
            continue
        if dq_node.name in blocked_node_names:
            continue
        if len(dq_node.output) != 1:
            continue

        dq_out = dq_node.output[0]
        if dq_out in blocked_tensor_names:
            continue

        if allow_substrings:
            haystack = " ".join([dq_node.name, dq_out])
            if not any(s in haystack for s in allow_substrings):
                continue

        original_src = q_node.input[0]

        rewires.append((dq_out, original_src))
        nodes_to_remove.add(id(q_node))
        nodes_to_remove.add(id(dq_node))
        removed_pairs.append({
            "q_node_name": q_node.name,
            "dq_node_name": dq_node.name,
            "q_out": q_out,
            "dq_out": dq_out,
            "rewired_to": original_src,
        })

    for old_name, new_name in rewires:
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == old_name:
                    node.input[i] = new_name
        for out in model.graph.output:
            if out.name == old_name:
                out.name = new_name

    kept_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    print(f"[INFO] Removed {len(removed_pairs)} selected QDQ pairs.")
    return model

def export_fixed_shape_copy(
    input_path: str,
    output_path: str,
    fixed_batch: int,
    image_height: int,
    image_width: int,
):
    model = maybe_prepare_model_for_rewrite(
        input_model_path=input_path,
        fix_input_shape_flag=True,
        fixed_batch=fixed_batch,
        image_height=image_height,
        image_width=image_width,
    )
    save_model(model, output_path)


def export_onnxoptimizer_with_preset(
    input_path: str,
    output_path: str,
    preset: str,
    fix_input_shape_flag: bool,
    fixed_batch: int,
    image_height: int,
    image_width: int,
):
    try:
        import onnxoptimizer
    except ImportError as e:
        raise RuntimeError(
            "onnxoptimizer is not installed. Install it with: pip install onnxoptimizer"
        ) from e

    model = maybe_prepare_model_for_rewrite(
        input_model_path=input_path,
        fix_input_shape_flag=fix_input_shape_flag,
        fixed_batch=fixed_batch,
        image_height=image_height,
        image_width=image_width,
    )

    passes = get_onnxoptimizer_passes(preset)
    optimized = onnxoptimizer.optimize(model, passes)
    save_model(optimized, output_path)


def export_onnxsim(
    input_path: str,
    output_path: str,
    fix_input_shape_flag: bool,
    fixed_batch: int,
    image_height: int,
    image_width: int,
    skip_constant_folding: bool,
    skip_shape_inference: bool,
):
    try:
        from onnxsim import simplify
    except ImportError as e:
        raise RuntimeError(
            "onnxsim is not installed. Install it with: pip install onnxsim"
        ) from e

    model = maybe_prepare_model_for_rewrite(
        input_model_path=input_path,
        fix_input_shape_flag=fix_input_shape_flag,
        fixed_batch=fixed_batch,
        image_height=image_height,
        image_width=image_width,
    )

    input_name = model.graph.input[0].name if len(model.graph.input) == 1 else None
    overwrite_input_shapes = None
    if fix_input_shape_flag and input_name is not None:
        overwrite_input_shapes = {
            input_name: [fixed_batch, 3, image_height, image_width]
        }

    simplified, ok = simplify(
        model,
        overwrite_input_shapes=overwrite_input_shapes,
        skip_constant_folding=skip_constant_folding,
        skip_shape_inference=skip_shape_inference,
    )
    if not ok:
        raise RuntimeError("onnxsim.simplify() returned ok=False")

    save_model(simplified, output_path)


def create_optimized_variants(
    input_model_path: str,
    output_dir: str,
    provider: str,
    variants: List[str],
    fix_input_shape_flag: bool,
    fixed_batch: int,
    image_height: int,
    image_width: int,
    onnxoptimizer_preset: str,
    onnxsim_skip_constant_folding: bool,
    onnxsim_skip_shape_inference: bool,
    block_tensors: List[str],
    block_nodes: List[str],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    def maybe_make_fixed_tmp(tag: str) -> str:
        if not fix_input_shape_flag:
            return input_model_path

        fixed_tmp = os.path.join(output_dir, f"_tmp_fixed_{tag}.onnx")
        export_fixed_shape_copy(
            input_path=input_model_path,
            output_path=fixed_tmp,
            fixed_batch=fixed_batch,
            image_height=image_height,
            image_width=image_width,
        )
        return fixed_tmp

    for variant in variants:
        out_path = os.path.join(output_dir, f"{variant}.onnx")

        try:
            if variant == "manual_hand_opt":
                manual_optimize_model(
                    input_model_path=input_model_path,
                    output_model_path=out_path,
                    blocked_tensor_names=set(block_tensors),
                    blocked_node_names=set(block_nodes),
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                )

            elif variant == "ort_basic":
                input_for_ort = maybe_make_fixed_tmp("ort_basic")
                save_ort_optimized_model(
                    input_model_path=input_for_ort,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="basic",
                )

            elif variant == "ort_extended":
                input_for_ort = maybe_make_fixed_tmp("ort_extended")
                save_ort_optimized_model(
                    input_model_path=input_for_ort,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="extended",
                )

            elif variant == "ort_all":
                input_for_ort = maybe_make_fixed_tmp("ort_all")
                save_ort_optimized_model(
                    input_model_path=input_for_ort,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="all",
                )

            elif variant == "onnxoptimizer_plus_ort_basic":
                tmp_path = os.path.join(output_dir, "_tmp_onnxoptimizer_for_ort_basic.onnx")
                export_onnxoptimizer_with_preset(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    preset=onnxoptimizer_preset,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="basic",
                )

            elif variant == "onnxoptimizer_plus_ort_extended":
                tmp_path = os.path.join(output_dir, "_tmp_onnxoptimizer_for_ort_extended.onnx")
                export_onnxoptimizer_with_preset(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    preset=onnxoptimizer_preset,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="extended",
                )

            elif variant == "onnxoptimizer_plus_ort_all":
                tmp_path = os.path.join(output_dir, "_tmp_onnxoptimizer_for_ort_all.onnx")
                export_onnxoptimizer_with_preset(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    preset=onnxoptimizer_preset,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="all",
                )

            elif variant == "onnxsim_plus_ort_basic":
                tmp_path = os.path.join(output_dir, "_tmp_onnxsim_for_ort_basic.onnx")
                export_onnxsim(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                    skip_constant_folding=onnxsim_skip_constant_folding,
                    skip_shape_inference=onnxsim_skip_shape_inference,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="basic",
                )

            elif variant == "onnxsim_plus_ort_extended":
                tmp_path = os.path.join(output_dir, "_tmp_onnxsim_for_ort_extended.onnx")
                export_onnxsim(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                    skip_constant_folding=onnxsim_skip_constant_folding,
                    skip_shape_inference=onnxsim_skip_shape_inference,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="extended",
                )

            elif variant == "onnxsim_plus_ort_all":
                tmp_path = os.path.join(output_dir, "_tmp_onnxsim_for_ort_all.onnx")
                export_onnxsim(
                    input_path=input_model_path,
                    output_path=tmp_path,
                    fix_input_shape_flag=fix_input_shape_flag,
                    fixed_batch=fixed_batch,
                    image_height=image_height,
                    image_width=image_width,
                    skip_constant_folding=onnxsim_skip_constant_folding,
                    skip_shape_inference=onnxsim_skip_shape_inference,
                )
                save_ort_optimized_model(
                    input_model_path=tmp_path,
                    output_model_path=out_path,
                    provider=provider,
                    opt_level_name="all",
                )

            else:
                raise ValueError(f"Unknown variant: {variant}")

            generated[variant] = out_path
            print(f"[INFO] Created variant: {variant} -> {out_path}")

        except Exception as e:
            print(f"[WARN] Failed to create {variant}: {e}")

    return generated


# -----------------------------
# evaluate_model wrapper
# -----------------------------
def build_ort_model_obj(
    model_path: str,
    provider: str,
    model_category: str,
):
    sess = make_session_from_model_path(
        model_path,
        provider=provider,
        enable_all_optimizations=False,
    )

    input_names = list_input_names(sess)
    output_names = list_output_names(sess)

    if len(input_names) != 1:
        raise ValueError(f"Expected exactly 1 input, got: {input_names}")

    return {
        "backend": "onnx",
        "model": None,
        "session": sess,
        "input_name": input_names[0],
        "output_names": output_names,
        "model_category_const": model_category,
    }


# -----------------------------
# main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--qdq_model", type=str, required=True)

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
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"])

    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    parser.add_argument("--pcc_threshold", type=float, default=0.99)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)
    parser.add_argument("--debug_batch", action="store_true")

    parser.add_argument("--output_dir", type=str, default="onnx_optimization_eval")
    parser.add_argument("--report_json", type=str, default="optimization_report.json")

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["manual_hand_opt"],
        choices=[
            "manual_hand_opt",
            "ort_basic",
            "ort_extended",
            "ort_all",
            "onnxoptimizer_plus_ort_basic",
            "onnxoptimizer_plus_ort_extended",
            "onnxoptimizer_plus_ort_all",
            "onnxsim_plus_ort_basic",
            "onnxsim_plus_ort_extended",
            "onnxsim_plus_ort_all",
        ],
    )

    parser.add_argument("--fix_input_shape", action="store_true")
    parser.add_argument("--fixed_batch", type=int, default=1)

    parser.add_argument(
        "--onnxoptimizer_preset",
        type=str,
        default="basic",
        choices=["basic", "conservative", "transpose", "extended"],
    )

    parser.add_argument("--onnxsim_skip_constant_folding", action="store_true")
    parser.add_argument("--onnxsim_skip_shape_inference", action="store_true")

    parser.add_argument(
        "--block_tensors",
        nargs="*",
        default=["4662", "4655", "output"],
        help="Tensor names to protect from manual QDQ removal.",
    )
    parser.add_argument(
        "--block_nodes",
        nargs="*",
        default=[],
        help="Node names to protect from manual QDQ removal.",
    )
    parser.add_argument(
        "--inspect_tensors",
        nargs="*",
        default=["4662"],
        help="Tensor names to inspect in original and optimized models.",
    )

    parser.add_argument(
        "--allow_qdq_substrings",
        nargs="*",
        default=[],
        help="Only remove QDQ pairs if node/output names contain one of these substrings."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    variants_dir = os.path.join(args.output_dir, "models")
    report_path = os.path.join(args.output_dir, args.report_json)

    print("[INFO] Creating optimized variants...")
    variants = create_optimized_variants(
        input_model_path=args.qdq_model,
        output_dir=variants_dir,
        provider=args.provider,
        variants=args.variants,
        fix_input_shape_flag=args.fix_input_shape,
        fixed_batch=args.fixed_batch,
        image_height=args.image_height,
        image_width=args.image_width,
        onnxoptimizer_preset=args.onnxoptimizer_preset,
        onnxsim_skip_constant_folding=args.onnxsim_skip_constant_folding,
        onnxsim_skip_shape_inference=args.onnxsim_skip_shape_inference,
        block_tensors=args.block_tensors,
        block_nodes=args.block_nodes,
    )

    if not variants:
        raise RuntimeError("No optimized variants were created.")

    print("[INFO] Collecting original model stats...")
    original_stats = collect_model_stats(args.qdq_model)

    original_inspection = {}
    for tensor_name in args.inspect_tensors:
        try:
            original_inspection[tensor_name] = inspect_tensor_context(args.qdq_model, tensor_name)
        except Exception as e:
            original_inspection[tensor_name] = {"error": str(e)}

    print("[INFO] Building loader...")
    loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        split=args.split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("[INFO] Building baseline sample...")
    ref_sess = make_session_from_model_path(
        args.qdq_model,
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

    print("[INFO] Evaluating original QDQ model...")
    baseline_obj = build_ort_model_obj(
        model_path=args.qdq_model,
        provider=args.provider,
        model_category=args.model_category,
    )
    baseline_eval = evaluate_model(
        model_obj=baseline_obj,
        model_category_const=baseline_obj["model_category_const"],
        loader=loader,
        device="cpu",
        max_samples=args.max_samples,
    )
    baseline_eval = to_jsonable(baseline_eval)

    results = {}

    for name, model_path in variants.items():
        print(f"\n[INFO] ===== Variant: {name} =====")
        item = {
            "model_path": model_path,
            "model_stats": None,
            "eval": None,
            "final_outputs": None,
            "first_bad_tensor": None,
            "worst_50_tensors": None,
            "total_compared_tensors": None,
            "debug_error": None,
            "is_numerically_good": None,
            "inspection": {},
            "status": "ok",
            "error": None,
        }

        try:
            item["model_stats"] = collect_model_stats(model_path)

            for tensor_name in args.inspect_tensors:
                try:
                    item["inspection"][tensor_name] = inspect_tensor_context(model_path, tensor_name)
                except Exception as e:
                    item["inspection"][tensor_name] = {"error": str(e)}

            model_obj = build_ort_model_obj(
                model_path=model_path,
                provider=args.provider,
                model_category=args.model_category,
            )

            eval_result = evaluate_model(
                model_obj=model_obj,
                model_category_const=model_obj["model_category_const"],
                loader=loader,
                device="cpu",
                max_samples=args.max_samples,
            )
            eval_result = to_jsonable(eval_result)
            item["eval"] = eval_result

            final_output_report = compare_model_outputs(
                original_model_path=args.qdq_model,
                optimized_model_path=model_path,
                sample=sample,
                provider=args.provider,
            )

            tensor_report = compare_all_tensors(
                original_model_path=args.qdq_model,
                optimized_model_path=model_path,
                sample=sample,
                provider=args.provider,
                pcc_threshold=args.pcc_threshold,
                cosine_threshold=args.cosine_threshold,
            )

            item["final_outputs"] = final_output_report
            item["first_bad_tensor"] = tensor_report["first_bad_tensor"]
            item["worst_50_tensors"] = tensor_report["worst_50_tensors"]
            item["total_compared_tensors"] = tensor_report["total_compared_tensors"]
            item["debug_error"] = tensor_report.get("debug_error")
            item["is_numerically_good"] = is_good_final_output(
                final_output_report,
                pcc_threshold=args.pcc_threshold,
                cosine_threshold=args.cosine_threshold,
            )

            print("[INFO] Eval result:")
            print(json.dumps(eval_result, indent=2))

            print("[INFO] Model stats:")
            print(json.dumps(to_jsonable(item["model_stats"]), indent=2))

            print("[INFO] Final output summary:")
            for out_name, metrics in final_output_report.items():
                print(
                    f"  {out_name}: "
                    f"pcc={metrics['pcc']:.6f}, "
                    f"cosine={metrics['cosine']:.6f}, "
                    f"max_abs={metrics['max_abs']:.6e}, "
                    f"mean_abs={metrics['mean_abs']:.6e}"
                )

            if tensor_report["first_bad_tensor"] is None:
                print("[INFO] No clearly bad tensor found.")
            else:
                bad = tensor_report["first_bad_tensor"]
                print(
                    "[INFO] First bad tensor: "
                    f"name={bad['name']}, "
                    f"pcc={bad['pcc']:.6f}, "
                    f"cosine={bad['cosine']:.6f}, "
                    f"max_abs={bad['max_abs']:.6e}, "
                    f"mean_abs={bad['mean_abs']:.6e}"
                )

            if item["debug_error"] is not None:
                print(f"[WARN] Intermediate tensor debug error: {item['debug_error']}")

            if not item["is_numerically_good"]:
                print(f"[WARN] Variant {name} is numerically bad. Ignore its speed result.")

            for tensor_name in args.inspect_tensors:
                print(f"[INFO] Inspection for tensor {tensor_name}:")
                print(json.dumps(to_jsonable(item["inspection"][tensor_name]), indent=2))

        except Exception as e:
            item["status"] = "failed"
            item["error"] = str(e)
            print(f"[WARN] Variant {name} failed: {e}")

        results[name] = item

    report = {
        "input_qdq_model": args.qdq_model,
        "input_qdq_model_stats": original_stats,
        "baseline_eval": baseline_eval,
        "original_inspection": original_inspection,
        "config": {
            "variants": args.variants,
            "fix_input_shape": args.fix_input_shape,
            "fixed_batch": args.fixed_batch,
            "onnxoptimizer_preset": args.onnxoptimizer_preset,
            "onnxsim_skip_constant_folding": args.onnxsim_skip_constant_folding,
            "onnxsim_skip_shape_inference": args.onnxsim_skip_shape_inference,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "batch_size": args.batch_size,
            "split": args.split,
            "pcc_threshold": args.pcc_threshold,
            "cosine_threshold": args.cosine_threshold,
            "block_tensors": args.block_tensors,
            "block_nodes": args.block_nodes,
            "inspect_tensors": args.inspect_tensors,
        },
        "variants_result": results,
    }

    with open(report_path, "w") as f:
        json.dump(to_jsonable(report), f, indent=2)

    print(f"\n[INFO] Wrote report to: {report_path}")


if __name__ == "__main__":
    main()