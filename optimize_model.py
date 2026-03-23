#!/usr/bin/env python3
import argparse
import copy
import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, shape_inference

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

EPS = 1e-12


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

    for name in names:
        if name in existing_output_names:
            continue

        if name in known_value_infos:
            model.graph.output.append(copy.deepcopy(known_value_infos[name]))
        else:
            model.graph.output.append(
                helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            )

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


# -----------------------------
# Model optimization variants
# -----------------------------
def export_baseline_copy(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copyfile(input_path, output_path)


def export_onnxoptimizer_basic(input_path: str, output_path: str):
    try:
        import onnxoptimizer
    except ImportError as e:
        raise RuntimeError(
            "onnxoptimizer is not installed. Install it with: pip install onnxoptimizer"
        ) from e

    model = onnx.load(input_path)

    passes = [
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

    optimized = onnxoptimizer.optimize(model, passes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(optimized, output_path)


def export_onnxsim(input_path: str, output_path: str):
    try:
        from onnxsim import simplify
    except ImportError as e:
        raise RuntimeError(
            "onnxsim is not installed. Install it with: pip install onnxsim"
        ) from e

    model = onnx.load(input_path)
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim.simplify() returned ok=False")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(simplified, output_path)


def create_optimized_variants(
    input_model_path: str,
    output_dir: str,
    provider: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    variants = {}

    baseline_path = os.path.join(output_dir, "baseline_copy.onnx")
    export_baseline_copy(input_model_path, baseline_path)
    variants["baseline_copy"] = baseline_path

    for level in ["basic", "extended", "all"]:
        out_path = os.path.join(output_dir, f"ort_{level}.onnx")
        try:
            save_ort_optimized_model(
                input_model_path=input_model_path,
                output_model_path=out_path,
                provider=provider,
                opt_level_name=level,
            )
            variants[f"ort_{level}"] = out_path
        except Exception as e:
            print(f"[WARN] Failed to create ort_{level}: {e}")

    onnxopt_path = os.path.join(output_dir, "onnxoptimizer_basic.onnx")
    try:
        export_onnxoptimizer_basic(input_model_path, onnxopt_path)
        variants["onnxoptimizer_basic"] = onnxopt_path
    except Exception as e:
        print(f"[WARN] Failed to create onnxoptimizer_basic: {e}")

    onnxsim_path = os.path.join(output_dir, "onnxsim.onnx")
    try:
        export_onnxsim(input_model_path, onnxsim_path)
        variants["onnxsim"] = onnxsim_path
    except Exception as e:
        print(f"[WARN] Failed to create onnxsim: {e}")

    if "onnxoptimizer_basic" in variants:
        combo_path = os.path.join(output_dir, "onnxoptimizer_plus_ort_all.onnx")
        try:
            save_ort_optimized_model(
                input_model_path=variants["onnxoptimizer_basic"],
                output_model_path=combo_path,
                provider=provider,
                opt_level_name="all",
            )
            variants["onnxoptimizer_plus_ort_all"] = combo_path
        except Exception as e:
            print(f"[WARN] Failed to create onnxoptimizer_plus_ort_all: {e}")

    if "onnxsim" in variants:
        combo_path = os.path.join(output_dir, "onnxsim_plus_ort_all.onnx")
        try:
            save_ort_optimized_model(
                input_model_path=variants["onnxsim"],
                output_model_path=combo_path,
                provider=provider,
                opt_level_name="all",
            )
            variants["onnxsim_plus_ort_all"] = combo_path
        except Exception as e:
            print(f"[WARN] Failed to create onnxsim_plus_ort_all: {e}")

    return variants


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
        # If your evaluate_model() expects "ort" instead of "onnx", change this line.
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

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"])

    parser.add_argument("--provider", type=str, default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"])

    parser.add_argument("--pcc_threshold", type=float, default=0.99)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)
    parser.add_argument("--debug_batch", action="store_true")

    parser.add_argument("--output_dir", type=str, default="onnx_optimization_eval")
    parser.add_argument("--report_json", type=str, default="optimization_report.json")

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
    )

    if not variants:
        raise RuntimeError("No optimized variants were created.")

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

    baseline_eval = None
    results = {}

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

    for name, model_path in variants.items():
        print(f"\n[INFO] ===== Variant: {name} =====")
        item = {
            "model_path": model_path,
            "eval": None,
            "final_outputs": None,
            "first_bad_tensor": None,
            "worst_50_tensors": None,
            "total_compared_tensors": None,
            "status": "ok",
            "error": None,
        }

        try:
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

            print("[INFO] Eval result:")
            print(json.dumps(eval_result, indent=2))

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

        except Exception as e:
            item["status"] = "failed"
            item["error"] = str(e)
            print(f"[WARN] Variant {name} failed: {e}")

        results[name] = item

    report = {
        "input_qdq_model": args.qdq_model,
        "baseline_eval": baseline_eval,
        "variants": results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Wrote report to: {report_path}")


if __name__ == "__main__":
    main()