#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import hashlib
import time
from datetime import datetime

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from model.pdl import (
    DEEPLAB_V3_PLUS,
    PANOPTIC_DEEPLAB,
    build_model,
)
from quantization.quantize_function import load_aimet_quantized_model

from utils.image_loader import load_images, preprocess_image
from utils.demo_utils import (
    create_deeplab_v3plus_visualization,
    create_panoptic_visualization,
    save_predictions,
)


QUANT_OP_TYPES_EXACT = {
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "ConvInteger",
    "MatMulInteger",
}

QUANT_OP_KEYWORDS = (
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinear",
    "Integer",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze and package PDL quantization handoff artifacts with demo outputs."
    )

    parser.add_argument("--package_root", type=str, default="handoff_packages")
    parser.add_argument("--artifact_name", type=str, default="pdl_quant_handoff")
    parser.add_argument("--artifact_version", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    parser.add_argument("--fp32_weights", type=str, default=None,
                        help="Path to FP32 torch weights for loadability check")
    parser.add_argument("--fp32_onnx", type=str, default=None,
                        help="Optional FP32 ONNX for graph diff")
    parser.add_argument("--quant_model", type=str, required=True,
                        help="Path to quantized artifact (.onnx/.pt/.pth/.pkl)")

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)

    parser.add_argument("--demo_images", type=str, required=True,
                        help="Image file or folder for demo inference")
    parser.add_argument("--demo_num_iters", type=int, default=-1,
                        help="Number of demo images to process; -1 means all")
    parser.add_argument("--demo_output_path", type=str, default=None,
                        help="Optional output directory for demo visualizations. Defaults inside package.")
    parser.add_argument("--center_threshold", type=float, default=0.05,
                        help="Center threshold for panoptic visualization")

    parser.add_argument("--accuracy_note", type=str,
                        default="Not evaluated in this freeze package; demo included instead",
                        help="Freeform note for accuracy field when runtime eval is omitted")
    parser.add_argument("--notes", type=str, default="",
                        help="Optional freeform note to include in package")

    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sha256_file(path):
    if path is None or not os.path.isfile(path):
        return None

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def copy_into_package(src_path, dst_dir):
    if not src_path:
        return None

    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

    if src_path.endswith(".onnx"):
        src_data = src_path + "_data"
        dst_data = dst_path + "_data"
        if os.path.exists(src_data):
            shutil.copy2(src_data, dst_data)

    return dst_path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def tensor_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_array(arr):
    arr = np.asarray(arr)
    info = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
    }
    if arr.size > 0 and arr.dtype.kind in ("f", "i", "u"):
        info.update({
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        })
    return info


def is_quant_op(op_type):
    if op_type in QUANT_OP_TYPES_EXACT:
        return True
    return any(keyword in op_type for keyword in QUANT_OP_KEYWORDS)


def collect_onnx_op_inventory(model_path):
    model = onnx.load(model_path, load_external_data=False)

    op_counts = {}
    quant_op_counts = {}
    non_quant_op_counts = {}
    node_rows = []

    total_nodes = len(model.graph.node)
    quant_node_count = 0

    for idx, node in enumerate(model.graph.node):
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

        quant_flag = is_quant_op(op_type)
        if quant_flag:
            quant_op_counts[op_type] = quant_op_counts.get(op_type, 0) + 1
            quant_node_count += 1
        else:
            non_quant_op_counts[op_type] = non_quant_op_counts.get(op_type, 0) + 1

        node_rows.append({
            "index": idx,
            "name": node.name,
            "op_type": op_type,
            "is_quant_op": quant_flag,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    return {
        "node_count": total_nodes,
        "quant_node_count": quant_node_count,
        "non_quant_node_count": total_nodes - quant_node_count,
        "quant_node_ratio": (float(quant_node_count) / float(total_nodes)) if total_nodes > 0 else 0.0,
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: x[0])),
        "quant_op_counts": dict(sorted(quant_op_counts.items(), key=lambda x: x[0])),
        "non_quant_op_counts": dict(sorted(non_quant_op_counts.items(), key=lambda x: x[0])),
        "all_ops": sorted(op_counts.keys()),
        "quant_ops": sorted(quant_op_counts.keys()),
        "non_quant_ops": sorted(non_quant_op_counts.keys()),
        "nodes": node_rows,
    }


def collect_onnx_qparams_and_tensor_metadata(model_path):
    model = onnx.load(model_path, load_external_data=False)
    qparams = []
    tensor_meta = []

    init_map = {init.name: init for init in model.graph.initializer}

    def initializer_to_full_json(init_obj):
        info = {
            "name": init_obj.name,
            "dims": list(init_obj.dims),
            "data_type": int(init_obj.data_type),
            "uses_external_data": bool(init_obj.external_data),
        }

        if not init_obj.external_data:
            try:
                arr = numpy_helper.to_array(init_obj)
                info["dtype"] = str(arr.dtype)
                info["value"] = arr.tolist()
                info["summary"] = summarize_array(arr)
            except Exception as e:
                info["value_error"] = str(e)

        return info

    for init in model.graph.initializer:
        meta_row = {
            "name": init.name,
            "dims": list(init.dims),
            "data_type": int(init.data_type),
            "uses_external_data": bool(init.external_data),
        }

        if not init.external_data:
            try:
                arr = numpy_helper.to_array(init)
                meta_row["dtype"] = str(arr.dtype)
                meta_row["summary"] = summarize_array(arr)
                meta_row["value"] = arr.tolist()
            except Exception as e:
                meta_row["summary_error"] = str(e)

        tensor_meta.append(meta_row)

    for node in model.graph.node:
        if is_quant_op(node.op_type):
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)

            entry = {
                "node_name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "axis": axis,
            }

            if len(node.input) >= 2 and node.input[1] in init_map:
                entry["scale"] = initializer_to_full_json(init_map[node.input[1]])

            if len(node.input) >= 3 and node.input[2] in init_map:
                entry["zero_point"] = initializer_to_full_json(init_map[node.input[2]])

            qparams.append(entry)

    return {
        "qparam_nodes": qparams,
        "initializer_tensor_metadata": tensor_meta,
    }


def compare_onnx_graphs(fp32_onnx_path, quant_onnx_path):
    if not fp32_onnx_path or not quant_onnx_path:
        return None

    fp32_inv = collect_onnx_op_inventory(fp32_onnx_path)
    quant_inv = collect_onnx_op_inventory(quant_onnx_path)

    fp32_ops = fp32_inv["op_counts"]
    quant_ops = quant_inv["op_counts"]

    all_ops = sorted(set(fp32_ops.keys()) | set(quant_ops.keys()))
    diff_rows = []

    for op in all_ops:
        diff_rows.append({
            "op_type": op,
            "fp32_count": fp32_ops.get(op, 0),
            "quant_count": quant_ops.get(op, 0),
            "delta": quant_ops.get(op, 0) - fp32_ops.get(op, 0),
            "is_quant_op": is_quant_op(op),
        })

    return {
        "fp32_node_count": fp32_inv["node_count"],
        "quant_node_count": quant_inv["node_count"],
        "fp32_ops": fp32_inv["all_ops"],
        "quant_ops": quant_inv["all_ops"],
        "quant_only_ops": sorted(set(quant_inv["all_ops"]) - set(fp32_inv["all_ops"])),
        "fp32_only_ops": sorted(set(fp32_inv["all_ops"]) - set(quant_inv["all_ops"])),
        "op_count_diff": diff_rows,
    }


def run_single_torch_model(model, image_np, device):
    x = torch.from_numpy(image_np).to(device)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        y = model(x)

    if was_training:
        model.train()

    if torch.is_tensor(y):
        return {"output_0": tensor_to_numpy(y)}

    if isinstance(y, dict):
        out = {}
        for k, v in y.items():
            if torch.is_tensor(v):
                out[k] = tensor_to_numpy(v)
        return out

    if isinstance(y, (tuple, list)):
        out = {}
        for i, v in enumerate(y):
            if torch.is_tensor(v):
                out["output_{}".format(i)] = tensor_to_numpy(v)
        return out

    raise RuntimeError("Unsupported torch model output type: {}".format(type(y)))


def run_single_backend(model_obj, image_np, device):
    if model_obj["backend"] == "torch":
        return run_single_torch_model(model_obj["model"], image_np, device)

    if model_obj["backend"] == "onnx":
        sess = model_obj["session"]
        input_name = model_obj["input_name"]
        output_names = model_obj["output_names"]

        outputs = sess.run(output_names, {input_name: image_np})
        out = {}

        if output_names is None:
            for i, arr in enumerate(outputs):
                out["output_{}".format(i)] = tensor_to_numpy(arr)
        else:
            for i, arr in enumerate(outputs):
                out[output_names[i]] = tensor_to_numpy(arr)

        return out

    raise RuntimeError("Unsupported backend: {}".format(model_obj["backend"]))


def collect_sample_io_from_files(model_obj, device, sample_input_paths, out_dir):
    ensure_dir(out_dir)
    rows = []

    for count, sample_path in enumerate(sample_input_paths):
        image_np = np.load(sample_path)

        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)

        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)

        outputs = run_single_backend(model_obj, image_np, device)

        input_dst = os.path.join(out_dir, "sample_{:03d}_input.npy".format(count))
        np.save(input_dst, image_np)

        out_entries = []
        for out_name, out_arr in outputs.items():
            safe_name = out_name.replace("/", "_").replace(":", "_")
            out_path = os.path.join(out_dir, "sample_{:03d}_{}.npy".format(count, safe_name))
            np.save(out_path, out_arr)
            out_entries.append({
                "name": out_name,
                "path": os.path.basename(out_path),
                "summary": summarize_array(out_arr),
            })

        rows.append({
            "sample_index": count,
            "source_input_file": sample_path,
            "input_path": os.path.basename(input_dst),
            "input_summary": summarize_array(image_np),
            "reference_outputs": out_entries,
        })

    return rows


def verify_model_loadability(fp32_weights, quant_model, args):
    results = {
        "fp32_load_ok": False,
        "quant_load_ok": False,
        "fp32_error": None,
        "quant_error": None,
        "quant_backend": None,
    }

    if fp32_weights:
        try:
            model, category = build_model(
                weights_path=fp32_weights,
                model_category=args.model_category,
                image_height=args.image_height,
                image_width=args.image_width,
                device=args.device,
            )
            _ = {
                "backend": "torch",
                "model": model,
                "session": None,
                "input_name": None,
                "output_names": None,
                "model_category_const": category,
            }
            results["fp32_load_ok"] = True
        except Exception as e:
            results["fp32_error"] = str(e)

    try:
        quant_obj = load_aimet_quantized_model(
            quant_weights=quant_model,
            model_category=args.model_category,
            device=args.device,
            provider=args.onnx_provider,
        )
        results["quant_load_ok"] = True
        results["quant_backend"] = quant_obj["backend"]
    except Exception as e:
        results["quant_error"] = str(e)

    return results


def inspect_torch_checkpoint(path):
    obj = torch.load(path, map_location="cpu")

    info = {
        "path": path,
        "python_type": str(type(obj)),
        "format": "unknown",
        "top_level_keys": None,
        "state_dict_key": None,
        "num_tensors": None,
    }

    if isinstance(obj, dict):
        keys = list(obj.keys())
        info["top_level_keys"] = keys[:200]

        if len(keys) > 0 and all(isinstance(k, str) for k in keys):
            tensor_like_values = 0
            for v in obj.values():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    tensor_like_values += 1

            if tensor_like_values > 0 and tensor_like_values == len(obj):
                info["format"] = "raw_state_dict"
                info["state_dict_key"] = None
                info["num_tensors"] = len(obj)
                return obj, info

        for candidate_key in ["state_dict", "model_state_dict", "model", "network", "net"]:
            if candidate_key in obj and isinstance(obj[candidate_key], dict):
                sd = obj[candidate_key]

                if len(sd) > 0 and all(isinstance(k, str) for k in sd.keys()):
                    tensor_like_values = 0
                    for v in sd.values():
                        if torch.is_tensor(v) or isinstance(v, np.ndarray):
                            tensor_like_values += 1

                    if tensor_like_values > 0:
                        info["format"] = "checkpoint_with_state_dict"
                        info["state_dict_key"] = candidate_key
                        info["num_tensors"] = len(sd)
                        return sd, info

        info["format"] = "generic_checkpoint_dict"
        return None, info

    info["format"] = "serialized_model_object"
    return None, info


def save_state_dict_artifact(state_dict_obj, output_path):
    ensure_dir(os.path.dirname(output_path))
    cpu_state_dict = {}

    for k, v in state_dict_obj.items():
        if torch.is_tensor(v):
            cpu_state_dict[k] = v.detach().cpu()
        elif isinstance(v, np.ndarray):
            cpu_state_dict[k] = torch.from_numpy(v)

    torch.save(cpu_state_dict, output_path)

    return {
        "path": output_path,
        "num_tensors": len(cpu_state_dict),
        "sha256": sha256_file(output_path),
        "keys_preview": list(cpu_state_dict.keys())[:100],
    }


def load_demo_model(weights_path, model_category, image_height, image_width, device, onnx_provider):
    ext = os.path.splitext(weights_path)[1].lower()

    if ext == ".onnx":
        print("[INFO] Loading ONNX model for demo...")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            weights_path,
            sess_options=so,
            providers=[onnx_provider],
        )

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        model_category_const = (
            PANOPTIC_DEEPLAB if model_category == "PANOPTIC_DEEPLAB" else DEEPLAB_V3_PLUS
        )

        return {
            "backend": "onnx",
            "session": session,
            "input_name": input_name,
            "output_names": output_names,
            "model": None,
            "model_category_const": model_category_const,
        }

    print("[INFO] Loading PyTorch model for demo...")
    model, model_category_const = build_model(
        weights_path=weights_path,
        model_category=model_category,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )

    return {
        "backend": "torch",
        "model": model,
        "session": None,
        "input_name": None,
        "output_names": None,
        "model_category_const": model_category_const,
    }


def run_demo_inference(model_obj, torch_input, model_category_const):
    if model_obj["backend"] == "torch":
        with torch.no_grad():
            outputs = model_obj["model"](torch_input)

        if model_category_const == DEEPLAB_V3_PLUS:
            if isinstance(outputs, torch.Tensor):
                return outputs
            if isinstance(outputs, (list, tuple)):
                return outputs[0]
            if isinstance(outputs, dict):
                if "out" in outputs:
                    return outputs["out"]
                return next(iter(outputs.values()))
            raise RuntimeError("Unsupported Deeplab torch output type: {}".format(type(outputs)))

        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError("Expected panoptic torch output tuple/list with at least 3 elements")
        semantic_logits, center_heatmap, offset_map = outputs[0], outputs[1], outputs[2]
        extra = outputs[3] if len(outputs) > 3 else None
        return semantic_logits, center_heatmap, offset_map, extra

    elif model_obj["backend"] == "onnx":
        session = model_obj["session"]
        input_name = model_obj["input_name"]

        input_np = torch_input.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = session.run(None, {input_name: input_np})

        if model_category_const == DEEPLAB_V3_PLUS:
            semantic_logits = torch.from_numpy(outputs[0]).float()
            if semantic_logits.ndim == 4 and semantic_logits.shape[1] != 19 and semantic_logits.shape[-1] == 19:
                semantic_logits = semantic_logits.permute(0, 3, 1, 2).contiguous()
            return semantic_logits

        semantic_logits = torch.from_numpy(outputs[0]).float()
        center_heatmap = torch.from_numpy(outputs[1]).float()
        offset_map = torch.from_numpy(outputs[2]).float()
        extra = torch.from_numpy(outputs[3]).float() if len(outputs) > 3 else None

        if semantic_logits.ndim == 4 and semantic_logits.shape[1] != 19 and semantic_logits.shape[-1] == 19:
            semantic_logits = semantic_logits.permute(0, 3, 1, 2).contiguous()

        if center_heatmap.ndim == 4 and center_heatmap.shape[1] not in (1, 2) and center_heatmap.shape[-1] in (1, 2):
            center_heatmap = center_heatmap.permute(0, 3, 1, 2).contiguous()

        if offset_map.ndim == 4 and offset_map.shape[1] not in (1, 2) and offset_map.shape[-1] in (1, 2):
            offset_map = offset_map.permute(0, 3, 1, 2).contiguous()

        return semantic_logits, center_heatmap, offset_map, extra

    raise ValueError(f"Unsupported backend: {model_obj['backend']}")


def save_demo_visualization(
    model_category_const,
    output,
    original_image,
    output_path,
    image_path,
    center_threshold=0.05,
):
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]
    output_dir = os.path.join(output_path, f"{image_stem}_output")
    os.makedirs(output_dir, exist_ok=True)

    if model_category_const == DEEPLAB_V3_PLUS:
        semantic_logits = output
        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_deeplab_v3plus_visualization(
            semantic_np,
            original_image=original_image,
        )
    else:
        semantic_logits, center_heatmap, offset_map, _ = output

        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        center_np = center_heatmap.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        offset_np = offset_map.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_panoptic_visualization(
            semantic_np,
            center_np,
            offset_np,
            original_image,
            center_threshold=center_threshold,
            score_threshold=center_threshold,
            stuff_area=1,
            top_k=1000,
            nms_kernel=11,
        )

    save_predictions(output_dir, image_name, original_image, vis)

    return {
        "image_name": image_name,
        "output_dir": output_dir,
    }


def run_demo_and_package_outputs(args, model_obj, demo_output_dir):
    os.makedirs(demo_output_dir, exist_ok=True)

    images = load_images(args.demo_images, args.demo_num_iters)
    if len(images) == 0:
        raise ValueError(f"No valid images found in: {args.demo_images}")

    rows = []
    total_start = time.time()

    for i, image_path in enumerate(images, start=1):
        preprocess_device = args.device
        if model_obj["backend"] == "onnx" and args.onnx_provider == "CPUExecutionProvider":
            preprocess_device = "cpu"

        original_image, torch_input = preprocess_image(
            image_path=image_path,
            input_width=args.image_width,
            input_height=args.image_height,
            device=preprocess_device,
        )

        if model_obj["backend"] == "torch" and args.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        output = run_demo_inference(model_obj, torch_input, model_obj["model_category_const"])
        if model_obj["backend"] == "torch" and args.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        vis_info = save_demo_visualization(
            model_category_const=model_obj["model_category_const"],
            output=output,
            original_image=original_image,
            output_path=demo_output_dir,
            image_path=image_path,
            center_threshold=args.center_threshold,
        )

        rows.append({
            "index": i - 1,
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "latency_ms": (end_time - start_time) * 1000.0,
            "visualization_output_dir": os.path.relpath(vis_info["output_dir"], demo_output_dir),
        })

        print(f"[{i}/{len(images)}] {os.path.basename(image_path)}: {(end_time - start_time) * 1000:.2f} ms")

    total_end = time.time()
    total_time = total_end - total_start
    fps = len(images) / total_time if total_time > 0 else 0.0

    summary = {
        "model_category": args.model_category,
        "backend": model_obj["backend"],
        "onnx_provider": args.onnx_provider if model_obj["backend"] == "onnx" else None,
        "num_inputs": len(images),
        "total_execution_time_s": total_time,
        "samples_per_second": fps,
        "per_image": rows,
    }

    return summary


def build_notes_md(manifest, accuracy_summary, graph_diff, op_inventory, args):
    lines = []
    lines.append("# PDL Quantization Handoff")
    lines.append("")
    lines.append("## Artifact")
    lines.append("- Name: {}".format(manifest["artifact"]["name"]))
    lines.append("- Version: {}".format(manifest["artifact"]["version"]))
    lines.append("- Created: {}".format(manifest["artifact"]["created_at_utc"]))
    lines.append("")
    lines.append("## Verification")
    lines.append("- Quantized model load on PC: {}".format(manifest["verification"]["quant_load_ok"]))
    lines.append("- FP32 model load on PC: {}".format(manifest["verification"]["fp32_load_ok"]))
    lines.append("- Quant backend: {}".format(manifest["verification"].get("quant_backend")))
    if manifest["verification"]["fp32_error"]:
        lines.append("- FP32 load error: {}".format(manifest["verification"]["fp32_error"]))
    if manifest["verification"]["quant_error"]:
        lines.append("- Quant load error: {}".format(manifest["verification"]["quant_error"]))
    lines.append("")
    lines.append("## Accuracy Result vs FP32 Baseline")
    if accuracy_summary is not None:
        for k, v in accuracy_summary.items():
            lines.append("- {}: {}".format(k, v))
    else:
        lines.append("- Not provided")
    lines.append("")
    lines.append("## Demo")
    lines.append("- Demo images: {}".format(args.demo_images))
    lines.append("- See `reports/demo_report.json` for per-image latency and demo throughput.")
    lines.append("- Demo visualizations are stored under `demo_outputs/` inside the package unless an explicit output path was used.")
    lines.append("")
    lines.append("## Quant Graph Summary")
    if isinstance(op_inventory, dict) and "node_count" in op_inventory:
        lines.append("- Total nodes: {}".format(op_inventory["node_count"]))
        lines.append("- Quant nodes: {}".format(op_inventory["quant_node_count"]))
        lines.append("- Non-quant nodes: {}".format(op_inventory["non_quant_node_count"]))
        lines.append("- Quant node ratio: {:.6f}".format(op_inventory["quant_node_ratio"]))
        lines.append("- Unique ops: {}".format(len(op_inventory["all_ops"])))
        lines.append("- Unique quant ops: {}".format(len(op_inventory["quant_ops"])))
        lines.append("- Unique non-quant ops: {}".format(len(op_inventory["non_quant_ops"])))
    else:
        lines.append("- Quant model is not ONNX, graph inventory unavailable.")
    lines.append("")
    lines.append("## Known Unsupported Ops")
    lines.append("- Not auto-detected in this package. See full op inventory for backend compatibility review.")
    lines.append("")
    lines.append("## Expected Fallback Ops")
    lines.append("- Not auto-detected in this package. See non-quant ops and runtime backend support matrix.")
    lines.append("")
    lines.append("## Graph Differences vs FP32")
    if graph_diff is None:
        lines.append("- FP32 ONNX not provided")
    else:
        lines.append("- FP32 node count: {}".format(graph_diff["fp32_node_count"]))
        lines.append("- Quant node count: {}".format(graph_diff["quant_node_count"]))
        lines.append("- Quant-only ops: {}".format(", ".join(graph_diff["quant_only_ops"]) if graph_diff["quant_only_ops"] else "None"))
        lines.append("- FP32-only ops: {}".format(", ".join(graph_diff["fp32_only_ops"]) if graph_diff["fp32_only_ops"] else "None"))
        lines.append("- See `notes/graph_diff.json` for full op-level diff")
    lines.append("")
    lines.append("## Torch Checkpoint Format")
    ckpt = manifest.get("checkpoint_inspection")
    if ckpt is not None:
        lines.append("- Format: {}".format(ckpt.get("format")))
        lines.append("- Python type: {}".format(ckpt.get("python_type")))
        lines.append("- Embedded state_dict key: {}".format(ckpt.get("state_dict_key")))
        lines.append("- Number of tensors: {}".format(ckpt.get("num_tensors")))
    else:
        lines.append("- Not applicable")
    lines.append("")
    if args.notes:
        lines.append("## Additional Notes")
        lines.append(args.notes)
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    package_dir = os.path.join(
        args.package_root,
        "{}_v{}_{}".format(args.artifact_name, args.artifact_version, timestamp),
    )

    assets_dir = os.path.join(package_dir, "assets")
    models_dir = os.path.join(assets_dir, "models")
    io_dir = os.path.join(assets_dir, "sample_io")
    notes_dir = os.path.join(package_dir, "notes")
    reports_dir = os.path.join(package_dir, "reports")

    for p in [package_dir, assets_dir, models_dir, io_dir, notes_dir, reports_dir]:
        ensure_dir(p)

    verification = verify_model_loadability(args.fp32_weights, args.quant_model, args)

    fp32_model_obj = None
    quant_obj = None
    accuracy_summary = {
        "status": "not_evaluated_in_this_freeze",
        "note": args.accuracy_note,
    }

    if args.fp32_weights:
        print("[INFO] Loading FP32 model...")
        fp32_model, fp32_category = build_model(
            weights_path=args.fp32_weights,
            model_category=args.model_category,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device,
        )
        fp32_model_obj = {
            "backend": "torch",
            "model": fp32_model,
            "session": None,
            "input_name": None,
            "output_names": None,
            "model_category_const": fp32_category,
        }

    print("[INFO] Loading quantized/exported model...")
    quant_obj = load_aimet_quantized_model(
        quant_weights=args.quant_model,
        model_category=args.model_category,
        device=args.device,
        provider=args.onnx_provider,
    )

    print("[INFO] Copying model assets...")
    packaged_fp32_weights = copy_into_package(args.fp32_weights, models_dir) if args.fp32_weights else None
    packaged_fp32_onnx = copy_into_package(args.fp32_onnx, models_dir) if args.fp32_onnx else None
    packaged_quant_model = copy_into_package(args.quant_model, models_dir)

    checkpoint_inspection = None
    extracted_state_dict_info = None

    quant_model_ext = os.path.splitext(args.quant_model)[1].lower()
    if quant_model_ext in [".pt", ".pth", ".pkl"]:
        print("[INFO] Inspecting torch checkpoint for state_dict...")
        try:
            extracted_sd, checkpoint_inspection = inspect_torch_checkpoint(args.quant_model)

            if checkpoint_inspection is not None:
                save_json(
                    checkpoint_inspection,
                    os.path.join(notes_dir, "quant_checkpoint_inspection.json"),
                )

            if extracted_sd is not None:
                normalized_state_dict_path = os.path.join(models_dir, "quant_state_dict.pth")
                extracted_state_dict_info = save_state_dict_artifact(
                    state_dict_obj=extracted_sd,
                    output_path=normalized_state_dict_path,
                )
                save_json(
                    extracted_state_dict_info,
                    os.path.join(notes_dir, "quant_state_dict_info.json"),
                )
                print("[INFO] Saved normalized state_dict: {}".format(normalized_state_dict_path))
            else:
                print("[WARN] No extractable state_dict found in quant checkpoint.")

        except Exception as e:
            checkpoint_inspection = {"error": str(e)}
            save_json(
                checkpoint_inspection,
                os.path.join(notes_dir, "quant_checkpoint_inspection.json"),
            )
            print("[WARN] Failed to inspect checkpoint: {}".format(e))

    quant_op_inventory = None
    quant_qparams = None
    graph_diff = None

    quant_model_packaged_path = os.path.join(models_dir, os.path.basename(args.quant_model))
    if quant_model_packaged_path.endswith(".onnx"):
        print("[INFO] Extracting ONNX quantization metadata...")
        quant_op_inventory = collect_onnx_op_inventory(quant_model_packaged_path)
        quant_qparams = collect_onnx_qparams_and_tensor_metadata(quant_model_packaged_path)

        if packaged_fp32_onnx:
            fp32_onnx_packaged_path = os.path.join(models_dir, os.path.basename(args.fp32_onnx))
            graph_diff = compare_onnx_graphs(fp32_onnx_packaged_path, quant_model_packaged_path)
    else:
        quant_op_inventory = {
            "warning": "Quant model is not ONNX. ONNX op inventory unavailable."
        }
        quant_qparams = {
            "warning": "Quant model is not ONNX. QParam extraction unavailable."
        }

    def create_sample_npys_from_demo_images(args, out_dir):
        ensure_dir(out_dir)

        images = load_images(args.demo_images, args.demo_num_iters)
        if len(images) == 0:
            raise ValueError(f"No valid images found in: {args.demo_images}")

        rows = []

        for idx, image_path in enumerate(images):
            preprocess_device = args.device
            if args.quant_model.endswith(".onnx") and args.onnx_provider == "CPUExecutionProvider":
                preprocess_device = "cpu"

            original_image, torch_input = preprocess_image(
                image_path=image_path,
                input_width=args.image_width,
                input_height=args.image_height,
                device=preprocess_device,
            )

            input_np = torch_input.detach().cpu().numpy().astype(np.float32, copy=False)

            npy_name = "sample_{:03d}_input.npy".format(idx)
            npy_path = os.path.join(out_dir, npy_name)
            np.save(npy_path, input_np)

            rows.append({
                "sample_index": idx,
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "input_npy": npy_name,
                "input_summary": summarize_array(input_np),
            })

            print(f"[INFO] Saved sample input: {npy_path}")

        return rows

    print("[INFO] Creating sample input NPYs from demo images...")
    sample_input_index = create_sample_npys_from_demo_images(
        args=args,
        out_dir=io_dir,
    )

    generated_sample_paths = [
        os.path.join(io_dir, row["input_npy"])
        for row in sample_input_index
    ]

    print("[INFO] Collecting reference outputs from generated sample NPYs...")
    sample_io_index = collect_sample_io_from_files(
        model_obj=quant_obj,
        device=args.device,
        sample_input_paths=generated_sample_paths,
        out_dir=io_dir,
    )

    for row, src in zip(sample_io_index, sample_input_index):
        row["source_image_path"] = src["image_path"]
        row["source_image_name"] = src["image_name"]

    print("[INFO] Running demo and saving visualizations...")
    demo_output_dir = args.demo_output_path
    if demo_output_dir is None:
        demo_output_dir = os.path.join(package_dir, "demo_outputs")

    demo_summary = run_demo_and_package_outputs(
        args=args,
        model_obj=quant_obj,
        demo_output_dir=demo_output_dir,
    )

    save_json(demo_summary, os.path.join(reports_dir, "demo_report.json"))
    save_json(accuracy_summary, os.path.join(reports_dir, "accuracy_compare.json"))
    save_json(quant_op_inventory, os.path.join(notes_dir, "op_inventory.json"))
    save_json(quant_qparams, os.path.join(notes_dir, "quant_assets.json"))
    save_json(sample_io_index, os.path.join(notes_dir, "sample_io_index.json"))
    if graph_diff is not None:
        save_json(graph_diff, os.path.join(notes_dir, "graph_diff.json"))

    manifest = {
        "artifact": {
            "name": args.artifact_name,
            "version": args.artifact_version,
            "created_at_utc": timestamp,
            "package_dir": package_dir,
        },
        "verification": verification,
        "models": {
            "fp32_weights": os.path.basename(packaged_fp32_weights) if packaged_fp32_weights else None,
            "fp32_onnx": os.path.basename(packaged_fp32_onnx) if packaged_fp32_onnx else None,
            "quant_model": os.path.basename(packaged_quant_model) if packaged_quant_model else None,
            "quant_state_dict": "quant_state_dict.pth" if extracted_state_dict_info is not None else None,
        },
        "checksums": {
            "fp32_weights_sha256": sha256_file(packaged_fp32_weights) if packaged_fp32_weights else None,
            "fp32_onnx_sha256": sha256_file(packaged_fp32_onnx) if packaged_fp32_onnx else None,
            "quant_model_sha256": sha256_file(packaged_quant_model) if packaged_quant_model else None,
            "quant_state_dict_sha256": (
                extracted_state_dict_info["sha256"] if extracted_state_dict_info is not None else None
            ),
        },
        "demo": {
            "images": args.demo_images,
            "num_iters": args.demo_num_iters,
            "output_dir": "demo_outputs" if args.demo_output_path is None else args.demo_output_path,
            "report_file": "reports/demo_report.json",
        },
        "packaged_assets": {
            "sample_io_dir": "assets/sample_io",
            "sample_io_index": "notes/sample_io_index.json",
            "quant_assets": "notes/quant_assets.json",
            "op_inventory": "notes/op_inventory.json",
            "graph_diff": "notes/graph_diff.json" if graph_diff is not None else None,
            "accuracy_compare": "reports/accuracy_compare.json",
            "demo_report": "reports/demo_report.json",
            "checkpoint_inspection": "notes/quant_checkpoint_inspection.json" if checkpoint_inspection is not None else None,
            "state_dict_info": "notes/quant_state_dict_info.json" if extracted_state_dict_info is not None else None,
        },
        "ops_summary": {
            "all_ops": quant_op_inventory.get("all_ops") if isinstance(quant_op_inventory, dict) else None,
            "quant_ops": quant_op_inventory.get("quant_ops") if isinstance(quant_op_inventory, dict) else None,
            "non_quant_ops": quant_op_inventory.get("non_quant_ops") if isinstance(quant_op_inventory, dict) else None,
            "quant_op_counts": quant_op_inventory.get("quant_op_counts") if isinstance(quant_op_inventory, dict) else None,
            "non_quant_op_counts": quant_op_inventory.get("non_quant_op_counts") if isinstance(quant_op_inventory, dict) else None,
        },
        "notes": {
            "known_unsupported_ops": [],
            "expected_fallback_ops": [],
            "freeform_notes": args.notes,
            "unsupported_fallback_note": "Not auto-detected. Review full op inventory against target backend support.",
        },
        "checkpoint_inspection": checkpoint_inspection,
        "state_dict_export": extracted_state_dict_info,
    }

    save_json(manifest, os.path.join(package_dir, "manifest.json"))

    notes_md = build_notes_md(
        manifest=manifest,
        accuracy_summary=accuracy_summary,
        graph_diff=graph_diff,
        op_inventory=quant_op_inventory,
        args=args,
    )
    save_text(notes_md, os.path.join(package_dir, "README.md"))

    print("[INFO] Package created successfully:")
    print(package_dir)


if __name__ == "__main__":
    main()