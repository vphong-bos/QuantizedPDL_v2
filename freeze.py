#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import hashlib
from datetime import datetime

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from model.pdl import build_model
from quantization.quantize_function import load_aimet_quantized_model
from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify, evaluate, and package PDL quantization handoff artifacts."
    )

    parser.add_argument("--cityscapes_root", type=str, required=True)
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
                        help="Path to FP32 torch weights")
    parser.add_argument("--fp32_onnx", type=str, default=None,
                        help="Optional FP32 ONNX for graph diff")
    parser.add_argument("--quant_model", type=str, required=True,
                        help="Path to quantized artifact (.onnx/.pt/.pth/.pkl)")

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])

    parser.add_argument("--num_sample_inputs", type=int, default=3)
    parser.add_argument("--notes", type=str, default="",
                        help="Optional freeform note to include in package")
    parser.add_argument("--known_unsupported_ops", type=str, nargs="*", default=[])
    parser.add_argument("--expected_fallback_ops", type=str, nargs="*", default=[])

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


def is_numeric_tensor_or_array(x):
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


def find_first_numeric_tensor(obj):
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


def collect_onnx_op_inventory(model_path):
    model = onnx.load(model_path)
    op_counts = {}
    node_rows = []

    for idx, node in enumerate(model.graph.node):
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        node_rows.append({
            "index": idx,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    return {
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: x[0])),
        "node_count": len(model.graph.node),
        "nodes": node_rows,
    }


def collect_onnx_qparams_and_tensor_metadata(model_path):
    model = onnx.load(model_path)
    qparams = []
    tensor_meta = []

    init_map = {init.name: init for init in model.graph.initializer}

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        tensor_meta.append({
            "name": init.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "summary": summarize_array(arr),
        })

    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
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
                scale_arr = numpy_helper.to_array(init_map[node.input[1]])
                entry["scale"] = {
                    "name": node.input[1],
                    "shape": list(scale_arr.shape),
                    "dtype": str(scale_arr.dtype),
                    "summary": summarize_array(scale_arr),
                }

            if len(node.input) >= 3 and node.input[2] in init_map:
                zp_arr = numpy_helper.to_array(init_map[node.input[2]])
                entry["zero_point"] = {
                    "name": node.input[2],
                    "shape": list(zp_arr.shape),
                    "dtype": str(zp_arr.dtype),
                    "summary": summarize_array(zp_arr),
                }

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
        })

    return {
        "fp32_node_count": fp32_inv["node_count"],
        "quant_node_count": quant_inv["node_count"],
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


def collect_sample_io(loader, model_obj, device, num_samples, out_dir):
    ensure_dir(out_dir)
    rows = []

    count = 0
    for batch_idx, batch in enumerate(loader):
        image = find_first_numeric_tensor(batch)
        if image is None:
            continue

        image_np = tensor_to_numpy(image).astype(np.float32)
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)

        outputs = run_single_backend(model_obj, image_np, device)

        input_path = os.path.join(out_dir, "sample_{:03d}_input.npy".format(count))
        np.save(input_path, image_np)

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
            "loader_batch_index": batch_idx,
            "input_path": os.path.basename(input_path),
            "input_summary": summarize_array(image_np),
            "reference_outputs": out_entries,
        })

        count += 1
        if count >= num_samples:
            break

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

        # raw state_dict
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

        # checkpoint with embedded state dict
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
    lines.append("## Accuracy")
    if accuracy_summary is not None:
        for k, v in accuracy_summary.items():
            lines.append("- {}: {}".format(k, v))
    else:
        lines.append("- FP32 baseline not provided, accuracy delta unavailable.")
    lines.append("")
    lines.append("## Known Unsupported Ops")
    if args.known_unsupported_ops:
        for op in args.known_unsupported_ops:
            lines.append("- {}".format(op))
    else:
        lines.append("- None explicitly declared")
    lines.append("")
    lines.append("## Expected Fallback Ops")
    if args.expected_fallback_ops:
        for op in args.expected_fallback_ops:
            lines.append("- {}".format(op))
    else:
        lines.append("- None explicitly declared")
    lines.append("")
    lines.append("## Quant Graph Summary")
    if isinstance(op_inventory, dict) and "node_count" in op_inventory:
        lines.append("- Total quant graph nodes: {}".format(op_inventory["node_count"]))
        lines.append("- Unique ops: {}".format(len(op_inventory["op_counts"])))
    else:
        lines.append("- Quant model is not ONNX, graph inventory unavailable.")
    lines.append("")
    lines.append("## Graph Differences vs FP32")
    if graph_diff is None:
        lines.append("- FP32 ONNX not provided")
    else:
        lines.append("- FP32 node count: {}".format(graph_diff["fp32_node_count"]))
        lines.append("- Quant node count: {}".format(graph_diff["quant_node_count"]))
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

    loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        split=args.split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    fp32_model_obj = None
    fp32_results = None
    quant_obj = None
    quant_results = None
    accuracy_summary = None

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

        # print("[INFO] Evaluating FP32 model...")
        # fp32_results = evaluate_model(
        #     model_obj=fp32_model_obj,
        #     model_category_const=fp32_category,
        #     loader=loader,
        #     device=args.device,
        #     max_samples=args.max_samples,
        # )

    print("[INFO] Loading quantized/exported model...")
    quant_obj = load_aimet_quantized_model(
        quant_weights=args.quant_model,
        model_category=args.model_category,
        device=args.device,
        provider=args.onnx_provider,
    )

    # print("[INFO] Evaluating quantized/exported model...")
    # quant_results = evaluate_model(
    #     model_obj=quant_obj,
    #     model_category_const=quant_obj["model_category_const"],
    #     loader=loader,
    #     device=args.device,
    #     max_samples=args.max_samples,
    # )

    # if fp32_results is not None:
    #     accuracy_summary = {
    #         "fp32_mIoU": fp32_results["mIoU"],
    #         "quant_mIoU": quant_results["mIoU"],
    #         "miou_drop": quant_results["mIoU"] - fp32_results["mIoU"],
    #         "fp32_fps": fp32_results["FPS"],
    #         "quant_fps": quant_results["FPS"],
    #         "speedup_x": (
    #             quant_results["FPS"] / fp32_results["FPS"]
    #             if fp32_results["FPS"] != 0 else None
    #         ),
    #         "fp32_latency_ms": fp32_results["Avg_Inference_Time_ms"],
    #         "quant_latency_ms": quant_results["Avg_Inference_Time_ms"],
    #         "latency_delta_ms": (
    #             quant_results["Avg_Inference_Time_ms"] - fp32_results["Avg_Inference_Time_ms"]
    #         ),
    #     }

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

    print("[INFO] Collecting sample inputs and reference outputs...")
    sample_io_index = collect_sample_io(
        loader=loader,
        model_obj=quant_obj,
        device=args.device,
        num_samples=args.num_sample_inputs,
        out_dir=io_dir,
    )

    save_json(quant_results, os.path.join(reports_dir, "quant_eval.json"))
    if fp32_results is not None:
        save_json(fp32_results, os.path.join(reports_dir, "fp32_eval.json"))
    if accuracy_summary is not None:
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
        "evaluation": {
            "split": args.split,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "onnx_provider": args.onnx_provider,
            "quant_results_file": "reports/quant_eval.json",
            "fp32_results_file": "reports/fp32_eval.json" if fp32_results is not None else None,
            "accuracy_compare_file": "reports/accuracy_compare.json" if accuracy_summary is not None else None,
        },
        "packaged_assets": {
            "sample_io_dir": "assets/sample_io",
            "sample_io_index": "notes/sample_io_index.json",
            "quant_assets": "notes/quant_assets.json",
            "op_inventory": "notes/op_inventory.json",
            "graph_diff": "notes/graph_diff.json" if graph_diff is not None else None,
            "checkpoint_inspection": "notes/quant_checkpoint_inspection.json" if checkpoint_inspection is not None else None,
            "state_dict_info": "notes/quant_state_dict_info.json" if extracted_state_dict_info is not None else None,
        },
        "notes": {
            "known_unsupported_ops": args.known_unsupported_ops,
            "expected_fallback_ops": args.expected_fallback_ops,
            "freeform_notes": args.notes,
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