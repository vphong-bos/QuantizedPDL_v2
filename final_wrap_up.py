#!/usr/bin/env python3
import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ONNX QOperator model to QDQ model by graph rewriting."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--infer_shapes", action="store_true")
    parser.add_argument("--fail_if_unsupported", action="store_true")
    return parser.parse_args()


def sanitize_name(name, fallback):
    if name:
        return name.replace("/", "_").replace(":", "_")
    return fallback


def get_initializer_map(model):
    return {init.name: init for init in model.graph.initializer}


def get_tensor_numpy(initializer_map, name):
    if name not in initializer_map:
        raise KeyError("Initializer not found: {}".format(name))
    return numpy_helper.to_array(initializer_map[name])


def add_initializer_if_missing(initializers_out, emitted_names, tensor):
    if tensor.name not in emitted_names:
        initializers_out.append(tensor)
        emitted_names.add(tensor.name)


def make_dq_node(x, scale, zero_point, y, name, axis=None):
    kwargs = {}
    if axis is not None:
        kwargs["axis"] = int(axis)
    return helper.make_node(
        "DequantizeLinear",
        inputs=[x, scale, zero_point],
        outputs=[y],
        name=name,
        **kwargs
    )


def make_q_node(x, scale, zero_point, y, name, axis=None):
    kwargs = {}
    if axis is not None:
        kwargs["axis"] = int(axis)
    return helper.make_node(
        "QuantizeLinear",
        inputs=[x, scale, zero_point],
        outputs=[y],
        name=name,
        **kwargs
    )


def copy_selected_attributes(node, names):
    selected = {}
    for attr in node.attribute:
        if attr.name in names:
            selected[attr.name] = helper.get_attribute_value(attr)
    return selected


def collect_model_op_counts(model):
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return op_counts


def print_onnx_op_counts(title, op_counts):
    print("[INFO] ONNX stats for: {}".format(title))
    interesting_ops = [
        "Conv",
        "MatMul",
        "Add",
        "Mul",
        "QuantizeLinear",
        "DequantizeLinear",
        "QLinearConv",
        "QLinearMatMul",
        "QLinearAdd",
        "QLinearMul",
        "ConvInteger",
        "MatMulInteger",
    ]
    for op_name in interesting_ops:
        print("  {:<16}: {}".format(op_name, op_counts.get(op_name, 0)))


def get_per_axis_axis_if_any(initializer_map, scale_name, default_axis=None):
    """
    Returns:
      - None if scale is scalar / per-tensor
      - int axis if scale is 1-D / per-axis
    """
    if scale_name not in initializer_map:
        return default_axis

    scale = get_tensor_numpy(initializer_map, scale_name)
    scale = np.asarray(scale)

    if scale.ndim == 0:
        return None
    if scale.ndim == 1:
        return default_axis

    # This script does not handle blocked quantization.
    raise ValueError(
        "Scale {} has rank {}. Only scalar or 1-D scales are supported.".format(
            scale_name, scale.ndim
        )
    )


def make_bias_float_initializer(node_name, bias_name, x_scale_name, w_scale_name, initializer_map):
    bias_i32 = get_tensor_numpy(initializer_map, bias_name)
    x_scale = get_tensor_numpy(initializer_map, x_scale_name)
    w_scale = get_tensor_numpy(initializer_map, w_scale_name)

    x_scale = np.asarray(x_scale, dtype=np.float32)
    w_scale = np.asarray(w_scale, dtype=np.float32)
    bias_i32 = np.asarray(bias_i32)

    if x_scale.size != 1:
        raise ValueError(
            "Expected scalar x_scale for bias conversion, got shape {} for {}".format(
                x_scale.shape, x_scale_name
            )
        )

    x_scale_scalar = np.float32(x_scale.reshape(()))
    bias_scale = w_scale.astype(np.float32) * x_scale_scalar

    bias_float = bias_i32.astype(np.float32)

    if bias_scale.ndim == 0:
        bias_float = bias_float * np.float32(bias_scale.reshape(()))
    else:
        bias_float = bias_float * bias_scale.reshape(bias_float.shape)

    out_name = "{}_bias_float".format(node_name)
    return numpy_helper.from_array(bias_float.astype(np.float32), name=out_name)


def convert_qlinearconv(node, initializer_map, new_nodes, new_initializers, emitted_initializer_names):
    # QLinearConv:
    # x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, bias]
    if len(node.input) not in (8, 9):
        print("[WARN] Skip {}: unexpected QLinearConv input count = {}".format(node.name, len(node.input)))
        return False

    node_name = sanitize_name(node.name, "QLinearConv")
    x = node.input[0]
    x_scale = node.input[1]
    x_zp = node.input[2]
    w = node.input[3]
    w_scale = node.input[4]
    w_zp = node.input[5]
    y_scale = node.input[6]
    y_zp = node.input[7]
    bias_name = node.input[8] if len(node.input) == 9 else None
    y = node.output[0]

    x_dq = "{}_x_dq".format(node_name)
    w_dq = "{}_w_dq".format(node_name)
    conv_out = "{}_conv_out".format(node_name)

    # Activation is usually per-tensor, so no axis unless x_scale is 1-D.
    x_axis = get_per_axis_axis_if_any(initializer_map, x_scale, default_axis=1)

    # Conv weight per-channel quantization is typically along output-channel axis 0.
    w_axis = get_per_axis_axis_if_any(initializer_map, w_scale, default_axis=0)

    # Output quant is usually per-tensor. If 1-D appears, axis=1 is a generic activation default.
    y_axis = get_per_axis_axis_if_any(initializer_map, y_scale, default_axis=1)

    new_nodes.append(
        make_dq_node(x, x_scale, x_zp, x_dq, "{}_DequantizeInput".format(node_name), axis=x_axis)
    )
    new_nodes.append(
        make_dq_node(w, w_scale, w_zp, w_dq, "{}_DequantizeWeight".format(node_name), axis=w_axis)
    )

    conv_inputs = [x_dq, w_dq]

    if bias_name:
        if bias_name not in initializer_map:
            raise ValueError(
                "QLinearConv bias {} is not an initializer. This script expects constant bias.".format(
                    bias_name
                )
            )

        bias_float_init = make_bias_float_initializer(
            node_name=node_name,
            bias_name=bias_name,
            x_scale_name=x_scale,
            w_scale_name=w_scale,
            initializer_map=initializer_map,
        )
        add_initializer_if_missing(new_initializers, emitted_initializer_names, bias_float_init)
        conv_inputs.append(bias_float_init.name)

    conv_attrs = copy_selected_attributes(
        node,
        ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"],
    )

    new_nodes.append(
        helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            name="{}_Conv".format(node_name),
            **conv_attrs
        )
    )

    new_nodes.append(
        make_q_node(conv_out, y_scale, y_zp, y, "{}_QuantizeOutput".format(node_name), axis=y_axis)
    )
    return True


def convert_qlinearmatmul(node, initializer_map, new_nodes):
    # QLinearMatMul:
    # a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
    if len(node.input) != 8:
        print("[WARN] Skip {}: unexpected QLinearMatMul input count = {}".format(node.name, len(node.input)))
        return False

    node_name = sanitize_name(node.name, "QLinearMatMul")
    a = node.input[0]
    a_scale = node.input[1]
    a_zp = node.input[2]
    b = node.input[3]
    b_scale = node.input[4]
    b_zp = node.input[5]
    y_scale = node.input[6]
    y_zp = node.input[7]
    y = node.output[0]

    a_dq = "{}_a_dq".format(node_name)
    b_dq = "{}_b_dq".format(node_name)
    mm_out = "{}_matmul_out".format(node_name)

    a_axis = get_per_axis_axis_if_any(initializer_map, a_scale, default_axis=1)
    b_axis = get_per_axis_axis_if_any(initializer_map, b_scale, default_axis=0)
    y_axis = get_per_axis_axis_if_any(initializer_map, y_scale, default_axis=1)

    new_nodes.append(
        make_dq_node(a, a_scale, a_zp, a_dq, "{}_DequantizeA".format(node_name), axis=a_axis)
    )
    new_nodes.append(
        make_dq_node(b, b_scale, b_zp, b_dq, "{}_DequantizeB".format(node_name), axis=b_axis)
    )

    new_nodes.append(
        helper.make_node(
            "MatMul",
            inputs=[a_dq, b_dq],
            outputs=[mm_out],
            name="{}_MatMul".format(node_name),
        )
    )

    new_nodes.append(
        make_q_node(mm_out, y_scale, y_zp, y, "{}_QuantizeOutput".format(node_name), axis=y_axis)
    )
    return True


def convert_qlinearadd(node, initializer_map, new_nodes):
    if len(node.input) != 8:
        print("[WARN] Skip {}: unexpected QLinearAdd input count = {}".format(node.name, len(node.input)))
        return False

    node_name = sanitize_name(node.name, "QLinearAdd")
    a = node.input[0]
    a_scale = node.input[1]
    a_zp = node.input[2]
    b = node.input[3]
    b_scale = node.input[4]
    b_zp = node.input[5]
    y_scale = node.input[6]
    y_zp = node.input[7]
    y = node.output[0]

    a_dq = "{}_a_dq".format(node_name)
    b_dq = "{}_b_dq".format(node_name)
    add_out = "{}_add_out".format(node_name)

    a_axis = get_per_axis_axis_if_any(initializer_map, a_scale, default_axis=1)
    b_axis = get_per_axis_axis_if_any(initializer_map, b_scale, default_axis=1)
    y_axis = get_per_axis_axis_if_any(initializer_map, y_scale, default_axis=1)

    new_nodes.append(make_dq_node(a, a_scale, a_zp, a_dq, "{}_DequantizeA".format(node_name), axis=a_axis))
    new_nodes.append(make_dq_node(b, b_scale, b_zp, b_dq, "{}_DequantizeB".format(node_name), axis=b_axis))
    new_nodes.append(
        helper.make_node(
            "Add",
            inputs=[a_dq, b_dq],
            outputs=[add_out],
            name="{}_Add".format(node_name),
        )
    )
    new_nodes.append(make_q_node(add_out, y_scale, y_zp, y, "{}_QuantizeOutput".format(node_name), axis=y_axis))
    return True


def convert_model(model, fail_if_unsupported=False):
    initializer_map = get_initializer_map(model)

    new_initializers = []
    emitted_initializer_names = set()

    for init in model.graph.initializer:
        add_initializer_if_missing(new_initializers, emitted_initializer_names, init)

    new_nodes = []
    unsupported_quantized_ops = []

    for node in model.graph.node:
        converted = False

        if node.op_type == "QLinearConv":
            converted = convert_qlinearconv(
                node=node,
                initializer_map=initializer_map,
                new_nodes=new_nodes,
                new_initializers=new_initializers,
                emitted_initializer_names=emitted_initializer_names,
            )
        elif node.op_type == "QLinearMatMul":
            converted = convert_qlinearmatmul(
                node=node,
                initializer_map=initializer_map,
                new_nodes=new_nodes,
            )
        elif node.op_type == "QLinearAdd":
            converted = convert_qlinearadd(
                node=node,
                initializer_map=initializer_map,
                new_nodes=new_nodes,
            )
        else:
            if node.op_type.startswith("QLinear") or node.op_type in ("ConvInteger", "MatMulInteger"):
                unsupported_quantized_ops.append("{}:{}".format(node.op_type, node.name))
            new_nodes.append(node)
            continue

        if converted:
            print("[INFO] Converted {}: {}".format(node.op_type, node.name))
        else:
            new_nodes.append(node)
            unsupported_quantized_ops.append("{}:{}".format(node.op_type, node.name))

    if unsupported_quantized_ops:
        print("[WARN] Unsupported quantized operators found:")
        for item in unsupported_quantized_ops:
            print("  - {}".format(item))
        if fail_if_unsupported:
            raise RuntimeError("Unsupported quantized operators found during conversion.")

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=new_initializers,
        value_info=model.graph.value_info,
    )

    new_model = helper.make_model(
        new_graph,
        producer_name="qoperator_to_qdq_converter",
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    del new_model.metadata_props[:]
    new_model.metadata_props.extend(model.metadata_props)

    return new_model, unsupported_quantized_ops


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError("Input model not found: {}".format(args.input))

    print("[INFO] Loading input ONNX: {}".format(args.input))
    model = onnx.load(args.input)

    input_counts = collect_model_op_counts(model)
    print_onnx_op_counts(args.input, input_counts)

    print("[INFO] Converting QOperator -> QDQ ...")
    converted_model, unsupported_quantized_ops = convert_model(
        model=model,
        fail_if_unsupported=args.fail_if_unsupported,
    )

    if args.infer_shapes:
        try:
            print("[INFO] Running ONNX shape inference...")
            converted_model = shape_inference.infer_shapes(converted_model)
        except Exception as e:
            print("[WARN] Shape inference failed: {}".format(e))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    onnx.save(converted_model, args.output)
    print("[INFO] Saved converted ONNX to: {}".format(args.output))

    output_model = onnx.load(args.output)
    output_counts = collect_model_op_counts(output_model)
    print_onnx_op_counts(args.output, output_counts)

    if output_counts.get("QLinearConv", 0) == 0 and output_counts.get("QLinearMatMul", 0) == 0:
        print("[INFO] Verified: QLinearConv and QLinearMatMul were removed from output.")

    if unsupported_quantized_ops:
        print("[WARN] Output may still contain other quantized operator forms not handled by this script.")


if __name__ == "__main__":
    main()