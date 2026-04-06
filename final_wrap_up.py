#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import onnx
from onnx import helper, TensorProto, shape_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX QOperator model to simple QDQ-style model."
    )
    parser.add_argument("--input", type=str, required=True, help="Input QOperator ONNX")
    parser.add_argument("--output", type=str, required=True, help="Output QDQ ONNX")
    return parser.parse_args()


def make_dq_node(x: str, scale: str, zp: str, y: str, name: str):
    return helper.make_node(
        "DequantizeLinear",
        inputs=[x, scale, zp],
        outputs=[y],
        name=name,
    )


def make_q_node(x: str, scale: str, zp: str, y: str, name: str):
    return helper.make_node(
        "QuantizeLinear",
        inputs=[x, scale, zp],
        outputs=[y],
        name=name,
    )


def get_attr(node, name: str):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def copy_conv_attrs(node) -> List:
    keep = {
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "pads",
        "strides",
    }
    return [attr for attr in node.attribute if attr.name in keep]


def convert_qlinearconv(node: onnx.NodeProto, new_nodes: List[onnx.NodeProto]) -> bool:
    # QLinearConv schema:
    # x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp [, bias]
    if len(node.input) not in (8, 9):
        print(f"[WARN] Skip {node.name}: unexpected QLinearConv input count = {len(node.input)}")
        return False

    x, x_scale, x_zp = node.input[0], node.input[1], node.input[2]
    w, w_scale, w_zp = node.input[3], node.input[4], node.input[5]
    y_scale, y_zp = node.input[6], node.input[7]
    bias = node.input[8] if len(node.input) == 9 else None

    y = node.output[0]

    x_dq = f"{node.name}_x_dq"
    w_dq = f"{node.name}_w_dq"
    conv_out = f"{node.name}_conv_out"
    q_out = y

    new_nodes.append(make_dq_node(x, x_scale, x_zp, x_dq, f"{node.name}_DequantizeInput"))
    new_nodes.append(make_dq_node(w, w_scale, w_zp, w_dq, f"{node.name}_DequantizeWeight"))

    conv_inputs = [x_dq, w_dq]
    if bias is not None:
        # Simple version: reuse bias as-is.
        # In some models bias may need special handling depending on how it was stored.
        conv_inputs.append(bias)

    new_nodes.append(
        helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_out],
            name=f"{node.name}_Conv",
            **{
                attr.name: helper.get_attribute_value(attr)
                for attr in copy_conv_attrs(node)
            },
        )
    )

    new_nodes.append(make_q_node(conv_out, y_scale, y_zp, q_out, f"{node.name}_QuantizeOutput"))
    return True


def convert_qlinearmatmul(node: onnx.NodeProto, new_nodes: List[onnx.NodeProto]) -> bool:
    # QLinearMatMul schema:
    # a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
    if len(node.input) != 8:
        print(f"[WARN] Skip {node.name}: unexpected QLinearMatMul input count = {len(node.input)}")
        return False

    a, a_scale, a_zp = node.input[0], node.input[1], node.input[2]
    b, b_scale, b_zp = node.input[3], node.input[4], node.input[5]
    y_scale, y_zp = node.input[6], node.input[7]
    y = node.output[0]

    a_dq = f"{node.name}_a_dq"
    b_dq = f"{node.name}_b_dq"
    mm_out = f"{node.name}_matmul_out"

    new_nodes.append(make_dq_node(a, a_scale, a_zp, a_dq, f"{node.name}_DequantizeA"))
    new_nodes.append(make_dq_node(b, b_scale, b_zp, b_dq, f"{node.name}_DequantizeB"))

    new_nodes.append(
        helper.make_node(
            "MatMul",
            inputs=[a_dq, b_dq],
            outputs=[mm_out],
            name=f"{node.name}_MatMul",
        )
    )

    new_nodes.append(make_q_node(mm_out, y_scale, y_zp, y, f"{node.name}_QuantizeOutput"))
    return True


def convert_model(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    new_nodes: List[onnx.NodeProto] = []

    converted = 0
    skipped = 0

    for node in graph.node:
        ok = False

        if node.op_type == "QLinearConv":
            ok = convert_qlinearconv(node, new_nodes)
        elif node.op_type == "QLinearMatMul":
            ok = convert_qlinearmatmul(node, new_nodes)
        else:
            new_nodes.append(node)
            continue

        if ok:
            converted += 1
            print(f"[INFO] Converted {node.op_type}: {node.name}")
        else:
            skipped += 1
            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info,
    )

    new_model = helper.make_model(
        new_graph,
        producer_name="qoperator_to_qdq_simple",
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )

    # Preserve metadata if present
    new_model.metadata_props.extend(model.metadata_props)

    print(f"[INFO] Converted nodes: {converted}")
    print(f"[INFO] Skipped nodes  : {skipped}")
    return new_model


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input model not found: {args.input}")

    print(f"[INFO] Loading: {args.input}")
    model = onnx.load(args.input)

    print("[INFO] Converting QOperator -> QDQ...")
    model_qdq = convert_model(model)

    try:
        print("[INFO] Running shape inference...")
        model_qdq = shape_inference.infer_shapes(model_qdq)
    except Exception as e:
        print(f"[WARN] Shape inference failed: {e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    onnx.save(model_qdq, args.output)
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()