#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


class QuantizedOnnxExtractor:
    WEIGHT_Q_SUFFIX = ".weight_q"
    BIAS_Q_SUFFIX = ".bias_q"

    PASSTHROUGH_OPS = {
        "Transpose",
        "Reshape",
        "Identity",
        "Cast",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
    }

    COMPUTE_OPS = {"Conv", "MatMul", "Gemm"}

    def __init__(self, ckpt_path):
        self.ckpt_path = str(ckpt_path)
        self.onnx_model = onnx.load(self.ckpt_path)
        self.initializers = self._build_initializer_map()
        self.producer, self.consumers = self._build_graph_maps()

    @staticmethod
    def _to_torch_tensor(initializer):
        return torch.from_numpy(numpy_helper.to_array(initializer).copy())

    def _build_initializer_map(self):
        return {
            initializer.name: initializer
            for initializer in self.onnx_model.graph.initializer
        }

    @classmethod
    def _normalize_prefix(self, name):
        # remove repeated wrappers like model.model.model...
        while name.startswith("model."):
            name = name[len("model.") :]

        while name.startswith("module."):
            name = name[len("module.") :]

        while name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]

        replacements = [
            ("patch_embeddings.", "patch_embed."),
            ("weight_zeropoint", "weight_zero_point"),
            ("bias_zeropoint", "bias_zero_point"),
            ("input_zeropoint", "input_zero_point"),
            ("output_zeropoint", "output_zero_point"),
        ]

        for src, dst in replacements:
            name = name.replace(src, dst)

        return name

    def _build_graph_maps(self):
        producer = {}
        consumers = defaultdict(list)

        for node in self.onnx_model.graph.node:
            for output_name in node.output:
                producer[output_name] = node
            for input_name in node.input:
                consumers[input_name].append(node)

        return producer, consumers

    def _find_quantized_prefixes(self):
        return sorted(
            initializer.name[: -len(self.WEIGHT_Q_SUFFIX)]
            for initializer in self.onnx_model.graph.initializer
            if initializer.name.endswith(self.WEIGHT_Q_SUFFIX)
        )

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        queue = [weight_tensor_name]
        seen_tensors = set()

        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)

            for consumer in self.consumers.get(tensor_name, []):
                if consumer.op_type in self.COMPUTE_OPS:
                    return consumer
                if consumer.op_type in self.PASSTHROUGH_OPS:
                    queue.extend(consumer.output)

        return None

    @staticmethod
    def _find_activation_input_name(compute_node, weight_prefix):
        weight_roots = {
            f"{weight_prefix}.weight_qdq",
            f"{weight_prefix}.weight_q",
            f"{weight_prefix}.weight",
        }

        for input_name in compute_node.input:
            if any(input_name.startswith(root) for root in weight_roots):
                continue
            if f"{weight_prefix}.weight" in input_name:
                continue
            return input_name

        if compute_node.op_type == "Conv":
            return compute_node.input[0]
        if len(compute_node.input) >= 2:
            return compute_node.input[0]
        return None

    def _extract_qparams_from_dq_tensor(self, tensor_name):
        dq_node = self.producer.get(tensor_name)
        if dq_node is None or dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 3:
            return None

        scale_name = dq_node.input[1]
        zero_point_name = dq_node.input[2]

        if scale_name not in self.initializers or zero_point_name not in self.initializers:
            return None

        return {
            "scale": self._to_torch_tensor(self.initializers[scale_name]),
            "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
        }

    def _find_output_quant_params(self, prefix, compute_node):
        output_tensor_name = compute_node.output[0]

        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor_name, []):
                if consumer.op_type != "Add":
                    continue
                if any(input_name == f"{prefix}.bias_qdq" for input_name in consumer.input):
                    output_tensor_name = consumer.output[0]
                    break

        for consumer in self.consumers.get(output_tensor_name, []):
            if consumer.op_type != "QuantizeLinear" or len(consumer.input) < 3:
                continue

            scale_name = consumer.input[1]
            zero_point_name = consumer.input[2]

            if scale_name in self.initializers and zero_point_name in self.initializers:
                return {
                    "scale": self._to_torch_tensor(self.initializers[scale_name]),
                    "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
                }

        return None

    def collect_quantized_layer_state(self):
        state_dict = {}
        missing_input_qparams = []
        missing_output_qparams = []

        for prefix in self._find_quantized_prefixes():
            export_prefix = self._normalize_prefix(prefix)

            weight_name = f"{prefix}.weight_q"
            weight_scale_name = f"{prefix}.weight_scale"
            weight_zero_point_name = f"{prefix}.weight_zero_point"

            state_dict[f"{export_prefix}.weight"] = self._to_torch_tensor(self.initializers[weight_name])
            state_dict[f"{export_prefix}.weight_scale"] = self._to_torch_tensor(self.initializers[weight_scale_name])
            state_dict[f"{export_prefix}.weight_zeropoint"] = self._to_torch_tensor(self.initializers[weight_zero_point_name])

            bias_name = f"{prefix}.bias_q"
            bias_scale_name = f"{prefix}.bias_scale"
            bias_zero_point_name = f"{prefix}.bias_zero_point"

            if bias_name in self.initializers:
                state_dict[f"{export_prefix}.bias"] = self._to_torch_tensor(self.initializers[bias_name])
            if bias_scale_name in self.initializers:
                state_dict[f"{export_prefix}.bias_scale"] = self._to_torch_tensor(self.initializers[bias_scale_name])
            if bias_zero_point_name in self.initializers:
                state_dict[f"{export_prefix}.bias_zeropoint"] = self._to_torch_tensor(self.initializers[bias_zero_point_name])

            compute_node = self._find_compute_node_from_weight_qdq(f"{prefix}.weight_qdq")
            if compute_node is None:
                missing_input_qparams.append(prefix)
                missing_output_qparams.append(prefix)
                continue

            activation_input_name = self._find_activation_input_name(compute_node, prefix)

            input_qparams = None
            if activation_input_name is not None:
                input_qparams = self._extract_qparams_from_dq_tensor(activation_input_name)

            if input_qparams is not None:
                state_dict[f"{export_prefix}.input_scale"] = input_qparams["scale"]
                state_dict[f"{export_prefix}.input_zeropoint"] = input_qparams["zeropoint"]
            else:
                missing_input_qparams.append(prefix)

            output_qparams = self._find_output_quant_params(prefix, compute_node)
            if output_qparams is not None:
                state_dict[f"{export_prefix}.output_scale"] = output_qparams["scale"]
                state_dict[f"{export_prefix}.output_zeropoint"] = output_qparams["zeropoint"]
            else:
                missing_output_qparams.append(prefix)

        return state_dict, missing_input_qparams, missing_output_qparams

    def save(self, output_path):
        state_dict, missing_input_qparams, missing_output_qparams = self.collect_quantized_layer_state()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, output_path)

        print(f"Saved {len(state_dict)} tensors to {output_path}")
        print(f"Layers without explicit input activation qparams: {len(missing_input_qparams)}")
        for prefix in missing_input_qparams:
            print(f"  - {prefix}")
        print(f"Layers without explicit output activation qparams: {len(missing_output_qparams)}")
        for prefix in missing_output_qparams:
            print(f"  - {prefix}")

        return state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Extract quantized tensors and activation qparams from a QDQ ONNX model."
    )
    parser.add_argument("ckpt_path", help="Path to the quantized ONNX model")
    parser.add_argument("output_path", help="Path to save the extracted torch state_dict")
    args = parser.parse_args()

    extractor = QuantizedOnnxExtractor(args.ckpt_path)
    extractor.save(args.output_path)


if __name__ == "__main__":
    main()