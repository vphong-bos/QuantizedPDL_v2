"""Generic QDQ ONNX extractor producing AIMET-compatible encodings."""

from collections import defaultdict
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


class QuantizedOnnxExtractor:
    WEIGHT_Q_SUFFIX = ".weight_q"
    BIAS_Q_SUFFIX = ".bias_q"

    PASSTHROUGH_OPS = {
        "Transpose", "Reshape", "Identity", "Cast",
        "Flatten", "Squeeze", "Unsqueeze",
    }

    COMPUTE_OPS = {"Conv", "MatMul", "Gemm"}

    def __init__(self, ckpt_path, compute_ops=None, passthrough_ops=None):
        self.ckpt_path = str(ckpt_path)
        self.onnx_model = onnx.load(self.ckpt_path)
        self.initializers = self._build_initializer_map()
        self.producer, self.consumers = self._build_graph_maps()
        self.compute_ops = set(compute_ops) if compute_ops is not None else self.COMPUTE_OPS
        self.passthrough_ops = set(passthrough_ops) if passthrough_ops is not None else self.PASSTHROUGH_OPS

    # ------------------------------------------------------------------
    # Tensor / graph helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_torch_tensor(initializer):
        return torch.from_numpy(numpy_helper.to_array(initializer).copy())

    def _build_initializer_map(self):
        return {
            init.name: init for init in self.onnx_model.graph.initializer
        }

    def _build_graph_maps(self):
        producer = {}
        consumers = defaultdict(list)
        for node in self.onnx_model.graph.node:
            for out in node.output:
                producer[out] = node
            for inp in node.input:
                consumers[inp].append(node)
        return producer, consumers

    # ------------------------------------------------------------------
    # Prefix discovery
    # ------------------------------------------------------------------

    def _find_quantized_prefixes(self):
        return sorted(
            init.name[: -len(self.WEIGHT_Q_SUFFIX)]
            for init in self.onnx_model.graph.initializer
            if init.name.endswith(self.WEIGHT_Q_SUFFIX)
        )

    @staticmethod
    def _normalize_prefix(name):
        """Strip common training wrappers from prefix."""
        for wrapper in ("model.", "module.", "_orig_mod."):
            while name.startswith(wrapper):
                name = name[len(wrapper):]
        return name

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        queue = [weight_tensor_name]
        seen = set()
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in self.compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    @staticmethod
    def _find_activation_input_name(compute_node, weight_prefix):
        weight_roots = {
            f"{weight_prefix}.weight_qdq",
            f"{weight_prefix}.weight_q",
            f"{weight_prefix}.weight",
        }
        for inp in compute_node.input:
            if any(inp.startswith(r) for r in weight_roots):
                continue
            return inp
        if compute_node.op_type == "Conv":
            return compute_node.input[0]
        if len(compute_node.input) >= 2:
            return compute_node.input[0]
        return None

    def _extract_qparams_from_dq_tensor(self, tensor_name):
        upstream_ops = {
            "Transpose", "Reshape", "Identity", "Cast", "Flatten",
            "Squeeze", "Unsqueeze", "GlobalAveragePool", "AveragePool",
            "MaxPool", "Relu", "Clip", "Concat"
        }
        visited = set()
        queue = [tensor_name]
        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            node = self.producer.get(name)
            if node is None:
                continue
            if node.op_type == "DequantizeLinear" and len(node.input) >= 3:
                s, zp = node.input[1], node.input[2]
                if s in self.initializers and zp in self.initializers:
                    return {
                        "scale": self._to_torch_tensor(self.initializers[s]),
                        "zeropoint": self._to_torch_tensor(self.initializers[zp]),
                    }
            if node.op_type in upstream_ops:
                queue.extend(node.input)
        return None

    def _find_output_quant_params(self, prefix, compute_node):
        downstream_ops = {
            "BatchNormalization", "Relu", "Clip", "Add", "Transpose",
            "Reshape", "Identity", "Cast", "Flatten", "Squeeze",
            "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool",
        }
        output_tensor = compute_node.output[0]
        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor, []):
                if consumer.op_type == "Add" and any(
                    inp == f"{prefix}.bias_qdq" for inp in consumer.input
                ):
                    output_tensor = consumer.output[0]
                    break
        visited = set()
        queue = [output_tensor]
        while queue:
            tensor = queue.pop(0)
            if tensor in visited:
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    s, zp = consumer.input[1], consumer.input[2]
                    if s in self.initializers and zp in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[s]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zp]),
                        }
                if consumer.op_type in downstream_ops:
                    queue.extend(consumer.output)
        return None

    # ------------------------------------------------------------------
    # ONNX node name → PyTorch module name
    # ------------------------------------------------------------------

    @staticmethod
    def _onnx_node_name_to_module(node_name):
        name = node_name.lstrip("/")
        parts = name.split("/")
        if len(parts) > 1:
            parts = parts[:-1]
        filtered = []
        for i, part in enumerate(parts):
            if i + 1 < len(parts) and parts[i + 1].startswith(part + "."):
                continue
            filtered.append(part)
        return ".".join(filtered)

    # ------------------------------------------------------------------
    # AIMET encoding conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_python(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        return value

    @staticmethod
    def _qparams_to_aimet_encoding(scale_tensor, zeropoint_tensor):
        scale = QuantizedOnnxExtractor._to_python(scale_tensor)
        zeropoint = QuantizedOnnxExtractor._to_python(zeropoint_tensor)

        def one_encoding(s, zp):
            zp, s = int(zp), float(s)
            if zp == 0:
                dtype, qmin, qmax, is_sym = "int", -128, 127, "True"
            elif zp == 128:
                dtype, qmin, qmax, is_sym = "uint", 0, 255, "True"
            else:
                dtype, qmin, qmax, is_sym = "uint", 0, 255, "False"
            return {
                "bitwidth": 8,
                "dtype": dtype,
                "is_symmetric": is_sym,
                "max": float((qmax - zp) * s),
                "min": float((qmin - zp) * s),
                "offset": -zp,
                "scale": s,
            }

        if isinstance(scale, list):
            if not isinstance(zeropoint, list):
                zeropoint = [zeropoint] * len(scale)
            return [one_encoding(s, zp) for s, zp in zip(scale, zeropoint)]
        return one_encoding(scale, zeropoint)

    # ------------------------------------------------------------------
    # Model-specific hooks (override in subclass)
    # ------------------------------------------------------------------

    def _get_activation_roles(self, export_prefix):
        """Return {'input'} / {'output'} / {'input','output'} / set().

        Default: extract both input and output for every quantized layer.
        Override in subclass to match AIMET QuantSim convention.
        """
        return {"input", "output"}

    def _collect_activation_only_encodings(self):
        """Collect encodings for activation-only ops (Relu, MaxPool, etc.).

        Default: extract output encoding for every Relu/MaxPool/Pool node
        that has a QuantizeLinear consumer.
        Override in subclass for model-specific naming/filtering.
        """
        encodings = {}
        act_ops = {"Relu", "MaxPool", "AveragePool", "GlobalAveragePool"}
        for node in self.onnx_model.graph.node:
            if node.op_type not in act_ops:
                continue
            module_name = self._onnx_node_name_to_module(node.name)
            layer_act = {}
            # input
            if node.input:
                qp = self._extract_qparams_from_dq_tensor(node.input[0])
                if qp:
                    layer_act["input"] = {
                        "0": self._qparams_to_aimet_encoding(qp["scale"], qp["zeropoint"])
                    }
            # output
            if node.output:
                for consumer in self.consumers.get(node.output[0], []):
                    if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                        s, zp = consumer.input[1], consumer.input[2]
                        if s in self.initializers and zp in self.initializers:
                            layer_act["output"] = {
                                "0": self._qparams_to_aimet_encoding(
                                    self._to_torch_tensor(self.initializers[s]),
                                    self._to_torch_tensor(self.initializers[zp]),
                                )
                            }
                            break
            if layer_act:
                encodings[module_name] = layer_act
        return encodings

    # ------------------------------------------------------------------
    # Main collection
    # ------------------------------------------------------------------

    def collect_aimet_encodings(self):
        activation_encodings = {}
        param_encodings = {}
        missing_input, missing_output = [], []

        for prefix in self._find_quantized_prefixes():
            export_prefix = self._normalize_prefix(prefix)

            # --- param encodings (weight only) ---
            ws = f"{prefix}.weight_scale"
            wzp = f"{prefix}.weight_zero_point"
            if ws in self.initializers and wzp in self.initializers:
                param_encodings[f"{export_prefix}.weight"] = self._qparams_to_aimet_encoding(
                    self._to_torch_tensor(self.initializers[ws]),
                    self._to_torch_tensor(self.initializers[wzp]),
                )

            # --- activation encodings ---
            roles = self._get_activation_roles(export_prefix)
            if not roles:
                continue

            compute_node = self._find_compute_node_from_weight_qdq(f"{prefix}.weight_qdq")
            if compute_node is None:
                if "input" in roles:
                    missing_input.append(prefix)
                if "output" in roles:
                    missing_output.append(prefix)
                continue

            layer_act = {}

            if "input" in roles:
                act_inp = self._find_activation_input_name(compute_node, prefix)
                qp = self._extract_qparams_from_dq_tensor(act_inp) if act_inp else None
                # print(f"Debug: prefix={prefix}, act_inp={act_inp}, has_qp={qp is not None}")
                if qp:
                    layer_act["input"] = {
                        "0": self._qparams_to_aimet_encoding(qp["scale"], qp["zeropoint"])
                    }
                else:
                    missing_input.append(prefix)

            if "output" in roles:
                qp = self._find_output_quant_params(prefix, compute_node)
                if qp:
                    layer_act["output"] = {
                        "0": self._qparams_to_aimet_encoding(qp["scale"], qp["zeropoint"])
                    }
                else:
                    missing_output.append(prefix)

            if layer_act:
                activation_encodings[export_prefix] = layer_act

        # Activation-only ops
        activation_encodings.update(self._collect_activation_only_encodings())

        return {
            "activation_encodings": activation_encodings,
            "excluded_layers": [],
            "param_encodings": param_encodings,
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 8,
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf_enhanced",
            },
            "version": "1.0.0",
        }, missing_input, missing_output

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    #TODO: Add support for saving in other formats (e.g., PyTorch state dict, etc.)
    def save(self, output_path):
        import json

        encodings, missing_in, missing_out = self.collect_aimet_encodings()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(encodings, f, indent=2)

        n_act = len(encodings["activation_encodings"])
        n_par = len(encodings["param_encodings"])
        print(f"Saved encodings to {output_path}  ({n_act} activation, {n_par} param)")
        if missing_in:
            print(f"Missing input qparams ({len(missing_in)}):")
            for p in missing_in:
                print(f"  - {p}")
        if missing_out:
            print(f"Missing output qparams ({len(missing_out)}):")
            for p in missing_out:
                print(f"  - {p}")

        return encodings
