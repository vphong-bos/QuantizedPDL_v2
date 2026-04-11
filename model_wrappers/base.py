import torch
import torch.nn as nn
import json
from typing import Dict, Any
from collections import OrderedDict
from pathlib import Path

class ActivationQuantWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        encoding_name: str,
        activation_encodings: dict,
        debug_activation_quant: bool = False,
    ):
        super().__init__()
        self.module = module
        self.encoding_name = encoding_name
        self.activation_encodings = activation_encodings
        self.debug_activation_quant = debug_activation_quant

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = object.__getattribute__(self, "_modules").get("module", None)
            if module is not None:
                return getattr(module, name)
            raise

    @staticmethod
    def _get_qrange(bitwidth: int = 8, signed: bool = True):
        if signed:
            return -(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1)) - 1
        return 0, (2 ** bitwidth) - 1

    def _debug_tensor(self, x: torch.Tensor):
        if not self.debug_activation_quant:
            return

        with torch.no_grad():
            shape = tuple(x.shape)
            xmin = float(x.min().item()) if x.numel() > 0 else 0.0
            xmax = float(x.max().item()) if x.numel() > 0 else 0.0

        print(
            f"[ACT-Q DEBUG] {self.encoding_name} "
            f"shape={shape} min={xmin:.6f} max={xmax:.6f}"
        )

    def quantize_activation(self, x: torch.Tensor):
        enc_info = self.activation_encodings.get(self.encoding_name)
        if enc_info is None:
            raise KeyError(f"Missing activation encoding: {self.encoding_name}")

        output_info = enc_info.get("output", {})
        if "0" not in output_info:
            raise KeyError(f"Missing output[0] encoding for: {self.encoding_name}")

        enc = output_info["0"]

        scale = float(enc["scale"])
        zero_point = int(enc["offset"])
        bitwidth = int(enc.get("bitwidth", 8))

        qmin, qmax = self._get_qrange(bitwidth, True)
        q = torch.round(x / scale) + zero_point
        q = torch.clamp(q, qmin, qmax)
        dq = (q - zero_point) * scale
        return dq

    def _quantize_output(self, out):
        if isinstance(out, torch.Tensor):
            self._debug_tensor(out)
            return self.quantize_activation(out)

        if isinstance(out, tuple):
            new_out = []
            for item in out:
                if isinstance(item, torch.Tensor):
                    self._debug_tensor(item)
                    new_out.append(self.quantize_activation(item))
                else:
                    new_out.append(item)
            return tuple(new_out)

        if isinstance(out, list):
            new_out = []
            for item in out:
                if isinstance(item, torch.Tensor):
                    self._debug_tensor(item)
                    new_out.append(self.quantize_activation(item))
                else:
                    new_out.append(item)
            return new_out

        return out

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return self._quantize_output(out)

class QuantModelWrapper(nn.Module):
    #TODO: Add support for loading and converting from various checkpoint formats (e.g., ONNX, etc.)
    def __init__(self, model, encodings_path, debug_activation_quant):
        assert model, "Model cannot be empty."
        assert encodings_path, "Encodings path cannot be empty."

        super(QuantModelWrapper, self).__init__()
        self.model = model
        self.encodings_path = str(encodings_path)

        self.encodings = None
        self.param_encodings = None
        self.activation_encodings = None
        
        self.debug_activation_quant = debug_activation_quant

        self._load_encodings()
        print(f"Loaded encodings from: {self.encodings_path}")

    # -------------------------
    # ENCODING ACCESS
    # -------------------------

    def get_param_key(self, limit: int = 25) -> str:
        param_keys = list(self.param_encodings.keys())
        return param_keys[:limit] if limit < len(param_keys) else param_keys

    def get_activation_keys(self, limit: int = 25) -> str:
        act_keys = list(self.activation_encodings.keys())
        return act_keys[:limit] if limit < len(act_keys) else act_keys

    def get_param_encoding(self, param_name: str):
        enc = self.param_encodings.get(param_name)
        if enc is None:
            raise KeyError(f"No encoding found for param: {param_name}")

        if isinstance(enc, dict):
            return enc

        if isinstance(enc, list) and len(enc) > 0 and all(isinstance(item, dict) for item in enc):
            return enc

        raise ValueError(f"Invalid encoding for param: {param_name}")

    def get_activation_encoding(self, act_name: str) -> Dict[str, Any]:
        enc = self.activation_encodings.get(act_name)
        if enc is None:
            raise KeyError(f"No encoding found for activation: {act_name}")
        if not isinstance(enc, dict):
            raise ValueError(f"Invalid encoding for activation: {act_name}")
        return enc

    # -------------------------
    # ENCODING LOADING
    # -------------------------

    def _load_encodings(self):
        suffix = Path(self.encodings_path).suffix.lower()
        if suffix in [".json", ".encodings", ".txt"]:
            with open(self.encodings_path, "r") as f:
                self.encodings = json.load(f)
                self.param_encodings = self.encodings.get("param_encodings", {})
                self.activation_encodings = self.encodings.get("activation_encodings", {})
        elif suffix in [".pt", ".pth"]:
            print("See you later, alligator! (Unsupported pytorch state dict quant format yet)")
            pass #TODO: Adding support for pt state dict
        else:
            raise ValueError(f"Unsupported encodings format: {suffix}")

    # -------------------------
    # QUANTIZATION LOGIC
    # -------------------------

    @staticmethod
    def _get_qrange(bitwidth: int = 8, signed: bool = True):
        if signed:
            return -(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1)) - 1, torch.int8
        return 0, (2 ** bitwidth) - 1, torch.uint8

    @staticmethod
    def quantize_tensor(
        x: torch.Tensor,
        scale: float,
        zero_point: int,
        bitwidth: int = 8,
        signed: bool = True,
    ) -> torch.Tensor:
        qmin, qmax, dtype = QuantModelWrapper._get_qrange(bitwidth, signed)
        q = torch.round(x / scale) + zero_point
        q = torch.clamp(q, qmin, qmax)
        return q.to(dtype)

    # -------------------------
    # PARAMETER QUANTIZATION
    # -------------------------

    def _map_per_tensor_qparams(self, x: torch.Tensor, enc: Dict[str, Any]):
        return self.quantize_tensor(
            x=x,
            scale=float(enc["scale"]),
            zero_point=int(enc["offset"]),
            bitwidth=int(enc.get("bitwidth", 8)),
            signed=True,
        )
    def _map_per_channel_qparams(self, x: torch.Tensor, enc_list: list):
        q_slices = []
        for ch, enc in enumerate(enc_list):
            q_slices.append(
                self.quantize_tensor(
                    x=x[ch],
                    scale=float(enc["scale"]),
                    zero_point=int(enc["offset"]),
                    bitwidth=int(enc.get("bitwidth", 8)),
                    signed=True,
                )
            )
        return torch.stack(q_slices, dim=0)

    def quantize_params(self) -> OrderedDict:
        q_state = OrderedDict()

        state_dict = self.model.state_dict() if hasattr(self.model, "state_dict") else self.model

        for name, tensor in state_dict.items():
            enc = self.param_encodings.get(name)

            # no encoding -> keep original
            if enc is None:
                q_state[name] = tensor
                continue

            try:
                # per-tensor encoding: single dict
                if isinstance(enc, dict):
                    q_state[name] = self._map_per_tensor_qparams(tensor, enc)
                    print(f"[INFO] Added QParams to: {name}")
                elif isinstance(enc, list) and len(enc) > 0 and all(isinstance(item, dict) for item in enc):
                    if tensor.shape[0] != len(enc):
                        print(
                            f"[WARN] Channel mismatch for {name}: "
                            f"tensor.shape[0]={tensor.shape[0]} vs encodings={len(enc)}. "
                            f"Keeping original tensor."
                        )
                        q_state[name] = tensor
                    else:
                        q_state[name] = self._map_per_channel_qparams(tensor, enc)

                    print(f"[INFO] Added QParams to: {name}")
                else:
                    print(f"[WARN] Skipping unsupported param encoding for: {name}")
                    q_state[name] = tensor

            except Exception as e:
                print(f"[WARN] Failed to quantize param {name}: {e}. Keeping original tensor.")
                q_state[name] = tensor

        return q_state

    # -------------------------
    # ACTIVATION QUANTIZATION
    # -------------------------

    def quantize_activation(self, x, name):
        enc_list = self.get_activation_encoding(name)
        if enc_list is None:
            raise KeyError(f"Missing activation encoding: {name}")

        enc = enc_list[0]

        return self.quantize_tensor(
            x,
            enc["scale"],
            enc["offset"],
            enc.get("bitwidth", 8),
            signed=True,
        )

        # -------------------------
    # MODULE WRAP HELPERS
    # -------------------------

    @staticmethod
    def _normalize_encoding_name(name: str) -> str:
        return name[6:] if name.startswith("model.") else name

    @staticmethod
    def _get_parent_module(root: nn.Module, module_name: str):
        parts = module_name.split(".")
        parent = root
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        return parent, parts[-1]

    @staticmethod
    def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
        parent, last = QuantModelWrapper._get_parent_module(root, module_name)
        if last.isdigit():
            parent[int(last)] = new_module
        else:
            setattr(parent, last, new_module)

    def add_activation_quant_wrapper(self):
        module_dict = dict(self.model.named_modules())

        for encoding_name in self.activation_encodings.keys():
            module_name = self._normalize_encoding_name(encoding_name)

            if module_name not in module_dict:
                print(f"[WARN] Module not found for activation encoding: {encoding_name}")
                continue

            original_module = module_dict[module_name]
            wrapped_module = ActivationQuantWrapper(
                module=original_module,
                encoding_name=encoding_name,
                activation_encodings=self.activation_encodings
                # quant_model_wrapper=self,
            )

            self._replace_module(self.model, module_name, wrapped_module)
            module_dict = dict(self.model.named_modules())

            print(f"[INFO] Added ActivationQuantWrapper to: {module_name}")


        
    # -------------------------
    # DEBUG
    # -------------------------
