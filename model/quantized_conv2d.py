import torch
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from model.conv2d import Conv2d


@QuantizationMixin.implements(Conv2d)
class QuantizedConv2d(QuantizationMixin, Conv2d):
    def __quant_init__(self):
        super().__quant_init__()

        # One input, one output for Conv2d
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

        if not hasattr(self, "param_quantizers") or self.param_quantizers is None:
            raise RuntimeError("AIMET did not initialize param_quantizers")

        # Disable bias quantization
        if "bias" in self.param_quantizers:
            self.param_quantizers["bias"] = None

        if "weight" not in self.param_quantizers:
            raise RuntimeError(
                f"weight quantizer missing. keys={list(self.param_quantizers.keys())}"
            )

        # Conv2d weight is [out_ch, in_ch, kH, kW]
        if self.weight.ndim != 4:
            raise RuntimeError(
                f"Expected 4D conv weight, got shape={tuple(self.weight.shape)}"
            )

        out_ch = int(self.weight.shape[0])

        # Replace the default weight quantizer with an explicit per-channel QDQ quantizer.
        # shape=(out_ch,1,1,1) means one encoding per output channel.
        self.param_quantizers["weight"] = QuantizeDequantize(
            shape=(out_ch, 1, 1, 1),
            bitwidth=8,
            symmetric=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0] is not None:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            x = super().forward(x)

        if self.output_quantizers[0] is not None:
            x = self.output_quantizers[0](x)

        return x