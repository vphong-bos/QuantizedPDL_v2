import torch
from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d

@QuantizationMixin.implements(Conv2d)
class QuantizedConv2d(QuantizationMixin, Conv2d):
    def __quant_init__(self):
        super().__quant_init__()

        # One input tensor, one output tensor
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activation
        if self.input_quantizers[0] is not None:
            x = self.input_quantizers[0](x)

        # Quantize weights/bias through AIMET context
        with self._patch_quantized_parameters():
            x = super().forward(x)

        # Quantize output activation
        if self.output_quantizers[0] is not None:
            x = self.output_quantizers[0](x)

        return x