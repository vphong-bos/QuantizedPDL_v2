import torch
from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d

@QuantizationMixin.implements(Conv2d)
class QuantizedConv2d(QuantizationMixin, Conv2d):
    """
    Quantized wrapper for custom Conv2d.

    Goal:
    - keep activation quantizers explicit
    - keep bias unquantized if config disables it
    - force weight quantizer toward per-channel on output-channel axis
    """

    def __quant_init__(self):
        super().__quant_init__()

        # One input tensor, one output tensor
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

        # Make sure param_quantizers exists
        if not hasattr(self, "param_quantizers") or self.param_quantizers is None:
            raise RuntimeError("AIMET did not initialize param_quantizers for QuantizedConv2d")

        # Bias: keep disabled to match your quantsim config
        if "bias" in self.param_quantizers:
            self.param_quantizers["bias"] = None

        # Weight: try to force per-channel along output-channel axis (axis 0 for Conv2d weights)
        wq = self.param_quantizers.get("weight", None)
        if wq is None:
            raise RuntimeError("AIMET did not create a weight quantizer for QuantizedConv2d")

        self._try_make_weight_quantizer_per_channel(wq)

    def _try_make_weight_quantizer_per_channel(self, wq):
        """
        Best-effort compatibility across AIMET versions.
        Conv2d weight layout is [out_channels, in_channels/groups, kH, kW],
        so per-channel should use axis 0.
        """
        changed = []

        # Common/simple flags some versions may expose
        if hasattr(wq, "per_channel"):
            try:
                wq.per_channel = True
                changed.append("per_channel=True")
            except Exception:
                pass

        if hasattr(wq, "channel_axis"):
            try:
                wq.channel_axis = 0
                changed.append("channel_axis=0")
            except Exception:
                pass

        # Some quantizers expose encoding analyzer/settings objects
        if hasattr(wq, "encoding_analyzer"):
            ea = getattr(wq, "encoding_analyzer")
            if hasattr(ea, "channel_axis"):
                try:
                    ea.channel_axis = 0
                    changed.append("encoding_analyzer.channel_axis=0")
                except Exception:
                    pass

        # Some versions may expose shape-related fields for encodings
        out_ch = int(self.weight.shape[0])
        if hasattr(wq, "shape"):
            try:
                # per-channel encoding shape for conv weights
                wq.shape = (out_ch, 1, 1, 1)
                changed.append(f"shape=({out_ch},1,1,1)")
            except Exception:
                pass

        # Some block quantizers may support block_size
        if hasattr(wq, "block_size"):
            try:
                wq.block_size = None
                changed.append("block_size=None")
            except Exception:
                pass

        if not changed:
            print(
                "[WARN] Could not explicitly force per-channel on weight quantizer. "
                "Inspect this object:",
                type(wq),
                getattr(wq, "__dict__", {}),
            )
        else:
            print("[INFO] QuantizedConv2d weight quantizer updated:", ", ".join(changed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0] is not None:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            x = super().forward(x)

        if self.output_quantizers[0] is not None:
            x = self.output_quantizers[0](x)

        return x