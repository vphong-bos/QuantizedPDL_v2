import torch
from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d


@QuantizationMixin.implements(Conv2d)
class QuantizedConv2d(QuantizationMixin, Conv2d):
    def __quant_init__(self):
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

        if not hasattr(self, "param_quantizers") or self.param_quantizers is None:
            raise RuntimeError("AIMET did not initialize param_quantizers")

        # bias off
        if "bias" in self.param_quantizers:
            self.param_quantizers["bias"] = None

        # weight quantizer must exist
        if "weight" not in self.param_quantizers:
            raise RuntimeError(
                f"weight quantizer missing. keys={list(self.param_quantizers.keys())}"
            )

        wq = self.param_quantizers["weight"]
        self._try_make_weight_quantizer_per_channel(wq)

    def _try_make_weight_quantizer_per_channel(self, wq):
        changed = []

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

        if hasattr(wq, "encoding_analyzer"):
            ea = wq.encoding_analyzer
            if hasattr(ea, "channel_axis"):
                try:
                    ea.channel_axis = 0
                    changed.append("encoding_analyzer.channel_axis=0")
                except Exception:
                    pass

        if hasattr(wq, "shape"):
            out_ch = int(self.weight.shape[0])
            try:
                wq.shape = (out_ch, 1, 1, 1)
                changed.append(f"shape=({out_ch},1,1,1)")
            except Exception:
                pass

        if not changed:
            print("[WARN] Could not force per-channel.")
            print("[WARN] weight quantizer type:", type(wq))
            print("[WARN] attrs:", getattr(wq, "__dict__", {}))
        else:
            print("[INFO] Updated weight quantizer:", ", ".join(changed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0] is not None:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            x = super().forward(x)

        if self.output_quantizers[0] is not None:
            x = self.output_quantizers[0](x)

        return x