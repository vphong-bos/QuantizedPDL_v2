import torch
from typing import Optional, Tuple, Dict

from model_wrappers.base import QuantModelWrapper
from model_wrappers.model.pdl.pdl import PytorchPanopticDeepLab


class QuantPDLWrapper(QuantModelWrapper):
    """
    Quantized wrapper for PytorchPanopticDeepLab model.

    Instead of quantizing only final outputs, this wraps internal modules
    whose names match activation encodings and applies fake quantization
    during the normal model forward path.
    """

    def __init__(
        self,
        model: PytorchPanopticDeepLab,
        encodings_path: str,
        debug_activation_quant: bool = False,
    ):
        super().__init__(
            model=model,
            encodings_path=encodings_path,
            debug_activation_quant=debug_activation_quant,
        )

        self.quantized_params = self.quantize_params()

        # Inject activation quantization into internal modules
        self.add_activation_quant_wrapper()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]]
    ]:
        if self.debug_activation_quant:
            self.reset_debug_state()

        out = self.model(x, return_features)

        if self.debug_activation_quant:
            self.print_activation_debug_summary()

        return out