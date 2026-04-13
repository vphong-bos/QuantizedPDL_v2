import torch
from typing import Optional, Tuple, Dict

from model_wrappers.base import QuantModelWrapper
from model_wrappers.model.pdl.pdl import PytorchPanopticDeepLab


class QuantPDLWrapper(QuantModelWrapper):
    """
    Quantized wrapper for PytorchPanopticDeepLab model.
    """

    def __init__(
        self,
        model: PytorchPanopticDeepLab,
        encodings_path: str,
    ):
        super().__init__(
            model=model,
            encodings_path=encodings_path,
        )

        self.quantized_params = self.quantize_params()

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
        out = self.model(x, return_features)

        return out