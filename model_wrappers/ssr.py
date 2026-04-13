import torch
from typing import Optional, Tuple, Dict

from model_wrappers.base import QuantModelWrapper


class QuantSSRWrapper(QuantModelWrapper):
    """
    Quantized wrapper for SSR model.
    """

    def __init__(
        self,
        model,
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
        data,
        return_loss: bool = False,
        rescale=True,
    ):
        out = self.model(return_loss=return_loss, rescale=rescale, **data)

        return out