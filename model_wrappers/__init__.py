from .base import QuantModelWrapper
# from .resnet import QuantResNetWrapper
# from .ssr import QuantSSRWrapper
from .pdl import QuantPDLWrapper
# from .vit import QuantViTWrapper

MODEL_REGISTRY = {
    # "resnet50": QuantResNetWrapper,
    # "resnet101": QuantResNetWrapper,
    # "resnet152": QuantResNetWrapper,
    # "ssr": QuantSSRWrapper,
    "pdl": QuantPDLWrapper,
    # "vit": QuantViTWrapper,
}


def get_wrapper(model_name):
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")
    return cls
