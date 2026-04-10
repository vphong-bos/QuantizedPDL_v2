from .base import QuantizedOnnxExtractor
from .resnet import ResNetExtractor
from .ssr import SSRExtractor
from .pdl import PDLExtractor
from .vit import ViTExtractor

MODEL_REGISTRY = {
    "resnet50": ResNetExtractor,
    "resnet101": ResNetExtractor,
    "resnet152": ResNetExtractor,
    "ssr": SSRExtractor,
    "pdl": PDLExtractor,
    "vit": ViTExtractor,
}


def get_extractor(model_name):
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")
    return cls
