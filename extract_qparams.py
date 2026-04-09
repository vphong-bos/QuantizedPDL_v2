#!/usr/bin/env python3

import argparse

from extractors.base import QuantizedOnnxExtractor
from extractors.resnet import ResNetExtractor

def choose_extractor(model_type):
    if model_type == "resnet":
        return ResNetExtractor
    else:
        return QuantizedOnnxExtractor

def main():
    parser = argparse.ArgumentParser(
        description="Extract quantized tensors and activation qparams from a QDQ ONNX model."
    )
    parser.add_argument("ckpt_path", help="Path to the quantized ONNX model")
    parser.add_argument("output_path", help="Path to save the extracted encodings")
    parser.add_argument(
        "--model",
        choices=["base", "resnet"],
        default="base",
        help="Model type for extraction (default: base)"
    )
    args = parser.parse_args()

    extractor_class = choose_extractor(args.model)

    extractor = extractor_class(args.ckpt_path)
    extractor.save(args.output_path)


if __name__ == "__main__":
    main()