#!/usr/bin/env python3
"""Extract AIMET-compatible encodings from a quantized (QDQ) ONNX model.

Usage:
    # Model-specific (AIMET-compatible output):
    python run_extract.py resnet50 model.onnx output.encodings.json

    # Generic (extract everything):
    python run_extract.py generic model.onnx output.encodings.json
"""

import argparse

from extractors import MODEL_REGISTRY, get_extractor
from extractors.base import QuantizedOnnxExtractor


def main():
    supported = ", ".join(sorted(MODEL_REGISTRY.keys())) + ", generic"

    parser = argparse.ArgumentParser(
        description="Extract AIMET-compatible encodings from a QDQ ONNX model.",
    )
    parser.add_argument("model", help=f"Model name ({supported})")
    parser.add_argument("ckpt_path", help="Path to the quantized ONNX model")
    parser.add_argument("output_path", help="Path to save the AIMET encodings JSON")
    args = parser.parse_args()

    if args.model == "generic":
        extractor = QuantizedOnnxExtractor(args.ckpt_path)
    else:
        cls = get_extractor(args.model)
        extractor = cls(args.ckpt_path)

    extractor.save(args.output_path)


if __name__ == "__main__":
    main()
