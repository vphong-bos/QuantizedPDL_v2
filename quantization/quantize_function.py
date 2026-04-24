import torch
import os
import math
from aimet_torch import quantsim
from torch.utils.data import DataLoader, Dataset
from aimet_torch.quantsim import QuantizationSimModel
import torchvision.transforms as T
from quantization.calibration_dataset import CalibrationDataset
from model.quantized_conv2d import QuantizedConv2d
from model.pdl import build_model

import torch
import torch.nn as nn

from model.pdl import DEEPLAB_V3_PLUS, PANOPTIC_DEEPLAB

class AimetTraceWrapper(nn.Module):
    def __init__(self, model, model_category_const):
        super().__init__()
        self.model = model
        self.model_category_const = model_category_const

    def forward(self, x):
        outputs = self.model(x)

        # Expecting original model to return:
        # semantic_logits, center_heatmap, offset_map, something_else
        if self.model_category_const == DEEPLAB_V3_PLUS:
            if isinstance(outputs, (tuple, list)):
                semantic_logits = outputs[0]
            else:
                semantic_logits = outputs
            return semantic_logits

        # PANOPTIC_DEEPLAB
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(f"Expected tuple/list output from model, got {type(outputs)}")

        semantic_logits = outputs[0]
        center_heatmap = outputs[1]
        offset_map = outputs[2]

        # Return only tensors
        return semantic_logits, center_heatmap, offset_map

def create_quant_sim(
    model,
    model_category_const,
    device,
    image_height,
    image_width,
    quant_scheme,
    default_output_bw,
    default_param_bw,
    config_file = None,
    skip_layer_names = None,
):
    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)

    # wrapped_model = AimetTraceWrapper(model, model_category_const).to(device)
    # model.eval()

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file = config_file,
    )

    name_to_module = dict(sim.model.named_modules())

    layers_to_exclude = []
    missing = []

    for name in skip_layer_names:
        if name in name_to_module:
            layers_to_exclude.append(name_to_module[name])
        else:
            missing.append(name)

    if missing:
        print("Missing layer names:")
        for name in missing:
            print("  ", name)

    if layers_to_exclude:
        sim.exclude_layers_from_quantization(layers_to_exclude)

    return sim, dummy_input

def calibration_forward_pass(model, forward_pass_args):
    dataloader, device, calib_size = forward_pass_args
    model.eval()

    if len(dataloader) == 0:
        raise ValueError("Calibration dataloader is empty.")

    if calib_size is None:
        dataset_len = len(dataloader.dataset)
        calib_size = dataset_len if dataset_len > 0 else len(dataloader)

    if calib_size <= 0:
        raise ValueError("calib_size must be > 0.")

    processed = 0
    loader_iter = iter(dataloader)

    with torch.no_grad():
        while processed < calib_size:
            try:
                images = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                images = next(loader_iter)

            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            _ = model(images)
            processed += images.shape[0]

def quantize_model_with_aimet(
    model,
    model_category_const,
    image_paths,
    device,
    image_height,
    image_width,
    num_calib=500,
    calib_size=None,
    quant_scheme="tf_enhanced",
    default_output_bw=8,
    default_param_bw=8,
):
    dataset = CalibrationDataset(
        image_paths=image_paths[:num_calib],
        image_width=image_width,
        image_height=image_height,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    sim, _ = create_quant_sim(
        model=model,
        model_category_const=model_category_const,
        device=device,
        image_height=image_height,
        image_width=image_width,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
    )

    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(loader, device, calib_size or min(num_calib, len(dataset))),
    )

    return sim

import onnxruntime as ort

def load_aimet_quantized_model(
    quant_weights,
    model_category,
    device,
    provider="CPUExecutionProvider",
):
    print("Loading quantized model...")

    model_category_const = (
        PANOPTIC_DEEPLAB if model_category == "PANOPTIC_DEEPLAB" else DEEPLAB_V3_PLUS
    )

    ext = os.path.splitext(quant_weights)[1].lower()

    # =========================
    # Case 1: ONNX QDQ model
    # =========================
    if ext == ".onnx":
        print("Detected ONNX model")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            quant_weights,
            sess_options=so,
            providers=[provider],
        )

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        print(f"[ONNX] input: {input_name}")
        print(f"[ONNX] outputs: {output_names}")

        return {
            "backend": "onnx",
            "session": session,
            "input_name": input_name,
            "output_names": output_names,
            "model": None,
            "model_category_const": model_category_const,
        }

    # =========================
    # Case 2: AIMET checkpoint
    # =========================
    print("Detected AIMET checkpoint")

    sim = quantsim.load_checkpoint(quant_weights)
    sim.model.to(device).eval()

    return {
        "backend": "torch",
        "model": sim.model,
        "session": None,
        "input_name": None,
        "output_names": None,
        "model_category_const": model_category_const,
    }
