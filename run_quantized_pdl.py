#!/usr/bin/env python3
import argparse
import os
import copy
import time

import torch

from model.pdl import build_model

from quantization.calibration_dataset import (
    create_calibration_loader,
    sample_calibration_images,
)
from quantization.quantize_function import (
    AimetTraceWrapper,
    create_quant_sim,
    calibration_forward_pass,
)
from quantization.bias_correction import apply_bias_correction, copy_biases
from utils.image_loader import load_images

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_common.utils import CallbackFunc
from aimet_torch import quantsim, onnx

import torch
from aimet_torch.v2.nn import QuantizationMixin
from model.conv2d import Conv2d
from model.quantized_conv2d import QuantizedConv2d

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
DEFAULT_EXPORT_PATH = os.path.join(pdl_home_path, "quantized_export")
DEFAULT_ANALYZER_PATH = os.path.join(pdl_home_path, "quant_analyzer_results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_height", type=int, default=512, help="input image height")
    parser.add_argument("--image_width", type=int, default=1024, help="input image width")

    parser.add_argument(
        "--weights_path",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="path to FP32 model weights",
    )
    parser.add_argument(
        "--model_category",
        type=str,
        default="PANOPTIC_DEEPLAB",
        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
        help="semantic-only or full panoptic model",
    )

    parser.add_argument(
        "--calib_images",
        type=str,
        required=True,
        help="image file or folder used for AIMET calibration",
    )

    parser.add_argument("--num_calib", type=int, default=800, help="number of calibration images")
    parser.add_argument("--batch_size", type=int, default=1, help="AIMET calibration batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=123, help="random seed for calibration sampling")

    parser.add_argument("--quant_scheme", type=str, default="tf_enhanced", help="AIMET quantization scheme")
    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")

    parser.add_argument(
        "--save_quant_checkpoint",
        type=str,
        default=None,
        help="optional path to save AIMET sim checkpoint",
    )

    parser.add_argument(
        "--export_path",
        type=str,
        default=DEFAULT_EXPORT_PATH,
        help="path to export quantized model",
    )
    parser.add_argument(
        "--export_prefix",
        type=str,
        default="panoptic_deeplab_int8",
        help="export filename prefix",
    )
    parser.add_argument("--no_export", action="store_true", help="skip AIMET export step")

    parser.add_argument("--no_operation_orient", action="store_true", help="skip operation orient ONNX export step")

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="config file for quantized model",
    )

    parser.add_argument(
        "--enable_cle",
        dest="enable_cle",
        action="store_true",
        help="enable Cross-Layer Equalization before quantization",
    )
    parser.add_argument(
        "--disable_cle",
        dest="enable_cle",
        action="store_false",
        help="disable Cross-Layer Equalization",
    )
    parser.set_defaults(enable_cle=False)

    parser.add_argument(
        "--enable_bn_fold",
        action="store_true",
        help="apply batch norm folding before creating QuantSim",
    )

    parser.add_argument(
        "--enable_adaround",
        action="store_true",
        help="apply AIMET AdaRound before creating QuantSim",
    )
    parser.add_argument(
        "--adaround_num_batches",
        type=int,
        default=32,
        help="number of calibration batches to use for AdaRound",
    )
    parser.add_argument(
        "--adaround_num_iterations",
        type=int,
        default=10000,
        help="number of iterations for AdaRound",
    )
    parser.add_argument(
        "--adaround_path",
        type=str,
        default=None,
        help="directory to save AdaRound encodings",
    )
    parser.add_argument(
        "--adaround_prefix",
        type=str,
        default="adaround",
        help="filename prefix for AdaRound encodings",
    )

    parser.add_argument(
        "--enable_bias_correction",
        action="store_true",
        help="apply AIMET Bias Correction before creating QuantSim",
    )
    parser.add_argument(
        "--bias_corr_num_quant_samples",
        type=int,
        default=256,
        help="number of samples used to build temporary quant sim during bias correction",
    )
    parser.add_argument(
        "--bias_corr_num_bias_samples",
        type=int,
        default=256,
        help="number of samples used for bias correction",
    )
    parser.add_argument(
        "--bias_corr_empirical_only",
        action="store_true",
        help="use empirical-only bias correction instead of analytical+empirical",
    )

    parser.add_argument(
        "--run_quant_analyzer",
        action="store_true",
        help="run AIMET QuantAnalyzer and save sensitivity reports",
    )
    parser.add_argument(
        "--quant_analyzer_dir",
        type=str,
        default=DEFAULT_ANALYZER_PATH,
        help="directory to save QuantAnalyzer outputs",
    )
    parser.add_argument(
        "--analyzer_num_batches",
        type=int,
        default=None,
        help="number of calibration batches for analyzer forward pass; default uses all",
    )
    parser.add_argument(
        "--cityscapes_root",
        type=str,
        default=None,
        help="Cityscapes root, required when --run_quant_analyzer is set",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="evaluation split for QuantAnalyzer",
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=-1,
        help="max eval samples for QuantAnalyzer, -1 means full split",
    )

    return parser.parse_args(argv)


def adaround_forward_fn(model, inputs):
    if isinstance(inputs, dict):
        images = inputs["image"]
    elif isinstance(inputs, (list, tuple)):
        images = inputs[0]
    else:
        images = inputs

    images = images.to(next(model.parameters()).device)
    return model(images)


def analyzer_forward_pass(model, callback_args):
    calib_loader, device, max_batches = callback_args

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                images = batch["image"]
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            _ = model(images)


def analyzer_eval_callback(model, callback_args):
    eval_loader, model_category_const, device, max_samples = callback_args

    results = evaluate_model(
        model=model,
        model_category_const=model_category_const,
        loader=eval_loader,
        device=device,
        max_samples=max_samples,
    )
    return float(results["mIoU"])


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if args.save_quant_checkpoint is not None:
        save_dir = os.path.dirname(args.save_quant_checkpoint)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    os.makedirs(args.export_path, exist_ok=True)

    if args.adaround_path is None:
        args.adaround_path = args.export_path
    os.makedirs(args.adaround_path, exist_ok=True)

    eval_loader = None
    if args.run_quant_analyzer:
        if args.cityscapes_root is None:
            raise ValueError("--cityscapes_root is required when --run_quant_analyzer is set")

        print("Building evaluation loader for QuantAnalyzer...")
        eval_loader = build_eval_loader(
            cityscapes_root=args.cityscapes_root,
            split=args.eval_split,
            image_width=args.image_width,
            image_height=args.image_height,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print("Loading FP32 model...")
    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    model = model.to(args.device).eval()

    if args.enable_cle:
        print("Applying Cross-Layer Equalization (CLE)...")
        cle_start = time.time()

        model = model.cpu().eval()
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        cle_wrapper = AimetTraceWrapper(
            model=model,
            model_category_const=model_category_const,
        ).cpu().eval()

        equalize_model(
            cle_wrapper,
            input_shapes=(1, 3, args.image_height, args.image_width),
            dummy_input=dummy_input_cpu,
        )

        model = model.to(args.device).eval()
        cle_time = time.time() - cle_start
        print(f"CLE finished in {cle_time:.2f} s")
    else:
        print("CLE disabled")

    print("Collecting calibration images...")
    all_calib_images = load_images(args.calib_images, num_iters=-1, recursive=True)
    calib_images = sample_calibration_images(all_calib_images, args.num_calib, args.seed)
    print(f"Found {len(all_calib_images)} candidate calibration images")
    print(f"Using {len(calib_images)} images for calibration")

    calib_loader = create_calibration_loader(
        calib_image_paths=calib_images,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Wrapping model for AIMET tracing...")
    wrapped_model = AimetTraceWrapper(
        model=model,
        model_category_const=model_category_const,
    ).cpu().eval()

    if args.enable_bn_fold:
        print("Applying batch norm folding...")
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        fold_all_batch_norms(
            model=wrapped_model,
            input_shapes=(1, 3, args.image_height, args.image_width),
            dummy_input=dummy_input_cpu,
        )
    else:
        print("BN folding disabled")

    if args.enable_bias_correction:
        print("Applying Bias Correction...")
        bc_start = time.time()

        bc_model = copy.deepcopy(wrapped_model).cpu().eval()

        bc_model = apply_bias_correction(
            model=bc_model,
            calib_loader=calib_loader,
            image_height=args.image_height,
            image_width=args.image_width,
            quant_scheme=args.quant_scheme,
            default_param_bw=args.default_param_bw,
            default_output_bw=args.default_output_bw,
            config_file=args.config_file,
            bias_corr_num_quant_samples=args.bias_corr_num_quant_samples,
            bias_corr_num_bias_samples=args.bias_corr_num_bias_samples,
            bias_corr_empirical_only=args.bias_corr_empirical_only,
        )

        copy_biases(bc_model, wrapped_model)
        del bc_model

        bc_time = time.time() - bc_start
        print(f"Bias Correction finished in {bc_time:.2f} s")
    else:
        print("Bias Correction disabled")

    wrapped_model = wrapped_model.to(args.device).eval()

    if args.enable_adaround:
        print("Applying AdaRound...")
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        adaround_params = AdaroundParameters(
            data_loader=calib_loader,
            num_batches=min(args.adaround_num_batches, len(calib_loader)),
            default_num_iterations=args.adaround_num_iterations,
            forward_fn=adaround_forward_fn,
        )

        wrapped_model = wrapped_model.cpu().eval()
        wrapped_model = Adaround.apply_adaround(
            model=wrapped_model,
            dummy_input=dummy_input_cpu,
            params=adaround_params,
            path=args.adaround_path,
            filename_prefix=args.adaround_prefix,
            default_param_bw=args.default_param_bw,
            default_quant_scheme=args.quant_scheme,
            default_config_file=args.config_file,
        )

        wrapped_model = wrapped_model.to(args.device).eval()
        print(
            f"AdaRound finished. Encodings saved under: "
            f"{os.path.join(args.adaround_path, args.adaround_prefix)}*.encodings"
        )
    else:
        print("AdaRound disabled")

    if args.run_quant_analyzer:
        print("Running AIMET QuantAnalyzer...")
        os.makedirs(args.quant_analyzer_dir, exist_ok=True)

        dummy_input = torch.randn(
            1, 3, args.image_height, args.image_width, device=args.device
        )

        forward_pass_callback = CallbackFunc(
            analyzer_forward_pass,
            func_callback_args=(calib_loader, args.device, args.analyzer_num_batches),
        )

        eval_callback = CallbackFunc(
            analyzer_eval_callback,
            func_callback_args=(
                eval_loader,
                model_category_const,
                args.device,
                args.eval_max_samples,
            ),
        )

        analyzer = QuantAnalyzer(
            model=wrapped_model,
            dummy_input=dummy_input,
            forward_pass_callback=forward_pass_callback,
            eval_callback=eval_callback,
            modules_to_ignore=None,
        )

        # analyzer.analyze(
        #     quant_scheme=args.quant_scheme,
        #     default_param_bw=args.default_param_bw,
        #     default_output_bw=args.default_output_bw,
        #     config_file=args.config_file,
        #     results_dir=args.quant_analyzer_dir,
        # )

        # print(f"QuantAnalyzer results saved to: {args.quant_analyzer_dir}")

    skip_layer_names = [
        # "model.backbone.stem.conv1",
        # "model.backbone.stem.conv1.norm",
        # "model.backbone.stem.conv2",
        # "model.backbone.stem.conv2.norm",
        # "model.backbone.stem.conv3",
    ]

    print("Creating AIMET QuantizationSimModel...")
    sim, _ = create_quant_sim(
        model=wrapped_model,
        model_category_const=model_category_const,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        quant_scheme=args.quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file=args.config_file,
        skip_layer_names=skip_layer_names
    )

    print("Computing encodings with calibration data...")
    calib_start = time.time()
    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(calib_loader, args.device),
    )
    calib_time = time.time() - calib_start
    print(f"Calibration finished in {calib_time:.2f} s")

    quantized_model = sim.model
    quantized_model.eval()

    if args.save_quant_checkpoint is not None:
        import pickle

        with open(args.save_quant_checkpoint, "wb") as f:
            pickle.dump(sim, f)

        print(f"Saved AIMET sim checkpoint to: {args.save_quant_checkpoint}")
    from aimet_torch import onnx as aimet_onnx

    if not args.no_export:
        print("Exporting quantized model to ONNX QDQ...")
        sim.model.cpu().eval()

        cpu_dummy_input = torch.randn(
            1, 3, args.image_height, args.image_width, device="cpu"
        )

        os.makedirs(args.export_path, exist_ok=True)
        onnx_path = os.path.join(args.export_path, f"{args.export_prefix}.onnx")

        aimet_onnx.export(
            sim.model,
            cpu_dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            export_int32_bias=True,
            dynamo=False,   # AIMET says dynamo=True is not supported here
        )

        print(f"Exported QDQ ONNX to: {onnx_path}")

    # if not args.no_operation_orient:
    #     print("Exporting operation orient quantized model to ONNX QDQ...")
    #     sim.model.cpu().eval()

    #     cpu_dummy_input = torch.randn(
    #         1, 3, args.image_height, args.image_width, device="cpu"
    #     )

    #     os.makedirs(args.export_path, exist_ok=True)
    #     onnx_path = os.path.join(args.export_path, f"{args.export_prefix}_operation_orient.onnx")

    #     onnx.export(
    #         sim.model,                 
    #         cpu_dummy_input,
    #         onnx_path,
    #         input_names=["input"],
    #         output_names=["output"],
    #         opset_version=21,
    #         export_int32_bias=True,
    #         dynamo=False,
    #         operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    #     )

    #     print(f"Exported QDQ ONNX to: {onnx_path}")

    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)