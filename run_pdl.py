import argparse
import os
import time

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from utils.demo_utils import (
    create_deeplab_v3plus_visualization,
    create_panoptic_visualization,
    save_predictions,
)
from model.pdl import (
    DEEPLAB_V3_PLUS,
    build_model
)

from utils.image_loader import load_images, preprocess_image

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = os.path.join(pdl_home_path, "data/images")
WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
OUTPUT_PATH = os.path.join(pdl_home_path, "output")

center_threshold = 0.05


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-n", "--num_iters", type=int, default=-1, help="number of images to process")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, currently only 1 is supported")
    parser.add_argument("--image_height", type=int, default=512, help="input image height")
    parser.add_argument("--image_width", type=int, default=1024, help="input image width")

    parser.add_argument("--images", type=str, default=IMAGES_PATH, help="image file or folder")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH, help="path to model weights")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="path to save outputs")
    parser.add_argument(
        "--model_category",
        type=str,
        default="PANOPTIC_DEEPLAB",
        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
        help="semantic-only or full panoptic model",
    )

    return parser.parse_args(argv)

def run_inference(model, torch_input, model_category_const):
    with torch.no_grad():
        semantic_logits, center_heatmap, offset_map, _ = model(torch_input)

    if model_category_const == DEEPLAB_V3_PLUS:
        return semantic_logits
    return semantic_logits, center_heatmap, offset_map, _


def save_visualization(
    model_category_const,
    output,
    original_image,
    output_path,
    image_path,
):
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]
    output_dir = os.path.join(output_path, f"{image_stem}_output")
    os.makedirs(output_dir, exist_ok=True)

    if model_category_const == DEEPLAB_V3_PLUS:
        semantic_logits = output
        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_deeplab_v3plus_visualization(
            semantic_np,
            original_image=original_image,
        )
    else:
        semantic_logits, center_heatmap, offset_map = output

        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        center_np = center_heatmap.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        offset_np = offset_map.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_panoptic_visualization(
            semantic_np,
            center_np,
            offset_np,
            original_image,
            center_threshold=center_threshold,
            score_threshold=center_threshold,
            stuff_area=1,
            top_k=1000,
            nms_kernel=11,
        )

    save_predictions(output_dir, image_name, original_image, vis)
    print(f"Saved output for image {image_name} to {output_dir}")

def panoptic_deeplab_runner(args):
    if args.batch_size != 1:
        raise ValueError("This simplified PyTorch runner currently supports batch_size=1 only.")

    os.makedirs(args.output_path, exist_ok=True)

    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    images = load_images(args.images, args.num_iters)
    if len(images) == 0:
        raise ValueError(f"No valid images found in: {args.images}")

    total_start = time.time()

    for i, image_path in enumerate(images, start=1):
        original_image, torch_input = preprocess_image(
            image_path=image_path,
            input_width=args.image_width,
            input_height=args.image_height,
            device=args.device,
        )

        start_time = time.time()
        output = run_inference(model, torch_input, model_category_const)
        end_time = time.time()

        print(f"[{i}/{len(images)}] {os.path.basename(image_path)}: {(end_time - start_time) * 1000:.2f} ms")

        save_visualization(
            model_category_const=model_category_const,
            output=output,
            original_image=original_image,
            output_path=args.output_path,
            image_path=image_path,
        )

    total_end = time.time()
    total_time = total_end - total_start
    fps = len(images) / total_time if total_time > 0 else 0.0

    print("\n================ Execution Results ================")
    print(f"Model Category: {args.model_category}")
    print(f"Number of Inputs: {len(images)}")
    print(f"Total Execution Time: {total_time:.4f} s")
    print(f"Samples per Second: {fps:.2f} samples/s")
    print("===================================================")

    return {
        "fps": fps,
    }


if __name__ == "__main__":
    args = parse_args()
    panoptic_deeplab_runner(args)