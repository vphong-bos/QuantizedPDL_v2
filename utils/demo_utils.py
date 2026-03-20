# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist

from model.pdl import DEEPLAB_V3_PLUS
from model.postprocessing import get_panoptic_segmentation

# Cityscapes dataset configuration
CITYSCAPES_CATEGORIES = [
    {"id": 0, "name": "road", "color": [128, 64, 128]},
    {"id": 1, "name": "sidewalk", "color": [244, 35, 232]},
    {"id": 2, "name": "building", "color": [70, 70, 70]},
    {"id": 3, "name": "wall", "color": [102, 102, 156]},
    {"id": 4, "name": "fence", "color": [190, 153, 153]},
    {"id": 5, "name": "pole", "color": [153, 153, 153]},
    {"id": 6, "name": "traffic light", "color": [250, 170, 30]},
    {"id": 7, "name": "traffic sign", "color": [220, 220, 0]},
    {"id": 8, "name": "vegetation", "color": [107, 142, 35]},
    {"id": 9, "name": "terrain", "color": [152, 251, 152]},
    {"id": 10, "name": "sky", "color": [70, 130, 180]},
    {"id": 11, "name": "person", "color": [220, 20, 60]},
    {"id": 12, "name": "rider", "color": [255, 0, 0]},
    {"id": 13, "name": "car", "color": [0, 0, 142]},
    {"id": 14, "name": "truck", "color": [0, 0, 70]},
    {"id": 15, "name": "bus", "color": [0, 60, 100]},
    {"id": 16, "name": "train", "color": [0, 80, 100]},
    {"id": 17, "name": "motorcycle", "color": [0, 0, 230]},
    {"id": 18, "name": "bicycle", "color": [119, 11, 32]},
]


def load_images(images_path: str) -> List[str]:
    """
    Load image paths from a file or directory using plain Python.

    Args:
        images_path: Path to an image file or a directory of images

    Returns:
        List of image file paths
    """
    if images_path is None:
        return []

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    if os.path.isfile(images_path):
        _, ext = os.path.splitext(images_path)
        return [images_path] if ext.lower() in valid_exts else []

    if not os.path.isdir(images_path):
        return []

    image_paths = []
    for file_name in sorted(os.listdir(images_path)):
        full_path = os.path.join(images_path, file_name)
        if not os.path.isfile(full_path):
            continue
        _, ext = os.path.splitext(file_name)
        if ext.lower() in valid_exts:
            image_paths.append(full_path)

    return image_paths


def resolve_demo_paths(demo_file: str) -> Dict[str, str]:
    """
    Resolve all demo-related paths from the demo file location.

    Args:
        demo_file: Path to the demo file (usually __file__)

    Returns:
        Dictionary with resolved paths for images, weights, outputs
    """
    demo_dir = os.path.dirname(os.path.abspath(demo_file))
    base_dir = os.path.dirname(demo_dir)  # panoptic_deeplab/

    return {
        "images": os.path.join(base_dir, "images"),
        "weights": os.path.join(base_dir, "weights", "model_final_bd324a.pkl"),
        "outputs": os.path.join(base_dir, "demo_outputs"),
    }


def preprocess_input_params(
    output_dir: str,
    model_category: str,
    current_dir: str,
    model_location_generator=None,
):
    """
    Pure Python replacement for old tt.common path helpers.

    model_location_generator is kept in signature for compatibility,
    but is not used anymore.
    """
    base_dir = os.path.abspath(current_dir)

    images_dir = os.path.join(base_dir, "images")
    weights_path = os.path.join(base_dir, "weights", "model_final_bd324a.pkl")
    output_dir = os.path.join(output_dir, model_category)

    images_paths = load_images(images_dir)
    if len(images_paths) == 0:
        logger.error(f"No images found in the specified path: {images_dir}")
        raise FileNotFoundError(f"No images found in the specified path: {images_dir}")

    return images_paths, weights_path, output_dir


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (512, 1024)) -> torch.Tensor:
    """
    Preprocess image for Panoptic DeepLab inference.

    Args:
        image_path: Path to input image
        target_size: Target size as (height, width)

    Returns:
        Preprocessed tensor in NCHW format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))  # cv2 uses (width, height)

    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def merge_nearby_instances(panoptic_seg: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """
    Merge nearby instances of the same class that are likely parts of the same object.
    """
    result = panoptic_seg.copy()
    unique_ids = np.unique(panoptic_seg)
    label_divisor = 1000

    class_instances = {}
    for segment_id in unique_ids:
        if segment_id == 0 or segment_id == 255:
            continue

        category_id = segment_id // label_divisor

        if 11 <= category_id <= 18:
            class_instances.setdefault(category_id, []).append(segment_id)

    for category_id, instance_ids in class_instances.items():
        if len(instance_ids) < 2:
            continue

        centroids = []
        valid_instance_ids = []

        for inst_id in instance_ids:
            mask = panoptic_seg == inst_id
            if np.sum(mask) == 0:
                continue
            y_coords, x_coords = np.where(mask)
            centroid = [np.mean(y_coords), np.mean(x_coords)]
            centroids.append(centroid)
            valid_instance_ids.append(inst_id)

        if len(centroids) < 2:
            continue

        distances = cdist(centroids, centroids)

        merged = set()
        for i in range(len(valid_instance_ids)):
            if i in merged:
                continue

            for j in range(len(valid_instance_ids)):
                if i == j or j in merged:
                    continue

                if distances[i, j] < max_distance:
                    mask_j = result == valid_instance_ids[j]
                    result[mask_j] = valid_instance_ids[i]
                    merged.add(j)
                    logger.debug(
                        f"Merged class {category_id} instance {valid_instance_ids[j]} into {valid_instance_ids[i]}"
                    )

    return result


def save_predictions(
    output_dir: str,
    image_name: str,
    original_image: np.ndarray,
    panoptic_vis: np.ndarray,
):
    """Save prediction results to files."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_original.jpg"),
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_panoptic.jpg"),
        cv2.cvtColor(panoptic_vis, cv2.COLOR_RGB2BGR),
    )

    logger.info(f"Saved predictions for {image_name} to {output_dir}")


def create_deeplab_v3plus_visualization(
    semantic_pred: np.ndarray,
    original_image: np.ndarray,
):
    semantic_classes = np.argmax(semantic_pred, axis=2)

    logger.info("Creating semantic segmentation visualization (no instance heads)")

    h, w = semantic_classes.shape
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(CITYSCAPES_CATEGORIES)):
        mask = semantic_classes == class_id
        if np.any(mask):
            color = CITYSCAPES_CATEGORIES[class_id]["color"]
            vis_image[mask] = color

    if original_image is not None:
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        if original_image.shape[:2] != (h, w):
            original_image = cv2.resize(original_image, (w, h))

        alpha = 0.6
        vis_image = cv2.addWeighted(original_image.astype(np.uint8), 1 - alpha, vis_image, alpha, 0)

    panoptic_info = {
        "mode": DEEPLAB_V3_PLUS,
        "num_classes": len(np.unique(semantic_classes)),
        "class_distribution": {
            int(cls): int(count) for cls, count in zip(*np.unique(semantic_classes, return_counts=True))
        },
    }

    return vis_image, panoptic_info


def create_panoptic_visualization(
    semantic_pred: np.ndarray,
    center_pred: np.ndarray = None,
    offset_pred: np.ndarray = None,
    original_image: np.ndarray = None,
    score_threshold: float = 0.05,
    center_threshold: float = 0.05,
    nms_kernel: int = 11,
    top_k: int = 1000,
    stuff_area: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Create panoptic or semantic segmentation visualization from model outputs.
    """
    logger.info("Creating full panoptic segmentation visualization (with instance heads)")

    semantic_tensor = torch.from_numpy(semantic_pred)
    semantic_tensor = torch.argmax(semantic_tensor, dim=2).unsqueeze(0)  # [1, H, W]

    center_tensor = torch.from_numpy(center_pred)
    if center_tensor.dim() == 3:
        center_tensor = center_tensor.squeeze(-1)
    center_tensor = center_tensor.unsqueeze(0)  # [1, H, W]

    offset_tensor = torch.from_numpy(offset_pred).permute(2, 0, 1)  # [2, H, W]

    logger.info(f"Using thresholds - center: {center_threshold}, score: {score_threshold}, stuff_area: {stuff_area}")
    logger.debug(f"Max center prediction: {center_tensor.max().item():.6f}")
    logger.debug(f"Pixels above center threshold: {(center_tensor > center_threshold).sum().item()}")
    logger.debug(f"Pixels with center > 0.01: {(center_tensor > 0.01).sum().item()}")
    logger.debug(f"Pixels with center > 0.05: {(center_tensor > 0.05).sum().item()}")

    panoptic_seg, center_points = get_panoptic_segmentation(
        semantic_tensor,
        center_tensor,
        offset_tensor,
        thing_ids=set(range(11, 19)),
        label_divisor=1000,
        stuff_area=stuff_area,
        void_label=255,
        threshold=center_threshold,
        nms_kernel=nms_kernel,
        top_k=top_k,
        foreground_mask=None,
    )

    logger.debug(f"Panoptic segmentation returned {len(np.unique(panoptic_seg))} unique segments")

    panoptic_seg = panoptic_seg.squeeze(0).numpy()
    semantic_classes = np.argmax(semantic_pred, axis=2)
    unique_raw = np.unique(panoptic_seg)

    max_valid_id = 18999
    invalid_ids = unique_raw[unique_raw > max_valid_id]
    if len(invalid_ids) > 0:
        logger.warning(f"Found invalid segment IDs: {invalid_ids}")
        for invalid_id in invalid_ids:
            panoptic_seg[panoptic_seg == invalid_id] = 0

    cleaned_panoptic = panoptic_seg.copy()

    for segment_id in unique_raw:
        if segment_id == 0 or segment_id == 255:
            continue

        category_id = segment_id // 1000
        mask = panoptic_seg == segment_id
        semantic_at_mask = semantic_classes[mask]

        if len(semantic_at_mask) > 0:
            semantic_counts = np.bincount(semantic_at_mask, minlength=19)
            most_common_semantic = np.argmax(semantic_counts)
            agreement_ratio = semantic_counts[most_common_semantic] / len(semantic_at_mask)

            if agreement_ratio < 0.85:
                wrong_semantic_mask = mask & (semantic_classes != category_id)
                cleaned_panoptic[wrong_semantic_mask] = 0

    panoptic_seg = cleaned_panoptic

    if len(np.unique(panoptic_seg)) > 15:
        logger.info("Detected over-segmentation, applying post-processing to merge instances...")
        panoptic_seg = merge_nearby_instances(panoptic_seg, max_distance=40)
        logger.info(f"After merging: {len(np.unique(panoptic_seg))} unique segments")

    segments_info = []
    unique_ids = np.unique(panoptic_seg)
    label_divisor = 1000

    logger.info(f"Panoptic segmentation shape: {panoptic_seg.shape}, unique IDs: {len(unique_ids)}")
    logger.info(f"Unique segment IDs: {unique_ids[:10]}...")

    background_pixels = np.sum(panoptic_seg == 0)
    void_pixels = np.sum(panoptic_seg == 255)
    total_pixels = panoptic_seg.size
    logger.info(f"Background pixels: {background_pixels}/{total_pixels} ({background_pixels/total_pixels*100:.1f}%)")
    logger.info(f"Void pixels: {void_pixels}/{total_pixels} ({void_pixels/total_pixels*100:.1f}%)")

    semantic_unique, semantic_counts = np.unique(semantic_classes, return_counts=True)
    for class_id, count in zip(semantic_unique, semantic_counts):
        if class_id < len(CITYSCAPES_CATEGORIES):
            class_name = CITYSCAPES_CATEGORIES[class_id]["name"]
            logger.info(f"Semantic class {class_id} ({class_name}): {count} pixels ({count/total_pixels*100:.1f}%)")
        else:
            logger.info(f"Unknown semantic class {class_id}: {count} pixels")

    for segment_id in unique_ids:
        if segment_id == 0 or segment_id == 255:
            continue

        category_id = segment_id // label_divisor
        mask = panoptic_seg == segment_id
        area = np.sum(mask)

        segments_info.append(
            {
                "id": int(segment_id),
                "category_id": int(category_id),
                "area": int(area),
                "iscrowd": 0,
            }
        )

    height, width = panoptic_seg.shape
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)

    logger.info(f"Using panoptic segmentation with {len(segments_info)} segments")

    background_mask = (panoptic_seg == 0) | (panoptic_seg == 255)

    for class_id in range(len(CITYSCAPES_CATEGORIES)):
        class_mask = (semantic_classes == class_id) & background_mask
        if np.any(class_mask):
            color = CITYSCAPES_CATEGORIES[class_id]["color"]

            if 11 <= class_id <= 18:
                vis_image[class_mask] = color
            else:
                color_dim = [int(c * 0.7) for c in color]
                vis_image[class_mask] = color_dim

    segments_sorted = sorted(segments_info, key=lambda x: x["area"], reverse=True)
    for segment in segments_sorted:
        segment_id = segment["id"]
        category_id = segment["category_id"]
        mask = panoptic_seg == segment_id

        if category_id < len(CITYSCAPES_CATEGORIES):
            color = CITYSCAPES_CATEGORIES[category_id]["color"]
        else:
            np.random.seed(segment_id)
            color = np.random.randint(128, 255, 3).tolist()

        vis_image[mask] = color

    alpha = 0.6
    blended = cv2.addWeighted(original_image, 1 - alpha, vis_image, alpha, 0)

    return blended, {"segments": segments_info, "panoptic_seg": panoptic_seg, "pure_vis": vis_image}