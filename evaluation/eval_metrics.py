import argparse
import glob
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model.pdl import (
    DEEPLAB_V3_PLUS,
    PANOPTIC_DEEPLAB,
    PytorchPanopticDeepLab,
    build_model
)

from evaluation.eval_dataset import EvalDataset, eval_collate, build_eval_loader

def normalize_logits_output(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ["semantic", "sem_seg", "out"]:
            if key in output:
                return output[key]
        raise KeyError(f"Cannot find semantic logits in dict keys: {list(output.keys())}")

    if isinstance(output, (list, tuple)):
        # only safe if you know index 0 is semantic
        return output[0]

    raise TypeError(f"Expected Tensor after unwrapping, got {type(output)}")

def get_semantic_logits(model_obj, image, model_category_const):
    backend = model_obj["backend"]
    use_cuda = str(image.device).startswith("cuda")

    if backend == "torch":
        if use_cuda:
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        output = model_obj["model"](image)
        if use_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        logits = normalize_logits_output(output)
        inference_time = end_time - start_time
        return logits, inference_time

    elif backend == "onnx":
        session = model_obj["session"]
        input_name = model_obj["input_name"]

        if image.device.type != "cpu":
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image.detach().numpy()

        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: image_np})
        end_time = time.perf_counter()

        logits = torch.from_numpy(outputs[0]).float()
        return logits, end_time - start_time

def update_confusion_matrix(conf_mat, pred, target, num_classes=19, ignore_index=255):
    if pred.device != target.device:
        pred = pred.to(target.device)

    valid = (
        (target != ignore_index) &
        (target >= 0) & (target < num_classes) &
        (pred >= 0) & (pred < num_classes)
    )
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return conf_mat

    inds = num_classes * target + pred
    conf_mat += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat

def compute_miou_from_confmat(conf_mat):
    conf_mat = conf_mat.float()
    tp = torch.diag(conf_mat)
    pos_gt = conf_mat.sum(dim=1)
    pos_pred = conf_mat.sum(dim=0)
    union = pos_gt + pos_pred - tp

    iou = tp / union.clamp(min=1)
    valid = union > 0
    miou = iou[valid].mean().item() * 100.0

    return {
        "mIoU": miou,
        "IoU_per_class": (iou * 100.0).cpu().numpy(),
    }

def evaluate_model(model_obj, model_category_const, loader, device, max_samples=-1):
    """
    Supports either:
      1) dict input:
         {
             "backend": "torch",
             "model": <nn.Module or engine>
         }

      2) plain model/module input:
         <nn.Module>, including AimetTraceWrapper
    """

    # Normalize model_obj so the rest of the function can use backend/model safely
    if isinstance(model_obj, dict):
        backend = model_obj.get("backend", "torch")
        model = model_obj.get("model", model_obj)
        normalized_model_obj = model_obj
    else:
        backend = getattr(model_obj, "backend", "torch")
        model = model_obj
        normalized_model_obj = {
            "backend": backend,
            "model": model,
        }

    if backend == "torch":
        model.eval()

    use_cuda = str(device).startswith("cuda")
    conf_mat = torch.zeros((19, 19), dtype=torch.int64, device=device)

    processed = 0
    total_inference_time = 0.0

    for batch in loader:
        for sample in batch:
            if backend == "torch":
                image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
            else:
                image = sample["image"].unsqueeze(0).to(dtype=torch.float32)

            label = sample["label"].to(device=device)
            orig_h, orig_w = sample["orig_size"]

            if use_cuda and backend == "torch":
                torch.cuda.synchronize()

            logits, inference_time = get_semantic_logits(
                normalized_model_obj, image, model_category_const
            )
            total_inference_time += inference_time

            if use_cuda and backend == "torch":
                torch.cuda.synchronize()

            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

            pred = logits.argmax(dim=1).squeeze(0)
            conf_mat = update_confusion_matrix(conf_mat, pred, label)

            processed += 1
            if processed % 50 == 0:
                current_fps = processed / total_inference_time if total_inference_time > 0 else 0.0
                print(f"Processed {processed} images | FPS: {current_fps:.2f}")

            if max_samples > 0 and processed >= max_samples:
                metrics = compute_miou_from_confmat(conf_mat)
                metrics["FPS"] = processed / total_inference_time if total_inference_time > 0 else 0.0
                metrics["Avg_Inference_Time_ms"] = (
                    (total_inference_time / processed) * 1000.0 if processed > 0 else 0.0
                )
                return metrics

    metrics = compute_miou_from_confmat(conf_mat)
    metrics["FPS"] = processed / total_inference_time if total_inference_time > 0 else 0.0
    metrics["Avg_Inference_Time_ms"] = (
        (total_inference_time / processed) * 1000.0 if processed > 0 else 0.0
    )
    return metrics