import os
from typing import List
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

def load_images(images_path: str, num_iters: int = -1, recursive: bool = True) -> List[str]:
    image_extensions = {".png", ".jpg", ".jpeg"}

    if images_path is None:
        return []

    if os.path.isfile(images_path):
        images = [images_path]
    else:
        images = []
        if recursive:
            for root, _, files in os.walk(images_path):
                for file in sorted(files):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in image_extensions:
                        images.append(os.path.join(root, file))
        else:
            for file in sorted(os.listdir(images_path)):
                full_path = os.path.join(images_path, file)
                _, ext = os.path.splitext(file)
                if os.path.isfile(full_path) and ext.lower() in image_extensions:
                    images.append(full_path)

    images = sorted(images)
    if num_iters != -1:
        images = images[:num_iters]
    return images

def preprocess_image(image_path, input_width, input_height, device):
    transform = T.Compose([T.ToTensor()])

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (input_width, input_height))

    torch_input = transform(Image.fromarray(original_image)).unsqueeze(0).to(device=device, dtype=torch.float32)

    return original_image, torch_input
