import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random
from typing import List, Optional

class CalibrationDataset(Dataset):
    def __init__(self, image_paths, image_width, image_height):
        self.image_paths = image_paths
        self.image_width = image_width
        self.image_height = image_height
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))
        tensor = self.transform(Image.fromarray(image)).to(dtype=torch.float32)

        return tensor
    
def create_calibration_loader(
    calib_image_paths: List[str],
    image_width: int,
    image_height: int,
    batch_size: int,
    num_workers: int,
):
    dataset = CalibrationDataset(
        image_paths=calib_image_paths,
        image_width=image_width,
        image_height=image_height,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader


def sample_calibration_images(all_image_paths: List[str], num_calib: int, seed: int) -> List[str]:
    if len(all_image_paths) == 0:
        raise ValueError("No calibration images found.")

    if num_calib >= len(all_image_paths):
        return all_image_paths

    rng = random.Random(seed)
    sampled = rng.sample(all_image_paths, num_calib)
    sampled.sort()
    return sampled