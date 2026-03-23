import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ----------------------------
# Cityscapes labelId -> trainId
# ----------------------------
# Standard Cityscapes mapping
IGNORE_LABEL = 255

ID_TO_TRAIN_ID = np.full((256,), IGNORE_LABEL, dtype=np.uint8)

# road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
# vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle
MAPPING = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
for k, v in MAPPING.items():
    ID_TO_TRAIN_ID[k] = v

class EvalDataset(Dataset):
    def __init__(self, cityscapes_root, split="val", image_width=2048, image_height=1024):
        self.cityscapes_root = cityscapes_root
        self.split = split
        self.image_width = image_width
        self.image_height = image_height
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        pattern = os.path.join(
            cityscapes_root,
            "leftImg8bit",
            split,
            "*",
            "*_leftImg8bit.png",
        )
        self.image_paths = sorted(glob.glob(pattern))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found: {pattern}")

    def __len__(self):
        return len(self.image_paths)

    def _get_label_path(self, image_path):
        city = os.path.basename(os.path.dirname(image_path))
        base = os.path.basename(image_path).replace("_leftImg8bit.png", "")
        return os.path.join(
            self.cityscapes_root,
            "gtFine",
            self.split,
            city,
            f"{base}_gtFine_labelIds.png",
        )

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self._get_label_path(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = self.transform(Image.fromarray(resized)).float()

        label_ids = np.array(Image.open(label_path), dtype=np.uint8)
        train_ids = ID_TO_TRAIN_ID[label_ids]
        label_tensor = torch.from_numpy(train_ids.astype(np.int64))

        return {
            "image": image_tensor,
            "label": label_tensor,
            "file_name": image_path,
            "orig_size": (orig_h, orig_w),
        }

def eval_collate(batch):
    return batch


def build_eval_loader(
    cityscapes_root,
    split="val",
    image_width=1024,
    image_height=512,
    batch_size=1,
    num_workers=2
):
    dataset = EvalDataset(
        cityscapes_root=cityscapes_root,
        split=split,
        image_width=image_width,
        image_height=image_height,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=eval_collate,
    )

    return loader