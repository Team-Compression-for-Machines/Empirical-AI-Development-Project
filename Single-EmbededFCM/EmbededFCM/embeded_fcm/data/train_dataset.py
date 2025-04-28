import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np

class OpenImageDatasetFPN(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.transform = transform  # expects a torchvision.transforms.Compose, etc.
        self.mode = split

    def __getitem__(self, index):
        # 1) OpenCV로 읽고 BGR→RGB 변환
        img_bgr = cv2.imread(str(self.samples[index]))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {self.samples[index]}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2) PIL Image로 변환
        img = Image.fromarray(img_rgb)

        # 3) torchvision Transform 적용 (e.g. Resize, ToTensor, Normalize 등)
        if self.transform is not None:
            img = self.transform(img)  
            # transform은 보통 ToTensor() → FloatTensor(C×H×W), [0,1] 스케일로 변환
        else:
            # transform이 없으면 기본 Tensor 변환
            img = torch.from_numpy(
                np.array(img, dtype=np.float32).transpose(2, 0, 1)
            )

        return img

    def __len__(self):
        return len(self.samples)
