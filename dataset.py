import os
import random
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset

from time import time
from torchvision.transforms import InterpolationMode, functional
BILINEAR = InterpolationMode.BILINEAR
from multiprocessing.pool import ThreadPool
from torchvision.datasets.utils import verify_str_arg
import pandas as pd
from imagenet import imagenet_templates, imagenet_classes
import linecache
from tqdm import tqdm
import ast

class TextOOD(Dataset):

    def __init__(self, filename):
        ext = filename.split(".")[-1]
        f = open(filename, 'r')
        words = f.read().splitlines()
        templates = ["This is a photo of a {}."]
        self.texts = np.array([template.format(word) for word in words for template in templates])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class OOD(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=None)
 
        self._data_dir = Path(self.root) / "images"
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")
        names = os.listdir(self._data_dir)
        self._image_files = [self._data_dir / n for n in names]

        name = self.root.split("/")[-1]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        image = Image.open(image_file).convert("RGB")
        label = 0

        if self.transform:
            image = self.transform(image)

        return image, label

    def _check_exists(self) -> bool:
        print(self._data_dir)
        return self._data_dir.is_dir()
