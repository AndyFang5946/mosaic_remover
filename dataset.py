import os
from typing import Optional

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode 

from mosiac import apply_random_mosiac

class MosaicDataset(Dataset):
    def __init__(
        self,
        directory: str,
        image_size: int = 256,
        min_mosaic_size: int = 48, 
        max_mosaic_size: int = 128,
        min_regions: int = 1,
        max_regions: int = 3,
        block_sizes=(8, 12 , 16, 24),
        training: bool = True,
        seed: int = 1234,
        augment: bool = False
    ):
        super().__init__()
        self.directory=directory
        self.image_size=image_size
        self.min_mosaic_size=min_mosaic_size
        self.max_mosaic_size=max_mosaic_size
        self.min_regions=min_regions
        self.max_regions=max_regions
        self.block_sizes=tuple(block_sizes)
        self.training=training
        self.seed=seed
        self.augment=augment

        self.images = [
            file for file in os.listdir(directory)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not self.images:
            raise RuntimeError(f"No images found in :{directory}")

    def __len__(self):
        return len(self.images)

    def _load(self, index: int) -> Image.Image:
        path = os.path.join(
            self.directory,
            self.images[index]
        )

        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, index):
        clean = self._load(index)
        clean = TF.resize(
            clean,
            [self.image_size, self.image_size],
            interpolation=(InterpolationMode.BILINEAR),
            antialias=True
        )

        if(self.training and self.augment):
            if torch.rand(()) < 0.5:
                clean = TF.hflip(clean)
            if torch.rand(()) < 0.2:
                clean = TF.vflip(clean)

        if self.training:
            corruption_seed = None
        else:
            corruption_seed = (self.seed + index)

        mosaic, mask = (
            apply_random_mosiac(
                clean,
                min_size=self.min_mosaic_size,
                max_size=self.max_mosaic_size,
                min_regions=self.min_regions,
                max_regions=self.max_regions,
                block_sizes=self.block_sizes,
                seed = corruption_seed
            )
        )

        clean_tensor = TF.to_tensor(clean)
        mosaic_tensor = TF.to_tensor(mosaic)
        mask_tensor = TF.to_tensor(mask)

        mask_tensor = (mask_tensor > 0.5).float()

        # print("PIL mask range:", mask.getextrema())
        # print("Tensor mask min/max:",
        #     mask_tensor.min().item(),
        #     mask_tensor.max().item())
        # print("Masked pixels:", mask_tensor.sum().item())

        return mosaic_tensor, clean_tensor, mask_tensor