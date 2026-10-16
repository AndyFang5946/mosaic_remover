import os

import torch

from torch.utils.data import (
    DataLoader
)

from torchvision.utils import (
    make_grid,
    save_image
)

from config import Config

from dataset import (
    MosaicDataset
)


def main():

    cfg = Config()

    os.makedirs(
        cfg.output_dir,
        exist_ok=True
    )

    dataset = MosaicDataset(

        cfg.train_dir,

        image_size=cfg.image_size,

        min_mosaic_size=(
            cfg.min_mosaic_size
        ),

        max_mosaic_size=(
            cfg.max_mosaic_size
        ),

        min_regions=(
            cfg.min_regions
        ),

        max_regions=(
            cfg.max_regions
        ),

        block_sizes=(
            cfg.block_sizes
        ),

        training=True,

        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    (
        mosaic,
        clean,
        mask
    ) = next(
        iter(loader)
    )

    mask_rgb = mask.repeat(
        1,
        3,
        1,
        1
    )

    grid = make_grid(

        torch.cat(
            [
                mosaic,
                mask_rgb,
                clean
            ],
            dim=0
        ),

        nrow=8,

        padding=4
    )

    save_image(

        grid,

        os.path.join(
            cfg.output_dir,
            "dataset_preview.png"
        )
    )

    print(
        "Saved "
        "outputs/dataset_preview.png"
    )


if __name__ == "__main__":

    main()