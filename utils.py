import os
import random

import numpy as np
import torch

from torchvision.utils import make_grid, save_image

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_comparison(mosaic, mask, prediction, clean, path, n=4):
    # Limit the number of images displayed.
    n = min(n, mosaic.shape[0])

    # Move tensors to CPU and select the first n images.
    mosaic = mosaic[:n].detach().cpu().clamp(0, 1)
    mask = mask[:n].detach().cpu().clamp(0, 1)
    prediction = prediction[:n].detach().cpu().clamp(0, 1)
    clean = clean[:n].detach().cpu().clamp(0, 1)

    # Convert one-channel masks to three channels for visualization.
    # [B, 1, H, W] -> [B, 3, H, W]
    if mask.shape[1] == 1:
        mask_display = mask.repeat(1, 3, 1, 1)
    else:
        mask_display = mask

    if prediction.shape[1] == 1:
        prediction_display = prediction.repeat(1, 3, 1, 1)
    else:
        prediction_display = prediction

    # Keep the original mosaic pixels outside the predicted mask.
    reconstructed = prediction * mask + mosaic * (1 - mask)

    # If prediction is one-channel, broadcasting applies the mask
    # to all three RGB channels. The result remains an RGB image.

    rows = []

    for i in range(n):
        row = torch.stack([
            mosaic[i],                 # RGB mosaic image
            mask_display[i],           # Ground-truth mask
            prediction_display[i],     # Predicted mask
            reconstructed[i],          # Image using predicted mask
            clean[i]                   # Original clean image
        ])

        rows.append(row)

    # Each row contains 5 RGB images.
    # Concatenate all rows and arrange 5 images per grid row.
    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=5,
        padding=4
    )

    save_image(grid, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
