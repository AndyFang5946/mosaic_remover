import random
from typing import Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np

def _make_shape_mask(
    width: int,
    height:int,
    rng: random.Random,
    min_size: int,
    max_size: int 
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    shape = rng.choice(["rec", "ell", "pol"])
    w = rng.randint(min_size, min(max_size, width))
    h = rng.randint(min_size, min(max_size, height))
    x = rng.randint(0, width-w)
    y = rng.randint(0, height-h)

    if shape == "rec":
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill = 255)
    elif shape == "ell":
        draw.ellipse((x, y, x + w - 1, y + h - 1), fill = 255)
    else: 
        points = []
        cx = x + w // 2
        cy = y + h // 2

        n = rng.randint(7, 12)

        for i in range(n):
            angle = 2 * np.pi * i / n
            radius = rng.uniform(0.65, 1.0)
            px = int(np.clip(cx + np.cos(angle) * (w/2) * radius, 0, width - 1))
            py = int(np.clip(cy + np.sin(angle) * (h/2) * radius, 0, height - 1))

            points.append((px, py))

        draw.polygon(points, fill=255)

    return mask

def apply_mosiac_to_mask(
    image:Image.Image,
    mask:Image.Image,
    block_size: int
) -> Image.Image:
    image_array = np.asanyarray(image).copy()
    mask_array = np.asanyarray(mask).copy()

    height, width = mask_array.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y2 = min(y + block_size, height)
            x2 = min(x + block_size, width)

            block_mask = (mask_array[y:y2, x:x2] > 0)
            if not block_mask.any():
                continue

            block = image_array[y:y2, x:x2]
            selected = block[block_mask]

            average_color = (selected.mean(axis=0).astype(np.uint8))
            block[block_mask] = average_color

    return Image.fromarray(image_array, mode = "RGB")

def apply_random_mosiac(
        image: Image.Image,
        min_size: int = 48,
        max_size: int = 128,
        min_regions: int = 1,
        max_regions: int = 3,
        block_sizes=(8, 12, 16, 24),
        seed: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]: 
    if image.mode != "RGB":
        image = image.convert("RGB")
    rng = random.Random(seed)
    width, height = image.size

    mask = Image.new("L", (width, height), 0)
    regions = rng.randint(min_regions, max_regions)

    for _ in range(regions):
        shape_mask = _make_shape_mask(
            width,
            height,
            rng,
            min_size,
            max_size
        )

        mask = Image.fromarray(
            np.maximum(
                np.asarray(mask),
                np.asarray(shape_mask)
            ).astype(np.uint8),
            mode="L"
        )

    block_size = rng.choice(list(block_sizes))
    mosaicked = apply_mosiac_to_mask(image, mask, block_size)

    return mosaicked, mask
