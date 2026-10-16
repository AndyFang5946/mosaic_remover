from dataclasses import dataclass

@dataclass
class Config:
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    test_dir = "dataset/test"

    image_size = 256

    min_mosaic_size = 48
    max_mosaic_size = 128
    min_regions = 1
    max_regions = 3
    block_sizes = (8, 12, 16, 24)

    batch_size = 8
    num_workers = 2
    base_channels = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 30
    patience = 8
    grad_clip = 1.0
    ssim_weight = 0.2

    seed = 42
    validation_seed = 1234

    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
