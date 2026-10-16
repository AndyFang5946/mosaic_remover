import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_detector import Config_D
from model import UNet
from dataset import MosaicDataset
from utils import save_comparison

def dice_loss(logits, targets, smooth=1.0):
    prob = torch.sigmoid(logits)

    intersection = (prob * targets).sum(dim = (2, 3))
    deno = prob.sum(dim = (2, 3)) + targets.sum(dim = (2, 3))

    dice = (2* intersection + smooth)/(deno + smooth)

    return 1 - dice.mean()

def evaluate(model, loader, bce_loss, device, output_dir, epoch):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_images = 0.0
    total_bce = 0.0

    first_batch = None

    with torch.no_grad():
        for images, clean, masks in loader: 
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            bce_l = bce_loss(logits, masks)
            loss = bce_l + dice_loss(logits, masks)
            prediction = (torch.sigmoid(logits) >= 0.5).float()

            intersection = (prediction * masks).sum(dim = (1, 2, 3))
            denominator = (prediction.sum(dim=(1, 2, 3)) + masks.sum(dim = (1, 2, 3)))
            dice = (2 * intersection + 1e-6)/(denominator + 1e-6)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_dice += dice.sum().item()
            total_bce += bce_l.item() * batch_size
            total_images += batch_size

            if first_batch is None:
                first_batch = (images, masks, prediction, clean)

    if first_batch is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_comparison(*first_batch, os.path.join(output_dir, f"detector_validation_epoch{epoch:03d}.png"))

    return total_loss/ max(total_images, 1), total_dice/max(total_images, 1), total_bce/max(total_images, 1)

def main():
    cfg = Config_D()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print("Using: ", device)

    train_dataset = MosaicDataset(
        cfg.train_dir,
        image_size=cfg.image_size,
        min_mosaic_size=cfg.min_mosaic_size,
        max_mosaic_size=cfg.max_mosaic_size,
        min_regions=cfg.min_regions,
        max_regions=cfg.max_regions,
        block_sizes=cfg.block_sizes,
        training=True,
        augment=True,
        seed=cfg.seed
    )

    val_dataset = MosaicDataset(
        cfg.val_dir,
        image_size=cfg.image_size,
        min_mosaic_size=cfg.min_mosaic_size,
        max_mosaic_size=cfg.max_mosaic_size,
        min_regions=cfg.min_regions,
        max_regions=cfg.max_regions,
        block_sizes=cfg.block_sizes,
        training=False,
        augment=False,
        seed=cfg.validation_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0
    )

    model = UNet(
        in_channels=3, 
        out_channels=1,
        base_channels=cfg.base_channels
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([3.0], device = device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6
    )

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        image_count = 0

        for images, clean, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            image_count += batch_size

        train_loss = running_loss/max(image_count, 1)

        val_loss, val_dice, val_bce = evaluate(model, val_loader, bce_loss, device, cfg.output_dir, epoch)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch: {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {val_dice:.4f} "
            f"Val BCE: {val_bce:.4f} | "
            f"LR: {lr:.2e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state":model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "val_loss":val_loss,
            "config":cfg.__dict__
        }

        torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "last_detector.pt"))

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0

            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "best_detector.pt"))

            print(" -> saved new best checkpoint")
        else:
            epochs_without_improvement += 1 

        if(epochs_without_improvement >= cfg.patience):
            print("Early stopping.")
            break

if __name__ == "__main__":
    main()

