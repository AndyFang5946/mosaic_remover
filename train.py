import os
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config
from dataset import MosaicDataset
from loss import reconstruction_loss 
from model import UNet
from utils import count_parameters, save_comparison, seed_everything

# import inspect

# print("UNet class:", UNet)
# print("UNet source file:", inspect.getfile(UNet))
# print("UNet constructor:", inspect.signature(UNet.__init__))

def run_validation(model, loader, device, ssim_weight, output_dir, epoch):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0 

    first_batch = None

    with torch.no_grad():
        for(batch_idx, (mosaic, clean, mask)) in enumerate(loader):
            mosaic = mosaic.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            model_input = torch.cat([mosaic, mask], dim = 1)

            logits = model(model_input)
            prediction = torch.sigmoid(logits)

            loss, l1, ssim = reconstruction_loss(
                prediction,
                clean,
                mask,
                ssim_weight
            )

            total_loss += loss.item()
            total_l1 += l1.item()
            total_ssim += ssim.item()

            if first_batch is None:
                first_batch = (
                    mosaic,
                    mask,
                    prediction,
                    clean
                )
    
    count = len(loader)
    avg_loss = total_loss/count
    avg_l1 = total_l1/count
    avg_ssim = total_ssim/count

    if first_batch is not None:
        os.makedirs(output_dir, exist_ok=True)

        save_comparison(*first_batch, os.path.join(output_dir, f"vallidation_epoch_{epoch:03d}.png"))

    return avg_loss, avg_l1, avg_ssim

def main():
    cfg = Config()
    seed_everything(cfg.seed)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device}")

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
        pin_memory=use_amp,
        persistent_workers=cfg.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_amp,
        persistent_workers=cfg.num_workers > 0
    )

    model = UNet(
        in_channels=4,
        out_channels=3,
        base_channels=cfg.base_channels
    ).to(device)

    print("trainable parameters:", f"{count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    scaler = GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0

        for (mosaic, clean, mask) in train_loader:
            # print("Mosaic shape:", mosaic.shape)
            # print("Clean shape:", clean.shape)
            # print("Mask shape:", mask.shape)
            # print("Mask min/max:", mask.min().item(), mask.max().item())
            # print("Masked pixels:", mask.sum().item())
            # print("Mosaic range:", mosaic.min().item(), mosaic.max().item())
            # print("Clean range:", clean.min().item(), clean.max().item())

            mosaic = mosaic.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            model_input = torch.cat([mosaic, mask], dim = 1)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                prediction = torch.sigmoid(model(model_input))

                loss, l1, ssim = reconstruction_loss(
                    prediction,
                    clean,
                    mask,
                    cfg.ssim_weight
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_l1 += l1.item()
            running_ssim += ssim.item() 

        train_loss = running_loss/len(train_loader)
        train_l1 = running_l1/len(train_loader)
        train_ssim = running_ssim/len(train_loader)

        val_loss, val_l1, val_ssim = run_validation(
            model, 
            val_loader,
            device,
            cfg.ssim_weight,
            cfg.output_dir,
            epoch
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg.epochs} | "

            f"train="
            f"{train_loss:.5f} "

            f"(L1="
            f"{train_l1:.5f}, "

            f"SSIM="
            f"{train_ssim:.5f}) | "

            f"val="
            f"{val_loss:.5f} "

            f"(L1="
            f"{val_l1:.5f}, "

            f"SSIM="
            f"{val_ssim:.5f}) | "

            f"lr="
            f"{lr:.2e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state":model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "val_loss":val_loss,
            "config":cfg.__dict__
        }

        torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0

            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, "best.pt"))

            print(" -> saved new best checkpoint")
        else:
            epochs_without_improvement += 1 

        if(epochs_without_improvement >= cfg.patience):
            print("Early stopping.")
            break

    print(
        f"Best validation loss: "
        f"{best_val:.6f}"
    )

if __name__ == "__main__":
    main()







