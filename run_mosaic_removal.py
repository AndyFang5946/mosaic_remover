import os
import torch
import numpy as np
from PIL import Image

from model import UNet
from dataset import MosaicDataset

def load_model(checkpoint_path, device, remover):
    if remover == True: 
        model = UNet(
            in_channels=4,
            out_channels=3,
            base_channels=32
        ).to(device)
    else:
        model = UNet(
            in_channels=3,
            out_channels=1,
            base_channels=32
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval() 

    return model

def load_image(image_path, image_size=256):
    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    scale = min(
        image_size / original_width,
        image_size / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        (0, 0, 0)
    )

    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2

    canvas.paste(image, (left, top))

    array = np.asarray(canvas, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    return tensor

def extract_sample(dataset, index=0, device="cpu"):
    mosaic, clean, true_mask = dataset[index]

    mosaic = mosaic.unsqueeze(0)
    clean = clean.unsqueeze(0)
    true_mask = true_mask.unsqueeze(0)

    mosaic = mosaic.to(device)
    clean = clean.to(device)
    true_mask = true_mask.to(device)

    return mosaic, clean, true_mask

def save_mask(mask, output_path):
    array = mask[0, 0].detach().cpu().numpy()
    array = (array * 255).astype(np.uint8)

    Image.fromarray(array).save(output_path)

def save_RGB(tensor, ouput_path):
    array = tensor[0].detach().cpu().clamp(0, 1).numpy()
    array = (array.transpose(1, 2, 0) * 255).round() 
    array = array.astype(np.uint8)

    Image.fromarray(array).save(ouput_path)

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(project_dir, "result")
    input_dir = os.path.join(project_dir, "input")
    checkpoint_dir = os.path.join(project_dir, "checkpoints")

    os.makedirs(output_dir, exist_ok=True)

    detector_path = os.path.join(checkpoint_dir, "best_detector.pt")
    remover_path = os.path.join(checkpoint_dir, "best.pt")
    input_path = os.path.join(input_dir, "input.png")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    detector = load_model(detector_path, device, False)
    remover = load_model(remover_path, device, True)

    #image = load_image(input_path)
    val_dataset = MosaicDataset(
        directory="dataset/validation",
        image_size=256,
        training=False,
        seed=1234,
        augment=False
    )

    # Extract one validation sample
    mosaic, clean, true_mask = extract_sample(
        val_dataset,
        index=0,
        device=device
    )

    image = mosaic.to(device)

    with torch.no_grad():
        logits = detector(image)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= 0.5).float()

    remover_input = torch.cat([image, mask], dim = 1)

    with torch.no_grad():
        prediction = remover(remover_input)
        prediction = torch.sigmoid(prediction)

    restored = prediction * mask + image * (1.0 - mask)

    save_mask(mask, os.path.join(output_dir, "mask.png"))
    save_RGB(restored, os.path.join(output_dir, "restored.png"))
    save_RGB(mosaic, os.path.join(output_dir, "mosaic.png"))
    save_RGB(clean, os.path.join(output_dir, "clean.png"))

if __name__ == "__main__":
    main()






