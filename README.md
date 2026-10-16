# Mosaic Remover

A deep learning computer vision project that aims to detect mosaicked regions in images and restore the affected areas using PyTorch and convolutional neural networks.

## Overview

Mosaic Remover explores how deep learning can be used to identify and reconstruct image regions obscured by mosaic pixelation. The project focuses on developing a pipeline that combines **mosaic region detection** with **image restoration**.

The goal is to locate mosaicked areas, generate a mask identifying the affected pixels, and use a neural network to predict a restored version of the image.

This project is built with Python and PyTorch, using a U-Net architecture as the foundation for image-to-image learning.

## Project Showcase

<!-- Replace the placeholders below with your own images. -->

### 1. Original Image

The original image before mosaic processing.

![Original Image](result/clean.png)

### 2. Mosaicked Image

An example of an image with a mosaic effect applied to a selected region.

![Mosaicked Image](result/mosaic.png)

### 3. Detected Mosaic Region

The mosaic detector predicts a mask identifying the region affected by mosaic pixelation.

![Detected Mosaic Region](result/mask.png)

### 4. Restored Image

The restoration model attempts to reconstruct the mosaicked region based on learned image patterns.

![Restored Image](result/restored.png)

### 5. Before and After Comparison

A comparison between the mosaicked input and the model's restoration output.

| Mosaicked Input | Restoration Output |
|---|---|
| ![Input comparison](result/mosaic_compare.png) | ![Restoration comparison](result/restored_compare.png) |

> **Note:** These image paths are placeholders. Add your own example images to the `images/` directory and update the filenames as needed.

## Key Features

- **Mosaic Region Detection:** Train a neural network to identify regions affected by mosaic pixelation.
- **Mask Prediction:** Represent the detected region as a pixel-level mask.
- **Image Restoration:** Use a U-Net-based neural network to predict reconstructed image content.
- **Custom Dataset Pipeline:** Prepare paired training examples consisting of mosaicked images, original images, and corresponding masks.
- **Model Training and Validation:** Monitor training and validation losses to evaluate model performance.
- **GPU Acceleration:** Support CUDA-enabled PyTorch environments for faster training when compatible hardware and software are available.

## Model Architecture

The project uses a U-Net-style convolutional neural network for image-to-image prediction.

U-Net consists of an encoder, a bottleneck, and a decoder connected through skip connections.

- **Encoder:** Extracts image features at progressively lower spatial resolutions.
- **Bottleneck:** Learns higher-level representations of the input.
- **Decoder:** Reconstructs spatial information and produces a prediction at the target resolution.
- **Skip Connections:** Transfer features from encoder layers to corresponding decoder layers, helping preserve fine-grained spatial information.

The convolutional blocks use convolution layers, activation functions, and other feature-processing operations to learn patterns in the input images.

### Project Pipeline

```text
Original Image
      |
      v
Mosaic Generation
      |
      v
Mosaicked Image
      |
      v
Mosaic Detection Model
      |
      v
Predicted Mosaic Mask
      |
      v
Image Restoration Model
      |
      v
Restoration Prediction
      |
      v
Final Output Image
```

The detector and restoration model have different responsibilities: the detector identifies *where* the mosaic is located, while the restoration model predicts *what the affected image content might look like*.

The exact degree of integration between these stages depends on the current implementation.

## Technologies Used

- **Python** — Main programming language
- **PyTorch** — Deep learning framework
- **Torchvision** — Image processing and computer vision utilities
- **Pillow (PIL)** — Image loading and manipulation
- **NumPy** — Numerical operations and array processing
- **Matplotlib** — Visualizing images, masks, and training results
- **CUDA** — Optional GPU acceleration for supported NVIDIA GPUs

## Project Structure

An example organization for the project is shown below. Adjust the filenames to match the actual repository.

```text
mosaic-remover/
│
├── dataset/
│   ├── original/
│   ├── mosaicked/
│   └── masks/
│
├── images/
│   ├── original.png
│   ├── mosaicked.png
│   ├── detected_mask.png
│   └── restored.png
│
├── models/
│   ├── mosaic_detector.py
│   └── unet.py
│
├── scripts/
│   └── prepare_dataset.py
│
├── train.py
├── train_detector.py
├── predict.py
├── requirements.txt
└── README.md
```

This is a suggested layout, not a requirement. Keep your existing project structure if your scripts and dataset are organized differently.

## Dataset Preparation

The model learns from examples of images before and after mosaic processing.

A supervised training dataset can contain three corresponding items:

- **Original images:** The target images used to teach the restoration model.
- **Mosaicked images:** The inputs containing artificially or previously applied mosaic effects.
- **Binary masks:** Labels identifying the regions affected by mosaic pixelation.

For paired training, each mosaicked image should correspond to the correct original image. The mask should align with the affected region.

### Important Dataset Considerations

- Use consistent image dimensions and preprocessing for paired inputs and targets.
- Generate masks that accurately represent the mosaicked regions.
- Include different mosaic block sizes, locations, shapes, and image contents.
- Split the dataset into training, validation, and test sets.
- Avoid putting near-duplicate versions of the same original image in different splits.

**Important:** Resizing an already mosaicked image can change the appearance of the mosaic blocks and reduce the visual evidence needed for detection. The preprocessing pipeline should preserve relevant mosaic patterns, and training examples should reflect the conditions expected during inference.

## Training

The detector and restoration model may require separate training scripts.

Example commands, depending on the filenames in your repository:

```bash
python train_detector.py
```

```bash
python train.py
```

During training, the models learn from the training dataset and are evaluated against a separate validation dataset.

Typical training metrics include:

- **Training loss:** Measures the model's error on training examples.
- **Validation loss:** Measures error on unseen validation examples during training.
- **L1 loss:** Measures the average absolute difference between predicted and target pixel values.
- **SSIM:** Measures structural similarity between images.

Lower L1 loss generally indicates smaller pixel-wise errors. Higher SSIM generally indicates greater structural similarity. Neither metric alone guarantees that reconstructed details are accurate.

The exact losses, optimizer, learning rate, and number of epochs depend on the training configuration.

```bash
python predict.py
```

The command and its arguments should be updated to match the actual implementation.

