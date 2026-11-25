# Satellite Image Semantic Segmentation using Deep Learning

A comparative study of deep learning architectures for semantic segmentation of high-resolution satellite imagery using the **LoveDA dataset** and TensorFlow/Keras.

## Overview

Satellite image segmentation is an important computer vision task used in applications such as urban planning, land-use monitoring, environmental analysis, and geographical information systems.

The objective of this project is to classify every pixel in a satellite image into one of seven land-cover categories.

I implemented and compared three architectures:

1. U-Net
2. ResNet50-based U-Net
3. DeepLabV3+

The experiments explore how encoder architectures, transfer learning, and multi-scale contextual feature extraction affect segmentation performance.

## Dataset

**LoveDA (Land-cover Dataset for Domain Adaptation)**
This project uses the LoveDA (Land-cover Dataset for Domain Adaptive Semantic Segmentation) dataset, obtained from Kaggle.

Data Source: https://www.kaggle.com/datasets/growinfame/loveda

- Total image-mask pairs: 4,191
- Original image resolution: 1024 × 1024
- Model input resolution: 256 × 256
- Number of segmentation classes: 7

### Classes

| Class ID | Land-cover category |
|---|---|
| 0 | Background |
| 1 | Building |
| 2 | Road |
| 3 | Water |
| 4 | Barren |
| 5 | Forest |
| 6 | Agriculture |

The original dataset's no-data label (0) is mapped to 255 and ignored during loss and metric calculations.

### Data split

- Training: 70%
- Validation: 30%


Random cropping is used to generate 256 × 256 training samples from the original images.

## Model Architectures

### 1. U-Net

A U-Net architecture with an encoder-decoder structure and skip connections.

- Four encoder levels
- Initial filters: 32
- Bottleneck filters: 1024
- Seven-class softmax output

### 2. ResNet50-U-Net

A U-Net decoder connected to a pretrained ResNet50 encoder.

- ImageNet-pretrained ResNet50
- Transfer learning
- Multi-level feature extraction
- Skip connections between encoder and decoder

### 3. DeepLabV3+

A ResNet50-based DeepLabV3+ implementation with Atrous Spatial Pyramid Pooling (ASPP).

Key components:

- ImageNet-pretrained ResNet50 backbone
- Low-level features from `conv2_block3_out`
- High-level features from `conv4_block6_out`
- ASPP branches with dilation rates 6, 12, and 18
- Global average pooling branch
- Low-level feature projection
- Decoder with two 3 × 3 convolution layers
- Seven-class pixel-wise softmax output

The implementation uses an OS16-style feature extraction configuration with the ResNet50 `conv4` stage as its high-level feature source. It does not modify the canonical ResNet50 stride/dilation configuration.

## Experimental Results

| Architecture | Best validation mIoU |
|---|---:|
| U-Net | 36.53% |
| ResNet50-U-Net | 60.33% | after 92 epochs
| DeepLabV3+ | **62.15%** | after 40 epochs

*These are observed validation results, not final test-set results. A fair comparison requires the same validation protocol and metric implementation.*

### Best result

**DeepLabV3+ — 62.15% validation mIoU**

Observed at epoch 40.

## Evaluation Metrics

The primary evaluation metric is **Mean Intersection over Union (mIoU)**.

mIoU measures the overlap between predicted segmentation masks and ground-truth masks across the seven classes.

The custom `MaskedMeanIoU` metric ignores pixels labelled 255 so that no-data regions do not affect the evaluation.

## Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab
- Google Colab T4 GPU

## Project Workflow

1. Load satellite images and corresponding segmentation masks.
2. Verify image-mask alignment.
3. Remap original mask labels.
4. Split the dataset into training, validation, and testing sets.
5. Generate 256 × 256 image crops.
6. Build and train U-Net.
7. Build and train ResNet50-U-Net.
8. Build and train DeepLabV3+.
9. Compare validation mIoU across architectures.
10. Evaluate the selected model on the held-out test dataset.

## Future Improvements

- Evaluate all architectures on the held-out test set.
- Perform per-class segmentation analysis.
- Visualize predicted masks against ground-truth masks.
- Investigate class imbalance and difficult land-cover categories.
- Experiment with fine-tuning strategies for pretrained backbones.
- Explore additional segmentation architectures and ablation studies.

## Current Status

**Completed:**
- Dataset preprocessing pipeline
- U-Net implementation
- ResNet50-U-Net implementation
- DeepLabV3+ implementation
- Initial architecture comparison

**Next:**
- Final test-set evaluation
- Prediction visualization
- Further experiments and analysis



**Project Focus:** Computer Vision | Semantic Segmentation | Satellite Image Analysis | Deep Learning

