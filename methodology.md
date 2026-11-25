# Methodology

## 1. Problem Definition

This project focuses on semantic segmentation of high-resolution satellite imagery using deep learning.

Semantic segmentation assigns a land-cover category to every pixel in an image. The objective is to identify seven different land-cover classes from satellite images using convolutional neural networks.

The project investigates and compares three segmentation architectures:

1. U-Net
2. ResNet50-U-Net
3. DeepLabV3+

## 2. Dataset Description

The project uses the LoveDA (Land-cover Dataset for Domain Adaptive Semantic Segmentation) dataset, obtained from Kaggle.

### Dataset configuration

| Parameter | Value |
|---|---|
| Dataset | LoveDA |
| Original image resolution | 1024 × 1024 |
| Model input resolution | 256 × 256 |
| Total image-mask pairs | 4,191 |
| Number of classes | 7 |
| Framework | TensorFlow / Keras |

### Land-cover classes

| Class ID | Category |
|---|---|
| 0 | Background |
| 1 | Building |
| 2 | Road |
| 3 | Water |
| 4 | Barren |
| 5 | Forest |
| 6 | Agriculture |

The original no-data label is mapped to 255 and excluded from loss and metric calculations.

## 3. Data Preprocessing

### 3.1 Image-mask alignment

Each satellite image is paired with its corresponding segmentation mask. Filename alignment is verified to ensure that every image is associated with the correct ground-truth annotation.

### 3.2 Mask remapping

Original mask labels are remapped to the model's seven-class label space.

The no-data label is assigned 255 as an ignore label.

### 3.3 Dataset splitting

The dataset is divided into three subsets:

- Training: 70%
- Validation: 30%

The test set is kept separate for final evaluation.

### 3.4 Image cropping

Original 1024 × 1024 images are converted into 256 × 256 crops.

Random cropping is used to expose the model to different spatial regions of satellite images.

The same crop coordinates are applied to the image and its corresponding segmentation mask to preserve pixel-level alignment.

## 4. Model Architectures

Three deep learning architectures are implemented and evaluated.

### 4.1 U-Net

U-Net follows an encoder-decoder architecture with skip connections.

The encoder extracts increasingly abstract visual features, while the decoder reconstructs spatial resolution for pixel-level prediction.

Configuration:

- Input shape: 256 × 256 × 3
- Four encoder levels
- Initial filters: 32
- Bottleneck filters: 1024
- Output classes: 7
- Output activation: Softmax

Skip connections transfer spatial information from the encoder to the decoder.

### 4.2 ResNet50-U-Net

This architecture combines a pretrained ResNet50 encoder with a U-Net-style decoder.

ResNet50 is initialized using ImageNet pretrained weights.

The encoder extracts hierarchical image features, and the decoder progressively upsamples them to produce segmentation maps.

The architecture uses multiple intermediate ResNet feature maps as skip connections.

### 4.3 DeepLabV3+

DeepLabV3+ combines multi-scale contextual feature extraction with a decoder that recovers spatial details.

#### Backbone

A pretrained ResNet50 backbone initialized with ImageNet weights is used.

Selected feature maps:

- Low-level features: `conv2_block3_out`
- High-level features: `conv4_block6_out`

The backbone is frozen during the transfer-learning experiment.

#### Atrous Spatial Pyramid Pooling (ASPP)

ASPP captures contextual information at multiple spatial scales using parallel branches.

The implementation contains:

1. 1 × 1 convolution branch
2. 3 × 3 atrous convolution with dilation rate 6
3. 3 × 3 atrous convolution with dilation rate 12
4. 3 × 3 atrous convolution with dilation rate 18
5. Global average pooling branch

Each branch produces 256 feature channels.

The outputs are concatenated and projected using a 1 × 1 convolution.

#### Decoder

The ASPP output is upsampled and combined with projected low-level ResNet features.

The decoder uses two 3 × 3 convolution layers with 256 filters each.

The resulting features are upsampled to the original input resolution.

#### Segmentation head

A final 1 × 1 convolution generates seven output channels.

Softmax activation produces per-pixel class probabilities.

**Implementation note:** This is a ResNet50-based OS16-style variant using the conv4 feature stage as its high-level input. It does not modify the canonical ResNet50 stride/dilation configuration.

## 5. Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.0001 for DeepLabV3+ |
| Loss | Weighted sparse categorical cross-entropy |
| Primary metric | Masked Mean IoU |
| Input resolution | 256 × 256 |
| Output classes | 7 |
| Maximum epochs | 200 |
| Hardware | Google Colab T4 GPU |

The same weighted loss and masked metric are used in the architecture experiments to support meaningful comparisons.

The masked metric excludes pixels with the ignore label 255.

## 6. Evaluation Metric

### Mean Intersection over Union (mIoU)

Intersection over Union measures the overlap between predicted and ground-truth segmentation regions.

\[
IoU = \frac{TP}{TP+FP+FN}
\]

Mean IoU is calculated by averaging IoU across the seven land-cover classes.

\[
mIoU = \frac{1}{7}\sum_{i=1}^{7} IoU_i
\]

Pixels assigned the ignore label are excluded from the evaluation.

## 7. Experimental Results

The following are the best observed validation mIoU values from the experiments.

| Architecture | Validation mIoU |
|---|---:|
| U-Net | 36.53% |
| ResNet50-U-Net | 60.33% |
| DeepLabV3+ | 62.15% |

DeepLabV3+ currently achieves the highest observed validation mIoU of 62.15%, recorded at epoch 40.

These values are validation results, not final test-set results.

## 8. Experimental Limitations

- The current experiments use a limited set of architecture configurations.
- The ResNet50-based DeepLabV3+ implementation uses a selected feature stage rather than the canonical modified-stride ResNet50 configuration.
- Validation performance alone does not establish generalization to unseen test data.
- Further experiments are required to understand class-wise performance and model behavior.

## 9. Future Work

Potential future improvements include:

- Final evaluation on the held-out test set.
- Per-class IoU analysis.
- Qualitative prediction visualization.
- Fine-tuning pretrained backbone layers.
- Ablation studies on ASPP dilation rates.
- Comparison with additional semantic segmentation architectures.
- Analysis of performance on different land-cover categories.

---

**Project:** Satellite Image Semantic Segmentation using Deep Learning

**Dataset:** LoveDA

**Framework:** TensorFlow / Keras