# Steering Angle Prediction using Vision Transformers

This repository contains the code and experimental results for predicting a vehicle’s steering angle from front-facing camera images using deep learning models. The project compares traditional Convolutional Neural Networks (CNNs) with modern Transformer-based architectures like the Vision Transformer (ViT) and Swin Transformer.

The models are trained and evaluated on driving video data captured from real-world driving scenarios. The goal is to assess the models' performance in both in-domain and cross-domain prediction of steering angles, based solely on visual input.

---

## Objective

To build a computer vision-based regression model that predicts the steering angle of a vehicle in real-time using only monocular dashboard camera footage. This contributes to research in autonomous driving by investigating the capability of ViT-based models in steering control tasks.

---

## Dataset

The dataset used for this project is publicly available at:

> [https://github.com/SullyChen/driving-datasets](https://github.com/SullyChen/driving-datasets)

### Description

- Dashcam videos captured from a front-facing camera during real driving sessions.
- Each video frame is paired with a continuous steering angle value in degrees.
- Videos were recorded under varying lighting and road conditions to simulate realistic driving environments.

### Preprocessing

- Frames extracted from video at 20 FPS.
- Resized to 224×224 pixels.
- Pixel values normalized to [-1, 1].
- Converted into PyTorch tensors.

### Data Split Strategy

- Training and validation conducted on one subset of driving videos.
- A different subset with distinct road/camera settings is reserved for cross-domain generalization testing.

---

## Assumptions

- Vehicle speed and IMU data are unavailable and assumed constant.
- No temporal context is used (single-frame prediction).
- Camera orientation is front-facing and fixed across samples.
- No latency between captured frame and steering label.

---

## Models

### 1. Convolutional Neural Network (CNN)

A simple architecture consisting of stacked convolutional layers followed by a fully connected regressor.

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

### 2. Vision Transformer (ViT)

Pretrained ViT-B/16 model (trained on ImageNet-21k) used with a custom regression head.

- Only the regression head is trained; the transformer encoder is frozen.
- Each image is divided into 16×16 patches and embedded as tokens.
- Reference: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)

### 3. Swin Transformer

A hierarchical vision transformer using local window-based self-attention. Also pretrained on ImageNet-1k.

- Encoder frozen; only regression head fine-tuned.
- Reference: [Liu et al., 2021](https://arxiv.org/abs/2103.14030)

---

## Results

### In-Domain Evaluation (Trained and Tested on Same Subset)

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 3.15    | 9.50     |
| ViT       | 14.70   | 30.02    |
| Swin ViT  | 11.08   | 26.15    |

### Cross-Domain Evaluation (Generalization Test)

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 16.20   | 28.39    |
| ViT       | 13.38   | 27.12    |
| Swin ViT  | 13.37   | 25.61    |

---

## Interpretability Tools

- **Grad-CAM**: Applied to CNN to visualize the spatial attention of convolutional filters.
- **Attention Maps**: Extracted from ViT and Swin Transformer to observe token-level importance.

---

## Key Observations

- CNNs outperform ViTs on small datasets but tend to overfit.
- Swin Transformer showed better cross-domain robustness than ViT.
- Label imbalance (dominated by straight-driving) and lack of speed context limit model performance.
- Vision-only models face limitations in generalization without temporal or multimodal inputs.

---

## Future Work

- Use time-series input (3D CNN or Transformer with temporal embeddings).
- Augment data to balance left/right turn samples.
- Incorporate additional sensors like IMU, speed, and GPS.
- Add auxiliary tasks like lane segmentation to enhance spatial reasoning.

---

  
