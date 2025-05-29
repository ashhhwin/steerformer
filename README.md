# Steering Angle Prediction using Vision Transformers

This repository contains the implementation and evaluation of deep learning models for predicting the steering angle of a vehicle using only front-facing camera images. The goal is to explore the use of computer vision architectures—including Convolutional Neural Networks (CNNs), Vision Transformers (ViT), and Swin Transformers—for real-time, vision-only autonomous driving tasks.

The project assesses how effectively these models can generalize across different domains and lighting conditions, and evaluates their ability to interpret key road features such as lane markings and road curvature from RGB image sequences.

---

## Project Overview

Autonomous driving systems require accurate perception of the road to generate safe control decisions. In this project, we investigate the problem of steering angle prediction from monocular RGB images using supervised regression.

We train and evaluate three model architectures:

1. A custom-built Convolutional Neural Network (CNN)
2. A pretrained Vision Transformer (ViT)
3. A pretrained Swin Transformer

Each model is trained on a labeled dataset of driving sequences containing synchronized RGB frames and corresponding steering angles, and is evaluated both in-domain and cross-domain.

---

## Team Members

- Ashwin Ram Venkatraman  
- Anuja Tipare  
- Kanav Goyal  
- Swetha Subramanian

---

## Problem Statement

The aim of this project is to predict the instantaneous steering angle of a vehicle solely from front-facing RGB camera images. The model should be able to infer turning direction and magnitude by analyzing road features such as lane lines, road edges, and curvature, without relying on other sensory inputs like speed, GPS, or IMU data.

---

## Dataset Description

Two real-world driving datasets were used:

- **Dataset A**: ~45,000 RGB frames (unseen during training)
- **Dataset B**: ~19,000 RGB frames with steering-angle labels

Each image has a resolution of 640×480 pixels and a sampling rate of 20 Hz. The steering angle labels are extracted from the vehicle’s CAN bus.

### Data Splits

- **Training**: 80% of Dataset B
- **Validation**: 20% of Dataset B
- **Cross-domain Test**: Dataset A (completely unseen during training)

### Observations

- **Label Imbalance**: Approximately 73% of steering angles fall within ±1°, reflecting mostly straight driving. Turns (>10°) are rare and underrepresented.
- **Domain Shift**: Datasets differ in lighting, road geometry, and camera properties, significantly affecting generalization performance.
- **Missing Modalities**: No access to speed, IMU, or acceleration data. The system is trained to rely exclusively on visual cues.

---

## Assumptions

- The camera is fixed in a front-facing orientation.
- Lighting and weather conditions are assumed to be non-critical.
- There is minimal delay between image capture and steering label.
- Steering angle is the only control signal used for training.
- Vehicle speed is assumed constant or slowly varying across the dataset.

---

## Data Preprocessing

The following transformations were applied to each image:

- Resize to 224×224 pixels
- Normalize pixel values to [-1, 1]
- Convert to PyTorch tensor format

No data augmentation or noise injection was applied for this baseline comparison.

---

## Model Architectures

### 1. Convolutional Neural Network (CNN)

A simple CNN was trained from scratch, composed of four convolutional blocks with increasing channel depth, followed by flattening and a fully connected regression head.

- Activation: ReLU  
- Loss: Mean Squared Error (MSE)  
- Optimizer: Adam

### 2. Vision Transformer (ViT)

The ViT model is a pretrained transformer encoder originally trained on ImageNet-21k. For our task:

- Only the regression head is trained; the backbone remains frozen.
- Each image is split into 16×16 patches, which are linearly embedded and positionally encoded.
- Paper reference: [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)

### 3. Swin Transformer

Swin Transformer introduces hierarchical feature maps and local self-attention using shifted windows. It is also pretrained on ImageNet-1k.

- The backbone is frozen; only the regressor is trained.
- Paper reference: [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)

---

## Evaluation Metrics

- **Mean Absolute Error (MAE)** in degrees
- **Root Mean Square Error (RMSE)** in degrees

Performance is reported both for in-domain (B → B) and cross-domain (B → A) settings.

---

## Results

### In-Domain: Trained and Tested on Dataset B

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 3.15    | 9.50     |
| ViT       | 14.70   | 30.02    |
| Swin ViT  | 11.08   | 26.15    |

### Cross-Domain: Trained on B, Tested on A

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 16.20   | 28.39    |
| ViT       | 13.38   | 27.12    |
| Swin ViT  | 13.37   | 25.61    |

---

## Visualization & Interpretability

- **Inference Outputs**: Visual comparison of predicted vs. true steering angles
- **Grad-CAM**: Applied to CNN for visualizing feature importance
- **Attention Maps**: Extracted from ViT and Swin Transformer to examine global context usage

---

## Key Insights

- The CNN performed best on in-domain data, though it showed signs of overfitting.
- Transformer models underperformed due to limited training data and sensitivity to domain shift.
- Swin Transformer was relatively more robust to cross-domain testing than ViT.

---

## Limitations

- Dataset is imbalanced and lacks sharp-turn examples.
- Pretrained models were not fine-tuned end-to-end due to computational constraints.
- No use of multimodal sensor data, which would be essential in real-world driving.

---

## Future Work

- Collect more diverse data including extreme steering angles and adverse conditions.
- Incorporate lane segmentation or depth estimation as auxiliary tasks.
- Explore data augmentation techniques and domain adaptation strategies.
- Fine-tune transformer backbones with regularization and scheduling.

---

## Citation

If you use this work in your research or projects, please cite our project report or acknowledge the authors.

---

## License

This project is intended for academic and educational purposes only. For reuse or collaboration, please contact the contributors.
