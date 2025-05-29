# Steering Angle Prediction using Vision Transformers

This repository contains the code and experimental results for predicting a vehicle’s steering angle from front-facing camera images using deep learning models. The project compares traditional Convolutional Neural Networks (CNNs) with modern Transformer-based architectures like the Vision Transformer (ViT) and Swin Transformer.

The models are trained and evaluated on real-world driving video data. The objective is to evaluate how well each architecture performs on both in-domain and cross-domain steering angle prediction tasks using only visual input.

---

## Objective

To build a computer vision-based regression model that predicts the steering angle of a vehicle in real-time using monocular dashboard camera footage. This work contributes to research in autonomous driving by exploring the feasibility of ViT-based models in steering control without requiring multimodal inputs.

---

## Team Members

- Ashwin Ram Venkatraman  
- Anuja Tipare  
- Kanav Goyal  
- Swetha Subramanian

---

## Dataset

The dataset used for this project is publicly available at:

> [https://github.com/SullyChen/driving-datasets](https://github.com/SullyChen/driving-datasets)

### Dataset Description

Two real-world driving datasets were used:

- **Dataset A**: ~45,000 RGB frames (completely unseen during training)
- **Dataset B**: ~19,000 RGB frames with synchronized steering-angle labels

Each frame has a resolution of **640×480 pixels**, captured at **20 Hz**. Steering angles are recorded from the vehicle’s CAN bus system.

### Data Splits and Observations

- **Training Set**: 80% of Dataset B
- **Validation Set**: 20% of Dataset B
- **Cross-Domain Test Set**: Dataset A
- **Label Imbalance**: 73% of frames have |θ| < 1°, reflecting straight-driving dominance. High-angle turns are rare.
- **Domain Shift**: Visual mismatch between Dataset A and B in lighting, road geometry, and camera type.
- **Missing Modalities**: No speed, IMU, or GPS data available; model relies entirely on RGB images.

---

## Preprocessing

- Video frames extracted at 20 FPS
- Resized to 224×224 pixels
- Pixel values normalized to [-1, 1]
- Converted to PyTorch tensors

---

## Assumptions

- Vehicle speed is constant or slowly varying across the dataset
- No significant delay between frame capture and steering label
- The camera is fixed in a front-facing orientation
- No temporal context is used (each frame is treated independently)

---

## Models

### 1. Convolutional Neural Network (CNN)

A simple convolutional architecture comprising multiple Conv2D layers followed by ReLU activations and a fully connected regression head.

- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

### 2. Vision Transformer (ViT)

Pretrained ViT-B/16 (ImageNet-21k) model with a custom regression head. The transformer encoder is frozen and only the head is trained.

- Each image is split into 16×16 patches and embedded as token sequences
- Uses self-attention to model long-range spatial dependencies
- Reference: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)

### 3. Swin Transformer

The Swin Transformer is a hierarchical vision transformer architecture developed by Microsoft, designed to be more efficient and scalable for vision tasks than standard ViTs.

Unlike ViTs that perform full global attention (which is computationally expensive), Swin uses a **local window-based self-attention mechanism** and introduces **shifted windows** in alternating layers to allow for **cross-window interactions**. This builds both **local and global representations** while maintaining computational efficiency.

- The **entire Swin Transformer model was fully fine-tuned** during training (encoder and regression head).
- It was pretrained on ImageNet-1k and adapted to the steering angle regression task.
- Hierarchical feature extraction allows it to resemble CNNs in multi-scale understanding while leveraging attention-based modeling.

**Reference**: [Liu et al., 2021](https://arxiv.org/abs/2103.14030)

---

## Results

### In-Domain Evaluation (Train & Test on Dataset B)

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 3.15    | 9.50     |
| ViT       | 14.70   | 30.02    |
| Swin ViT  | 11.08   | 26.15    |

### Cross-Domain Evaluation (Train on B, Test on A)

| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 16.20   | 28.39    |
| ViT       | 13.38   | 27.12    |
| Swin ViT  | 13.37   | 25.61    |

---

## Interpretability Tools

- **Grad-CAM**: Applied to CNN to understand spatial focus during prediction.
- **Attention Maps**: Extracted from ViT and Swin to visualize patch-level importance during inference.

---

## Key Observations

- CNNs outperformed transformer models on smaller, in-domain datasets but overfit easily.
- Swin Transformer showed more consistent cross-domain generalization compared to ViT.
- Both transformers struggled due to dataset size and imbalance.
- All models are limited by the lack of multimodal context (e.g., speed, IMU).

---

## Future Work

- Incorporate temporal models (e.g., 3D CNNs, TimeSformer)
- Augment training data for better distribution across steering angles
- Introduce multi-sensor fusion (e.g., speed, IMU, GPS)
- Add auxiliary tasks such as lane segmentation or depth estimation to strengthen spatial reasoning

---

## References

- Sully Chen. “driving-datasets.” GitHub: https://github.com/SullyChen/driving-datasets  
- Dosovitskiy et al., *An Image is Worth 16x16 Words*, arXiv:2010.11929  
- Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2103.14030

---

## License

This repository is for academic and research purposes only. For reuse or collaboration, please contact the authors.
