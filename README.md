# 🚗 Steering Angle Prediction using Vision Transformers

This project investigates the use of computer vision models—specifically CNNs and Transformers (ViT & Swin ViT)—to predict a vehicle’s steering angle from front-facing camera images. The ultimate goal is to enable real-time, vision-only autonomous steering prediction for self-driving cars.

---

## 📌 Team Members
- Ashwin Ram Venkatraman  
- Anuja Tipare  
- Kanav Goyal  
- Swetha Subramanian

---

## 📍 Problem Statement

Predict the steering angle of a car using deep learning models trained on dashboard camera footage. The system should learn to mimic human steering behavior by identifying features like lane lines, curves, and road boundaries from visual data alone.

---

## 📊 Dataset

### 🗂 Sources
- **Source A**: Udacity SD Car Dataset (~45K RGB frames)
- **Source B**: Comma.ai 2016 Dataset (~19K RGB frames)  
  (Resolution: 640×480 @ 20Hz, includes steering-angle labels from CAN bus)

### 🧪 Strategy
- **Training**: Source B (80%)
- **Validation**: Source B (20%)
- **Cross-domain Test**: Source A (unseen dataset)

### ⚠ Challenges
- Imbalanced data: ~73% of angles < 1°
- Domain shift: Source A vs B have different lighting, road types, and cameras
- No auxiliary data: No speed/IMU/GPS, only vision used for steering prediction

---

## ⚙️ Preprocessing

- Resized images to **224×224**
- Normalized pixel values to **[-1, 1]**
- Converted to PyTorch tensors

---

## 🧠 Models Compared

### ✅ CNN (Baseline)
- 4 Conv2D layers with ReLU and flatten
- Loss: Mean Squared Error  
- Optimizer: Adam

### 🧠 ViT (Vision Transformer)
- Pretrained on ImageNet-21k  
- Frozen backbone, trained regression head  
- Loss: MSE | Optimizer: Adam  
- Paper: [ViT - An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### 🌀 Swin Transformer
- Hierarchical architecture with local self-attention  
- Pretrained on ImageNet-1k  
- Loss: MSE | Optimizer: Adam  
- Paper: [Swin Transformer](https://arxiv.org/abs/2103.14030)

---

## 📈 Results

### ▶️ Source B → Source B (Validation)
| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 3.15    | 9.50     |
| ViT       | 14.70   | 30.02    |
| Swin ViT  | 11.08   | 26.15    |

### 🌐 Source B → Source A (Cross-domain Test)
| Model     | MAE (°) | RMSE (°) |
|-----------|---------|----------|
| CNN       | 16.20   | 28.39    |
| ViT       | 13.38   | 27.12    |
| Swin ViT  | 13.37   | 25.61    |

---

## 🔍 Visualizations

- **Inference Samples**: Steering predictions from all models on test images
- **Grad-CAM & Attention Maps**: Heatmaps showing where the model focuses

---

## 🚀 Learnings & Future Work

- CNN performed best on both in-domain and cross-domain tests but showed signs of overfitting.
- ViTs and Swin ViTs struggled due to limited training size and domain shift.
- Future directions:
  - Collect a more diverse dataset with balanced steering angles
  - Use lane segmentation as an auxiliary task
  - Explore domain adaptation techniques (e.g., fine-tuning, adversarial training)

---

## 📎 References

- [ViT Paper](https://arxiv.org/abs/2010.11929) – An Image is Worth 16x16 Words  
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)

---

> “A model that knows when not to trust itself is often more valuable than a model that is occasionally wrong but always confident.”
