# AI-Based Chest X-Ray Screening System  
**Hackathon Project | Healthcare AI**

---

## 🩺 Problem Statement

### Inadequate Medical Screening  
In rural and semi-urban regions, chest X-ray reports often take **3–7 days** due to limited diagnostic resources. During this delay, diseases like **Tuberculosis (TB)** and **Pneumonia** may worsen or spread, leading to preventable complications.

### Poor Doctor–Patient Ratio  
There is a critical shortage of radiologists, with approximately **1 radiologist per 100,000 people** in underserved areas. This results in delayed diagnosis, overworked doctors, and reduced access to early screening.

---

## 💡 Our Solution

We developed an **AI-powered chest X-ray screening assistant** that analyzes X-ray images in **seconds** and supports doctors in early detection of lung diseases.

**Detects:**
- Normal  
- Pneumonia  
- Tuberculosis (TB)

**Provides:**
- Instant predictions with confidence scores  
- **Grad-CAM heatmaps** highlighting affected lung regions  
- Fast, low-cost screening support  

⚠️ **This system assists doctors — it does not replace them.**  
Doctors remain responsible for final diagnosis and treatment decisions.

---

## 🚀 Key Features

- Real-time X-ray analysis (~3 seconds)  
- **DenseNet-based deep learning model** optimized for medical imaging  
- Handles class imbalance in medical datasets  
- Explainable AI using Grad-CAM visualizations  
- Lightweight and deployment-ready  
- Designed for rural and resource-limited healthcare settings  

---

## 🧠 How It Works

1. User uploads a chest X-ray image  
2. Image is resized and normalized (`224 × 224`)  
3. **DenseNet model** extracts deep visual features  
4. AI predicts disease class with confidence score  
5. Grad-CAM highlights suspicious lung regions  
6. Doctor reviews AI output and makes the final decision  

---

## 📊 Dataset

Publicly available datasets:
- **Normal & Pneumonia** – Kaggle Chest X-ray Dataset  
- **Tuberculosis (TB)** – Shenzhen TB Dataset  
```
xray_dataset/
├── train/
├── val/
└── test/
├── NORMAL
├── PNEUMONIA
└── TB

```

---

## 🧪 Model Overview

- Architecture: **DenseNet (DenseNet-121 / DenseNet-169)**  
- Why DenseNet:
  - Strong feature reuse  
  - Excellent performance on medical images  
  - Fewer parameters with deeper representations  
- Input Size: `224 × 224`  
- Optimizer: AdamW  
- Loss Function: Weighted Cross-Entropy  
- Accuracy: ~85-90%  
- Explainability: **Grad-CAM**

---

## 🖥️ Tech Stack

- Python  
- PyTorch  
- Torchvision  
- DenseNet  
- Grad-CAM  
- NumPy, Matplotlib  
- Google Colab  

---

## 🔮 Advanced Use Cases

### 1. Clinical Triage Support  
Automatically flags **high-risk X-rays** so doctors can prioritize urgent cases in crowded hospitals.

### 2. Rural Clinic Assistance  
Provides instant screening support in clinics without on-site radiologists.

### 3. Second Opinion for Doctors  
Acts as an AI-powered second reader to reduce diagnostic errors and improve confidence.

### 4. Medical Education Tool  
Grad-CAM heatmaps help students learn **where pathological patterns appear** in chest X-rays.

### 5. Mass Screening Programs  
Ideal for TB and pneumonia screening camps requiring fast, large-scale preliminary analysis.

### 6. Telemedicine Integration  
Can be integrated with telemedicine platforms for faster remote diagnosis support.

---

## 📦 Outputs

- `best_xray_model.pth` – Best checkpoint  
- `xray_model_weights.pth` – Weights only  
- `xray_model_complete.pth` – Full model  
- `model_info.json` – Model metadata  

---

## ⚠️ Disclaimer

This project is intended for **hackathon, research, and clinical assistance purposes only**.  
It **does not replace doctors**.  
Final diagnosis and treatment decisions must always be made by licensed medical professionals.

---

## 🌍 Impact

- Reduces screening time from **days to seconds**  
- Assists doctors under heavy patient loads  
- Enables early detection of TB and Pneumonia  
- Low-cost AI support for underserved regions  
- Improves healthcare access while preserving human oversight  

---
---

## 🏆 Conclusion

This project demonstrates how **DenseNet-based AI systems can responsibly assist doctors**, improving speed and accessibility of chest X-ray screening while keeping medical professionals in full control of clinical decisions.
