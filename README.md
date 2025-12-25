# XRAI - AI-Based Chest X-Ray Screening System  
**Hackathon Project | Healthcare AI**

---

## 🩺 Problem Statement

### Inadequate Medical Screening  
In rural and semi-urban regions, chest X-ray reports often take **3–7 days** due to limited diagnostic resources. During this delay, diseases like **Tuberculosis (TB)** and **Pneumonia** may worsen or spread, leading to preventable complications.

### Poor Doctor–Patient Ratio  
There is a severe shortage of radiologists, with approximately **1 radiologist per 100,000 people** in underserved areas. This imbalance delays diagnosis and reduces the quality of early medical intervention.

---

## 💡 Our Solution

We developed an **AI-powered chest X-ray screening assistant** that analyzes X-ray images in **seconds** and supports clinicians in identifying high-risk cases early.

**Detects:**
- Normal  
- Pneumonia  
- Tuberculosis (TB)

**Provides:**
- Instant predictions with confidence scores  
- **Grad-CAM heatmaps** showing affected lung regions  
- Fast, low-cost screening support  

⚠️ **Important:** This system is designed to **assist doctors, not replace them**.  
It serves as a **decision-support tool** to help clinicians prioritize cases and reduce workload.

---

## 🚀 Key Features

- Near real-time X-ray analysis (~3 seconds)  
- Deep learning–based medical image classification  
- Explainable AI using visual heatmaps  
- Robust performance on imbalanced datasets  
- Lightweight and deployment-ready  
- Designed for rural and resource-limited clinics  

---

## 🧠 How It Works

1. User uploads a chest X-ray image  
2. Image is resized and normalized (`224 × 224`)  
3. AI model predicts disease class  
4. Confidence score is calculated  
5. Grad-CAM highlights suspicious lung areas  
6. Doctor reviews AI output and makes the final decision  

---

## 📊 Dataset

Publicly available datasets:
- **Normal & Pneumonia** – Kaggle Chest X-ray Dataset  
- **Tuberculosis (TB)** – Shenzhen TB Dataset  

xray_dataset/
├── train/
├── val/
└── test/
├── NORMAL
├── PNEUMONIA
└── TB


---

## 🧪 Model Overview

- Architecture: **EfficientNet-B0 (pretrained)**  
- Input Size: `224 × 224`  
- Optimizer: AdamW  
- Loss: Weighted Cross-Entropy  
- Accuracy: ~80–85% (hackathon MVP range)  
- Explainability: Grad-CAM  

---

## 🖥️ Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Grad-CAM  
- NumPy, Matplotlib  
- Google Colab  

---

## 🔮 Advanced Use Cases

Beyond basic screening, this system can be extended to:

### 1. Clinical Triage Support  
Automatically flags **high-risk X-rays** so doctors can prioritize urgent cases first, especially in crowded hospitals.

### 2. Rural Clinic Assistance  
Acts as a **first-line screening tool** in clinics without on-site radiologists, reducing unnecessary referrals.

### 3. Second Opinion for Doctors  
Provides a quick AI-based second opinion, helping doctors validate findings and reduce human error.

### 4. Training Tool for Medical Students  
Grad-CAM heatmaps can help students understand **where and why** abnormalities appear in X-rays.

### 5. Mass Screening Programs  
Useful for TB or pneumonia screening campaigns where thousands of X-rays need quick preliminary review.

### 6. Telemedicine Integration  
Can be integrated into telemedicine platforms to assist remote consultations and faster diagnosis.

---

## 📦 Outputs

- `best_xray_model.pth` – Best checkpoint  
- `xray_model_weights.pth` – Weights only  
- `xray_model_complete.pth` – Full model  
- `model_info.json` – Model metadata  

---

## ⚠️ Disclaimer

This project is intended for **hackathon, research, and clinical assistance purposes only**.  
It **does not replace medical professionals**.  
Final diagnosis and treatment decisions must always be made by licensed doctors.

---

## 🌍 Impact

- Reduces diagnosis time from **days to seconds**  
- Assists doctors in high patient-load environments  
- Enables early detection of TB and Pneumonia  
- Low-cost AI screening for underserved regions  
- Improves healthcare access without replacing human expertise  

---

## 🏁 Hackathon Value

- Clear real-world healthcare problem  
- Working end-to-end AI system  
- Explainable and trustworthy outputs  
- Strong social impact  
- Scalable MVP built within 48 hours  

---

## 🏆 Conclusion

This project demonstrates how **AI can responsibly assist healthcare professionals** by improving efficiency, speed, and accessibility—while keeping doctors firmly in control of final medical decisions.
