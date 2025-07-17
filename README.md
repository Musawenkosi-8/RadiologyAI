# 🧠 RadiologyAI – Chest X-ray Diagnosis with AI

RadiologyAI is a deep learning–powered mobile solution designed to assist in diagnosing **Pneumonia**, **Tuberculosis (TB)**, and identifying **Normal** chest X-rays using **PyTorch**, **OpenCV**, and **PyTorch Mobile**. 

This project is built to serve under-resourced clinics and rural healthcare settings, where radiologists are scarce, by enabling fast, offline diagnosis on Android devices — with support for **multilingual labels** and **Grad-CAM visualizations** to promote transparency.

---

## 📱 App Demo

<!-- Add screenshots of your Android app UI and Grad-CAM heatmaps here -->
<!-- ![Screenshot](screenshots/app_ui.png) -->

---

## 🚀 Features

- ✅ Diagnose **Normal**, **Pneumonia**, and **TB** from chest X-ray images.
- ✅ Offline Android app built using **PyTorch Mobile**.
- ✅ **Grad-CAM** heatmaps to highlight areas that influenced the model's decision.
- ✅ **Multilingual label support** (English, French, isiZulu).
- ✅ Lightweight TorchScript model optimized for mobile.
- ✅ End-to-end pipeline: from preprocessing to deployment.

---

## 🧠 Model Details

- **Architecture:** ResNet18
- **Framework:** PyTorch → TorchScript for mobile
- **Input Size:** 224x224 RGB images
- **Training Dataset:** Chest X-ray datasets including TB and Pneumonia
- **Performance:**
  - 📊 Accuracy: **93.4%**
  - 📉 Loss: 0.21
  - 🧪 Evaluation Metrics: F1 Score, Confusion Matrix, Grad-CAM explainability

---

## 🛠️ Tech Stack

| Component      | Tech Used                  |
|----------------|----------------------------|
| Model Training | PyTorch, torchvision       |
| Image Processing | OpenCV, PIL               |
| Mobile App     | Android (Java/Kotlin) + PyTorch Mobile |
| Visualization  | Grad-CAM, Matplotlib       |
| Deployment     | TorchScript (.pt file)     |
| Languages      | English, French, isiZulu   |

---

## 🌍 Multilingual Support

`labels.json` contains multilingual translations:
```json
{
  "0": {"en": "Normal", "fr": "Normal", "zu": "Okuvamile"},
  "1": {"en": "Pneumonia", "fr": "Pneumonie", "zu": "Umkhuhlane we-pneumonia"},
  "2": {"en": "TB", "fr": "Tuberculose", "zu": "Isifo sofuba"}
}
<img width="567" height="488" alt="radiology3" src="https://github.com/user-attachments/assets/0c197676-cc8f-409d-8a03-f805480c8c14" />
<img width="514" height="468" alt="radiology1" src="https://github.com/user-attachments/assets/e6d8d9a5-d4b2-4f09-a317-94ae6d8259d6" />
<img width="514" height="468" alt="radiology1" src="https://github.com/user-attachments/assets/59cb13ac-2805-4152-bebd-3728724c238d" />
