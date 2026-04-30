# 🧵 Fabric Defect Detection using Deep Learning (Jetson Nano Deployment)

## 🚀 Project Overview

This project presents a **real-time fabric defect detection system** built using **deep learning and computer vision**, and successfully deployed on an **edge device (Jetson Nano)**.

The system:

* Detects defects in fabric using a trained CNN model
* Works in real-time using a camera feed
* Runs efficiently on low-power hardware (Jetson Nano)
* Uses ONNX for optimized deployment

---

## 🎯 Objectives

* Build a **custom-trained model** (not just pretrained usage)
* Achieve **high accuracy in defect detection**
* Deploy on **edge hardware (Jetson Nano)**
* Enable **real-time inference**
* Maintain a **scalable pipeline (training → conversion → deployment)**

---

## 🧠 Models Used

### 🔹 Model 1: Custom CNN (Baseline Model)

* Built from scratch using TensorFlow/Keras
* Trained on fabric dataset
* Purpose: Understand data and establish baseline

**Limitations:**

* Lower accuracy
* Overfitting issues
* Poor generalization

---

### 🔹 Model 2: MobileNetV2-based Model (Final Model)

* Transfer learning using MobileNetV2 backbone
* Fine-tuned on fabric dataset
* Used custom classification head

**Advantages:**

* High accuracy
* Better generalization
* Lightweight (good for edge deployment)
* Faster inference

---

## 📊 Model Comparison

| Feature                | Custom CNN | MobileNetV2 |
| ---------------------- | ---------- | ----------- |
| Accuracy               | Medium     | High        |
| Training Time          | High       | Moderate    |
| Generalization         | Poor       | Good        |
| Deployment Suitability | Low        | Excellent   |
| Final Selection        | ❌          | ✅           |

---

## 📁 Dataset

* Used: **MVTec AD (Carpet subset)**
* Classes:

  * `Good` (no defect)
  * `Defective`

### Preprocessing:

* Resized to **224x224**
* Normalized using:

```python
(img / 127.5) - 1.0
```

---

## 🏗️ Project Pipeline

```text
Dataset → Training (TensorFlow)
        → Model (.h5/.keras)
        → ONNX Conversion
        → Jetson Deployment
        → Real-time Inference
```

---

## 🧪 Training (Google Colab)

* Framework: TensorFlow / Keras
* Loss: Binary Crossentropy
* Optimizer: Adam
* Input shape: (224, 224, 3)

Model saved as:

```text
fabric_model2.h5
```

---

## 🔄 Model Conversion (Critical Step)

Since Jetson Nano has limited support for TensorFlow:

### Converted model to ONNX:

```bash
python -m tf2onnx.convert \
--keras fabric_model2.h5 \
--output model.onnx \
--opset 13
```

---

## ⚙️ Deployment on Jetson Nano

### Why ONNX?

* Lightweight
* Faster inference
* Compatible with edge devices

---

### 🖥️ Setup Steps

1. Connect Jetson to WiFi
2. Install dependencies:

```bash
pip3 install numpy opencv-python onnxruntime
```

---

### 📂 Transfer Files

```bash
scp model.onnx jetson@<IP>:/home/jetson/
scp main.py jetson@<IP>:/home/jetson/
```

---

### ▶️ Run on Jetson

```bash
python3 main.py
```

---

## 🎥 Real-Time Inference

* Captures video using OpenCV
* Preprocesses frame
* Runs ONNX model
* Displays result on screen

---

## 🧠 Key Engineering Challenges (Solved)

### ❌ TensorFlow not supported on Jetson

✔ Solution: Converted to ONNX

---

### ❌ ONNX Runtime installation failure

✔ Solution: Installed Jetson-compatible version

---

### ❌ Input shape mismatch error

✔ Solution:

* Removed incorrect transpose
* Maintained NHWC format

---

### ❌ GUI not displaying over SSH

✔ Solution:

* Used VNC for remote desktop

---

### ❌ Network issues (hotspot)

✔ Solution:

* Connected both devices to same network
* Used correct IP

---

## ⚡ Performance Optimization

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 🧩 Final Architecture

```text
Camera → OpenCV → Preprocessing → ONNX Model → Prediction → Display
```

---

## 📌 Features

* Real-time defect detection
* Edge deployment (Jetson Nano)
* Lightweight model
* Modular pipeline
* Scalable for industrial use

---

## 🚧 Limitations

* Only classification (no localization)
* Sensitive to lighting variations
* Requires camera calibration

---

## 🔥 Future Improvements

* 🔍 Defect localization (Grad-CAM)
* ⚡ TensorRT optimization (faster inference)
* 🔊 Audio feedback (Bluetooth speaker)
* 📊 Confidence smoothing
* 📱 Mobile interface

---

## 🏆 Conclusion

This project demonstrates a **complete deep learning pipeline**:

* Data → Model → Optimization → Deployment

It goes beyond theory and achieves:

> ✅ Real-world, real-time AI system on edge hardware

---

## 🙌 Author

**Ronit Saini**

---

## ⭐ If you found this useful

Give it a ⭐ and feel free to fork!

---
