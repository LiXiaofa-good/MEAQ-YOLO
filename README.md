# MEAQ-YOLO: Lightweight YOLOv11n-Based Apple Detector for Edge Devices

Official implementation of **MEAQ-YOLO**, a lightweight apple detection model based on YOLOv11n for real-time orchard perception and edge-device deployment.

---

## 📌 Overview

MEAQ-YOLO is a lightweight apple detection model built on YOLOv11n and designed for deployment in complex orchard environments. It can be used for robotic harvesting, fruit localization, yield estimation, and precision orchard management.

**Key features:**

- Lightweight and efficient for low-resource edge devices
- Designed for orchard scenes with occlusion, dense fruit distribution, and illumination variation
- Reduced model size and computational cost while maintaining detection capability
- Supports training, validation, inference, and model export based on the Ultralytics YOLO framework

---

## 🔧 Key Modules

MEAQ-YOLO introduces four targeted modules:

| Module | Description |
|---|---|
| **MRepConv** | Multi-branch re-parameterized convolution for enhancing shallow texture and contour features. The training-time multi-branch structure is fused into a single convolution during inference. |
| **EShuffleNetV2** | Enhanced ShuffleNetV2 module with Efficient Channel Attention (ECA) for lightweight and channel-aware feature extraction. |
| **ADown** | Adaptive downsampling module that reduces spatial information loss during feature aggregation. |
| **QDD-Head** | Quality-guided decoupled detection head that improves the alignment between classification confidence and localization quality. |

---

## 🏗️ Architecture

```text
Input Image
    │
    ▼
Backbone
    ├── Conv
    ├── MRepConv
    ├── EShuffleNetV2
    ├── SPPF
    └── C2PSA
    │
    ▼
Neck
    ├── Upsample
    ├── Concat
    ├── C3k2
    ├── ADown
    └── Multi-scale Feature Fusion
    │
    ▼
QDD-Head
    ├── Classification Branch
    ├── Regression Branch
    ├── Quality-Guided Task Alignment
    └── Detection Output
```

---

## 📁 Repository Structure

```text
MEAQ-YOLO/
├── train.py
├── ultralytics/
│   ├── cfg/
│   │   └── models/
│   │       └── 11/
│   │           └── MEAQ-YOLO.yaml
│   └── nn/
│       └── modules/
│           ├── MRepConv.py
│           ├── EShuffleNetV2.py
│           ├── ADown.py
│           └── QDDHead.py
├── examples/
├── docs/
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

```bash
git clone https://github.com/LiXiaofa-good/MEAQ-YOLO.git
cd MEAQ-YOLO
pip install -e .
```

---

## 🚀 Quick Inference

```python
from ultralytics import YOLO

# Load a trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on a sample image
results = model("examples/apple.jpg")
results[0].show()
```

---

## 🏋️ Training

Users can train MEAQ-YOLO on their own datasets in YOLO format.

### Dataset YAML example

```yaml
path: /path/to/apple_dataset
train: images/train
val: images/val
test: images/test

names:
  0: apple
```

### Training example

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/MEAQ-YOLO.yaml")

model.train(
    data="your_dataset.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
)
```

---

## ✅ Validation

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(
    data="your_dataset.yaml",
    imgsz=640,
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

---

## 📦 Export

```python
# Export to ONNX
model.export(format="onnx")

# Export to NCNN for mobile or ARM devices
model.export(format="ncnn")

# Export to TensorRT for NVIDIA edge devices
model.export(format="engine")
```

---

## 🔒 Safety Notes

- This repository **does not include unpublished raw experimental data**.
- Only code, model definitions, and optional example files are provided.
- Users can reproduce the model using their own datasets.
- Experimental details and quantitative results should be referenced from the corresponding paper.

---

## 📄 License

This project is based on the Ultralytics YOLO framework. Please follow the corresponding open-source license requirements when using or modifying this repository.

---

## 🙏 Acknowledgements

This project is developed based on the Ultralytics YOLO framework. We thank the open-source community for providing valuable tools and resources for object detection research.

---

## 📬 Contact

For questions or collaboration, please open an issue in this repository.
