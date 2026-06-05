# MEAQ-YOLO: A Lightweight YOLOv11n-Based Detector for Real-Time Apple Detection on Edge Devices

Official implementation of **MEAQ-YOLO**, a lightweight apple detection model based on YOLOv11n for real-time orchard perception and edge-device deployment.

## Overview

Accurate apple detection is important for robotic harvesting, fruit localization, yield estimation, and precision orchard management. However, natural orchard scenes are challenging due to branch and leaf occlusion, dense fruit distribution, fruit overlap, illumination variation, scale changes, and complex backgrounds.

MEAQ-YOLO is designed to improve the accuracy–efficiency trade-off of YOLOv11n. It reduces model parameters, computational cost, and storage size while maintaining competitive apple detection performance in complex orchard environments.

## Highlights

- Lightweight YOLOv11n-based apple detector for edge deployment.
- Designed for complex orchard conditions, including occlusion, dense targets, and illumination variation.
- Reduced model size and computational cost compared with YOLOv11n.
- Supports training, validation, inference, and export using the Ultralytics YOLO framework.

## Key Modules

MEAQ-YOLO introduces four targeted modules:

- **MRepConv**: A multi-branch re-parameterized convolution module used to enhance shallow texture and contour representation. During inference, the training-time multi-branch structure is fused into a single convolution.
- **EShuffleNetV2**: An enhanced ShuffleNetV2 module with Efficient Channel Attention (ECA), designed to reduce redundant backbone computation while preserving discriminative channel features.
- **ADown**: An adaptive downsampling module that combines pre-smoothing and multi-branch downsampling to reduce spatial information loss during feature aggregation.
- **QDD-Head**: A quality-guided decoupled detection head that improves the consistency between classification confidence and localization quality.

## Architecture

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
    └── Multi-scale feature fusion
    │
    ▼
QDD-Head
    ├── Classification branch
    ├── Regression branch
    ├── Quality-guided task alignment
    └── Detection output
```

## Repository Structure

```text
MEAQ-YOLO/
├── train.py
├── ultralytics/
│   ├── cfg/
│   │   └── models/11/
│   │       └── MEAQ-YOLO.yaml
│   └── nn/modules/
│       ├── MRepConv.py
│       ├── EShuffleNetV2.py
│       ├── ADown.py
│       └── QDDHead.py
├── examples/
├── docs/
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/LiXiaofa-good/MEAQ-YOLO.git
cd MEAQ-YOLO
pip install -e .
```

## Training

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/MEAQ-YOLO.yaml")

model.train(
    data="your_dataset.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    workers=6,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    seed=42,
    deterministic=True,
)
```

Or run:

```bash
python train.py
```

## Validation

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

## Inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("examples/apple.jpg")
results[0].show()
```

## Export

```python
model.export(format="onnx")
model.export(format="ncnn")
model.export(format="engine")
```

## Experimental Results

Experiments were conducted on a fused apple detection dataset constructed from public orchard image datasets. The same data split, input size, and training configuration were used for YOLOv11n and MEAQ-YOLO.

| Model     | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | GFLOPs | Model Size (MB) |
| --------- | ------------- | ---------- | ----------- | ---------------- | ---------- | ------ | --------------- |
| YOLOv11n  | 88.4          | 81.0       | 90.2        | 57.7             | 2.58       | 6.3    | 5.2             |
| MEAQ-YOLO | 89.0          | 80.9       | 90.1        | 57.3             | 1.30       | 3.4    | 2.9             |

Compared with YOLOv11n, MEAQ-YOLO reduces the number of parameters by **49.6%**, GFLOPs by **46.0%**, and model size by **44.2%**, while maintaining comparable detection accuracy.

## Reproducibility Notes

To improve reproducibility, please keep the following settings consistent with the paper:

- Input image size: `640 × 640`
- Training epochs: `200`
- Optimizer: `SGD`
- Initial learning rate: `0.01`
- Momentum: `0.937`
- Weight decay: `0.0005`
- Random seed: `42`

Dataset paths and class names should be configured in your own `your_dataset.yaml` file.

## Dataset

The training and evaluation dataset is constructed by integrating public apple image datasets from orchard environments. Due to dataset license restrictions, this repository does not redistribute the original datasets directly.

Users should download the datasets from their official sources and organize them in YOLO format before training.

Example dataset configuration:

```yaml
path: /path/to/apple_dataset
train: images/train
val: images/val
test: images/test

names:
  0: apple
```

## License

This project is based on the Ultralytics YOLO framework. Please follow the corresponding open-source license requirements of Ultralytics YOLO when using or modifying this repository.

## Acknowledgements

This work is developed based on the Ultralytics YOLO framework. We thank the open-source community for providing valuable tools and resources for object detection research.

## Contact

For questions, please open an issue in this repository.
