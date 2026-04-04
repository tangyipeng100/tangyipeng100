# Small-Sample Model Training Framework

A lightweight training framework for image classification with **small datasets / few-shot learning**, designed for rapid experimentation and mobile deployment.

Targeted at: industrial inspection, defect detection, and other scenarios where labeled data is scarce.

## Features

- **Few-shot training** — configure `shots_per_class` to train with as few as 5 samples per class
- **Transfer learning** — pretrained MobileNetV3 / EfficientNet backbones (optionally freeze backbone)
- **Strong augmentation** — random crop, flip, color jitter, rotation for data efficiency
- **Early stopping** — avoid overfitting on small datasets
- **ONNX export** — deploy on Android / iOS / edge devices

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your data
```
data/
  train/
    class_a/  img1.jpg  img2.jpg  ...
    class_b/  img1.jpg  ...
  val/
    class_a/  ...
    class_b/  ...
```

Or generate a demo dataset:
```bash
cd model_training
python scripts/prepare_demo_data.py --classes 5 --shots 10
```

### 3. Train
```bash
# Few-shot training (5 samples/class)
python train.py --config configs/fewshot.yaml

# Standard training with all data
python train.py --config configs/default.yaml

# Override parameters
python train.py --config configs/fewshot.yaml --shots 20 --epochs 100
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint outputs/best_model.pth --data data/test
```

### 5. Export for mobile
```bash
python export.py --checkpoint outputs/best_model.pth --output exports/model.onnx
```

## Configuration

Key parameters in `configs/`:

| Parameter | Description |
|---|---|
| `model.backbone` | `mobilenetv3_small_100`, `efficientnet_b0`, etc. |
| `model.freeze_backbone` | `true` = only train the head (good for very few samples) |
| `data.shots_per_class` | Number of training images per class (`null` = use all) |
| `training.label_smoothing` | Regularization for small datasets (try 0.1–0.2) |

## Mobile Deployment

The exported ONNX model can be integrated into:
- **Android**: ONNX Runtime Mobile (`com.microsoft.onnxruntime`)
- **iOS**: ONNX Runtime / CoreML (via `onnx-coreml`)
- **Edge/embedded**: ONNX Runtime on Raspberry Pi, Jetson Nano

Model sizes (approximate):
- `mobilenetv3_small_100`: ~6 MB
- `efficientnet_b0`: ~20 MB

## Project Structure

```
model_training/
├── configs/
│   ├── default.yaml      # Standard training
│   └── fewshot.yaml      # Few-shot (5-shot) training
├── data/
│   └── dataset.py        # Dataset + DataLoader
├── models/
│   └── classifier.py     # Backbone + classification head
├── utils/
│   ├── metrics.py        # Accuracy, classification report
│   └── logger.py         # Logging
├── scripts/
│   └── prepare_demo_data.py  # Generate toy dataset
├── train.py              # Training loop
├── evaluate.py           # Evaluation
├── export.py             # ONNX export
└── requirements.txt
```
