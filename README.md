---
title: Wildlife Detector
emoji: 🦌
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.33.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Wildlife Detector

Real-time wildlife detection using YOLOv8n trained **from scratch** on camera trap imagery. Deployed on Hugging Face Spaces with ZeroGPU support.

**[Live Demo](https://huggingface.co/spaces/Danx10/wildlife-detector)** | **[W&B Training Run](https://wandb.ai/e11901844-tu-wien/runs-YOLOv8/runs/yolov8n-megadetector_20260531_235238)**

## Model Performance

| Metric | Value |
|--------|-------|
| **mAP50** | 0.682 |
| **mAP50-95** | 0.477 |
| **Precision** | 0.702 |
| **Recall** | 0.606 |
| **Parameters** | 3.0M |
| **GFLOPs** | 8.1 |
| **Model Size** | 6.2 MB |
| **Inference Speed** | 13.1ms (Apple M4 Pro) |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| deer | 115 | 143 | 0.643 | 0.579 | 0.655 | 0.441 |
| coyote | 121 | 127 | 0.704 | 0.630 | 0.738 | 0.553 |
| rabbit | 98 | 102 | 0.650 | 0.500 | 0.598 | 0.445 |
| squirrel | 117 | 121 | 0.622 | 0.463 | 0.531 | 0.294 |
| bird | 101 | 123 | 0.734 | 0.407 | 0.487 | 0.249 |
| bobcat | 132 | 134 | 0.711 | 0.717 | 0.794 | 0.623 |
| raccoon | 105 | 119 | 0.717 | 0.689 | 0.747 | 0.500 |
| skunk | 107 | 109 | 0.832 | 0.864 | 0.902 | 0.712 |

## Training Details

### Architecture

- **Model**: YOLOv8n (nano variant) - smallest and fastest YOLOv8
- **Trained from scratch**: Random weight initialization using `yolov8n.yaml`, no pretrained COCO weights
- **Why from scratch?**: Full control over learned features, ensures model learns wildlife-specific patterns without COCO bias

### Dataset

- **Source**: [LILA Science Camera Traps](https://lila.science/) - largest repository of camera trap images
- **Bounding Boxes**: Generated using **MegaDetector v5** (Microsoft AI for Earth)
  - MegaDetector detects animals in camera trap images with >95% accuracy
  - Provides species-agnostic bounding boxes that we label with our 8 classes
- **Classes**: 8 North American wildlife species
- **Split**: 80% train / 20% validation (seed=42)
- **Total**: ~4,500 images, 978 validation instances

### Training Configuration

```yaml
# Hyperparameters
epochs: 100
batch_size: 32
image_size: 640
optimizer: AdamW
learning_rate: 0.001
lr_final: 0.01 (cosine decay)
weight_decay: 0.0005
warmup_epochs: 5
patience: 20 (early stopping)

# Augmentation
mosaic: 1.0
mixup: 0.1
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10
translate: 0.1
scale: 0.5
flipud: 0.5
fliplr: 0.5

# Hardware
device: Apple M4 Pro (MPS)
training_time: 4.68 hours
```

### Training Curves

Training and validation metrics over 100 epochs:

![Training Results](results/results.png)

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix_normalized.png)

### Precision-Recall Curve

![PR Curve](results/BoxPR_curve.png)

## Project Structure

```
wildlife-detector/
├── app.py                 # Gradio demo (ZeroGPU enabled)
├── configs/
│   └── dataset.yaml       # Dataset paths and class mapping
├── src/
│   ├── download_data.py   # LILA data downloader
│   ├── prepare_data.py    # MegaDetector bbox generation + YOLO format
│   └── train.py           # Training script with W&B integration
├── models/
│   └── best.pt            # Trained model weights
├── results/               # Training plots and metrics
│   ├── results.png        # Training curves
│   ├── confusion_matrix.png
│   ├── BoxPR_curve.png
│   ├── BoxF1_curve.png
│   └── results.csv        # Epoch-by-epoch metrics
├── examples/              # Demo images (28 images, 4 per class)
└── requirements.txt       # HF Spaces dependencies
```

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/Danx101/wildlife-detector.git
cd wildlife-detector

# Install dependencies
pip install -e .
# or
uv sync
```

### Run Inference

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

### Local Demo

```bash
python app.py
# Open http://localhost:7860
```

## Data Pipeline

### 1. Download from LILA

```bash
python src/download_data.py --samples-per-class 500
```

Downloads camera trap images for 8 target species from LILA Science datasets.

### 2. Generate Bounding Boxes with MegaDetector

```bash
python src/prepare_data.py --megadetector
```

- Runs MegaDetector v5 on all images
- Filters detections with confidence > 0.5
- Converts to YOLO format: `class_id x_center y_center width height`
- Creates train/val split

### 3. Train

```bash
python src/train.py \
    --data configs/dataset.yaml \
    --epochs 100 \
    --batch-size 32 \
    --device mps \
    --name yolov8n-wildlife
```

Training logs automatically sync to Weights & Biases.

## Deployment

### Hugging Face Spaces

The app is deployed with ZeroGPU support for faster inference:

```python
import spaces

@spaces.GPU  # Requests GPU for this function
def detect_wildlife(image, confidence_threshold=0.25):
    results = model.predict(source=image, conf=confidence_threshold)
    return results[0].plot()
```

To deploy your own:

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/wildlife-detector

# Push (requires Git LFS for binary files)
git lfs install
git lfs track "*.pt" "*.jpg" "*.png"
git push hf main
```

Then enable ZeroGPU in Space Settings > Hardware.

## Experiment Tracking

All training runs are logged to Weights & Biases:

- Loss curves (box, cls, dfl)
- Validation metrics (mAP, precision, recall)
- Learning rate schedules
- Confusion matrices
- Sample predictions
- Model checkpoints as artifacts

View the training run: [wandb.ai/e11901844-tu-wien/runs-YOLOv8](https://wandb.ai/e11901844-tu-wien/runs-YOLOv8/runs/yolov8n-megadetector_20260531_235238)

## Limitations

- **Small dataset**: ~4,500 images limits generalization
- **Camera trap bias**: Model may underperform on non-camera-trap wildlife images
- **Class imbalance**: Some species (skunk, bobcat) have better detection than others (bird, squirrel)
- **Single bounding box**: MegaDetector sometimes misses multiple animals in frame

## Future Improvements

- [ ] Larger dataset from multiple LILA sources
- [ ] YOLOv8s/m for improved accuracy
- [ ] Multi-animal detection refinement
- [ ] Night vision / infrared image support
- [ ] Video inference with tracking

## License

MIT

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [LILA Science](https://lila.science/) for camera trap datasets
- [MegaDetector](https://github.com/microsoft/CameraTraps) by Microsoft AI for Earth
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Hugging Face Spaces](https://huggingface.co/spaces) for deployment
