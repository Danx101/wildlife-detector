# Wildlife Detector

YOLOv8n object detection model trained from scratch on the Caltech Camera Traps dataset for detecting North American wildlife.

## Features

- **Train from scratch**: No pretrained weights, full understanding of the training process
- **W&B integration**: Track experiments, visualize metrics, log model artifacts
- **HF Spaces deployment**: Interactive Gradio demo with CPU inference
- **8 wildlife classes**: deer, coyote, rabbit, squirrel, bird, bobcat, raccoon, skunk

## Project Structure

```
wildlife-detector/
├── app.py                 # Gradio demo for HF Spaces
├── configs/
│   ├── dataset.yaml       # Dataset configuration
│   └── train_config.yaml  # Training hyperparameters
├── src/
│   ├── prepare_data.py    # Data download & preprocessing
│   └── train.py           # Training script with W&B
├── data/
│   ├── raw/               # Downloaded dataset
│   └── processed/         # YOLO-formatted data
├── models/                # Trained model weights
├── examples/              # Demo images
├── requirements.txt       # HF Spaces dependencies
└── pyproject.toml         # Project dependencies
```

## Setup

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### 2. Download the dataset

Visit [LILA Science - Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) and download:
- Images (or a subset)
- Annotation JSON file

Place files in `data/raw/`:
```
data/raw/
├── caltech_images/
│   ├── image1.jpg
│   └── ...
└── caltech_camera_traps.json
```

### 3. Prepare the dataset

```bash
python src/prepare_data.py
```

This converts COCO annotations to YOLO format and creates train/val splits.

## Training

### Basic training

```bash
python src/train.py --epochs 100 --batch-size 16 --device mps
```

### With W&B tracking

```bash
# Login to W&B first
wandb login

# Train with entity specified
python src/train.py \
    --epochs 100 \
    --batch-size 16 \
    --device mps \
    --entity your-wandb-entity \
    --project wildlife-detector \
    --name yolov8n-experiment-1
```

### Training options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | configs/dataset.yaml | Dataset config path |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--img-size` | 640 | Input image size |
| `--device` | mps | Device (mps/cuda/cpu) |
| `--project` | wildlife-detector | W&B project name |
| `--name` | yolov8n-caltech | Run name |
| `--entity` | None | W&B entity/team |
| `--pretrained` | False | Use pretrained weights |
| `--resume` | False | Resume from checkpoint |

## Validation

```bash
python src/train.py --validate runs/wildlife-detector/yolov8n-caltech/weights/best.pt
```

## Export for deployment

```bash
# Export to ONNX (faster CPU inference)
python src/train.py --export runs/wildlife-detector/yolov8n-caltech/weights/best.pt

# Copy to models/ for deployment
cp runs/wildlife-detector/yolov8n-caltech/weights/best.onnx models/
cp runs/wildlife-detector/yolov8n-caltech/weights/best.pt models/
```

## Local demo

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `models/best.pt` (or `best.onnx`)
   - `examples/` (sample images)

Or use the HF CLI:

```bash
# Login
huggingface-cli login

# Create and push
huggingface-cli repo create wildlife-detector --type space --space-sdk gradio
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/wildlife-detector
git push hf main
```

## Expected Performance

| Metric | Value |
|--------|-------|
| mAP50 | ~0.60-0.75 |
| mAP50-95 | ~0.35-0.50 |
| Inference (CPU) | ~200-400ms |
| Model size | ~6MB (nano) |

*Results vary based on dataset size and training duration.*

## License

MIT
