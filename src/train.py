"""
YOLOv8 Training Script with Weights & Biases Integration

Train YOLOv8n from scratch on wildlife detection dataset.
"""

import argparse
from pathlib import Path

import wandb
from ultralytics import YOLO


def train(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "mps",
    project_name: str = "wildlife-detector",
    run_name: str = "yolov8n-caltech",
    wandb_entity: str | None = None,
    from_scratch: bool = True,
    resume: bool = False,
) -> None:
    """
    Train YOLOv8 model with W&B logging.

    Args:
        data_yaml: Path to dataset configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to train on ('mps', 'cuda', 'cpu')
        project_name: W&B project name
        run_name: W&B run name
        wandb_entity: W&B entity/team name
        from_scratch: If True, train from scratch; else use pretrained
        resume: Resume from last checkpoint
    """
    # Initialize W&B
    wandb.init(
        entity=wandb_entity,
        project=project_name,
        name=run_name,
        config={
            "model": "yolov8n",
            "from_scratch": from_scratch,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "device": device,
            "data": data_yaml,
        },
    )

    # Initialize model
    if from_scratch:
        # Build model from YAML (random weights)
        model = YOLO("yolov8n.yaml")
        print("Training from scratch (random initialization)")
    else:
        # Load pretrained weights
        model = YOLO("yolov8n.pt")
        print("Training from pretrained COCO weights")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=f"runs/{project_name}",
        name=run_name,
        exist_ok=True,
        resume=resume,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Early stopping
        patience=20,
        # Workers
        workers=8,
        # Verbosity
        verbose=True,
    )

    # Log final metrics to W&B
    if results.results_dict:
        wandb.log({
            "final/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "final/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            "final/precision": results.results_dict.get("metrics/precision(B)", 0),
            "final/recall": results.results_dict.get("metrics/recall(B)", 0),
        })

    # Save best model path
    best_model_path = Path(f"runs/{project_name}/{run_name}/weights/best.pt")
    print(f"\nBest model saved to: {best_model_path}")

    # Log model artifact to W&B
    if best_model_path.exists():
        artifact = wandb.Artifact(
            name=f"{run_name}-model",
            type="model",
            description="Best YOLOv8n wildlife detector model",
        )
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)

    wandb.finish()

    return results


def validate(model_path: str, data_yaml: str, device: str = "mps") -> None:
    """Run validation on trained model."""
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device)

    print("\nValidation Results:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")


def export_model(model_path: str, format: str = "onnx") -> str:
    """Export model for deployment."""
    model = YOLO(model_path)
    export_path = model.export(format=format, imgsz=640, half=False)
    print(f"Model exported to: {export_path}")
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Wildlife Detector")

    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="wildlife-detector",
        help="W&B project name",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n-caltech",
        help="Run name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/team name",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights instead of training from scratch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Path to model for validation only",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Path to model for export",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="onnx",
        help="Export format (onnx, torchscript, etc.)",
    )

    args = parser.parse_args()

    if args.validate:
        validate(args.validate, args.data, args.device)
    elif args.export:
        export_model(args.export, args.export_format)
    else:
        train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            project_name=args.project,
            run_name=args.name,
            wandb_entity=args.entity,
            from_scratch=not args.pretrained,
            resume=args.resume,
        )
