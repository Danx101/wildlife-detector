"""
Caltech Camera Traps Data Preparation Script

Downloads and prepares the dataset for YOLOv8 training.
Converts annotations to YOLO format (normalized xywh).
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# Target classes - common North American wildlife
TARGET_CLASSES = [
    "deer",
    "coyote",
    "rabbit",
    "squirrel",
    "bird",
    "bobcat",
    "raccoon",
    "skunk",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def download_caltech_cameratraps(data_dir: Path) -> Path:
    """
    Download Caltech Camera Traps dataset.

    Note: The full dataset is very large. This function downloads
    the LILA subset which is more manageable.

    For the full dataset, you may need to request access from:
    https://lila.science/datasets/caltech-camera-traps
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # COCO Camera Traps annotation file URL (from LILA)
    annotation_url = (
        "https://lila.science/wp-content/uploads/2023/06/"
        "caltech_camera_traps.json.zip"
    )

    print("=" * 60)
    print("CALTECH CAMERA TRAPS DATASET")
    print("=" * 60)
    print()
    print("The full dataset requires manual download from LILA Science:")
    print("https://lila.science/datasets/caltech-camera-traps")
    print()
    print("Steps:")
    print("1. Visit the URL above")
    print("2. Download the images (or a subset)")
    print("3. Download the annotation JSON")
    print("4. Place files in:", raw_dir)
    print()
    print("Expected structure:")
    print("  data/raw/")
    print("  ├── caltech_images/")
    print("  │   ├── image1.jpg")
    print("  │   └── ...")
    print("  └── caltech_camera_traps.json")
    print()

    return raw_dir


def load_annotations(annotation_file: Path) -> tuple[dict, dict, dict]:
    """Load COCO-format annotations."""
    print(f"Loading annotations from {annotation_file}...")

    with open(annotation_file) as f:
        data = json.load(f)

    # Build lookups
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"].lower() for cat in data["categories"]}

    # Group annotations by image
    annotations_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    return images, categories, annotations_by_image


def convert_bbox_to_yolo(bbox: list, img_width: int, img_height: int) -> tuple:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format.
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    # Clip to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def prepare_dataset(
    raw_dir: Path,
    processed_dir: Path,
    train_split: float = 0.8,
    max_images_per_class: int = 2000,
) -> dict:
    """
    Prepare dataset for YOLOv8 training.

    Args:
        raw_dir: Directory containing raw images and annotations
        processed_dir: Output directory for YOLO-formatted data
        train_split: Fraction of data for training
        max_images_per_class: Maximum images per class to balance dataset

    Returns:
        Statistics about the prepared dataset
    """
    annotation_file = raw_dir / "caltech_camera_traps.json"
    images_dir = raw_dir / "caltech_images"

    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        print("Please download the dataset first.")
        return {}

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        print("Please download the images first.")
        return {}

    # Load annotations
    images, categories, annotations_by_image = load_annotations(annotation_file)

    # Create output directories
    for split in ["train", "val"]:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Filter to target classes and collect valid images
    print("Filtering images with target classes...")
    valid_images = []
    class_counts = {cls: 0 for cls in TARGET_CLASSES}

    for img_id, img_info in tqdm(images.items()):
        if img_id not in annotations_by_image:
            continue

        anns = annotations_by_image[img_id]
        valid_anns = []

        for ann in anns:
            cat_name = categories.get(ann["category_id"], "").lower()

            # Check if this category matches any target class
            matched_class = None
            for target_cls in TARGET_CLASSES:
                if target_cls in cat_name:
                    matched_class = target_cls
                    break

            if matched_class and "bbox" in ann and len(ann["bbox"]) == 4:
                valid_anns.append((ann, matched_class))

        if valid_anns:
            valid_images.append((img_id, img_info, valid_anns))

    print(f"Found {len(valid_images)} images with target classes")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(valid_images)

    split_idx = int(len(valid_images) * train_split)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    stats = {"train": {}, "val": {}}

    # Process each split
    for split, split_images in [("train", train_images), ("val", val_images)]:
        print(f"\nProcessing {split} split ({len(split_images)} images)...")
        split_class_counts = {cls: 0 for cls in TARGET_CLASSES}

        for img_id, img_info, valid_anns in tqdm(split_images):
            # Find source image
            img_filename = img_info.get("file_name", f"{img_id}.jpg")
            src_path = images_dir / img_filename

            if not src_path.exists():
                continue

            # Get image dimensions
            try:
                with Image.open(src_path) as img:
                    img_width, img_height = img.size
            except Exception:
                continue

            # Create YOLO label file
            label_lines = []
            for ann, cls_name in valid_anns:
                cls_idx = CLASS_TO_IDX[cls_name]
                bbox = convert_bbox_to_yolo(ann["bbox"], img_width, img_height)
                label_lines.append(f"{cls_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                split_class_counts[cls_name] += 1

            # Copy image
            dst_img_path = processed_dir / "images" / split / f"{img_id}.jpg"
            shutil.copy2(src_path, dst_img_path)

            # Write label
            dst_label_path = processed_dir / "labels" / split / f"{img_id}.txt"
            with open(dst_label_path, "w") as f:
                f.write("\n".join(label_lines))

        stats[split] = split_class_counts

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        total = 0
        for cls, count in stats[split].items():
            print(f"  {cls}: {count}")
            total += count
        print(f"  TOTAL: {total}")

    return stats


def update_dataset_yaml(processed_dir: Path, config_path: Path) -> None:
    """Update dataset.yaml with correct paths and classes."""
    yaml_content = f"""# Caltech Camera Traps Dataset Configuration
# Auto-generated by prepare_data.py

path: {processed_dir.absolute()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(TARGET_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    yaml_content += f"\nnc: {len(TARGET_CLASSES)}\n"

    with open(config_path, "w") as f:
        f.write(yaml_content)

    print(f"\nUpdated {config_path}")


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    config_path = project_root / "configs" / "dataset.yaml"

    # Check if data exists, otherwise show download instructions
    annotation_file = raw_dir / "caltech_camera_traps.json"

    if not annotation_file.exists():
        download_caltech_cameratraps(data_dir)
        print("\nAfter downloading, run this script again to prepare the data.")
    else:
        stats = prepare_dataset(raw_dir, processed_dir)
        if stats:
            update_dataset_yaml(processed_dir, config_path)
            print("\nDataset preparation complete!")
            print("You can now run training with: python src/train.py")
