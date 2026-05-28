"""
Wildlife Detector - Gradio Demo

Upload an image to detect wildlife using YOLOv8.
Deployed on Hugging Face Spaces.
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Use ONNX Runtime for faster CPU inference if available
try:
    import onnxruntime as ort

    USE_ONNX = True
except ImportError:
    USE_ONNX = False

from ultralytics import YOLO

# Model path - will be set based on available model
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "models/best.onnx")

# Class names
CLASS_NAMES = [
    "deer",
    "coyote",
    "rabbit",
    "squirrel",
    "bird",
    "bobcat",
    "raccoon",
    "skunk",
]

# Colors for bounding boxes (RGB)
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
]


def load_model():
    """Load the detection model."""
    # Prefer ONNX for CPU inference
    if USE_ONNX and Path(ONNX_MODEL_PATH).exists():
        print(f"Loading ONNX model from {ONNX_MODEL_PATH}")
        return YOLO(ONNX_MODEL_PATH)
    elif Path(MODEL_PATH).exists():
        print(f"Loading PyTorch model from {MODEL_PATH}")
        return YOLO(MODEL_PATH)
    else:
        # Fallback to pretrained for demo
        print("No trained model found, using pretrained YOLOv8n")
        return YOLO("yolov8n.pt")


# Load model at startup
model = load_model()


def detect_wildlife(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> tuple[Image.Image, str]:
    """
    Run wildlife detection on an image.

    Args:
        image: Input PIL Image
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        Annotated image and detection summary
    """
    if image is None:
        return None, "Please upload an image."

    # Run inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    # Get annotated image
    result = results[0]
    annotated_image = result.plot()

    # Convert BGR to RGB (OpenCV uses BGR)
    annotated_image = annotated_image[:, :, ::-1]
    annotated_pil = Image.fromarray(annotated_image)

    # Build detection summary
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get class name
            if cls_id < len(CLASS_NAMES):
                cls_name = CLASS_NAMES[cls_id]
            else:
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

            detections.append(f"- {cls_name}: {conf:.1%}")

    if detections:
        summary = f"**Detected {len(detections)} object(s):**\n" + "\n".join(detections)
    else:
        summary = "No wildlife detected in this image."

    return annotated_pil, summary


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="Wildlife Detector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🦌 Wildlife Detector

            Upload an image to detect wildlife using YOLOv8 trained from scratch
            on the Caltech Camera Traps dataset.

            **Detectable animals:** deer, coyote, rabbit, squirrel, bird, bobcat, raccoon, skunk
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                    )

                detect_btn = gr.Button("Detect Wildlife", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Detection Results")
                output_text = gr.Markdown(label="Summary")

        # Example images
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["examples/deer.jpg"],
                ["examples/coyote.jpg"],
                ["examples/bird.jpg"],
            ],
            inputs=[input_image],
            outputs=[output_image, output_text],
            fn=detect_wildlife,
            cache_examples=False,
        )

        # Event handlers
        detect_btn.click(
            fn=detect_wildlife,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_text],
        )

        input_image.change(
            fn=detect_wildlife,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_text],
        )

        gr.Markdown(
            """
            ---
            **Model:** YOLOv8n trained from scratch | **Dataset:** Caltech Camera Traps |
            **Training:** Tracked with Weights & Biases

            [GitHub](https://github.com/yourusername/wildlife-detector) |
            [W&B Report](https://wandb.ai/your-report)
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
