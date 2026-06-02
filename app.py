"""
Wildlife Detector - Gradio Demo

Upload an image to detect wildlife using YOLOv8.
Deployed on Hugging Face Spaces with ZeroGPU support.
"""

import os
import random
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

# ZeroGPU support - only available on HF Spaces
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

# Model path
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")

# Examples directory
EXAMPLES_DIR = Path("examples")

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

# Load model at startup
if Path(MODEL_PATH).exists():
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Please ensure best.pt is uploaded."
    )


def get_random_examples(num_examples=4):
    """Get random example images from the examples directory."""
    all_images = list(EXAMPLES_DIR.glob("*.jpg")) + list(EXAMPLES_DIR.glob("*.png"))
    if not all_images:
        return []
    selected = random.sample(all_images, min(num_examples, len(all_images)))
    return [[str(img)] for img in selected]


def gpu_decorator(func):
    """Apply @spaces.GPU decorator only when running on HF Spaces."""
    if SPACES_AVAILABLE:
        return spaces.GPU(func)
    return func


@gpu_decorator
def detect_wildlife(image, confidence_threshold=0.25):
    """
    Run wildlife detection on an image.
    """
    if image is None:
        return None, "Please upload an image."

    # Run inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=0.45,
        verbose=False,
    )

    # Get annotated image
    result = results[0]
    annotated_image = result.plot()

    # Convert BGR to RGB (OpenCV uses BGR)
    annotated_image = annotated_image[:, :, ::-1]

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
        summary = f"Detected {len(detections)} object(s):\n" + "\n".join(detections)
    else:
        summary = "No wildlife detected in this image."

    return annotated_image, summary


def refresh_examples():
    """Refresh the example gallery with new random images."""
    examples = get_random_examples(8)
    images = [str(ex[0]) for ex in examples]
    return images


# Build UI with Blocks for more control
with gr.Blocks(title="Wildlife Detector") as demo:
    gr.Markdown(
        """
        # Wildlife Detector
        Upload an image to detect wildlife using YOLOv8 trained on LILA Camera Traps dataset.

        **Detectable animals:** deer, coyote, rabbit, squirrel, bird, bobcat, raccoon, skunk

        **Metrics:** mAP50: 0.682 | Precision: 0.702 | Recall: 0.606
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Image")
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
            )
            detect_btn = gr.Button("Detect Wildlife", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detection Results")
            output_text = gr.Textbox(label="Summary")

    gr.Markdown("### Example Images")
    with gr.Row():
        refresh_btn = gr.Button("🔄 Load New Examples", size="sm")

    example_gallery = gr.Gallery(
        value=refresh_examples(),
        label="Click an image to use it",
        columns=4,
        height="auto",
        object_fit="cover",
        allow_preview=False,
    )

    # Event handlers
    detect_btn.click(
        fn=detect_wildlife,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_text],
    )

    input_image.change(
        fn=detect_wildlife,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_text],
    )

    refresh_btn.click(fn=refresh_examples, outputs=[example_gallery])

    def load_example(evt: gr.SelectData):
        # Gradio 5.x: evt.value can be dict with 'image' key or direct path
        if isinstance(evt.value, dict):
            return evt.value.get("image", {}).get("path", evt.value)
        return evt.value

    example_gallery.select(fn=load_example, outputs=[input_image])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
