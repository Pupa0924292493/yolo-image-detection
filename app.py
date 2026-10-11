
import os
import io
import base64
import threading
from pathlib import Path
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

MODEL_PATH = Path(__file__).parent / "best (43).onnx"

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

COLORS = [
    "#ff5252", "#42a5f5", "#ab47bc",
    "#ffa726", "#66bb6a"
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading ONNX model...", flush=True)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)

INPUT_NAME = session.get_inputs()[0].name
INPUT_SHAPE = session.get_inputs()[0].shape
MODEL_SIZE = (
    INPUT_SHAPE[2]
    if isinstance(INPUT_SHAPE[2], int)
    else 640
)

print("Model loaded successfully!", flush=True)

lock = threading.Lock()


def preprocess(image):
    original_w, original_h = image.size
    size = MODEL_SIZE

    scale = min(size / original_w, size / original_h)
    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized = image.resize((new_w, new_h))

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, 0)

    return array, scale, pad_x, pad_y


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    return intersection / (area1 + area2 - intersection + 1e-6)


def detect(image, confidence=0.25):
    original_w, original_h = image.size

    input_tensor, scale, pad_x, pad_y = preprocess(image)

    with lock:
        outputs = session.run(None, {INPUT_NAME: input_tensor})

    predictions = np.asarray(outputs[0])

    # Standard YOLO11 detection output: (1, 4+classes, boxes)
    if predictions.ndim != 3:
        raise ValueError(f"Unexpected ONNX output: {predictions.shape}")

    predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    num_classes = len(CLASS_NAMES)

    if predictions.shape[1] != 4 + num_classes:
        raise ValueError(
            f"Expected {4 + num_classes} channels, "
            f"got {predictions.shape[1]}. "
            "Check that this is the correct YOLO11 model."
        )

    scores = predictions[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= confidence

    boxes_xywh = predictions[mask, :4]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return image, {}

    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)

    # Class-aware non-maximum suppression
    keep = []

    for cls in np.unique(class_ids):
        indices = np.where(class_ids == cls)[0]
        indices = indices[np.argsort(confidences[indices])[::-1]]

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            overlaps = iou(boxes[current], boxes[indices[1:]])
            indices = indices[1:][overlaps < 0.45]

    keep = sorted(keep, key=lambda i: confidences[i], reverse=True)[:30]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    counts = Counter()

    for idx in keep:
        cls = int(class_ids[idx])
        conf = float(confidences[idx])
        x1, y1, x2, y2 = boxes[idx].astype(int)

        counts[cls] += 1
        color = COLORS[cls % len(COLORS)]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=max(2, original_w // 250)
        )

        label = f"{CLASS_NAMES[cls]} {conf:.0%}"
        text_box = draw.textbbox((x1, y1), label)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]

        label_y = max(0, y1 - text_h - 8)

        draw.rectangle(
            [x1, label_y, x1 + text_w + 10, label_y + text_h + 8],
            fill=color
        )
        draw.text(
            (x1 + 5, label_y + 3),
            label,
            fill="white"
        )

    summary = {
        CLASS_NAMES[cls]: count
        for cls, count in sorted(counts.items())
    }

    return annotated, summary


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO Image Detection</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #f4f6fb;
    font-family: Arial, sans-serif;
    color: #202636;
}
.container {
    max-width: 1200px;
    margin: 35px auto;
    padding: 20px;
}
h1 { margin-bottom: 8px; }
.subtitle { color: #667085; margin-bottom: 24px; }
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
.card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 15px #0000000c;
}
.card h3 { margin-top: 0; }
.dropzone {
    border: 2px dashed #a7b4cb;
    border-radius: 12px;
    min-height: 350px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    cursor: pointer;
    overflow: hidden;
    background: #fafbff;
    transition: 0.2s;
}
.dropzone.dragover {
    border-color: #3478f6;
    background: #eaf2ff;
}
.dropzone img, .result img {
    max-width: 100%;
    max-height: 470px;
    object-fit: contain;
}
.result {
    min-height: 350px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #fafbff;
    border-radius: 12px;
    overflow: hidden;
}
.placeholder { color: #98a2b3; text-align: center; padding: 20px; }
.controls { margin-top: 22px; }
input[type=range] { width: 100%; accent-color: #3478f6; }
button {
    background: #3478f6;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    cursor: pointer;
    margin-top: 15px;
}
.summary { margin-top: 20px; line-height: 1.8; }
.status { margin-top: 12px; color: #667085; }
@media (max-width: 750px) {
    .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">
    <h1>YOLO Image Detection Application</h1>
    <p class="subtitle">
        Upload, drag & drop, or paste an image using Ctrl+V.
        Detection runs automatically.
    </p>

    <div class="grid">
        <div class="card">
            <h3>Input Image</h3>
            <div class="dropzone" id="dropzone">
                <div class="placeholder" id="placeholder">
                    Click to upload, drag an image here,<br>
                    or press Ctrl+V anywhere on this page.
                </div>
                <img id="preview" style="display:none">
            </div>
            <input id="fileInput" type="file" accept="image/*" hidden>
            <button onclick="document.getElementById('fileInput').click()">
                Choose Image
            </button>
            <button onclick="clearImage()" style="background:#667085">
                Clear
            </button>
        </div>

        <div class="card">
            <h3>Detection Result</h3>
            <div class="result" id="result">
                <span class="placeholder">Result will appear here</span>
            </div>
            <div class="status" id="status">Ready</div>
        </div>
    </div>

    <div class="card controls">
        <label>
            Confidence Threshold:
            <strong id="confidenceValue">0.25</strong>
        </label>
        <input
            type="range" id="confidence"
            min="0.10" max="0.90"
            step="0.05" value="0.25"
        >
        <div class="summary">
            <strong>Detected Behaviors</strong>
            <div id="summary">No image selected.</div>
        </div>
    </div>
</div>

<script>
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const result = document.getElementById("result");
const statusText = document.getElementById("status");
const summary = document.getElementById("summary");
const confidence = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");

let currentFile = null;
let requestNumber = 0;
let controller = null;
let debounceTimer = null;

function loadImage(file) {
    if (!file || !file.type.startsWith("image/")) return;

    currentFile = file;
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    placeholder.style.display = "none";

    runDetection();
}

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (event) => {
    loadImage(event.target.files[0]);
    fileInput.value = "";
});

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    loadImage(event.dataTransfer.files[0]);
});

document.addEventListener("paste", (event) => {
    const items = event.clipboardData?.items || [];

    for (const item of items) {
        if (item.type.startsWith("image/")) {
            event.preventDefault();
            loadImage(item.getAsFile());
            return;
        }
    }
});

confidence.addEventListener("input", () => {
    confidenceValue.textContent = confidence.value;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runDetection, 350);
});

async function runDetection() {
    if (!currentFile) return;

    const myRequest = ++requestNumber;

    if (controller) controller.abort();
    controller = new AbortController();

    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("confidence", confidence.value);

    statusText.textContent = "Detecting...";
    summary.textContent = "Processing image...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
            signal: controller.signal
        });

        const data = await response.json();

        if (myRequest !== requestNumber) return;

        if (!response.ok) {
            throw new Error(data.error || "Detection failed");
        }

        result.innerHTML = "";

        const img = document.createElement("img");
        img.src = data.image;
        result.appendChild(img);

        const entries = Object.entries(data.summary);

        summary.textContent = entries.length
            ? entries.map(([name, count]) => name + ": " + count).join(" | ")
            : "No behavior detected.";

        statusText.textContent = "Detection completed";

    } catch (error) {
        if (error.name === "AbortError") return;
        if (myRequest !== requestNumber) return;

        statusText.textContent = "Error: " + error.message;
        summary.textContent = "Unable to detect image.";
    }
}

function clearImage() {
    requestNumber++;
    if (controller) controller.abort();

    currentFile = null;
    preview.removeAttribute("src");
    preview.style.display = "none";
    placeholder.style.display = "block";

    result.innerHTML =
        '<span class="placeholder">Result will appear here</span>';

    summary.textContent = "No image selected.";
    statusText.textContent = "Ready";
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        confidence = float(request.form.get("confidence", 0.25))
        confidence = max(0.1, min(confidence, 0.9))

        image = Image.open(file.stream).convert("RGB")

        # Resize large images to reduce CPU processing time
        image.thumbnail((960, 960))

        annotated, summary = detect(image, confidence)

        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=85)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({
            "image": "data:image/jpeg;base64," + encoded,
            "summary": summary
        })

    except Exception as e:
        print(f"Prediction error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

