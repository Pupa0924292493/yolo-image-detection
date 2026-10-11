
import os
import io
from pathlib import Path
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CLASS_NAMES = {
    0: "Forgot to Zip Up",
    1: "Sleep",
    2: "Talk with Friend",
    3: "Using Phone",
    4: "Yawn",
}

MODEL_PATH = Path(__file__).parent / "best.onnx"
session = None

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO11n Image Detection</title>
<style>
body {font-family:Arial,sans-serif;background:#f5f7fb;margin:0;padding:25px}
main {max-width:1100px;margin:auto}
h1 {text-align:center}
p {text-align:center;color:#555}
#drop {background:white;border:3px dashed #aaa;border-radius:14px;
padding:30px;text-align:center;cursor:pointer;margin:20px 0}
#drop.over {border-color:#2563eb;background:#eff6ff}
.grid {display:grid;grid-template-columns:1fr 1fr;gap:20px}
.panel {background:white;padding:15px;border-radius:12px}
img {max-width:100%;max-height:500px;display:block;margin:auto}
input[type=file] {display:none}
button {padding:10px 18px;cursor:pointer}
#status {font-weight:bold;margin:15px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<main>
<h1>YOLO11n Image Detection</h1>
<p>Upload, drag & drop, or paste an image using Ctrl+V</p>
<div id="drop">
<h3>Click here to upload an image</h3>
<p>Or drag an image here / Press Ctrl+V</p>
<input type="file" id="file" accept="image/*">
</div>
<div style="text-align:center">
Confidence:
<input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.25">
<span id="confvalue">0.25</span>
<button id="clear">Clear</button>
</div>
<p id="status">Ready</p>
<div class="grid">
<div class="panel"><h3>Input Image</h3><img id="input"></div>
<div class="panel"><h3>Detection Result</h3><img id="output"></div>
</div>
<div class="panel" style="margin-top:20px">
<h3>Detected Behaviors</h3>
<pre id="summary">No image selected.</pre>
</div>
</main>
<script>
const drop=document.getElementById("drop");
const file=document.getElementById("file");
const input=document.getElementById("input");
const output=document.getElementById("output");
const status=document.getElementById("status");
const summary=document.getElementById("summary");
const conf=document.getElementById("conf");
let currentFile=null;
let requestId=0;

drop.onclick=()=>file.click();
file.onchange=()=>{if(file.files[0])load(file.files[0]);};
drop.ondragover=e=>{e.preventDefault();drop.classList.add("over");};
drop.ondragleave=()=>drop.classList.remove("over");
drop.ondrop=e=>{
 e.preventDefault();drop.classList.remove("over");
 const f=Array.from(e.dataTransfer.files).find(x=>x.type.startsWith("image/"));
 if(f)load(f);
};
document.addEventListener("paste",e=>{
 const items=Array.from(e.clipboardData?.items||[]);
 const item=items.find(x=>x.type.startsWith("image/"));
 if(item){e.preventDefault();load(item.getAsFile());}
});
conf.oninput=()=>document.getElementById("confvalue").textContent=conf.value;
conf.onchange=()=>{if(currentFile)detect(currentFile);};
document.getElementById("clear").onclick=()=>{
 requestId++;currentFile=null;file.value="";
 input.removeAttribute("src");output.removeAttribute("src");
 summary.textContent="No image selected.";status.textContent="Ready";
};
function load(f){
 if(!f||!f.type.startsWith("image/"))return;
 currentFile=f;
 input.src=URL.createObjectURL(f);
 output.removeAttribute("src");
 detect(f);
}
async function detect(f){
 const id=++requestId;
 status.textContent="Detecting...";
 const data=new FormData();
 data.append("image",f,"image.png");
 data.append("confidence",conf.value);
 try{
  const res=await fetch("/predict",{method:"POST",body:data});
  const json=await res.json();
  if(id!==requestId)return;
  if(!res.ok)throw new Error(json.error||"Detection failed");
  output.src="data:image/jpeg;base64,"+json.image;
  summary.textContent=json.summary;
  status.textContent="Detection completed";
 }catch(err){
  if(id!==requestId)return;
  status.textContent="Error: "+err.message;
 }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

def get_session():
    global session
    if session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Upload best.onnx to GitHub first.")
        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
        )
    return session

@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(error="No image uploaded"), 400

        conf = float(request.form.get("confidence", 0.25))
        conf = max(0.1, min(conf, 0.9))
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(uploaded.read()))
        ).convert("RGB")
        image.thumbnail((960, 960))
        original = np.array(image)
        h, w = original.shape[:2]

        size = 416
        scale = min(size / w, size / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(original, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        px, py = (size - nw) // 2, (size - nh) // 2
        canvas[py:py+nh, px:px+nw] = resized

        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
        sess = get_session()
        inp = sess.get_inputs()[0]
        shape = inp.shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            expected = shape[2]
            if expected != size:
                raise ValueError(
                    f"ONNX expects {expected}px. Export model with imgsz=416."
                )
        predictions = sess.run(None, {inp.name: tensor})[0]
        predictions = np.squeeze(predictions)

        if predictions.ndim != 2:
            raise ValueError("Unexpected ONNX output shape")

        # Standard Ultralytics YOLO11 detect export: (4+classes, anchors)
        if predictions.shape[0] == 4 + len(CLASS_NAMES):
            predictions = predictions.T
        elif predictions.shape[1] != 4 + len(CLASS_NAMES):
            raise ValueError(
                f"Unexpected ONNX output: {predictions.shape}"
            )

        boxes, scores, classes = [], [], []
        for row in predictions:
            cls = int(np.argmax(row[4:]))
            score = float(row[4 + cls])
            if score < conf:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2 - px) / scale
            y1 = (cy - bh / 2 - py) / scale
            x2 = (cx + bw / 2 - px) / scale
            y2 = (cy + bh / 2 - py) / scale
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            scores.append(score)
            classes.append(cls)

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)
        annotated = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        counts = Counter()

        for i in np.array(indices).reshape(-1)[:30]:
            x, y, bw, bh = boxes[int(i)]
            cls = classes[int(i)]
            score = scores[int(i)]
            counts[cls] += 1
            cv2.rectangle(annotated, (x,y), (x+bw,y+bh), (0,180,80), 2)
            label = f"{CLASS_NAMES[cls]} {score:.2f}"
            cv2.putText(
                annotated,label,(x,max(y-7,15)),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,180,80),2
            )

        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise ValueError("Could not encode result")

        import base64
        summary = "\\n".join(
            f"{CLASS_NAMES[c]}: {n}" for c,n in sorted(counts.items())
        ) if counts else "No behavior detected."

        return jsonify(
            image=base64.b64encode(encoded.tobytes()).decode(),
            summary=summary,
        )
    except Exception as e:
        app.logger.exception("Detection failed")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
