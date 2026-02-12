import os
import numpy as np
import cv2
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image
from skimage.color import rgb2lab, rgb2hsv

# ------------------ TRY IMPORT CAFFE ------------------
CAFFE_AVAILABLE = True
try:
    import caffe
except Exception as e:
    CAFFE_AVAILABLE = False
    print("⚠️ Caffe not available:", e)

# ------------------ FLASK APP ------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ LOAD CAFFE MODEL (ONLY IF AVAILABLE) ------------------
if CAFFE_AVAILABLE:
    PROTOTXT = "models_colorization_deploy_v2.prototxt"
    MODEL = "colorization_release_v2.caffemodel"
    POINTS = "pts_in_hull.npy"

    net = caffe.Net(PROTOTXT, MODEL, caffe.TEST)
    pts = np.load(POINTS)
    net.params['class8_ab'][0].data[:] = pts.transpose((1,0))[:,:,None,None]

# ------------------ HTML UI ------------------
HTML = """
<!doctype html>
<title>BW Image Colorization</title>

<h2>Grayscale Image Colorization (Caffe)</h2>

{% if not caffe %}
<p style="color:red;">
⚠️ Caffe is NOT available on this system.<br>
Colorization works only on Linux / Kaggle / Colab.
</p>
{% endif %}

<form method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button type="submit">Colorize</button>
</form>

{% if img %}
<hr>
<h3>Output</h3>
<img src="{{ img }}" width="300">

<h3>No-Reference Metrics</h3>
<pre>{{ metrics }}</pre>
{% endif %}
"""

# ------------------ NO-REFERENCE METRICS ------------------
def no_reference_metrics(rgb):
    lab = rgb2lab(rgb)
    a, b = lab[:,:,1], lab[:,:,2]

    chroma = np.sqrt(a*a + b*b)
    hsv = rgb2hsv(rgb)
    sat = hsv[:,:,1]

    hist, _ = np.histogram(hsv[:,:,0], bins=100, range=(0,1), density=True)
    hist += 1e-9

    return {
        "Colorfulness": np.std(a) + np.std(b),
        "Mean Saturation": sat.mean(),
        "Saturation Std": sat.std(),
        "Mean Chroma": chroma.mean(),
        "Chroma Std": chroma.std(),
        "Neutral Pixels (%)": np.mean(chroma < 5) * 100,
        "Hue Entropy": -np.sum(hist * np.log2(hist))
    }

# ------------------ ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    img_path, metrics_text = None, ""

    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # Load grayscale image
        gray = Image.open(path).convert("L").resize((224,224))
        gray_np = np.array(gray)

        # --------- IF CAFFE AVAILABLE → REAL COLORIZATION ---------
        if CAFFE_AVAILABLE:
            lab = cv2.cvtColor(np.dstack([gray_np]*3), cv2.COLOR_RGB2LAB)
            L = lab[:,:,0]

            net.blobs['data'].data[...] = L[np.newaxis,np.newaxis,:,:] - 50
            net.forward()

            ab = net.blobs['class8_ab'].data[0].transpose((1,2,0))
            ab = cv2.resize(ab, (224,224))

            lab_out = np.zeros((224,224,3))
            lab_out[:,:,0] = L
            lab_out[:,:,1:] = ab

            rgb = cv2.cvtColor(lab_out.astype(np.float32), cv2.COLOR_LAB2RGB)
            rgb = np.clip(rgb, 0, 1)

        # --------- IF CAFFE NOT AVAILABLE → FAKE COLOR (DEMO) ---------
        else:
            rgb = np.dstack([gray_np, gray_np, gray_np]) / 255.0

        # Save output
        out_img = (rgb*255).astype(np.uint8)
        out_path = os.path.join(UPLOAD_FOLDER, "colorized_"+filename)
        Image.fromarray(out_img).save(out_path)

        metrics = no_reference_metrics(rgb)
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k,v in metrics.items()])
        img_path = out_path

    return render_template_string(
        HTML,
        img=img_path,
        metrics=metrics_text,
        caffe=CAFFE_AVAILABLE
    )

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
