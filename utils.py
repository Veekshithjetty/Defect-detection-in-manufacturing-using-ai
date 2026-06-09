
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime
import os
import json

CLASS_NAMES = ["Crazing", "Inclusion", "Patches",
               "Pitted_Surface", "Rolled-in_Scale", "Scratches"]
IMG_SIZE    = (224, 224)



def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given image and model.

    Args:
        model              : Trained Keras model
        img_array          : Preprocessed image (1, H, W, C) in [0,1]
        last_conv_layer_name: Name of the last Conv layer (e.g. 'Conv_1')
        pred_index         : Class index to explain (None = argmax)

    Returns:
        heatmap : np.ndarray (H, W) in [0, 1]
    """
    grad_model = keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads    = tape.gradient(class_channel, conv_outputs)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap on the original image.

    Args:
        img_rgb  : Original image as np.ndarray (H, W, 3) uint8
        heatmap  : Grad-CAM heatmap (H', W') in [0,1]
        alpha    : Transparency of heatmap overlay
        colormap : OpenCV colormap constant

    Returns:
        superimposed_img : np.ndarray (H, W, 3) uint8
    """
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed


def visualise_gradcam(model, image_path, last_conv_layer_name="Conv_1",
                       class_names=CLASS_NAMES, save_path=None):
    """
    Full Grad-CAM visualisation pipeline: loads image, computes heatmap,
    displays side-by-side comparison.
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE)
    img_arr = img_res.astype("float32") / 255.0
    img_bat = np.expand_dims(img_arr, 0)

    probs    = model.predict(img_bat, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf     = float(probs[pred_idx]) * 100
    pred_cls = class_names[pred_idx]

    heatmap    = get_gradcam_heatmap(model, img_bat, last_conv_layer_name, pred_idx)
    cam_img    = overlay_gradcam(img_res, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_res)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    heatmap_display = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(heatmap_display, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(cam_img)
    axes[2].set_title(
        f"Overlay\nPrediction: {pred_cls}\nConfidence: {conf:.1f}%",
        fontweight="bold", color="red" if conf >= 90 else "orange"
    )
    axes[2].axis("off")

    plt.suptitle(" Grad-CAM Explainability", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Grad-CAM saved: {save_path}")
    plt.show()

    return {"class": pred_cls, "confidence": conf, "heatmap": heatmap}



def batch_infer_folder(model, folder_path, output_csv="batch_results.csv",
                        class_names=CLASS_NAMES):
    """
    Runs inference on all images in a folder and saves results to CSV.

    Args:
        model       : Trained Keras model
        folder_path : Path to folder of images
        output_csv  : Path to save results CSV

    Returns:
        pd.DataFrame with columns: filename, predicted_class, confidence, severity
    """
    records = []
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"❌ No images found in: {folder_path}")
        return pd.DataFrame()

    print(f"🔍 Running batch inference on {len(image_files)} images...")

    for fname in image_files:
        fpath  = os.path.join(folder_path, fname)
        img    = cv2.imread(fpath)
        if img is None:
            continue
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img    = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
        batch  = np.expand_dims(img, 0)

        probs    = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[pred_idx]) * 100

        if conf >= 90:   sev = "HIGH"
        elif conf >= 70: sev = "MEDIUM"
        else:            sev = "LOW"

        records.append({
            "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename"       : fname,
            "predicted_class": class_names[pred_idx],
            "confidence_pct" : round(conf, 2),
            "severity"       : sev,
            **{f"prob_{c}": round(float(p)*100, 2) for c, p in zip(class_names, probs)}
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f" Batch inference complete. Results saved: {output_csv}")
    print(f"\n{df[['filename','predicted_class','confidence_pct','severity']].to_string(index=False)}")
    return df



def export_tflite(model, output_path="defect_model.tflite", quantize=True):
    """
    Converts a Keras model to TFLite format for edge deployment.

    Args:
        model       : Trained Keras model
        output_path : Save path for .tflite file
        quantize    : If True, applies dynamic range quantisation (reduces size ~4x)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  Applying dynamic range quantisation...")

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f" TFLite model saved: {output_path}")
    print(f"   File size: {size_mb:.2f} MB")
    return output_path


def tflite_predict(tflite_path, image_path, class_names=CLASS_NAMES):
    """
    Runs inference using a TFLite model (simulates edge device execution).

    Args:
        tflite_path : Path to .tflite file
        image_path  : Path to input image

    Returns:
        dict with predicted class, confidence, and all probabilities
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(probs))
    conf     = float(probs[pred_idx]) * 100

    result = {
        "class"        : class_names[pred_idx],
        "confidence"   : round(conf, 2),
        "probabilities": {c: round(float(p)*100,2) for c,p in zip(class_names, probs)}
    }
    print(f"TFLite Prediction: {result['class']} ({result['confidence']}%)")
    return result



def benchmark_model(model, n_runs=100, img_size=IMG_SIZE):
    """
    Measures average inference time per image.

    Args:
        model  : Keras model
        n_runs : Number of inference runs to average

    Returns:
        dict with avg_ms, min_ms, max_ms, fps
    """
    import time
    dummy = np.random.rand(1, *img_size, 3).astype("float32")

    for _ in range(5):
        model.predict(dummy, verbose=0)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    result = {
        "avg_ms"  : round(avg, 2),
        "min_ms"  : round(np.min(times), 2),
        "max_ms"  : round(np.max(times), 2),
        "fps"     : round(1000 / avg, 1)
    }
    print(f"\n Benchmark Results ({n_runs} runs):")
    print(f"   Avg latency : {result['avg_ms']} ms")
    print(f"   Min latency : {result['min_ms']} ms")
    print(f"   Max latency : {result['max_ms']} ms")
    print(f"   Throughput  : {result['fps']} FPS")
    return result



def compare_predictions(model, X_test, y_test, class_names=CLASS_NAMES,
                         n_samples=16, save_path=None):
    """
    Displays a grid of test images with True vs Predicted labels.
    Correct predictions shown in green, incorrect in red.
    """
    indices  = np.random.choice(len(X_test), n_samples, replace=False)
    cols     = 4
    rows     = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for ax, idx in zip(axes.flat, indices):
        img      = X_test[idx]
        true_idx = int(np.argmax(y_test[idx]))
        probs    = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[pred_idx]) * 100

        correct = true_idx == pred_idx
        color   = "green" if correct else "red"
        icon    = "✅" if correct else "❌"

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{icon} True: {class_names[true_idx]}\n"
            f"Pred: {class_names[pred_idx]} ({conf:.1f}%)",
            fontsize=8, color=color, fontweight="bold"
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    plt.suptitle(" Ground Truth vs Model Predictions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved: {save_path}")
    plt.show()



def generate_html_report(log_csv="inspection_log.csv",
                          output_html="inspection_report.html"):
    """
    Generates a styled HTML inspection report from the CSV log.
    """
    if not os.path.exists(log_csv):
        print(f" Log not found: {log_csv}")
        return

    df = pd.read_csv(log_csv)
    total      = len(df)
    high_count = (df["severity"] == "HIGH").sum()
    avg_conf   = df["confidence_pct"].mean()

    sev_color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}

    rows_html = ""
    for _, row in df.iterrows():
        color = sev_color.get(row["severity"], "#95a5a6")
        rows_html += (
            f"<tr>"
            f"<td>{row['timestamp']}</td>"
            f"<td>{row['image_name']}</td>"
            f"<td>{row['predicted_class']}</td>"
            f"<td>{row['confidence_pct']:.1f}%</td>"
            f"<td style='color:{color}; font-weight:bold'>{row['severity']}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Defect Inspection Report</title>
<style>
  body   {{ font-family: 'Segoe UI', sans-serif; background:#f0f2f5; margin:0; padding:20px; }}
  h1     {{ color:#2c3e50; }}
  .kpis  {{ display:flex; gap:20px; margin:20px 0; }}
  .kpi   {{ background:#fff; border-radius:10px; padding:16px 24px;
             box-shadow:0 2px 8px rgba(0,0,0,.1); flex:1; text-align:center; }}
  .kpi h2{{ margin:0; font-size:2rem; color:#e74c3c; }}
  .kpi p {{ margin:4px 0 0; color:#666; }}
  table  {{ width:100%; border-collapse:collapse; background:#fff;
             border-radius:10px; overflow:hidden;
             box-shadow:0 2px 8px rgba(0,0,0,.1); }}
  th     {{ background:#2c3e50; color:#fff; padding:12px; text-align:left; }}
  td     {{ padding:10px 12px; border-bottom:1px solid #eee; }}
  tr:hover{{ background:#f9f9f9; }}
  .footer{{ text-align:center; margin-top:20px; color:#999; font-size:.85rem; }}
</style>
</head>
<body>
<h1> Manufacturing Defect Inspection Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="kpis">
  <div class="kpi"><h2>{total}</h2><p>Total Inspections</p></div>
  <div class="kpi"><h2 style="color:#e74c3c">{high_count}</h2><p>High Severity</p></div>
  <div class="kpi"><h2 style="color:#3498db">{avg_conf:.1f}%</h2><p>Avg Confidence</p></div>
  <div class="kpi"><h2 style="color:#2ecc71">{total - high_count}</h2><p>Passed</p></div>
</div>

<table>
<thead>
<tr><th>Timestamp</th><th>Image</th><th>Defect Class</th>
    <th>Confidence</th><th>Severity</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="footer">
  Defect Detection System v1.0 | GITAM University | Sai Veekshith (2023003590)
</div>
</body>
</html>"""

    with open(output_html, "w") as f:
        f.write(html)

    print(f" HTML report saved: {output_html}")
    return output_html


if __name__ == "__main__":
    print("utils.py loaded successfully.")
    print("\nAvailable utilities:")
    utilities = [
        "visualise_gradcam(model, image_path)       — Grad-CAM heatmap",
        "batch_infer_folder(model, folder_path)      — Bulk inference",
        "export_tflite(model)                        — Edge AI export",
        "tflite_predict(tflite_path, image_path)     — TFLite inference",
        "benchmark_model(model)                      — Latency benchmarking",
        "compare_predictions(model, X_test, y_test)  — GT vs Pred grid",
        "generate_html_report()                      — HTML audit report",
    ]
    for u in utilities:
        print(f"  • {u}")
