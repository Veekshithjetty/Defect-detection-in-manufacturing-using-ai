
# streamlit_app.py — Manufacturing Defect Detection Dashboard
# Run: streamlit run streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import csv
from datetime import datetime
from PIL import Image
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "mobilenetv2_defect.h5"
LOG_CSV     = "inspection_log.csv"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Crazing", "Inclusion", "Patches",
               "Pitted_Surface", "Rolled-in_Scale", "Scratches"]

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏭 AI Defect Inspector",
    page_icon="🔍",
    layout="wide"
)

st.title("🏭 Manufacturing Defect Detection System")
st.markdown("**AI-powered real-time quality inspection using MobileNetV2**")
st.divider()

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Severity Helper ──────────────────────────────────────────────────────────
def get_severity(conf):
    if conf >= 90:   return "HIGH",   "🔴"
    elif conf >= 70: return "MEDIUM", "🟡"
    else:            return "LOW",    "🟢"

# ── Log Helper ───────────────────────────────────────────────────────────────
def log_prediction(image_name, pred_class, confidence, severity):
    headers = ["timestamp","image_name","predicted_class","confidence_pct","severity"]
    record  = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_name, pred_class, round(confidence, 2), severity]
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(headers)
        w.writerow(record)

# ── Layout: Two Columns ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Product Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_arr = np.array(pil_img.resize(IMG_SIZE)).astype("float32") / 255.0
        img_bat = np.expand_dims(img_arr, 0)

        # Predict
        probs    = model.predict(img_bat, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[pred_idx]) * 100
        pred_cls = CLASS_NAMES[pred_idx]
        sev, sev_icon = get_severity(conf)

        log_prediction(uploaded.name, pred_cls, conf, sev)

with col2:
    if uploaded:
        st.subheader("🎯 Prediction Results")

        color = {"HIGH":"red", "MEDIUM":"orange", "LOW":"green"}[sev]
        st.markdown(f"""<div style="background:{color}22; border-left:5px solid {color};
            padding:15px; border-radius:8px;">
            <h3 style="color:{color}; margin:0">{sev_icon} Defect Type: {pred_cls}</h3>
            <p style="margin:5px 0">Confidence: <b>{conf:.1f}%</b></p>
            <p style="margin:0">Severity: <b>{sev}</b></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.progress(int(conf))

        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({
            "Class"      : CLASS_NAMES,
            "Probability": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probability", ascending=False)

        fig = px.bar(prob_df, x="Probability", y="Class", orientation="h",
                     color="Probability", color_continuous_scale="RdYlGn",
                     range_x=[0, 100])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── Inspection History ───────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Inspection History")

if os.path.exists(LOG_CSV):
    log_df = pd.read_csv(LOG_CSV)
    st.dataframe(log_df.tail(20), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig2 = px.pie(log_df, names="predicted_class",
                       title="Defect Distribution",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        fig3 = px.pie(log_df, names="severity",
                       title="Severity Distribution",
                       color_discrete_map={"HIGH":"#e74c3c",
                                           "MEDIUM":"#f39c12",
                                           "LOW":"#2ecc71"})
        st.plotly_chart(fig3, use_container_width=True)

    # Download button
    st.download_button(
        label     = "⬇️ Download Inspection Report (CSV)",
        data      = log_df.to_csv(index=False),
        file_name = "inspection_report.csv",
        mime      = "text/csv"
    )
else:
    st.info("No inspection history yet. Upload an image to begin.")

st.sidebar.markdown("""
## ℹ️ About
**AI Defect Inspector v1.0**  
Built with TensorFlow + Streamlit  
Dataset: NEU Metal Surface Defects  

**Severity Rules:**  
🔴 HIGH   → ≥90% confidence  
🟡 MEDIUM → 70–89%  
🟢 LOW    → <70%  
""")
