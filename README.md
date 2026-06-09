
#  Defect Detection in Manufacturing Using AI

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **AI-powered computer vision system for real-time manufacturing quality inspection.**  
> Classifies 6 types of metal surface defects using MobileNetV2 Transfer Learning.

---

##  Project Overview

Manual visual inspection in manufacturing is slow, inconsistent, and expensive. This project builds an end-to-end AI defect detection pipeline that:
- Classifies 6 steel surface defect types from product images
- Provides defect inspection through an interactive Streamlit application.
- Logs every prediction with timestamp, class, confidence, and severity
- Offers an interactive Streamlit dashboard for production monitoring

---

##  Key Features

| Feature | Details |
|---------|--------|
|  Dual Model Architecture | Custom CNN + MobileNetV2 Fine-Tuned |
|  Severity Analysis | HIGH / MEDIUM / LOW confidence thresholds |
|  Audit Logging | Automatic CSV logging of all predictions |
|  Analytics Dashboard | Plotly interactive charts |
|  Web App | Streamlit deployment with upload & history |
|  Model Persistence | .h5 and SavedModel format exports |

---

##  Project Structure

```
defect-detection/
├── defect_detection.ipynb      # Main Jupyter Notebook (all 20 sections)
├── streamlit_app.py            # Streamlit web application
├── mobilenetv2_defect.h5       # Saved best model
├── inspection_log.csv          # Auto-generated prediction log
├── inspection_dashboard.html   # Interactive Plotly dashboard
├── requirements.txt            # Python dependencies
├── data/                       # Dataset folder (download from Kaggle)
│   ├── Cr/  In/  Pa/  PS/  RS/  Sc/
└── README.md
```

---

##  Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/defect-detection-ai.git
cd defect-detection-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
pip install kaggle
kaggle datasets download -d fantacher/neu-metal-surface-defects-data
unzip neu-metal-surface-defects-data.zip -d data/

# 5. Run Jupyter Notebook
jupyter notebook defect_detection.ipynb

# 6. OR run Streamlit app (after training model)
streamlit run streamlit_app.py
```

---

##  Dataset

**NEU Metal Surface Defects Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data))

| Class | Description | Images |
|-------|-------------|--------|
| Cr | Crazing | 200 |
| In | Inclusion | 200 |
| Pa | Patches | 200 |
| PS | Pitted Surface | 200 |
| RS | Rolled-in Scale | 200 |
| Sc | Scratches | 200 |

Total: **1,200 images** | Split: **70/15/15** train/val/test

---

##  Results

| Model | Test Accuracy | F1 Score | Inference Time |
|-------|-------------|---------|----------------|
| Custom CNN | ~83% | ~0.82 |
| MobileNetV2 (fine-tuned) | **~95%** | **~0.95** |

---

##  Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras, MobileNetV2
- **Computer Vision**: OpenCV
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualisation**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit

---
