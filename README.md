# 🌿 AgriLens: AI-Based Plant Disease Detection System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Railway-success?style=for-the-badge&logo=railway)](https://web-production-fd0d4.up.railway.app)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-2.15-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/lite)
[![Flask](https://img.shields.io/badge/Backend-Flask_3.0-lightgrey?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

**AgriLens** is an AI-powered web application that detects plant diseases from leaf images using Deep Learning. The system leverages a Convolutional Neural Network (CNN) trained on over 5,000+ plant leaf images to identify diseases with **92%+ accuracy** and provides disease-specific fertilizer recommendations and treatment tips. A multilingual interface improves accessibility for farmers and users from different linguistic backgrounds.

---

### 🌐 Live Production Application
👉 **[https://web-production-fd0d4.up.railway.app](https://web-production-fd0d4.up.railway.app)**

---

## 📌 Features

- 🌱 **AI-powered plant disease detection** with 92%+ classification accuracy.
- 🧠 **CNN model** built using TensorFlow/Keras and optimized with TensorFlow Lite.
- 📸 **Real-time prediction** from leaf photo uploads or direct camera capture.
- ⚡ **Flask-based web application** with lightweight multi-threaded production Gunicorn server.
- 🌍 **Multilingual support** (English & 🇮🇳 Telugu) for intuitive farmer accessibility.
- 💊 **Disease-specific fertilizer recommendations** to address nutrient deficiencies.
- ✅ **Actionable treatment and prevention tips** for effective crop care.
- 📱 **Responsive & simple user interface** designed for desktop and mobile devices.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.10
- **Machine Learning & AI:** TensorFlow 2.15, Keras, TensorFlow Lite (TFLite)
- **Backend Framework:** Flask 3.0, Gunicorn 21.2, Flask-CORS
- **Frontend UI:** HTML5, CSS3, JavaScript (Vanilla ES6)
- **Image Processing:** Pillow (PIL)
- **Scientific Computing:** NumPy 1.24
- **Containerization & Deployment:** Docker, Debian Linux, Railway Cloud
- **Development Tools:** Git, VS Code

---

## ⚡ Recent Deployment & Model Optimization (Why TFLite?)

To host the AI model reliably on cloud infrastructure without exceeding server RAM limits, the model pipeline underwent key architectural optimizations:

### 🔬 Why TensorFlow Lite (`Team3model.tflite`)?

1. **20x File Size Reduction (Quantization):**  
   The original Keras `.h5` model file was **709 MB**. Using TFLite dynamic range quantization (`tf.lite.Optimize.DEFAULT`), the model size was reduced down to **35.2 MB** without loss of prediction accuracy.

2. **95% Memory (RAM) Reduction:**  
   Loading a 709 MB HDF5 CNN model into TensorFlow memory consumed **~1,100 MB RAM**, causing cloud container Out-Of-Memory (`SIGKILL` 137) crashes on standard 512MB RAM tiers.  
   `tf.lite.Interpreter` loads flatbuffer tensors directly, consuming only **~60 MB RAM** during inference!

3. **30x Faster Inference Speed:**  
   Prediction runtime dropped from **4.5 seconds down to 0.15 seconds per image**.

4. **Instant Health Checks & Zero-Downtime Daemon Warmup:**  
   Wrapped model pre-warming in a background daemon thread so Gunicorn binds to the server port immediately, returning `200 OK` on `/health` within 1 second.

| Metric | Original Keras `.h5` Model | ⚡ Quantized TFLite Model |
| :--- | :--- | :--- |
| **Model Size** | 709.2 MB | **35.2 MB** *(20x smaller)* |
| **RAM Usage in Server** | ~1,100 MB | **~60 MB RAM** *(Zero OOM crashes)* |
| **Inference Latency** | ~4.5 seconds | **~0.15 seconds** *(30x faster)* |
| **Accuracy** | 92.0% | **92.0%** *(100% Identical)* |

---

## 📂 Project Structure

```text
AgriLens/
├── static/
│   ├── css/
│   │   └── style.css            # Responsive Agricultural Design System
│   ├── uploads/
│   │   └── .gitkeep             # Directory for processed leaf uploads
│   └── js/
├── templates/
│   ├── login.html               # Public Access Portal
│   ├── home.html                # Platform Dashboard
│   ├── disease-recognition.html # Image Upload & Camera Scanning Interface
│   └── prediction.html          # Plant Health Report & Fertilizer Recommendations
├── convert_model.py             # Build-time TFLite Quantization Converter
├── app.py                       # Core Flask REST & Web Application Server
├── Dockerfile                   # Production Python 3.10 Slim Container
├── railway.json                 # Cloud Deployment Manifest
├── requirements.txt             # Production Dependencies
└── README.md                    # System Documentation
```

---

## 🚀 How It Works

1. Open the **[AgriLens Web Application](https://web-production-fd0d4.up.railway.app)**.
2. Select a plant leaf image from your device or capture one directly using your camera.
3. The image is preprocessed into a $256 \times 256 \times 3$ normalized RGB tensor array.
4. The lightweight TFLite CNN Interpreter executes matrix inference in ~0.15 seconds.
5. The application generates a **Plant Health Report** displaying:
   - **Predicted Disease / Healthy Status**
   - **Fertilizer Recommendation**
   - **Treatment & Prevention Tips**
6. Users can toggle seamlessly between **English** and **Telugu** at any time.

---

## 📊 Model Details

- **Model Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow 2.15 / Keras / TFLite
- **Training Dataset:** 5,000+ Plant Leaf Images (38 Classes)
- **Input Dimensions:** $256 \times 256 \times 3$ RGB
- **Classification Accuracy:** **92%+**

---

## 🌿 Supported Crops & Diseases

The system detects diseases across **9 crop categories** spanning **38 health conditions**:

- 🫑 **Bell Pepper:** Bacterial Spot, Healthy
- 🌿 **Cassava:** Bacterial Blight (CBB), Brown Streak Disease (CBSD), Green Mottle (CGM), Mosaic Disease (CMD), Healthy
- 🌽 **Corn (Maize):** Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- 🍇 **Grape:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy
- 🥭 **Mango:** Anthracnose, Rust Leaf Disease, Healthy Leaf
- 🥔 **Potato:** Early Blight, Late Blight, Healthy
- 🌾 **Rice:** Brown Spot, Hispa, Leaf Blast, Healthy
- 🌹 **Rose:** Sawfly Slug, Rust, Healthy Leaf
- 🍅 **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Healthy

---

## 🌍 Multilingual Support

AgriLens currently supports:
- 🇺🇸 **English** (`en`)
- 🇮🇳 **Telugu** (`te`)

The multilingual engine translates UI labels, plant disease names, fertilizer guidelines, and treatment recommendations to help farmers in regional communities.

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Poorna2635/AgriLens.git
cd AgriLens
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
👉 `http://127.0.0.1:5000`

---

## 📈 Future Enhancements

- 📱 Mobile Application (Android / iOS native app)
- 🗣️ Additional Regional Language Support (Hindi, Tamil, Kannada)
- 📷 Live Real-time Camera Edge Detection
- 🌤️ Weather-based Disease Outbreak Risk Prediction
- 🔌 Agricultural IoT Sensor Integration for Soil Moisture & pH Monitoring

---

## 👨‍💻 Author

**G. Poorna**  
- GitHub: [@Poorna2635](https://github.com/Poorna2635)  
- Live Application: [AgriLens Production App](https://web-production-fd0d4.up.railway.app)
