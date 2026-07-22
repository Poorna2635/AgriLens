# 🌿 AgriLens - Intelligent Plant Disease Recognition System

AgriLens is a deep-learning-powered Flask web application that classifies 38 different plant disease categories from leaf images using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras. It provides multi-language support (English & Telugu), detailed disease diagnoses, fertilizer recommendations, and actionable treatment tips.

---

## 🛠️ Tech Stack & Model Specs

- **Backend / Framework:** Python 3.10, Flask, Gunicorn
- **Machine Learning:** TensorFlow (CPU), Keras (CNN), NumPy, Pillow
- **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Templates (with optional REST API support)
- **Model Architecture:** Custom CNN trained on 5,000+ leaf images
- **Accuracy:** 92% across 38 crop disease classes
- **Input Size:** 256x256 RGB images

---

## 📁 Directory Structure

```text
AgriLens/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   │   └── .gitkeep
│   └── images/
├── templates/
│   ├── login.html
│   ├── home.html
│   ├── disease-recognition.html
│   └── prediction.html
├── .dockerignore
├── .env.example
├── .gitignore
├── app.py
├── Dockerfile
├── Procfile
├── railway.json
├── requirements.txt
├── runtime.txt
├── Team3model.h5
└── README.md
```

---

## 🚀 Local Development (Without Docker)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AgriLens.git
   cd AgriLens
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

5. Access the app in your browser at `http://localhost:5000`.

---

## 🐳 Docker Deployment & Testing

1. **Build Docker Image:**
   ```bash
   docker build -t agrilens-app .
   ```

2. **Run Container:**
   ```bash
   docker run -p 5000:5000 -e PORT=5000 agrilens-app
   ```

3. **View Logs:**
   ```bash
   docker logs -f <container_id>
   ```

---

## 🌐 Production Deployment Guide

### Deploying on Railway.app
1. Push your project to GitHub (ensure `Team3model.h5` is pushed via Git LFS if >100MB).
2. Connect your GitHub account to [Railway.app](https://railway.app).
3. Create a **New Project** -> **Deploy from GitHub repo**.
4. Railway will auto-detect the `Dockerfile` and build the container.
5. Set environment variable: `PORT=5000` (or allow Railway to assign its own PORT).
6. Click **Generate Domain** under Settings to obtain your public URL!

---

## 📜 License
This project is open-source under the MIT License.
