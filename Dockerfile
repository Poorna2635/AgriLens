# Step 1: Base Image - Lightweight Python 3.10 Linux Image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set working directory inside the container
WORKDIR /app

# Step 2: Install system dependencies required by OpenCV / Pillow / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy requirements first for Docker layer caching
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application code
COPY . .

# Step 6: Create upload folder if not existing
RUN mkdir -p static/uploads

# Step 7: Expose container port
EXPOSE 5000

# Step 8: Production Start Command using Gunicorn WSGI Server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
