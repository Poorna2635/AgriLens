# Step 1: Base Image - Lightweight Python 3.10 Linux Image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set working directory inside the container
WORKDIR /app

# Step 2: Install system dependencies required by OpenCV / Pillow / TensorFlow / Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy requirements first for Docker layer caching
COPY requirements.txt .

# Step 4: Upgrade pip & install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application code (including .git repository for Git LFS resolution)
COPY . .

# Step 6: Pull actual 709MB binary file for Git LFS tracked model (Team3model.h5)
RUN git lfs install && git lfs pull || true

# Step 7: Create upload folder if not existing
RUN mkdir -p static/uploads

# Step 8: Expose container port
EXPOSE 5000

# Step 9: Production Start Command using shell wrapper for $PORT expansion
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120"]
