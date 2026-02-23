# Base image
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install PyTorch (CPU - change to cu118 URL for GPU)
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Cache directory for model files
RUN mkdir -p /workspace/kinyarwanda-tts-model

# Expose Gradio port
EXPOSE 7861

# Run app
CMD ["python", "app.py"]
