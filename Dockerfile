# Base image with Python 3.10
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install PyTorch FIRST (separate layer so it caches)
RUN pip install --no-cache-dir \
    torch==2.3.0 torchaudio==2.3.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --timeout 300

# Install heavy TTS deps separately (another cached layer)
RUN pip install --no-cache-dir \
    TTS==0.22.0 \
    --timeout 300

# Install gradio and huggingface
RUN pip install --no-cache-dir \
    gradio==4.44.0 \
    huggingface_hub==1.4.0 \
    datasets==2.20.0 \
    soundfile==0.13.1 \
    --timeout 300

# Copy and install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --timeout 300 \
    --ignore-installed

# Copy app code
COPY app.py .

# Create cache directories
RUN mkdir -p /workspace/kinyarwanda-tts-model

# Expose port
EXPOSE 7861

# Run
CMD ["python", "app.py"]
