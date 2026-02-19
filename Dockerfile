# Step 1: Base image with Python 3.10
FROM python:3.10-slim

# Step 2: Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Install CUDA + PyTorch (GPU support)
# Adjust torch/torchaudio versions to match Kaggle GPU drivers
RUN pip install torch==2.2.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Step 4: Install TTS (Coqui)
RUN pip install TTS

# Step 5: Set working directory
WORKDIR /workspace

# Step 6: Copy your app code into the container
COPY app.py .

# Step 7: Expose the port your app runs on
EXPOSE 7861

# Step 8: Default command to run your app
CMD ["python", "app.py"]
