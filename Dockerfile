FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps for Pillow / OpenCV-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-turbo8 libpng16-16 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ src/
COPY configs/ configs/

# Data and outputs are expected to be mounted at runtime
VOLUME ["/app/data", "/app/outputs"]

# Default: run training
ENTRYPOINT ["python", "-m", "src.train", "--config", "configs/default.yaml"]
