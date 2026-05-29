# Production image for the NVIDIA Jetson Orin Nano (JetPack 6.x).
#
# The Ultralytics Jetson base ships CUDA, cuDNN, TensorRT, a GStreamer-enabled
# OpenCV and torch/torchvision built for aarch64 — do NOT reinstall those.
#
# Build ON the Jetson:
#   docker build -t vision-ai:jetson .
ARG BASE=ultralytics/ultralytics:latest-jetson-jetpack6
FROM ${BASE}

WORKDIR /opt/vision-service

# Only the extras not already in the base image.
COPY requirements-jetson.txt .
RUN pip install --no-cache-dir -r requirements-jetson.txt

COPY app ./app
COPY config ./config
COPY scripts ./scripts
COPY models ./models

ENV VISION_MODE=jetson \
    VISION_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
