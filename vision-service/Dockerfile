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

# The Ultralytics Jetson base currently includes a pip opencv-python-headless
# wheel in /usr/local, which shadows NVIDIA's apt OpenCV in /usr/lib. The apt
# package is the one built with GStreamer/NVIDIA multimedia support.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    && python3 - <<'PY'
import cv2

info = cv2.getBuildInformation()
if "GStreamer:                   YES" not in info and "GStreamer: YES" not in info:
    raise SystemExit(f"OpenCV imported from {cv2.__file__} without GStreamer support")
print(f"OpenCV {cv2.__version__} imported from {cv2.__file__}")
PY

# PyGObject + GStreamer introspection: only needed for the optional hardware
# JPEG encoder (server.hw_jpeg -> nvjpegenc). The app falls back to CPU encoding
# when this is absent, so it's safe to drop if hw_jpeg is never enabled. Installs
# into the system python3 the app runs under (/usr/bin/python3).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi gir1.2-glib-2.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY config ./config
COPY scripts ./scripts
COPY models ./models

ENV VISION_MODE=jetson \
    VISION_PORT=7000 \
    PYTHONUNBUFFERED=1

# Run as a non-root user to reduce blast radius (mirrors Dockerfile.cpu). The
# TensorRT engine is pre-built on the host and bind-mounted read-only at runtime
# (see docker-compose.jetson.yml), so the app only needs to READ it. NB: when
# this image is reused as the deepstream export base, that stage re-escalates to
# root for its build steps (see Dockerfile.deepstream).
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /opt/vision-service
USER appuser

EXPOSE 7000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
