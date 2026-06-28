# Production image for the NVIDIA Jetson Orin Nano (JetPack 6.x).
#
# The Ultralytics Jetson base ships CUDA, cuDNN, TensorRT, a GStreamer-enabled
# OpenCV and torch/torchvision built for aarch64 — do NOT reinstall those.
#
# Build ON the Jetson:
#   docker build -t vision-ai:jetson .
#
# RELEASE PINNING: the default tag below floats (`latest-jetson-jetpack6`), so a
# rebuilt image can silently pick up a new base. For a reproducible safety release
# pin to a digest resolved ON the Jetson (the tag is multi-arch; resolve the
# aarch64 digest on the target box), e.g.:
#   docker pull ultralytics/ultralytics:latest-jetson-jetpack6
#   docker inspect --format='{{index .RepoDigests 0}}' ultralytics/ultralytics:latest-jetson-jetpack6
#   docker build --build-arg BASE=ultralytics/ultralytics@sha256:<digest> -t vision-ai:jetson .
# (Override via --build-arg BASE=... or set it in docker-compose.jetson.yml.)
ARG BASE=ultralytics/ultralytics:latest-jetson-jetpack6
FROM ${BASE}

WORKDIR /opt/vision-service

# Only the extras not already in the base image. constraints.txt pins the web
# stack (FastAPI/Starlette/httpx) to the verified versions — pure-Python, so the
# same pins are valid on aarch64.
COPY requirements-jetson.txt constraints.txt ./
RUN pip install --no-cache-dir -c constraints.txt -r requirements-jetson.txt

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
# Liveness: probe /health and report `unhealthy` on unreachable or degraded
# (camera/RTSP stall). The start-period is generous because the first start may
# build the TensorRT engine from the .pt/.onnx, which takes a while on the Orin
# Nano; failing probes during start-period don't count against retries. See
# scripts/healthcheck.py.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python3", "/opt/vision-service/scripts/healthcheck.py"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
