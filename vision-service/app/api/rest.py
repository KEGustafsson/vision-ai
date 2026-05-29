"""REST control + introspection endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, Response

from ..schemas import Backend, ControlRequest, HealthResponse
from .overlay import encode_jpeg

router = APIRouter()


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    p = _pipeline(request)
    backend = Backend(p.settings.detector.backend)
    errors = p.camera_errors()
    return HealthResponse(
        status="degraded" if errors else "ok",
        mode=p.settings.mode,
        backend=backend,
        cameras=[c.name for c in p.settings.cameras],
        uptime_s=time.time() - p.started_at,
        active_camera=p.active_camera,
        camera_errors=errors,
    )


@router.get("/config")
def get_config(request: Request):
    cfg = _pipeline(request).settings.model_dump()
    # RTSP URLs often embed credentials (rtsp://user:pass@host) — never expose them.
    for cam in cfg.get("cameras", []):
        if cam.get("url"):
            cam["url"] = "***redacted***"
    return cfg


@router.get("/cameras")
def cameras(request: Request):
    return [c.name for c in _pipeline(request).settings.cameras]


@router.get("/events/recent")
def recent_events(request: Request, n: int = 20):
    return _pipeline(request).events.recent(n)


@router.post("/control")
def control(request: Request, body: ControlRequest):
    p = _pipeline(request)
    applied = {}
    if body.confidence is not None:
        p.set_confidence(body.confidence)
        applied["confidence"] = body.confidence
    if body.active_camera is not None:
        if body.active_camera not in p.workers:
            raise HTTPException(status_code=404, detail=f"unknown camera {body.active_camera}")
        p.set_active_camera(body.active_camera)
        applied["active_camera"] = p.active_camera
    if body.mode_hint is not None:
        p.mode_hint = body.mode_hint
        applied["mode_hint"] = body.mode_hint
    return {"applied": applied}


@router.get("/snapshot/{camera}")
def snapshot(request: Request, camera: str):
    p = _pipeline(request)
    jpeg = p.frames.get(camera)
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"no frame for camera {camera}")
    return Response(content=jpeg, media_type="image/jpeg")
