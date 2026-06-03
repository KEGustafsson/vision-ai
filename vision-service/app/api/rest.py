"""REST control + introspection endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, Response

from ..detector.classmap import MODEL_LABELS
from ..schemas import Backend, ControlRequest, HealthResponse, PtzRequest

router = APIRouter()


def _pipeline(request: Request):
    return request.app.state.pipeline


def _ptz(request: Request):
    return request.app.state.ptz


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    p = _pipeline(request)
    backend = Backend(p.settings.detector.backend)
    errors = p.camera_errors()
    model = p.settings.detector.model
    return HealthResponse(
        status="degraded" if errors else "ok",
        mode=p.settings.mode,
        backend=backend,
        cameras=[c.name for c in p.settings.cameras],
        uptime_s=time.time() - p.started_at,
        active_camera=p.active_camera,
        camera_errors=errors,
        detection_enabled=p.enabled,
        max_targets=p.max_targets(),
        labels=p.labels(),
        model=model,
        model_labels=MODEL_LABELS.get(model, []),
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
    if body.max_targets is not None:
        p.set_max_targets(body.max_targets)
        applied["max_targets"] = p.max_targets()
    if body.active_camera is not None:
        if body.active_camera not in p.workers:
            raise HTTPException(status_code=404, detail=f"unknown camera {body.active_camera}")
        p.set_active_camera(body.active_camera)
        applied["active_camera"] = p.active_camera
    if body.mode_hint is not None:
        p.mode_hint = body.mode_hint
        applied["mode_hint"] = body.mode_hint
    if body.labels is not None:
        p.set_labels(body.labels)
        applied["labels"] = p.labels()
    if body.enabled is not None:
        p.set_enabled(body.enabled)
        applied["enabled"] = body.enabled
    return {"applied": applied}


@router.get("/ptz")
def ptz_cameras(request: Request):
    """Names of cameras that have ONVIF PTZ control enabled."""
    return {"cameras": _ptz(request).ptz_cameras()}


@router.post("/ptz/{camera}")
def ptz(request: Request, camera: str, body: PtzRequest):
    mgr = _ptz(request)
    if camera not in mgr.ptz_cameras():
        raise HTTPException(status_code=404, detail=f"camera {camera} has no PTZ")
    try:
        if body.action == "stop":
            mgr.stop(camera)
        elif body.action == "home":
            mgr.home(camera)
        elif body.action == "move":
            mgr.move(camera, body.pan, body.tilt, body.zoom)
        else:
            raise HTTPException(status_code=400, detail=f"unknown action {body.action}")
    except HTTPException:
        raise
    except Exception as e:  # ONVIF/network failure -> 502 (camera unreachable)
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True, "action": body.action}


@router.get("/snapshot/{camera}")
def snapshot(request: Request, camera: str):
    p = _pipeline(request)
    jpeg = p.frames.get(camera)
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"no frame for camera {camera}")
    return Response(content=jpeg, media_type="image/jpeg")
