"""REST control + introspection endpoints."""

from __future__ import annotations

import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response

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
    # DeepStream exposes auto-restart telemetry; the CPU/Jetson pipelines don't.
    restart_info = p.restart_info() if hasattr(p, "restart_info") else {}
    restarts = int(restart_info.get("restarts", 0))
    last_error = restart_info.get("last_error")
    # A restart only degrades health while it's recent (the pipeline is actively
    # recovering/flapping). A single restart that has since stayed up shouldn't
    # latch the container unhealthy for its whole life — the cumulative count is
    # still surfaced below as pipeline_restarts for diagnostics.
    restart_degraded = bool(restart_info.get("recent")) if restart_info else restarts > 0
    # Optional NVIDIA OFA diagnostics (DeepStream only). Stale/absent optical
    # flow does NOT degrade health on its own — it is a measurement, not part of
    # the detection path — unless the operator declared it required.
    optical_flow = p.optical_flow_status() if hasattr(p, "optical_flow_status") else {}
    of_degraded = bool(
        optical_flow
        and getattr(p.settings.detector, "optical_flow_required", False)
        and any(not cam.get("active") for cam in optical_flow.values())
    )
    return HealthResponse(
        status="degraded" if (errors or restart_degraded or of_degraded) else "ok",
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
        pipeline_restarts=restarts,
        pipeline_last_error=last_error,
        optical_flow=optical_flow,
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
def recent_events(request: Request, n: int = Query(20, ge=1, le=1000)):
    return _pipeline(request).events.recent(n)


# How long a client_generation stays authoritative. The fence only has to cover
# the seconds around a client restart, so it self-disarms when the client goes
# quiet: a boat computer without an RTC boots at some arbitrary earlier time and
# its plugin would then stamp a LOWER generation than the container remembers
# from before the reboot. Expiring the fence means that client is accepted after
# one idle window instead of being locked out for the life of the container.
# The plugin re-pushes every 5 s, so a live client keeps the fence fresh.
_CONTROL_FENCE_TTL_S = 30.0
_fence_lock = threading.Lock()


def _fence_allows(state, generation: Optional[int]) -> bool:
    """Whether a control request may be applied, updating the fence if so.

    Requests without a generation are always applied (an older client that
    doesn't send one) and never move the fence.
    """
    if generation is None:
        return True
    now = time.monotonic()
    with _fence_lock:
        seen = getattr(state, "control_generation", None)
        seen_at = getattr(state, "control_generation_at", 0.0)
        if seen is not None and now - seen_at <= _CONTROL_FENCE_TTL_S and generation < seen:
            return False
        state.control_generation = generation
        state.control_generation_at = now
        return True


@router.post("/control")
def control(request: Request, body: ControlRequest):
    p = _pipeline(request)
    if not _fence_allows(request.app.state, body.client_generation):
        # A request composed by an older client instance, overtaken by the one
        # currently talking to us. Applying it would silently restore settings
        # the operator has already changed.
        raise HTTPException(
            status_code=409,
            detail=("stale control request: client_generation "
                    f"{body.client_generation} is older than the active client"),
        )
    applied = {}
    if body.confidence is not None:
        p.set_confidence(body.confidence)
        applied["confidence"] = body.confidence
    if body.max_targets is not None:
        p.set_max_targets(body.max_targets)
        applied["max_targets"] = p.max_targets()
    if body.min_target_range_m is not None:
        p.set_min_target_range(body.min_target_range_m)
        applied["min_target_range_m"] = body.min_target_range_m
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


# A snapshot waits at most this long for the next annotated frame. Generous
# against a slow frame (inference + encode at a low target_fps) while still
# answering promptly when a camera is genuinely dead.
_SNAPSHOT_WAIT_S = 2.0


@router.get("/snapshot/{camera}")
async def snapshot(request: Request, camera: str):
    p = _pipeline(request)
    # Reject an unknown camera up front: no worker will ever publish for it, so
    # otherwise the request waits the full timeout only to 404 anyway, and each
    # new name left a demand entry behind (the store keys on the name, and only
    # a real camera's entries are ever replaced).
    if camera not in p.workers:
        raise HTTPException(status_code=404, detail=f"unknown camera {camera}")
    # Mark demand: the pipeline skips annotating and encoding while nobody is
    # watching, so this both asks for a frame and keeps the display path warm
    # for a client that polls snapshots.
    p.frames.note_demand(camera)
    jpeg = p.frames.get(camera)
    if jpeg is None:
        # Nothing stored: either the camera hasn't produced yet or the display
        # path was idle. Wait briefly for the next frame instead of returning a
        # 404 the caller would only have to retry.
        jpeg = await p.frames.wait_for_frame(camera, timeout=_SNAPSHOT_WAIT_S)
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"no frame for camera {camera}")
    return Response(content=jpeg, media_type="image/jpeg")
