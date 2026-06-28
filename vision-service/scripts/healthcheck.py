#!/usr/bin/env python3
"""Container HEALTHCHECK probe.

Queries the service's own /health endpoint over loopback and maps it to a Docker
health status:

  exit 0 (healthy)    — HTTP 200 and status == "ok"
  exit 1 (unhealthy)  — unreachable, non-200, malformed body, or status != "ok"
                        (i.e. "degraded": a camera/RTSP stall or a pipeline
                        restart, surfaced by app/api/rest.py:/health)

Docker won't restart an `unhealthy` container on its own (the compose files use
`restart: unless-stopped`, which acts on exit, not health), so flagging
"degraded" here is purely diagnostic — it shows up in `docker ps` / `docker
inspect` and any orchestration that gates on health, without risking a restart
loop on a single wedged camera. Stdlib only so it runs in the cpu, jetson and
deepstream images without extra deps.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

PORT = os.environ.get("VISION_PORT", "7000")
URL = f"http://127.0.0.1:{PORT}/health"
TIMEOUT_S = 5.0


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT_S) as resp:
            if resp.status != 200:
                print(f"healthcheck: {URL} -> HTTP {resp.status}", file=sys.stderr)
                return 1
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # unreachable / timeout / bad JSON
        print(f"healthcheck: {URL} unreachable: {exc}", file=sys.stderr)
        return 1

    status = body.get("status")
    if status == "ok":
        return 0

    # Degraded: include the camera/restart detail so `docker inspect` shows why.
    errors = body.get("camera_errors") or {}
    restarts = body.get("pipeline_restarts") or 0
    print(
        f"healthcheck: status={status} restarts={restarts} camera_errors={errors}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
