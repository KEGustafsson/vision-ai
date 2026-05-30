"""Vision service entrypoint: build the FastAPI app, wire the pipeline, run uvicorn."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import mjpeg, rest, ws
from .config import Settings, load_settings
from .pipeline import Pipeline
from .ptz import PtzManager
from .util import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    log = setup_logging()
    settings = settings or load_settings()
    log.info("starting vision service in mode=%s backend=%s",
             settings.mode, settings.detector.backend)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.events.bind_loop(asyncio.get_running_loop())
        pipeline.start()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                pipeline.stop()

    app = FastAPI(title="Marine Vision-AI Service", version="1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins or ["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    pipeline = Pipeline(settings, log)
    app.state.pipeline = pipeline
    app.state.ptz = PtzManager(settings)

    app.include_router(rest.router)
    app.include_router(ws.router)
    app.include_router(mjpeg.router)
    return app


app = create_app()


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    settings = load_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":  # pragma: no cover
    main()
