"""Minimal structured-ish logging setup."""

from __future__ import annotations

import logging
import os


def setup_logging() -> logging.Logger:
    level = os.environ.get("VISION_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger("vision")
