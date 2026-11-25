"""Logging guard: faces never reach logs. Drops raw bytes and redacts base64 blobs and stored image paths."""

from __future__ import annotations

import logging
import re

_BASE64 = re.compile(r"[A-Za-z0-9+/=]{200,}")
_IMAGE_PATH = re.compile(r"\S*[\\/][0-9a-f-]{36}\.(?:jpg|jpeg|png|webp)", re.IGNORECASE)


class NoFaceDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if any(isinstance(a, (bytes, bytearray, memoryview)) for a in
                   (record.args if isinstance(record.args, tuple) else (record.args,))):
                record.msg, record.args = "[log record dropped: contained raw bytes]", None
                return True
        msg = record.getMessage()
        clean = _IMAGE_PATH.sub("[image-path]", _BASE64.sub("[redacted]", msg))
        if clean != msg:
            record.msg, record.args = clean, None
        return True


def install() -> None:
    f = NoFaceDataFilter()
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "ml", "app"):
        logger = logging.getLogger(name)
        if not any(isinstance(x, NoFaceDataFilter) for x in logger.filters):
            logger.addFilter(f)
    for h in logging.getLogger().handlers:
        if not any(isinstance(x, NoFaceDataFilter) for x in h.filters):
            h.addFilter(f)
