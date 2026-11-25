"""Keep uploads in memory: cap request bodies and raise Starlette's spool threshold above that cap,
so an unconsented photo is never written to a temporary file."""

from __future__ import annotations

import json

from starlette.formparsers import MultiPartParser

FORM_OVERHEAD = 64 * 1024


class _TooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.limit = max_bytes + FORM_OVERHEAD
        MultiPartParser.spool_max_size = self.limit + 1
        MultiPartParser.max_part_size = self.limit + 1

    async def _reject(self, send):
        body = json.dumps({"detail": f"Ukuran foto maksimal {(self.limit - FORM_OVERHEAD) // (1024 * 1024)} MB."}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            return await self._reject(send)
        seen = 0
        started = False

        async def limited_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge
            return msg

        async def tracked_send(msg):
            nonlocal started
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _TooLarge:
            if not started:
                await self._reject(send)
