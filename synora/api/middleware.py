"""
SYNORA — Pre-Multipart Request Size Limit ASGI Middleware
Protects the application by enforcing request body limits before
FastAPI's router and Starlette's multipart parser spool uploads to disk.
"""

from typing import Optional
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from synora.config import MAX_REQUEST_BODY_BYTES


class RequestBodyLimitExceeded(StarletteHTTPException):
    """Exception raised when an incoming request body exceeds the configured limit."""

    def __init__(self, detail: str = "Upload size exceeds maximum permitted limit of 25 MB.") -> None:
        super().__init__(status_code=413, detail=detail)


def _extract_content_length(scope: Scope) -> Optional[int]:
    """Extracts and parses the Content-Length header from an ASGI HTTP scope."""
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() == b"content-length":
            try:
                length = int(raw_value.decode("latin-1").strip())
                return length if length >= 0 else None
            except (ValueError, UnicodeDecodeError):
                return None
    return None


def _create_413_json_response(max_audio_mb: int = 25) -> JSONResponse:
    """Creates a standardized SYNORA HTTP 413 JSON response."""
    return JSONResponse(
        status_code=413,
        content={
            "success": False,
            "error": "HTTP_413",
            "detail": f"Upload size exceeds maximum permitted limit of {max_audio_mb} MB.",
        },
    )


class RequestSizeLimitMiddleware:
    """
    Pure ASGI middleware for pre-multipart request size protection.

    1. Executes before FastAPI's router, dependency injection, and multipart form parser.
    2. Inspects declared Content-Length: if it exceeds `max_body_size`, immediately terminates
       the request with an HTTP 413 JSON response without invoking downstream routing or spooling.
    3. For chunked or streaming requests (where Content-Length is missing or omitted), wraps
       the ASGI `receive` callable and counts cumulative `http.request` body bytes.
    4. Immediately aborts streaming and returns HTTP 413 when cumulative bytes exceed `max_body_size`,
       halting disk spooling and network consumption.
    5. Correctly forwards non-HTTP scopes (e.g. lifespan, websocket) and `http.disconnect` events.
    6. Does not buffer the request body in memory.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int = MAX_REQUEST_BODY_BYTES,
        max_audio_mb: int = 25,
    ) -> None:
        self.app = app
        self.max_body_size = max_body_size
        self.max_audio_mb = max_audio_mb

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Pass non-HTTP scopes (lifespan, websocket) through directly
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. Early Content-Length header check (Pre-Multipart Guard)
        content_length = _extract_content_length(scope)
        if content_length is not None and content_length > self.max_body_size:
            response = _create_413_json_response(self.max_audio_mb)
            await response(scope, receive, send)
            return

        # 2. Wrapped streaming receive for chunked/streaming requests or undeclared length
        cumulative_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal cumulative_bytes
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                cumulative_bytes += len(chunk)
                if cumulative_bytes > self.max_body_size:
                    raise RequestBodyLimitExceeded(
                        detail=f"Upload size exceeds maximum permitted limit of {self.max_audio_mb} MB."
                    )
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyLimitExceeded as exc:
            if not response_started:
                response = _create_413_json_response(self.max_audio_mb)
                await response(scope, receive, send)
            else:
                raise

