"""
SYNORA — FastAPI Application Factory
Initializes lifespan lifecycle, CORS configuration, and route registrations.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from synora import __version__
from synora.config import MAX_REQUEST_BODY_BYTES
from synora.inference.engine import SynoraInferenceEngine
from synora.api.routes import router
from synora.api.middleware import RequestSizeLimitMiddleware

logger = logging.getLogger("synora.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager:
    Initializes the frozen inference engine on startup with SHA256 checksum verification.
    Reuses the engine instance across all requests.
    """
    logger.info("Initializing SYNORA inference engine and verifying model artifacts...")
    try:
        engine = SynoraInferenceEngine(verify_checksums=True)
        app.state.engine = engine
        logger.info("SYNORA inference engine loaded and cryptographically verified successfully.")
    except Exception as e:
        logger.error(f"FATAL: SYNORA inference engine failed to initialize: {e}")
        app.state.engine = None
        app.state.engine_error = str(e)

    yield

    logger.info("SYNORA service shutting down.")


def parse_cors_origins() -> List[str]:
    """
    Parses CORS origins from the SYNORA_CORS_ORIGINS environment variable.
    Defaults to standard local development origins. Never defaults to '*'.
    """
    raw_origins = os.getenv(
        "SYNORA_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="SYNORA Detection API",
        description="Audio deepfake and synthetic speech detection for Indian languages and telephony channels.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Pre-Multipart Request-Size Guard (Pure ASGI - executes before router & multipart parser)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_size=MAX_REQUEST_BODY_BYTES,
        max_audio_mb=25,
    )

    # 2. CORS Configuration (Outer wrapper - handles OPTIONS preflight and adds CORS headers)
    origins = parse_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Standardized JSON error response handler (catches both Starlette and FastAPI HTTPExceptions)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": f"HTTP_{exc.status_code}",
                "detail": exc.detail,
            },
        )

    # Mount routes
    app.include_router(router)

    return app


# Default ASGI application instance
app = create_app()

