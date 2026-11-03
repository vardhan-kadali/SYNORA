"""
SYNORA — API Routes
Endpoint routers for service identity, health checks, and audio analysis.
"""

from pathlib import Path
from typing import Set

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse

from synora import __version__
from synora.config import (
    MODEL_NAME,
    FEATURE_DIMENSION,
    MAX_UPLOAD_BYTES,
    MAX_AUDIO_DURATION_SECONDS,
    ALLOWED_AUDIO_EXTENSIONS,
)
from synora.dsp import decode_audio, SilentAudioError, InvalidAudioError
from synora.api.schemas import (
    RootResponse,
    HealthResponse,
    AnalysisResponse,
    AudioMetadataSchema,
    ModelInfoSchema,
    ProbabilityDistributionSchema,
    ErrorResponse,
)

router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["Service"])
async def root() -> RootResponse:
    """Service identity endpoint."""
    return RootResponse(
        name="SYNORA",
        tagline="Detecting the Synthetic. Protecting the Real.",
        status="running",
        api_version=__version__,
    )


@router.get("/health", response_model=HealthResponse, tags=["Service"])
async def health(request: Request) -> JSONResponse:
    """
    Health check endpoint reporting inference engine readiness.
    Does not expose sensitive system paths or artifact contents.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_name=MODEL_NAME,
                feature_dimensions=FEATURE_DIMENSION,
                version=__version__,
            ).model_dump(),
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=MODEL_NAME,
            feature_dimensions=FEATURE_DIMENSION,
            version=__version__,
        ).model_dump(),
    )


@router.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty, corrupted, or silent audio input"},
        413: {"model": ErrorResponse, "description": "Upload size (>25 MB) or audio duration (>60s) limit exceeded"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        500: {"model": ErrorResponse, "description": "Internal server or inference failure"},
        503: {"model": ErrorResponse, "description": "Inference engine unavailable"},
    },
    tags=["Inference"],
)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze (WAV, FLAC, OGG, MP3; max 25 MB, max 60s)"),
) -> AnalysisResponse:
    """
    Analyzes an uploaded audio recording for synthetic / AI-generated speech.
    
    1. Validates upload size (<= 25 MB).
    2. Validates format compatibility via strict allowlist.
    3. Validates decoded audio duration (<= 60 seconds).
    4. Streams bytes directly to in-memory DSP feature extraction (never saved permanently).
    5. Cleans up file resources reliably in all execution paths.
    6. Returns calibrated probabilities, binary prediction, and scientific disclaimer.
    """
    try:
        # 1. Verify engine initialization
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service is currently unavailable or model failed to initialize.",
            )

        # 2. Check filename and strict extension allowlist
        filename = file.filename or ""
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing or empty filename in upload request.",
            )

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension '{ext}'. Supported audio formats are WAV, FLAC, OGG, and MP3.",
            )

        # 3. Read bytes with strict size enforcement (up to MAX_UPLOAD_BYTES + 1)
        # Note: Pre-multipart request-level size limit is enforced upstream by
        # RequestSizeLimitMiddleware (pure ASGI). This step enforces the exact
        # 25 MB payload limit specifically on the extracted audio bytes.
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty upload: uploaded audio file contains 0 bytes.",
            )

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload size exceeds maximum permitted limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        # 4. Decode audio in-memory to validate duration and signal integrity
        try:
            _, audio_meta = decode_audio(content)
        except SilentAudioError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Silent audio detected: zero frames exceed speech activity threshold.",
            ) from e
        except InvalidAudioError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or corrupted audio stream: {e}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode audio file: {e}",
            ) from e

        # 6. Enforce maximum duration limit
        duration = float(audio_meta.get("duration_seconds", 0.0))
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Audio duration ({duration:.2f} s) exceeds maximum permitted limit "
                    f"of {int(MAX_AUDIO_DURATION_SECONDS)} seconds."
                ),
            )

        # 7. Execute inference using frozen engine
        try:
            result = engine.predict(content)
        except SilentAudioError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Silent audio detected: zero frames exceed speech activity threshold.",
            ) from e
        except InvalidAudioError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or corrupted audio stream: {e}",
            ) from e
        except Exception as e:
            # Safe internal failure without leaking stack trace or local paths
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during acoustic inference.",
            ) from e

        # 8. Structured contract response
        return AnalysisResponse(
            success=True,
            prediction=result.prediction,
            probabilities=ProbabilityDistributionSchema(
                fake=result.fake_probability,
                real=result.real_probability,
            ),
            decision_threshold=result.decision_threshold,
            audio=AudioMetadataSchema(
                duration_seconds=result.audio_metadata["duration_seconds"],
                sample_rate=result.audio_metadata["processed_sample_rate"],
                channels=result.audio_metadata["channels"],
            ),
            model=ModelInfoSchema(
                name="SYNORA A-Combined",
                feature_dimensions=result.feature_dimension,
            ),
            disclaimer=result.disclaimer,
        )
    finally:
        await file.close()
