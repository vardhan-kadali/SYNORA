"""
SYNORA — API Request & Response Schemas
Pydantic schemas for web API serialization, validation, and contracts.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Service identity returned on GET /."""
    name: str = "SYNORA"
    tagline: str = "Detecting the Synthetic. Protecting the Real."
    status: str = "running"
    api_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health status returned on GET /health."""
    status: str = Field(..., description="Service health state: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the inference engine is loaded and verified")
    model_name: str = Field(..., description="Name of the deployable detection model")
    feature_dimensions: int = Field(..., description="Input feature dimensionality (40)")
    version: str = Field("1.0.0", description="SYNORA package version")


class AudioMetadataSchema(BaseModel):
    """Metadata describing the decoded and processed audio."""
    duration_seconds: float = Field(..., description="Duration of decoded audio in seconds")
    sample_rate: int = Field(16000, description="Processing sample rate in Hz (16,000 Hz)")
    channels: int = Field(1, description="Native channel count prior to mono downmixing")


class ModelInfoSchema(BaseModel):
    """Identity and dimensionality of the active ML pipeline."""
    name: str = Field("SYNORA A-Combined", description="Human-readable model name")
    feature_dimensions: int = Field(40, description="Number of acoustic features extracted")


class ProbabilityDistributionSchema(BaseModel):
    """Calibrated probability scores for Fake and Real hypotheses."""
    fake: float = Field(..., ge=0.0, le=1.0, description="Calibrated P(fake | x)")
    real: float = Field(..., ge=0.0, le=1.0, description="Calibrated P(real | x) = 1 - P(fake)")


class AnalysisResponse(BaseModel):
    """
    Structured successful response returned on POST /api/v1/analyze.
    Enforces humility mandates: probabilistic assessment without overconfident categories.
    """
    success: bool = Field(True, description="Indicates request was successfully analyzed")
    prediction: str = Field(..., description="Binary classification: 'Fake' or 'Real'")
    probabilities: ProbabilityDistributionSchema = Field(..., description="Calibrated probabilities")
    decision_threshold: float = Field(0.50, description="Fixed operating threshold (0.50)")
    audio: AudioMetadataSchema = Field(..., description="Extracted audio file metadata")
    model: ModelInfoSchema = Field(..., description="Underlying model specification")
    disclaimer: str = Field(..., description="Mandatory scientific humility limitation statement")


class ErrorResponse(BaseModel):
    """Structured error payload for failed requests."""
    success: bool = False
    error: str = Field(..., description="Machine-readable error code")
    detail: str = Field(..., description="Human-readable safe explanation of the error")

