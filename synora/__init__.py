"""
SYNORA — Detecting the Synthetic. Protecting the Real.
Audio Deepfake Detection for Indian Languages & Telephony Channels.
"""

from synora.config import (
    MODEL_NAME,
    FEATURE_DIMENSION,
    FROZEN_TEMPERATURE,
    FIXED_DECISION_THRESHOLD,
    LABEL_FAKE,
    LABEL_REAL,
    DISCLAIMER_TEXT,
)
from synora.dsp import (
    decode_audio,
    extract_features,
    SilentAudioError,
    InvalidAudioError,
)
from synora.inference import (
    SynoraInferenceEngine,
    InferenceResult,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "SynoraInferenceEngine",
    "InferenceResult",
    "extract_features",
    "decode_audio",
    "SilentAudioError",
    "InvalidAudioError",
    "MODEL_NAME",
    "FEATURE_DIMENSION",
    "FROZEN_TEMPERATURE",
    "FIXED_DECISION_THRESHOLD",
    "LABEL_FAKE",
    "LABEL_REAL",
    "DISCLAIMER_TEXT",
]

