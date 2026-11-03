"""
SYNORA — DSP Package
Exports the thin adapter and custom exceptions.
"""

from synora.dsp.extractor import (
    decode_audio,
    extract_features,
    SilentAudioError,
    InvalidAudioError,
)

__all__ = [
    "decode_audio",
    "extract_features",
    "SilentAudioError",
    "InvalidAudioError",
]

