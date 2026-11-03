"""
SYNORA — DSP Feature Extraction Deployment Adapter
"Detecting the Synthetic. Protecting the Real."

Thin deployment wrapper around the authoritative DSP pipeline in cache_features_v2.py.
Preserves bit-exact 40-dimensional feature ordering, STFT framing, VAD thresholding,
and numerical stabilization without rewriting or duplicating the mathematical pipeline.
"""

import io
import math
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from synora.config import (
    WORKSPACE_ROOT,
    TARGET_SAMPLE_RATE,
    FEATURE_DIMENSION,
    assert_path_safety,
)

# Ensure workspace root is in sys.path to import the authoritative cache_features_v2 module
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cache_features_v2


class SilentAudioError(ValueError):
    """Raised when audio contains zero frames exceeding the VAD threshold."""
    pass


class InvalidAudioError(ValueError):
    """Raised when audio input cannot be decoded or is structurally malformed."""
    pass


def decode_audio(
    audio_input: Union[str, Path, bytes, np.ndarray],
    native_sr: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decodes audio from a file path, raw bytes, or numpy array.
    Validates path safety, downmixes multi-channel audio to mono,
    and resamples to the target 16,000 Hz using rational polyphase filtering.

    Returns:
        audio: 1D np.ndarray of shape (N,) and dtype float32 at 16,000 Hz.
        metadata: Dictionary of audio file metadata (duration, native sample rate, etc.).
    """
    if isinstance(audio_input, (str, Path)):
        file_path = Path(audio_input)
        assert_path_safety(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")
        
        try:
            with sf.SoundFile(str(file_path)) as sfile:
                input_sr = int(sfile.samplerate)
                channels = int(sfile.channels)
                frames = int(sfile.frames)
                duration = float(frames / input_sr) if input_sr > 0 else 0.0
            
            # Delegate directly to authoritative loader
            audio = cache_features_v2.load_audio_mono_16k(str(file_path))
        except PermissionError:
            raise
        except Exception as e:
            raise InvalidAudioError(f"Failed to decode audio file {file_path}: {e}") from e

    elif isinstance(audio_input, bytes):
        if len(audio_input) == 0:
            raise InvalidAudioError("Empty audio byte stream provided.")
        
        try:
            byte_io = io.BytesIO(audio_input)
            audio, input_sr = sf.read(byte_io, dtype="float32")
            channels = audio.shape[1] if audio.ndim > 1 else 1
            duration = float(len(audio) / input_sr) if input_sr > 0 else 0.0
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            
            if input_sr != TARGET_SAMPLE_RATE:
                gcd = math.gcd(input_sr, TARGET_SAMPLE_RATE)
                up = TARGET_SAMPLE_RATE // gcd
                down = input_sr // gcd
                audio = resample_poly(audio, up, down).astype(np.float32)
        except Exception as e:
            raise InvalidAudioError(f"Failed to decode audio bytes: {e}") from e

    elif isinstance(audio_input, np.ndarray):
        audio = np.asarray(audio_input, dtype=np.float32)
        if audio.ndim > 1:
            channels = audio.shape[1]
            audio = np.mean(audio, axis=-1)
        else:
            channels = 1
            
        input_sr = native_sr or TARGET_SAMPLE_RATE
        duration = float(len(audio) / input_sr) if input_sr > 0 else 0.0
        
        if input_sr != TARGET_SAMPLE_RATE:
            gcd = math.gcd(input_sr, TARGET_SAMPLE_RATE)
            up = TARGET_SAMPLE_RATE // gcd
            down = input_sr // gcd
            audio = resample_poly(audio, up, down).astype(np.float32)
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio_input)}. "
            "Expected str, Path, bytes, or np.ndarray."
        )

    metadata = {
        "duration_seconds": round(duration, 4),
        "native_sample_rate": input_sr,
        "processed_sample_rate": TARGET_SAMPLE_RATE,
        "channels": channels,
        "num_samples_16k": len(audio),
    }
    return audio.astype(np.float32), metadata


def extract_features(
    audio_input: Union[str, Path, bytes, np.ndarray],
    native_sr: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts the authoritative 40-dimensional A-Combined feature vector.
    Reuses cache_features_v2.extract_candidate_a_combined().

    Returns:
        features: 1D np.ndarray of shape (40,), dtype float32, strictly finite.
        metadata: Audio metadata dictionary.
    """
    audio, metadata = decode_audio(audio_input, native_sr=native_sr)
    
    if len(audio) == 0:
        raise InvalidAudioError("Decoded audio array is empty.")

    try:
        features = cache_features_v2.extract_candidate_a_combined(
            audio, sr=TARGET_SAMPLE_RATE
        )
    except ValueError as e:
        err_msg = str(e)
        if "Silent audio detected" in err_msg or "zero frames exceed" in err_msg:
            raise SilentAudioError(
                "Silent audio detected: zero frames exceed the speech activity threshold (1e-4 RMS)."
            ) from e
        raise

    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Feature dimension mismatch: expected ({FEATURE_DIMENSION},), got {features.shape}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite values (NaN or Inf) detected in extracted feature vector.")

    return features.astype(np.float32), metadata

