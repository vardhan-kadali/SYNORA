"""
SYNORA — Deployment Configuration
"Detecting the Synthetic. Protecting the Real."

Centralized configuration for frozen model artifacts, cryptographic hashes,
DSP constants, decision policy, and security path guards.
"""

import os
from pathlib import Path

# Workspace Root (D:\SYNORA)
PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent

# ==============================================================================
# 1. FROZEN MODEL ARTIFACT PATHS
# ==============================================================================
# Configurable model directory: supports optional SYNORA_MODEL_DIR env var for deployment,
# defaulting to the authoritative local development directory.
_env_model_dir = os.getenv("SYNORA_MODEL_DIR")
if _env_model_dir and _env_model_dir.strip():
    FINAL_MODEL_DIR = Path(_env_model_dir.strip()).resolve()
else:
    FINAL_MODEL_DIR = WORKSPACE_ROOT / "experiments" / "phase_3_2_4_2B_final_model"

SCALER_PATH = FINAL_MODEL_DIR / "final_scaler.joblib"
CLASSIFIER_PATH = FINAL_MODEL_DIR / "final_classifier.joblib"
TEMPERATURE_PATH = FINAL_MODEL_DIR / "temperature_scaling.json"
METADATA_PATH = FINAL_MODEL_DIR / "final_model_metadata.json"

# ==============================================================================
# 2. AUTHORITATIVE ARTIFACT SHA256 CHECKSUMS
# ==============================================================================
EXPECTED_SCALER_SHA256 = "d2b6030b2e753c7a8e345281d5cd6c9c260dcbadc5e2c04df8acf3523bf143c3"
EXPECTED_CLASSIFIER_SHA256 = "052ccc2e26a2b8a9bf2cb56723f6258c32d21faa26d43a7f5054fca4c6d7f67b"
EXPECTED_TEMPERATURE_SHA256 = "df54e415cca030b1f3f2d476959e0afddc61f483b53856c3e75c7b8e856beb33"
EXPECTED_METADATA_SHA256 = "dfe5548ecadd9467d039519620902a6a79e75e30bf9c9ac2fc40229b00d1ea9f"

EXPECTED_CHECKSUMS = {
    SCALER_PATH: EXPECTED_SCALER_SHA256,
    CLASSIFIER_PATH: EXPECTED_CLASSIFIER_SHA256,
    TEMPERATURE_PATH: EXPECTED_TEMPERATURE_SHA256,
    METADATA_PATH: EXPECTED_METADATA_SHA256,
}

# ==============================================================================
# 3. DSP & INFERENCE CONSTANTS
# ==============================================================================
TARGET_SAMPLE_RATE: int = 16000
FEATURE_DIMENSION: int = 40
CANDIDATE_ID: str = "candidate_a_combined"
MODEL_NAME: str = "SYNORA-A-Combined-LogisticRegression"

# Decision Policy (Frozen)
FROZEN_TEMPERATURE: float = 1.1229901304
FIXED_DECISION_THRESHOLD: float = 0.50

# Class Mapping: Positive class 1 is Fake (Synthetic), Class 0 is Real (Genuine)
LABEL_FAKE: str = "Fake"
LABEL_REAL: str = "Real"

# Humility Disclaimer (Mandatory on every inference result)
DISCLAIMER_TEXT: str = (
    "These probabilities are statistical estimates based on acoustic features. "
    "They do not establish definitive proof that an audio recording is genuine or AI-generated."
)

# ==============================================================================
# 4. API & UPLOAD SECURITY CONSTANTS
# ==============================================================================
# Audio file payload limit (25 MiB)
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

# Documented multipart framing overhead allowance (64 KiB)
# Accommodates MIME boundary delimiters, Content-Disposition/Content-Type headers,
# form field metadata, and browser headers without falsely rejecting near-limit files.
MULTIPART_OVERHEAD_ALLOWANCE: int = 64 * 1024

# Maximum permitted HTTP request entity-body limit (25 MiB + 64 KiB = 26,279,936 bytes)
MAX_REQUEST_BODY_BYTES: int = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_ALLOWANCE

# Decoded audio signal limits
MAX_AUDIO_DURATION_SECONDS: float = 60.0

# Strict audio format allowlist (supported by installed soundfile decoder)
ALLOWED_AUDIO_EXTENSIONS: frozenset = frozenset({".wav", ".flac", ".ogg", ".mp3"})

# ==============================================================================
# 5. PATH GUARDS & DATA QUARANTINE ASSERTIONS
# ==============================================================================
FORBIDDEN_PATHS = frozenset({
    "benchmark/test",
    "benchmark\\test",
    "test_features.npz",
    "test_features_v2.npz",
})

def assert_path_safety(path: str | Path) -> None:
    """
    Mandatory safety guard: ensures no script accesses the locked Test partition
    or creates/accesses test feature caches.
    """
    path_str = str(path).replace("/", "\\").lower()
    for forbidden in FORBIDDEN_PATHS:
        forbidden_norm = forbidden.replace("/", "\\").lower()
        if forbidden_norm in path_str:
            raise PermissionError(
                f"CRITICAL SAFETY VIOLATION: Access to locked path attempted: {path}"
            )

