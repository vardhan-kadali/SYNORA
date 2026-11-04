"""
SYNORA — Inference Engine
"Detecting the Synthetic. Protecting the Real."

Loads frozen StandardScaler and LogisticRegression models with startup SHA256 integrity verification.
Applies frozen temperature scaling (T=1.1229901304) and fixed decision threshold (0.50).
Produces structured, bounded inference outputs with required scientific humility disclaimers.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Dict, Any

import joblib
import numpy as np
from scipy.special import expit

from synora.config import (
    SCALER_PATH,
    CLASSIFIER_PATH,
    TEMPERATURE_PATH,
    METADATA_PATH,
    EXPECTED_CHECKSUMS,
    FEATURE_DIMENSION,
    MODEL_NAME,
    FROZEN_TEMPERATURE,
    FIXED_DECISION_THRESHOLD,
    LABEL_FAKE,
    LABEL_REAL,
    DISCLAIMER_TEXT,
)
from synora.dsp.extractor import extract_features


def compute_file_sha256(filepath: Path | str) -> str:
    """
    Computes SHA256 checksum of a file.

    For JSON text artifacts (.json), CRLF line endings (b'\\r\\n') are normalized
    to LF (b'\\n') in-memory before hashing to guarantee deterministic, identical
    checksums across Windows and Linux environments.

    For binary artifacts (.joblib, etc.), raw bytes are hashed directly without
    alteration. Files on disk are never modified.
    """
    path = Path(filepath)
    sha = hashlib.sha256()

    if path.suffix.lower() == ".json":
        content = path.read_bytes().replace(b"\r\n", b"\n")
        sha.update(content)
    else:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)

    return sha.hexdigest().lower()


@dataclass(frozen=True)
class InferenceResult:
    """Structured inference result returned by SynoraInferenceEngine."""
    prediction: str                      # "Fake" or "Real"
    fake_probability: float              # P(fake | x) in [0.0, 1.0]
    real_probability: float              # P(real | x) in [0.0, 1.0]
    decision_threshold: float            # Fixed 0.50
    raw_decision_score: float            # Raw logit z from classifier
    temperature_applied: float           # Frozen T = 1.1229901304
    model_name: str                      # "SYNORA-A-Combined-LogisticRegression"
    feature_dimension: int               # 40
    audio_metadata: Dict[str, Any]       # Duration, native SR, num samples
    disclaimer: str                      # Mandatory scientific disclaimer

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a clean Python dictionary."""
        return asdict(self)


class SynoraInferenceEngine:
    """
    Production inference engine for the frozen SYNORA AI-generated speech detection model.
    """

    def __init__(self, verify_checksums: bool = True):
        """
        Initializes the inference engine.
        Optionally verifies SHA256 checksums of all four frozen model artifacts before loading.
        """
        if verify_checksums:
            self._verify_artifact_integrity()

        self._load_artifacts()

    def _verify_artifact_integrity(self) -> None:
        """Verifies that all four model artifacts exist and match expected SHA256 checksums."""
        for path, expected_hash in EXPECTED_CHECKSUMS.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"CRITICAL INTEGRITY FAILURE: Missing required model artifact: {path}"
                )
            actual_hash = compute_file_sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"CRITICAL CHECKSUM MISMATCH on {path.name}: "
                    f"Actual={actual_hash} != Expected={expected_hash}"
                )

    def _load_artifacts(self) -> None:
        """Loads and validates frozen scaler, classifier, and temperature parameters."""
        # 1. Load Scaler
        self.scaler = joblib.load(SCALER_PATH)
        if not hasattr(self.scaler, "n_features_in_") or self.scaler.n_features_in_ != FEATURE_DIMENSION:
            raise ValueError(
                f"Loaded scaler has invalid feature dimension: {getattr(self.scaler, 'n_features_in_', None)}, "
                f"expected {FEATURE_DIMENSION}."
            )

        # 2. Load Classifier
        self.classifier = joblib.load(CLASSIFIER_PATH)
        if not hasattr(self.classifier, "n_features_in_") or self.classifier.n_features_in_ != FEATURE_DIMENSION:
            raise ValueError(
                f"Loaded classifier has invalid feature dimension: {getattr(self.classifier, 'n_features_in_', None)}, "
                f"expected {FEATURE_DIMENSION}."
            )
        if not np.array_equal(self.classifier.classes_, [0, 1]):
            raise ValueError(
                f"Loaded classifier classes mismatch: {self.classifier.classes_}, expected [0, 1]."
            )

        # 3. Load Temperature JSON
        with open(TEMPERATURE_PATH, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        
        self.temperature = float(temp_data.get("temperature", 0.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(f"Invalid temperature value loaded: {self.temperature}")
        
        # Verify parity with frozen constant
        if abs(self.temperature - FROZEN_TEMPERATURE) > 1e-6:
            raise ValueError(
                f"Temperature mismatch: loaded {self.temperature} != config {FROZEN_TEMPERATURE}"
            )

    def predict(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        native_sr: int | None = None,
    ) -> InferenceResult:
        """
        Executes end-to-end deepfake detection on input audio:
        1. Decodes and resamples audio to 16 kHz.
        2. Extracts 40-dimensional A-Combined feature vector.
        3. Standardizes features with frozen StandardScaler.
        4. Calculates decision score z via LogisticRegression.decision_function.
        5. Computes temperature-adjusted probability: P(fake) = sigmoid(z / T).
        6. Applies fixed 0.5 decision threshold.

        Returns:
            InferenceResult dataclass.
        """
        # 1. Feature Extraction (includes VAD, framing, and finiteness checks)
        features, metadata = extract_features(audio_input, native_sr=native_sr)

        # 2. Standardization
        scaled_features = self.scaler.transform(features.reshape(1, -1))

        # 3. Raw Logit Decision Score
        raw_logit = float(self.classifier.decision_function(scaled_features)[0])

        # 4. Temperature-Adjusted Probability
        scaled_logit = raw_logit / self.temperature
        fake_prob = float(expit(scaled_logit))

        # Strict bounds enforcement
        fake_prob = max(0.0, min(1.0, fake_prob))
        real_prob = float(max(0.0, min(1.0, 1.0 - fake_prob)))

        # 5. Fixed Decision Threshold Rule
        prediction = LABEL_FAKE if fake_prob >= FIXED_DECISION_THRESHOLD else LABEL_REAL

        return InferenceResult(
            prediction=prediction,
            fake_probability=round(fake_prob, 6),
            real_probability=round(real_prob, 6),
            decision_threshold=FIXED_DECISION_THRESHOLD,
            raw_decision_score=round(raw_logit, 6),
            temperature_applied=round(self.temperature, 10),
            model_name=MODEL_NAME,
            feature_dimension=FEATURE_DIMENSION,
            audio_metadata=metadata,
            disclaimer=DISCLAIMER_TEXT,
        )

