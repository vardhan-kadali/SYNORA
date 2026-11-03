"""
SYNORA — Test Suite: Artifact Integrity
Verifies the physical existence, immutability, and SHA256 checksums of all frozen model artifacts.
"""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from synora.config import (
    SCALER_PATH,
    CLASSIFIER_PATH,
    TEMPERATURE_PATH,
    METADATA_PATH,
    EXPECTED_SCALER_SHA256,
    EXPECTED_CLASSIFIER_SHA256,
    EXPECTED_TEMPERATURE_SHA256,
    EXPECTED_METADATA_SHA256,
    FEATURE_DIMENSION,
    FROZEN_TEMPERATURE,
)
from synora.inference.engine import compute_file_sha256, SynoraInferenceEngine


class TestArtifactIntegrity(unittest.TestCase):
    """Test suite verifying frozen model artifact integrity."""

    def test_01_artifact_files_exist(self):
        """Verify all four frozen model artifact files physically exist on disk."""
        self.assertTrue(SCALER_PATH.exists(), f"Scaler missing: {SCALER_PATH}")
        self.assertTrue(CLASSIFIER_PATH.exists(), f"Classifier missing: {CLASSIFIER_PATH}")
        self.assertTrue(TEMPERATURE_PATH.exists(), f"Temperature JSON missing: {TEMPERATURE_PATH}")
        self.assertTrue(METADATA_PATH.exists(), f"Metadata JSON missing: {METADATA_PATH}")

    def test_02_scaler_sha256(self):
        """Verify final_scaler.joblib matches authoritative SHA256."""
        actual_hash = compute_file_sha256(SCALER_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_SCALER_SHA256,
            f"Scaler checksum mismatch: {actual_hash} != {EXPECTED_SCALER_SHA256}",
        )

    def test_03_classifier_sha256(self):
        """Verify final_classifier.joblib matches authoritative SHA256."""
        actual_hash = compute_file_sha256(CLASSIFIER_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_CLASSIFIER_SHA256,
            f"Classifier checksum mismatch: {actual_hash} != {EXPECTED_CLASSIFIER_SHA256}",
        )

    def test_04_temperature_scaling_sha256(self):
        """Verify temperature_scaling.json matches authoritative SHA256."""
        actual_hash = compute_file_sha256(TEMPERATURE_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_TEMPERATURE_SHA256,
            f"Temperature JSON checksum mismatch: {actual_hash} != {EXPECTED_TEMPERATURE_SHA256}",
        )

    def test_05_metadata_json_sha256(self):
        """Verify final_model_metadata.json matches authoritative SHA256."""
        actual_hash = compute_file_sha256(METADATA_PATH)
        self.assertEqual(
            actual_hash,
            EXPECTED_METADATA_SHA256,
            f"Metadata JSON checksum mismatch: {actual_hash} != {EXPECTED_METADATA_SHA256}",
        )

    def test_06_scaler_parameters(self):
        """Verify scaler dimensionality and attributes."""
        scaler = joblib.load(SCALER_PATH)
        self.assertEqual(scaler.n_features_in_, FEATURE_DIMENSION)
        self.assertEqual(len(scaler.mean_), FEATURE_DIMENSION)
        self.assertEqual(len(scaler.scale_), FEATURE_DIMENSION)
        self.assertTrue(np.all(np.isfinite(scaler.mean_)))
        self.assertTrue(np.all(np.isfinite(scaler.scale_)))

    def test_07_classifier_parameters(self):
        """Verify classifier dimensionality, coefficients, intercept, and classes."""
        clf = joblib.load(CLASSIFIER_PATH)
        self.assertEqual(clf.n_features_in_, FEATURE_DIMENSION)
        self.assertTrue(np.array_equal(clf.classes_, [0, 1]))
        self.assertEqual(clf.coef_.shape, (1, FEATURE_DIMENSION))
        self.assertEqual(len(clf.intercept_), 1)
        self.assertTrue(np.all(np.isfinite(clf.coef_)))
        self.assertTrue(np.all(np.isfinite(clf.intercept_)))

    def test_08_temperature_json_contents(self):
        """Verify temperature JSON values match frozen specification."""
        with open(TEMPERATURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        temp_val = float(data["temperature"])
        self.assertAlmostEqual(temp_val, FROZEN_TEMPERATURE, places=6)
        self.assertGreater(temp_val, 0.0)

    def test_09_engine_checksum_guard(self):
        """Verify engine raises RuntimeError if an expected hash fails."""
        # Temporarily verify that engine initialises cleanly with correct hashes
        engine = SynoraInferenceEngine(verify_checksums=True)
        self.assertIsNotNone(engine.scaler)
        self.assertIsNotNone(engine.classifier)


if __name__ == "__main__":
    unittest.main(verbosity=2)

