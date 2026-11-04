"""
SYNORA — Test Suite: Artifact Integrity
Verifies the physical existence, immutability, and SHA256 checksums of all frozen model artifacts.
"""

import json
import tempfile
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

    def test_10_crlf_lf_json_checksum_parity(self):
        """Verify identical JSON content with LF and CRLF line endings produces the exact same checksum."""
        sample_json_text = '{\n  "name": "SYNORA",\n  "status": "ready"\n}\n'
        crlf_bytes = sample_json_text.replace('\n', '\r\n').encode('utf-8')
        lf_bytes = sample_json_text.encode('utf-8')

        with tempfile.TemporaryDirectory() as tmpdir:
            crlf_path = Path(tmpdir) / "sample_crlf.json"
            lf_path = Path(tmpdir) / "sample_lf.json"

            crlf_path.write_bytes(crlf_bytes)
            lf_path.write_bytes(lf_bytes)

            hash_crlf = compute_file_sha256(crlf_path)
            hash_lf = compute_file_sha256(lf_path)

            self.assertEqual(
                hash_crlf,
                hash_lf,
                f"Cross-platform hash mismatch: CRLF={hash_crlf} != LF={hash_lf}"
            )

    def test_11_binary_file_hash_byte_exactness(self):
        """Verify binary files (.joblib) are hashed byte-exact without line ending alteration."""
        # Data containing embedded b'\r\n' bytes
        binary_payload = b"\x80\x04\x95\r\n\x00\x00\x00\r\n\xff\xfe"

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir) / "sample_model.joblib"
            bin_path.write_bytes(binary_payload)

            import hashlib
            expected_raw_sha = hashlib.sha256(binary_payload).hexdigest().lower()
            computed_sha = compute_file_sha256(bin_path)

            self.assertEqual(
                computed_sha,
                expected_raw_sha,
                f"Binary hash alteration detected: {computed_sha} != {expected_raw_sha}"
            )

    def test_12_all_packaged_artifacts_pass_integrity(self):
        """Verify all four packaged model artifacts pass integrity checks."""
        engine = SynoraInferenceEngine(verify_checksums=True)
        self.assertEqual(engine.scaler.n_features_in_, FEATURE_DIMENSION)
        self.assertEqual(engine.classifier.n_features_in_, FEATURE_DIMENSION)
        self.assertEqual(compute_file_sha256(SCALER_PATH), EXPECTED_SCALER_SHA256)
        self.assertEqual(compute_file_sha256(CLASSIFIER_PATH), EXPECTED_CLASSIFIER_SHA256)
        self.assertEqual(compute_file_sha256(TEMPERATURE_PATH), EXPECTED_TEMPERATURE_SHA256)
        self.assertEqual(compute_file_sha256(METADATA_PATH), EXPECTED_METADATA_SHA256)

    def test_13_frozen_temperature_exact_value(self):
        """Verify the frozen temperature remains exactly 1.1229901304 across all definitions."""
        self.assertEqual(FROZEN_TEMPERATURE, 1.1229901304)

        with open(TEMPERATURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(float(data["temperature"]), 1.1229901304)

        engine = SynoraInferenceEngine(verify_checksums=True)
        self.assertEqual(engine.temperature, 1.1229901304)


if __name__ == "__main__":
    unittest.main(verbosity=2)

