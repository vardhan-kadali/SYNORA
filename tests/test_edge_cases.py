"""
SYNORA — Test Suite: Edge Cases
Verifies graceful handling of silence, invalid/corrupted files, empty inputs,
multi-channel audio, non-16kHz sampling rates, and path security guards.
"""

import unittest
import numpy as np

from synora.config import TARGET_SAMPLE_RATE, FEATURE_DIMENSION, assert_path_safety
from synora.dsp.extractor import (
    extract_features,
    decode_audio,
    SilentAudioError,
    InvalidAudioError,
)
from synora.inference.engine import SynoraInferenceEngine
from tests.test_feature_parity import generate_synthetic_speech_tone


class TestEdgeCases(unittest.TestCase):
    """Test suite verifying robustness under malformed, silent, or extreme inputs."""

    @classmethod
    def setUpClass(cls):
        cls.engine = SynoraInferenceEngine(verify_checksums=False)

    def test_01_silent_audio_rejection(self):
        """Verify all-zero audio deterministically raises SilentAudioError."""
        silent_audio = np.zeros(16000, dtype=np.float32)
        with self.assertRaises(SilentAudioError):
            self.engine.predict(silent_audio)

    def test_02_near_silent_below_vad_rejection(self):
        """Verify audio with amplitude below 1e-4 RMS raises SilentAudioError."""
        quiet_noise = (np.random.RandomState(42).randn(16000) * 1e-6).astype(np.float32)
        with self.assertRaises(SilentAudioError):
            self.engine.predict(quiet_noise)

    def test_03_corrupted_bytes_rejection(self):
        """Verify corrupted audio bytes raise InvalidAudioError."""
        corrupted_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPTED_DATA_STREAM"
        with self.assertRaises(InvalidAudioError):
            self.engine.predict(corrupted_bytes)

    def test_04_empty_bytes_rejection(self):
        """Verify empty byte stream raises InvalidAudioError."""
        with self.assertRaises(InvalidAudioError):
            self.engine.predict(b"")

    def test_05_empty_array_rejection(self):
        """Verify empty numpy array raises ValueError / InvalidAudioError."""
        with self.assertRaises((ValueError, InvalidAudioError)):
            self.engine.predict(np.array([], dtype=np.float32))

    def test_06_short_audio_processing(self):
        """Verify short audio (0.15s, 2400 samples) processes successfully."""
        short_audio = generate_synthetic_speech_tone(duration=0.15)
        result = self.engine.predict(short_audio)
        self.assertIn(result.prediction, ["Fake", "Real"])
        self.assertEqual(result.feature_dimension, FEATURE_DIMENSION)

    def test_07_stereo_to_mono_downmixing(self):
        """Verify 2-channel audio is automatically downmixed to mono."""
        mono_signal = generate_synthetic_speech_tone(duration=1.0)
        # Create stereo signal with different channel amplitudes
        stereo_signal = np.stack([mono_signal, mono_signal * 0.8], axis=-1)
        self.assertEqual(stereo_signal.ndim, 2)
        self.assertEqual(stereo_signal.shape[1], 2)

        result = self.engine.predict(stereo_signal)
        self.assertEqual(result.audio_metadata["channels"], 2)
        self.assertEqual(result.feature_dimension, FEATURE_DIMENSION)

    def test_08_resampling_from_24k(self):
        """Verify 24 kHz audio is cleanly resampled to 16 kHz."""
        tone_24k = generate_synthetic_speech_tone(duration=1.0, sr=24000)
        audio_16k, meta = decode_audio(tone_24k, native_sr=24000)

        self.assertEqual(meta["native_sample_rate"], 24000)
        self.assertEqual(meta["processed_sample_rate"], TARGET_SAMPLE_RATE)
        self.assertEqual(len(audio_16k), TARGET_SAMPLE_RATE)

        # Confirm inference functions cleanly on resampled array
        result = self.engine.predict(tone_24k, native_sr=24000)
        self.assertEqual(result.feature_dimension, FEATURE_DIMENSION)

    def test_09_path_safety_guards(self):
        """Verify path safety guard raises PermissionError on Test partition access."""
        with self.assertRaises(PermissionError):
            assert_path_safety("benchmark/test/clean/sample_001.wav")
        with self.assertRaises(PermissionError):
            assert_path_safety("benchmark\\test\\telephone\\sample_001.wav")
        with self.assertRaises(PermissionError):
            assert_path_safety("data/cache/test_features.npz")
        with self.assertRaises(PermissionError):
            assert_path_safety("data/cache/test_features_v2.npz")

        # Verify engine predict rejects locked path
        with self.assertRaises(PermissionError):
            self.engine.predict("benchmark/test/clean/sample_001.wav")


if __name__ == "__main__":
    unittest.main(verbosity=2)

