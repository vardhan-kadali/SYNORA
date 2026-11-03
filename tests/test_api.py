"""
SYNORA — Test Suite: Web API (FastAPI TestClient)
Verifies all API endpoints, contract schemas, error statuses, boundary limits, and hardening measures.
"""

import io
import unittest
from unittest.mock import patch, AsyncMock
from pathlib import Path

import numpy as np
import soundfile as sf
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartParser
from fastapi.testclient import TestClient

from synora.config import (
    WORKSPACE_ROOT,
    MODEL_NAME,
    FEATURE_DIMENSION,
    DISCLAIMER_TEXT,
    MAX_UPLOAD_BYTES,
    MULTIPART_OVERHEAD_ALLOWANCE,
    MAX_REQUEST_BODY_BYTES,
)
from synora.api.app import create_app
from synora.api.routes import MAX_AUDIO_DURATION_SECONDS
from synora.api.middleware import RequestSizeLimitMiddleware, RequestBodyLimitExceeded
from synora.inference.engine import InferenceResult


def generate_audio_bytes(format: str = "WAV", duration: float = 1.0, sr: int = 16000, silent: bool = False) -> bytes:
    """Generates pure in-memory audio bytes for testing in WAV, FLAC, OGG, or MP3."""
    bio = io.BytesIO()
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if silent:
        audio = np.zeros(len(t), dtype=np.float32)
    else:
        # Multi-harmonic synthetic speech-like tone
        f0 = 150.0
        audio = (
            0.40 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        ).astype(np.float32)
        # Apply envelope to prevent boundary clicks
        audio *= 0.5 * (1.0 - np.cos(2 * np.pi * t / duration))
    
    if format == "WAV":
        sf.write(bio, audio, sr, format="WAV", subtype="PCM_16")
    else:
        sf.write(bio, audio, sr, format=format)
    return bio.getvalue()


def generate_wav_bytes(duration: float = 1.0, sr: int = 16000, silent: bool = False) -> bytes:
    """Generates pure in-memory WAV audio bytes for testing without filesystem persistence."""
    return generate_audio_bytes(format="WAV", duration=duration, sr=sr, silent=silent)


class TestSynoraAPI(unittest.TestCase):
    """FastAPI TestClient test suite verifying all Stage 2 & Stage 4 API requirements."""

    @classmethod
    def setUpClass(cls):
        """Create application and TestClient."""
        cls.app = create_app()
        # Using context manager to run lifespan startup events
        cls.client_cm = TestClient(cls.app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_root_endpoint(self):
        """1. Verify GET / returns service identity and tagline."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "SYNORA")
        self.assertEqual(data["tagline"], "Detecting the Synthetic. Protecting the Real.")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["api_version"], "1.0.0")

    def test_02_health_endpoint(self):
        """2. Verify GET /health reports healthy status and model readiness without leaking paths."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_name"], MODEL_NAME)
        self.assertEqual(data["feature_dimensions"], FEATURE_DIMENSION)
        # Verify no absolute local paths or secrets in response
        self.assertNotIn("D:\\", str(data))
        self.assertNotIn("experiments", str(data))

    def test_03_successful_audio_upload(self):
        """3. Verify POST /api/v1/analyze with authorized Dev sample syn_001_clean returns 200 OK."""
        dev_path = WORKSPACE_ROOT / "data" / "audio" / "clean" / "syn_001_clean.wav"
        with open(dev_path, "rb") as f:
            audio_bytes = f.read()

        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("syn_001_clean.wav", audio_bytes, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"], "Fake")

    def test_04_json_response_structure(self):
        """4. Verify response contains all required contract fields and metadata."""
        audio_bytes = generate_wav_bytes(duration=1.0)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Contract fields
        self.assertIn("success", data)
        self.assertIn("prediction", data)
        self.assertIn("probabilities", data)
        self.assertIn("decision_threshold", data)
        self.assertIn("audio", data)
        self.assertIn("model", data)
        self.assertIn("disclaimer", data)

        # Sub-schemas
        self.assertIn("fake", data["probabilities"])
        self.assertIn("real", data["probabilities"])
        self.assertIn("duration_seconds", data["audio"])
        self.assertIn("sample_rate", data["audio"])
        self.assertIn("channels", data["audio"])
        self.assertIn("name", data["model"])
        self.assertIn("feature_dimensions", data["model"])

        # Scientific disclaimer
        self.assertEqual(data["disclaimer"], DISCLAIMER_TEXT)

    def test_05_probability_bounds_and_sum(self):
        """5. Verify probabilities are bounded in [0, 1] and sum to approximately 1.0."""
        audio_bytes = generate_wav_bytes(duration=1.2)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
        )
        data = response.json()
        fake_p = data["probabilities"]["fake"]
        real_p = data["probabilities"]["real"]

        self.assertGreaterEqual(fake_p, 0.0)
        self.assertLessEqual(fake_p, 1.0)
        self.assertGreaterEqual(real_p, 0.0)
        self.assertLessEqual(real_p, 1.0)
        self.assertAlmostEqual(fake_p + real_p, 1.0, places=5)

    def test_06_prediction_label_mapping(self):
        """6. Verify prediction label mapping: Fake if P(fake) >= 0.50 else Real."""
        audio_bytes = generate_wav_bytes(duration=1.0)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
        )
        data = response.json()
        expected_label = "Fake" if data["probabilities"]["fake"] >= 0.50 else "Real"
        self.assertEqual(data["prediction"], expected_label)

    def test_07_empty_upload_rejection(self):
        """7. Verify empty 0-byte upload is rejected with HTTP 400 Bad Request."""
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "HTTP_400")

    def test_08_corrupted_audio_rejection(self):
        """8. Verify corrupted audio stream is rejected with HTTP 400 Bad Request."""
        corrupted = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00NOT_VALID_AUDIO_BYTES"
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("corrupted.wav", corrupted, "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])

    def test_09_unsupported_media_rejection(self):
        """9. Verify non-audio media (e.g. .txt, .exe, .pdf) is rejected with HTTP 415."""
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        self.assertEqual(response.status_code, 415)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "HTTP_415")

    def test_10_upload_size_limit(self):
        """10. Verify upload exceeding 25 MB is rejected with HTTP 413 Payload Too Large."""
        oversized = b"RIFF" + b"\x00" * (MAX_UPLOAD_BYTES + 100)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("large.wav", oversized, "audio/wav")},
        )
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "HTTP_413")

    def test_11_duration_limit(self):
        """11. Verify audio longer than 60 seconds is rejected with HTTP 413 Payload Too Large."""
        long_wav = generate_wav_bytes(duration=65.0, sr=16000)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("long.wav", long_wav, "audio/wav")},
        )
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("60 seconds", data["detail"])

    def test_12_silent_audio_rejection(self):
        """12. Verify silent audio below VAD threshold is rejected with HTTP 400 Bad Request."""
        silent_wav = generate_wav_bytes(duration=1.0, silent=True)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("silent.wav", silent_wav, "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Silent audio detected", data["detail"])

    def test_13_initialization_failure_handling(self):
        """13. Verify engine unready state returns HTTP 503 Service Unavailable."""
        test_app = create_app()
        test_app.state.engine = None
        client = TestClient(test_app)

        health_res = client.get("/health")
        self.assertEqual(health_res.status_code, 503)
        self.assertFalse(health_res.json()["model_loaded"])

        predict_res = client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", generate_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(predict_res.status_code, 503)

    def test_14_no_permanent_upload_storage(self):
        """14. Verify no uploaded audio files or temporary files persist on disk."""
        before_files = set(Path(WORKSPACE_ROOT).glob("*.wav"))
        audio_bytes = generate_wav_bytes(duration=0.5)

        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("temporary_upload.wav", audio_bytes, "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)

        after_files = set(Path(WORKSPACE_ROOT).glob("*.wav"))
        self.assertEqual(before_files, after_files, "Temporary upload file persisted in workspace!")

    def test_15_valid_flac_upload(self):
        """15. Verify valid FLAC upload returns 200 OK and valid prediction."""
        flac_bytes = generate_audio_bytes(format="FLAC", duration=0.8)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("sample.flac", flac_bytes, "audio/flac")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ["Fake", "Real"])

    def test_16_valid_ogg_upload(self):
        """16. Verify valid OGG upload returns 200 OK and valid prediction."""
        ogg_bytes = generate_audio_bytes(format="OGG", duration=0.8)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("sample.ogg", ogg_bytes, "audio/ogg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ["Fake", "Real"])

    def test_17_valid_mp3_upload(self):
        """17. Verify valid MP3 upload returns 200 OK and valid prediction."""
        mp3_bytes = generate_audio_bytes(format="MP3", duration=0.8)
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("sample.mp3", mp3_bytes, "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction"], ["Fake", "Real"])

    def test_18_strict_allowlist_unsupported_extension(self):
        """18. Verify unsupported audio extensions (.xyz, .aac, .m4a) are rejected with HTTP 415."""
        dummy_bytes = b"RIFFfakeaudiobytes"
        for ext in [".xyz", ".aac", ".m4a", ".wma"]:
            response = self.client.post(
                "/api/v1/analyze",
                files={"file": (f"audio{ext}", dummy_bytes, "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 415, f"Extension {ext} was not rejected with 415!")
            data = response.json()
            self.assertEqual(data["error"], "HTTP_415")
            self.assertIn("Unsupported file extension", data["detail"])

    def test_19_incorrect_mime_with_valid_audio_bytes(self):
        """19. Verify valid audio bytes with spoofed/generic MIME type are decoded successfully."""
        wav_bytes = generate_wav_bytes(duration=1.0)
        # Client sends MIME type "text/plain" or "application/octet-stream", but filename has .wav
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("valid.wav", wav_bytes, "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    def test_20_invalid_audio_bytes_with_valid_extension(self):
        """20. Verify text or random binary bytes inside a .wav filename fail decoding with HTTP 400."""
        text_bytes = b"Hello, this is just plain text content masquerading as a wav file."
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("spoofed.wav", text_bytes, "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid or corrupted audio stream", data["detail"])

    def test_21_early_content_length_limit(self):
        """21. Verify request with declared Content-Length exceeding middleware limit is rejected with HTTP 413."""
        response = self.client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", b"small bytes", "audio/wav")},
            headers={"Content-Length": str(MAX_REQUEST_BODY_BYTES + 5000)},
        )
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertEqual(data["error"], "HTTP_413")
        self.assertIn("Upload size exceeds maximum permitted limit", data["detail"])

    def test_22_file_resource_cleanup_on_success(self):
        """22. Verify UploadFile.close() is called upon successful analysis."""
        wav_bytes = generate_wav_bytes(duration=0.5)
        orig_close = StarletteUploadFile.close
        with patch.object(StarletteUploadFile, "close", autospec=True, side_effect=orig_close) as mock_close:
            response = self.client.post(
                "/api/v1/analyze",
                files={"file": ("test_cleanup.wav", wav_bytes, "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            mock_close.assert_awaited()

    def test_23_file_resource_cleanup_on_failure(self):
        """23. Verify UploadFile.close() is called even when validation fails."""
        orig_close = StarletteUploadFile.close
        with patch.object(StarletteUploadFile, "close", autospec=True, side_effect=orig_close) as mock_close:
            # 0-byte upload causes HTTP 400
            response = self.client.post(
                "/api/v1/analyze",
                files={"file": ("empty.wav", b"", "audio/wav")},
            )
            self.assertEqual(response.status_code, 400)
            mock_close.assert_awaited()

    def test_24_cors_configuration(self):
        """24. Verify CORS configuration allows expected local origins and disallows credentials."""
        # Preflight OPTIONS request from localhost:5173
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        response = self.client.options("/api/v1/analyze", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        # Credentials must not be allowed
        self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_25_near_limit_valid_payload_with_oversized_multipart_envelope(self):
        """25. Verify valid audio payload (25 MiB - 100 B) whose multipart envelope exceeds 25 MiB is accepted."""
        dummy_wav = b"RIFF" + b"\x00" * (MAX_UPLOAD_BYTES - 104)
        self.assertEqual(len(dummy_wav), MAX_UPLOAD_BYTES - 100)

        mock_waveform = np.zeros(16000, dtype=np.float32)
        mock_waveform[100:200] = 0.5
        mock_meta = {
            "duration_seconds": 1.0,
            "native_sample_rate": 16000,
            "processed_sample_rate": 16000,
            "channels": 1,
            "num_samples_16k": 16000,
        }
        mock_inference = InferenceResult(
            prediction="Real",
            fake_probability=0.15,
            real_probability=0.85,
            decision_threshold=0.5,
            raw_decision_score=-1.8,
            temperature_applied=1.1229901304,
            model_name=MODEL_NAME,
            feature_dimension=FEATURE_DIMENSION,
            audio_metadata=mock_meta,
            disclaimer=DISCLAIMER_TEXT,
        )

        with patch("synora.api.routes.decode_audio", return_value=(mock_waveform, mock_meta)), \
             patch.object(self.client.app.state.engine, "predict", return_value=mock_inference):
            response = self.client.post(
                "/api/v1/analyze",
                files={"file": ("near_limit.wav", dummy_wav, "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["prediction"], "Real")

    def test_26_rejection_before_multipart_parser_runs(self):
        """26. Verify oversized declared requests are rejected before Starlette MultiPartParser runs."""
        with patch.object(MultiPartParser, "parse", autospec=True) as mock_parse:
            response = self.client.post(
                "/api/v1/analyze",
                files={"file": ("test.wav", b"small bytes", "audio/wav")},
                headers={"Content-Length": str(MAX_REQUEST_BODY_BYTES + 10000)},
            )
            self.assertEqual(response.status_code, 413)
            data = response.json()
            self.assertEqual(data["error"], "HTTP_413")
            mock_parse.assert_not_called()

    def test_27_chunked_streaming_request_exceeding_limit(self):
        """27. Verify streaming request without Content-Length exceeding limit returns HTTP 413."""
        from fastapi import FastAPI, Request
        from starlette.exceptions import HTTPException as StarletteHTTPException
        from starlette.responses import JSONResponse

        stream_app = FastAPI()
        @stream_app.exception_handler(StarletteHTTPException)
        async def stream_exc_handler(req: Request, exc: StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "error": f"HTTP_{exc.status_code}", "detail": exc.detail},
            )
        stream_app.add_middleware(RequestSizeLimitMiddleware, max_body_size=500, max_audio_mb=25)

        @stream_app.post("/stream_test")
        async def stream_endpoint(req: Request):
            body = await req.body()
            return {"bytes": len(body)}

        with TestClient(stream_app) as stream_client:
            def exceeding_chunks():
                for _ in range(8):
                    yield b"x" * 100  # 800 bytes total > 500 max_body_size

            response = stream_client.post("/stream_test", content=exceeding_chunks())
            self.assertEqual(response.status_code, 413)
            data = response.json()
            self.assertEqual(data["error"], "HTTP_413")

    def test_28_chunked_streaming_request_within_limit(self):
        """28. Verify streaming request without Content-Length within limit returns HTTP 200."""
        from fastapi import FastAPI, Request
        from starlette.exceptions import HTTPException as StarletteHTTPException
        from starlette.responses import JSONResponse

        stream_app = FastAPI()
        @stream_app.exception_handler(StarletteHTTPException)
        async def stream_exc_handler(req: Request, exc: StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "error": f"HTTP_{exc.status_code}", "detail": exc.detail},
            )
        stream_app.add_middleware(RequestSizeLimitMiddleware, max_body_size=500, max_audio_mb=25)

        @stream_app.post("/stream_test")
        async def stream_endpoint(req: Request):
            body = await req.body()
            return {"bytes": len(body)}

        with TestClient(stream_app) as stream_client:
            def within_chunks():
                for _ in range(3):
                    yield b"x" * 100  # 300 bytes total <= 500 max_body_size

            response = stream_client.post("/stream_test", content=within_chunks())
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["bytes"], 300)

    def test_underreported_content_length_with_oversized_body(self):
        """29. Verify request declaring Content-Length: 50 but transmitting 800 bytes is rejected with HTTP 413."""
        from fastapi import FastAPI, Request
        from starlette.exceptions import HTTPException as StarletteHTTPException
        from starlette.responses import JSONResponse

        test_app = FastAPI()

        @test_app.exception_handler(StarletteHTTPException)
        async def exc_handler(req: Request, exc: StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"success": False, "error": f"HTTP_{exc.status_code}", "detail": exc.detail},
            )

        test_app.add_middleware(RequestSizeLimitMiddleware, max_body_size=500, max_audio_mb=25)

        endpoint_processed = False

        @test_app.post("/underreported_test")
        async def underreported_endpoint(req: Request):
            nonlocal endpoint_processed
            body = await req.body()
            endpoint_processed = True
            return {"bytes": len(body)}

        with TestClient(test_app) as client:
            # Middleware limit: 500 bytes. Declared Content-Length: 50. Actual transmitted body: 800 bytes.
            response = client.post(
                "/underreported_test",
                content=b"X" * 800,
                headers={"Content-Length": "50"},
            )
            # 1. Verify HTTP 413
            self.assertEqual(response.status_code, 413)
            # 2. Verify structured JSON error contract
            data = response.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"], "HTTP_413")
            self.assertIn("Upload size exceeds maximum permitted limit", data["detail"])
            # 3. Verify downstream endpoint did not process the oversized body
            self.assertFalse(endpoint_processed, "Downstream endpoint must not process oversized body!")


class TestPreMultipartMiddleware(unittest.IsolatedAsyncioTestCase):
    """Direct ASGI test suite for RequestSizeLimitMiddleware edge cases."""

    async def test_non_http_scope_passthrough(self):
        """Verify non-HTTP scopes (lifespan) pass through to downstream app unaffected."""
        called = False
        async def inner_app(scope, receive, send):
            nonlocal called
            called = True

        mw = RequestSizeLimitMiddleware(inner_app, max_body_size=100)
        await mw({"type": "lifespan"}, None, None)
        self.assertTrue(called)

    async def test_client_disconnect_handling(self):
        """Verify http.disconnect message is forwarded to downstream without triggering size error."""
        received_msg = None
        async def receive_disconnect():
            return {"type": "http.disconnect"}

        async def inner_app(scope, receive, send):
            nonlocal received_msg
            received_msg = await receive()

        mw = RequestSizeLimitMiddleware(inner_app, max_body_size=100)
        scope = {"type": "http", "method": "POST", "headers": []}
        await mw(scope, receive_disconnect, None)
        self.assertEqual(received_msg, {"type": "http.disconnect"})

    async def test_controlled_asgi_declared_content_length_exceeded(self):
        """Verify declared Content-Length > limit returns 413 immediately without calling inner app."""
        called = False
        async def inner_app(scope, receive, send):
            nonlocal called
            called = True

        mw = RequestSizeLimitMiddleware(inner_app, max_body_size=100, max_audio_mb=25)
        scope = {"type": "http", "method": "POST", "headers": [(b"content-length", b"200")]}
        messages = []
        async def mock_send(msg):
            messages.append(msg)

        await mw(scope, None, mock_send)
        self.assertFalse(called, "Inner app was invoked despite declared Content-Length exceeding limit!")
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 413)

    async def test_controlled_asgi_streaming_byte_counting(self):
        """Verify cumulative byte counting over multiple chunks aborts stream on limit breach."""
        chunks = [
            {"type": "http.request", "body": b"A" * 60, "more_body": True},
            {"type": "http.request", "body": b"B" * 60, "more_body": False},
        ]
        chunk_iter = iter(chunks)
        async def mock_receive():
            return next(chunk_iter)

        messages = []
        async def mock_send(msg):
            messages.append(msg)

        async def inner_app(scope, receive, send):
            await receive()
            await receive()

        mw = RequestSizeLimitMiddleware(inner_app, max_body_size=100, max_audio_mb=25)
        scope = {"type": "http", "method": "POST", "headers": []}
        await mw(scope, mock_receive, mock_send)
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 413)


if __name__ == "__main__":
    unittest.main(verbosity=2)
