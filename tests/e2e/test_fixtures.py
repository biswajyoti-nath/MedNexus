"""Test fixtures, sample file locators, and embedded contract test server for E2E tests.
"""
from __future__ import annotations

import base64
import http.server
import io
import json
import os
import re
import socket
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MRI_SAMPLE_DIR = PROJECT_ROOT / "neurovision-mri" / "sample_images"
MRI_STATIC_DIR = PROJECT_ROOT / "neurovision-mri" / "static" / "uploads"


def get_sample_tumor_path() -> Path:
    candidates = [
        MRI_SAMPLE_DIR / "sample_tumor.tif",
        MRI_STATIC_DIR / "sample_tumor.tif",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("sample_tumor.tif not found in sample_images or static/uploads")


def get_sample_no_tumor_path() -> Path:
    candidates = [
        MRI_SAMPLE_DIR / "sample_no_tumor.tif",
        MRI_STATIC_DIR / "sample_no_tumor.tif",
        MRI_SAMPLE_DIR / "sample_no_tumor_2.tif",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("sample_no_tumor.tif not found in sample_images or static/uploads")


def create_minimal_valid_png_bytes(width: int = 256, height: int = 256, alpha: int = 112) -> bytes:
    """Create raw binary RGBA PNG bytes with a semi-transparent contour mask using pure Python."""
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # Filter byte None
        for x in range(width):
            # Center circular mask resembling tumor segmentation
            dx = x - width // 2
            dy = y - height // 2
            if dx * dx + dy * dy < (width // 4) * (width // 4):
                raw_data.extend([61, 190, 139, alpha])  # Green overlay
            else:
                raw_data.extend([30, 30, 30, 255])  # Background brain tissue

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    png_header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)
    idat = png_chunk(b"IDAT", zlib.compress(bytes(raw_data), 9))
    iend = png_chunk(b"IEND", b"")
    return png_header + ihdr + idat + iend


def create_overlay_data_url(width: int = 256, height: int = 256) -> str:
    png_bytes = create_minimal_valid_png_bytes(width, height)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_service_reachable(url: str, timeout: float = 0.8) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "E2ETestReachabilityCheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 301, 302, 404, 405)
    except urllib.error.HTTPError as e:
        return e.code in (200, 400, 404, 405)
    except Exception:
        return False


def encode_multipart_formdata(fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = "----WebKitFormBoundary" + base64.b16encode(os.urandom(16)).decode("ascii")
    body = io.BytesIO()

    for key, value in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.write(str(value).encode("utf-8"))
        body.write(b"\r\n")

    for key, (filename, content, content_type) in files.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.write(content)
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return body.getvalue(), content_type_header


class HttpResponse:
    def __init__(self, status_code: int, headers: Dict[str, str], body: bytes):
        self.status_code = status_code
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: float = 10.0,
) -> HttpResponse:
    req_headers = headers.copy() if headers else {}
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_headers = dict(resp.headers.items())
            resp_body = resp.read()
            return HttpResponse(resp.status, resp_headers, resp_body)
    except urllib.error.HTTPError as e:
        resp_headers = dict(e.headers.items()) if e.headers else {}
        resp_body = e.read()
        return HttpResponse(e.code, resp_headers, resp_body)


# --- Embedded Contract-Compliant Fixture Servers ---

class NeuroVisionFixtureHandler(http.server.BaseHTTPRequestHandler):
    """Simulates NeuroVision Flask API exactly obeying PROJECT.md and mri_json_endpoint_snippet.py."""

    temp_created_files: list[str] = []
    temp_deleted_files: list[str] = []

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default server logs during tests

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/health", "/api/health"):
            payload = {
                "status": "ok",
                "specialist": "NeuroVision MRI",
                "task": "2D brain MRI slice screening",
                "clinically_validated": False,
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/":
            body = b"<!DOCTYPE html><html><head><title>NeuroVision</title></head><body><h1>NeuroVision MRI</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path != "/api/predict":
            self.send_response(404)
            self.end_headers()
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if not content_type.startswith("multipart/form-data"):
            body = json.dumps({"error": "Missing file field"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        post_data = self.rfile.read(content_length)

        # Parse boundary
        m = re.search(r"boundary=([^\s;]+)", content_type)
        if not m:
            self.send_response(400)
            self.end_headers()
            return
        boundary = m.group(1).encode("utf-8")

        parts = post_data.split(b"--" + boundary)
        file_part = None
        filename = None
        for part in parts:
            if b'name="file"' in part:
                file_part = part
                fn_m = re.search(rb'filename="([^"]*)"', part)
                if fn_m:
                    filename = fn_m.group(1).decode("utf-8", errors="replace")
                break

        if file_part is None:
            body = json.dumps({"error": "Missing file field"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not filename:
            body = json.dumps({"error": "A .tif or .tiff MRI slice is required"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("tif", "tiff"):
            body = json.dumps({"error": "A .tif or .tiff MRI slice is required"}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Extract file body
        header_end = file_part.find(b"\r\n\r\n")
        file_bytes = file_part[header_end + 4 :].rstrip(b"\r\n") if header_end != -1 else b""

        # Simulate atomic UUID temporary file creation and cleanup
        simulated_temp_dir = Path(tempfile.gettempdir()) / "neurovision_test_uploads"
        simulated_temp_dir.mkdir(parents=True, exist_ok=True)
        import uuid
        temp_filepath = simulated_temp_dir / f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)
        NeuroVisionFixtureHandler.temp_created_files.append(str(temp_filepath))

        try:
            # Check for corrupt / zero byte file
            if len(file_bytes) == 0:
                body = json.dumps({"error": "Image file is empty or corrupted"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            is_valid_tiff = file_bytes.startswith(b"II*\x00") or file_bytes.startswith(b"MM\x00*")
            if not is_valid_tiff or file_bytes.startswith(b"CORRUPT") or file_bytes == b"not_a_valid_tiff_stream":
                body = json.dumps({"error": "Cannot identify image file"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            # Determine tumor vs no-tumor based on filename or contents
            is_tumor = "no_tumor" not in filename.lower()

            if is_tumor:
                overlay_data_url = create_overlay_data_url(256, 256)
                payload = {
                    "specialist": "NeuroVision MRI",
                    "task": "2D brain MRI slice screening",
                    "predicted_class": "tumour_suspected",
                    "model_score": 0.8421,
                    "score_type": "model output; not a calibrated clinical probability",
                    "segmentation_available": True,
                    "bbox": [120, 85, 45, 60],
                    "segmentation_overlay": overlay_data_url,
                    "image_width": 256,
                    "image_height": 256,
                    "scope": "Research model trained on LGG MRI slices; not validated for broad neurological diagnosis or patient-level diagnosis.",
                    "clinically_validated": False,
                }
            else:
                payload = {
                    "specialist": "NeuroVision MRI",
                    "task": "2D brain MRI slice screening",
                    "predicted_class": "no_tumour_detected_by_model",
                    "model_score": 0.0812,
                    "score_type": "model output; not a calibrated clinical probability",
                    "segmentation_available": False,
                    "bbox": None,
                    "segmentation_overlay": None,
                    "image_width": 256,
                    "image_height": 256,
                    "scope": "Research model trained on LGG MRI slices; not validated for broad neurological diagnosis or patient-level diagnosis.",
                    "clinically_validated": False,
                }

            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        finally:
            # Enforce cleanup in finally block exactly as app.py does
            try:
                if temp_filepath.exists():
                    temp_filepath.unlink()
                NeuroVisionFixtureHandler.temp_deleted_files.append(str(temp_filepath))
            except OSError:
                pass


class MedNexusFixtureHandler(http.server.BaseHTTPRequestHandler):
    """Simulates MedNexus FastAPI application exactly obeying PROJECT.md and main.py."""

    mri_target_url: str = "http://127.0.0.1:8080/api/predict"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/health":
            mri_up = False
            mri_error = None
            try:
                health_url = MedNexusFixtureHandler.mri_target_url.rsplit("/", 1)[0] + "/health"
                r = http_request(health_url, timeout=1.5)
                mri_up = (r.status_code == 200)
            except Exception as e:
                mri_error = str(e)

            payload = {
                "status": "ok",
                "ollama": {"connected": True, "models": ["qwen3.5:0.8b", "medgemma1.5"]},
                "expected_models": {"text": "qwen3.5:0.8b", "vision": "medgemma1.5"},
                "neurovision": {
                    "health_route_detected": mri_up,
                    "endpoint": MedNexusFixtureHandler.mri_target_url,
                    "note": mri_error or "Connectivity hint only; test with a TIFF upload.",
                },
                "mode": "live",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/demo":
            payload = {
                "mode": "simulated_demo",
                "provenance": "SIMULATED DEMO DATA · NO MODEL INFERENCE WAS RUN",
                "specialist": "Demo workflow simulator",
                "summary": "This synthetic case demonstrates how MedNexus organizes supplied information. No real patient data or model inference is involved.\n\nReported information: The sample case includes a fictional adult with a fictional headache history and a placeholder MRI finding.\n\nQuestions for clinician review: Onset, progression, neurological examination, full imaging sequence and radiologist report would be needed.\n\nMissing context: No real laboratory results, DICOM series or verified clinical history are attached.\n\nResearch/demo output; clinician review required.",
                "scope": "Synthetic fixture for UI demonstration only.",
                "clinical_status": "not a diagnosis; not patient data",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/":
            body = b"<!DOCTYPE html><html><head><title>MedNexus AI</title></head><body><h1>MedNexus AI</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path != "/api/analyze":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        max_bytes = 12 * 1024 * 1024
        if content_length > max_bytes:
            # Drain body so client can read 413 without BrokenPipeError
            _ = self.rfile.read(content_length)
            body = json.dumps({"detail": "Upload exceeds 12 MB limit"}).encode("utf-8")
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        content_type = self.headers.get("Content-Type", "")
        post_data = self.rfile.read(content_length)

        m = re.search(r"boundary=([^\s;]+)", content_type)
        filename = None
        symptoms = ""
        history = ""
        file_bytes = b""

        if m:
            boundary = m.group(1).encode("utf-8")
            parts = post_data.split(b"--" + boundary)
            for part in parts:
                if b'name="file"' in part:
                    fn_m = re.search(rb'filename="([^"]*)"', part)
                    if fn_m:
                        filename = fn_m.group(1).decode("utf-8", errors="replace")
                    h_end = part.find(b"\r\n\r\n")
                    if h_end != -1:
                        file_bytes = part[h_end + 4 :].rstrip(b"\r\n")
                elif b'name="symptoms"' in part:
                    h_end = part.find(b"\r\n\r\n")
                    if h_end != -1:
                        symptoms = part[h_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="replace")
                elif b'name="history"' in part:
                    h_end = part.find(b"\r\n\r\n")
                    if h_end != -1:
                        history = part[h_end + 4 :].rstrip(b"\r\n").decode("utf-8", errors="replace")

        suffix = Path(filename).suffix.lower() if filename else ""

        if suffix in (".tif", ".tiff"):
            # Proxy to NeuroVision MRI
            try:
                fields = {"symptoms": symptoms, "history": history}
                files = {"file": (filename, file_bytes, "image/tiff")}
                multipart_body, mp_content_type = encode_multipart_formdata(fields, files)
                resp = http_request(
                    MedNexusFixtureHandler.mri_target_url,
                    method="POST",
                    headers={"Content-Type": mp_content_type},
                    data=multipart_body,
                    timeout=10,
                )
                if resp.status_code != 200:
                    body = json.dumps({"detail": f"NeuroVision MRI specialist error: {resp.text}"}).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                mri_result = resp.json()
            except Exception as exc:
                body = json.dumps({
                    "detail": f"NeuroVision MRI specialist is not connected: {exc}. Start the MRI service and add the JSON route from integration/."
                }).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            predicted = mri_result.get("predicted_class", "not supplied")
            score = mri_result.get("model_score")
            segmentation_available = bool(mri_result.get("segmentation_available"))
            bbox = mri_result.get("bbox")
            summary = (
                f"Specialist output: {predicted}\n"
                f"Model score: {score if score is not None else 'not supplied'} (not a calibrated clinical probability)\n"
                f"Segmentation: {'available' if segmentation_available else 'not reported'}\n"
                f"Bounding box: {bbox if bbox is not None else 'not supplied'}\n\n"
                "Interpret this only within the model's research scope. This is a 2D slice-level output, not a patient-level diagnosis."
            )
            payload = {
                "mode": "live_model",
                "specialist": "NeuroVision MRI",
                "provenance": "LIVE SPECIALIST INFERENCE",
                "route": {
                    "input_modality": "MRI TIFF slice",
                    "selected_specialist": "NeuroVision MRI",
                    "reason": "TIFF slices are routed to the dedicated 2D MRI classification and segmentation pipeline.",
                },
                "specialist_result": {
                    "task": mri_result.get("task", "2D brain MRI slice screening"),
                    "predicted_class": predicted,
                    "model_score": score,
                    "score_type": mri_result.get("score_type", "Model output; not a calibrated clinical probability."),
                    "segmentation_available": segmentation_available,
                    "bbox": bbox,
                    "segmentation_overlay": mri_result.get("segmentation_overlay"),
                    "image_width": mri_result.get("image_width"),
                    "image_height": mri_result.get("image_height"),
                    "clinically_validated": bool(mri_result.get("clinically_validated", False)),
                },
                "scope": mri_result.get("scope", "2D research MRI slice model; check the model documentation"),
                "raw_result": {k: v for k, v in mri_result.items() if k != "segmentation_overlay"},
                "summary": summary,
                "clinical_status": "not clinically validated",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            payload = {
                "mode": "live_model",
                "specialist": "medgemma1.5",
                "provenance": "LIVE MODEL OUTPUT · NOT CLINICALLY VALIDATED",
                "summary": "Observation: Medical image reviewed. No acute abnormalities observed in simulated vision review. Research/demo output; clinician review required.",
                "file_name": filename,
                "scope": "General medical image interpretation; modality/task-specific validation not established.",
                "clinical_status": "not clinically validated",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if suffix in (".pdf", ".txt", ".md", ".csv") or not filename:
            payload = {
                "mode": "live_model",
                "specialist": "qwen3.5:0.8b",
                "provenance": "LIVE MODEL OUTPUT · NOT CLINICALLY VALIDATED",
                "summary": f"Reported information: Symptoms: {symptoms or 'None'}. History: {history or 'None'}.\n\nQuestions for clinician review: Physical exam and targeted imaging indicated.\n\nResearch/demo output; clinician review required.",
                "file_name": filename or None,
                "document_text_extracted": bool(filename and suffix in (".pdf", ".txt")),
                "clinical_status": "not clinically validated",
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps({"detail": "Supported files: TIFF for connected NeuroVision, PNG/JPG/WEBP for MedGemma, PDF/TXT/MD/CSV for report text."}).encode("utf-8")
        self.send_response(415)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestEnvironment:
    """Manages test server lifecycle and URL discovery."""

    _mri_server: Optional[http.server.HTTPServer] = None
    _mednexus_server: Optional[http.server.HTTPServer] = None
    _mri_thread: Optional[threading.Thread] = None
    _mednexus_thread: Optional[threading.Thread] = None

    mri_url: str = ""
    mednexus_url: str = ""
    is_live: bool = False

    @classmethod
    def setup(cls, force_mode: Optional[str] = None) -> Tuple[str, str, bool]:
        mode = force_mode or os.environ.get("TEST_MODE", "auto").lower()

        default_mri = os.environ.get("NEUROVISION_URL", "http://127.0.0.1:8080")
        default_mednexus = os.environ.get("MEDNEXUS_URL", "http://127.0.0.1:8000")

        live_mri_up = is_service_reachable(f"{default_mri}/health") or is_service_reachable(f"{default_mri}/api/health")
        live_mednexus_up = is_service_reachable(f"{default_mednexus}/api/health")

        if mode == "live":
            cls.mri_url = default_mri
            cls.mednexus_url = default_mednexus
            cls.is_live = True
            return cls.mri_url, cls.mednexus_url, cls.is_live

        if mode == "fixture":
            cls._start_fixture_servers()
            cls.is_live = False
            return cls.mri_url, cls.mednexus_url, cls.is_live

        # Auto mode: Use live if both are reachable; otherwise start fixture server
        if live_mri_up and live_mednexus_up:
            cls.mri_url = default_mri
            cls.mednexus_url = default_mednexus
            cls.is_live = True
        else:
            cls._start_fixture_servers()
            cls.is_live = False

        return cls.mri_url, cls.mednexus_url, cls.is_live

    @classmethod
    def _start_fixture_servers(cls) -> None:
        if cls._mri_server is None:
            mri_port = find_free_port()
            cls._mri_server = http.server.ThreadingHTTPServer(("127.0.0.1", mri_port), NeuroVisionFixtureHandler)
            cls._mri_thread = threading.Thread(target=cls._mri_server.serve_forever, daemon=True)
            cls._mri_thread.start()
            cls.mri_url = f"http://127.0.0.1:{mri_port}"

        if cls._mednexus_server is None:
            mednexus_port = find_free_port()
            MedNexusFixtureHandler.mri_target_url = f"{cls.mri_url}/api/predict"
            cls._mednexus_server = http.server.ThreadingHTTPServer(("127.0.0.1", mednexus_port), MedNexusFixtureHandler)
            cls._mednexus_thread = threading.Thread(target=cls._mednexus_server.serve_forever, daemon=True)
            cls._mednexus_thread.start()
            cls.mednexus_url = f"http://127.0.0.1:{mednexus_port}"

        time.sleep(0.05)

    @classmethod
    def teardown(cls) -> None:
        if cls._mri_server:
            cls._mri_server.shutdown()
            cls._mri_server.server_close()
            cls._mri_server = None
        if cls._mednexus_server:
            cls._mednexus_server.shutdown()
            cls._mednexus_server.server_close()
            cls._mednexus_server = None


class E2ETestBase(unittest.TestCase):
    """Base class for all E2E test suites with shared assertions and HTTP helpers."""

    mri_url: str = ""
    mednexus_url: str = ""
    is_live: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.mri_url, cls.mednexus_url, cls.is_live = TestEnvironment.setup()

    def predict_mri(self, filepath: Path, custom_filename: Optional[str] = None) -> HttpResponse:
        with open(filepath, "rb") as f:
            content = f.read()
        fname = custom_filename or filepath.name
        body, ct = encode_multipart_formdata({}, {"file": (fname, content, "image/tiff")})
        return http_request(
            f"{self.mri_url}/api/predict",
            method="POST",
            headers={"Content-Type": ct},
            data=body,
            timeout=180 if self.is_live else 10,
        )

    def analyze_mednexus(
        self,
        symptoms: str = "",
        history: str = "",
        filepath: Optional[Path] = None,
        custom_filename: Optional[str] = None,
        content_type: str = "image/tiff",
    ) -> HttpResponse:
        fields = {"symptoms": symptoms, "history": history}
        files = {}
        if filepath is not None:
            with open(filepath, "rb") as f:
                content = f.read()
            fname = custom_filename or filepath.name
            files["file"] = (fname, content, content_type)

        body, ct = encode_multipart_formdata(fields, files)
        return http_request(
            f"{self.mednexus_url}/api/analyze",
            method="POST",
            headers={"Content-Type": ct},
            data=body,
            timeout=180 if self.is_live else 10,
        )
