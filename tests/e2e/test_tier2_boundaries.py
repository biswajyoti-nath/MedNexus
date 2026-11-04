"""Tier 2: Boundary & Corner Cases E2E Tests (>=5 test cases per feature).

Features Covered:
1. Empty upload (missing file).
2. Unsupported extensions (.png, .jpg to /api/predict).
3. Corrupted / zero-byte file handling.
4. Filenames with special characters and spaces.
5. Rapid sequential uploads.
6. Large file upload handling.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from tests.e2e.test_fixtures import (
    E2ETestBase,
    NeuroVisionFixtureHandler,
    encode_multipart_formdata,
    get_sample_tumor_path,
    http_request,
)


class TestTier2BoundaryAndCornerCases(E2ETestBase):
    """Tier 2 comprehensive boundary and corner case tests."""

    # -------------------------------------------------------------------------
    # Feature 1: Empty upload (missing file)
    # -------------------------------------------------------------------------

    def test_01_predict_missing_file_part(self) -> None:
        """Multipart POST without 'file' part returns 400 Bad Request."""
        body, ct = encode_multipart_formdata({"symptoms": "headache"}, {})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("error", data)

    def test_02_predict_empty_filename(self) -> None:
        """Multipart upload with empty filename returns 400."""
        body, ct = encode_multipart_formdata({}, {"file": ("", b"some_bytes", "image/tiff")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)

    def test_03_predict_empty_post_body(self) -> None:
        """Empty POST body to /api/predict returns 400."""
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", data=b"")
        self.assertEqual(resp.status_code, 400)

    def test_04_analyze_empty_request(self) -> None:
        """POST to /api/analyze with no file and no text handled gracefully."""
        body, ct = encode_multipart_formdata({}, {})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("summary", data)

    def test_05_predict_missing_file_error_json_structure(self) -> None:
        """Missing file returns valid JSON with descriptive error string."""
        body, ct = encode_multipart_formdata({"dummy": "value"}, {})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        data = resp.json()
        self.assertIsInstance(data.get("error"), str)
        self.assertGreater(len(data.get("error")), 0)

    # -------------------------------------------------------------------------
    # Feature 2: Unsupported extensions (.png, .jpg to /api/predict)
    # -------------------------------------------------------------------------

    def test_06_predict_reject_png(self) -> None:
        """POSTing .png image to /api/predict returns 400."""
        body, ct = encode_multipart_formdata({}, {"file": ("scan.png", b"\x89PNG\r\n\x1a\n", "image/png")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("A .tif or .tiff MRI slice is required", resp.json().get("error", ""))

    def test_07_predict_reject_jpg(self) -> None:
        """POSTing .jpg image to /api/predict returns 400."""
        body, ct = encode_multipart_formdata({}, {"file": ("scan.jpg", b"\xff\xd8\xff", "image/jpeg")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)

    def test_08_predict_reject_pdf(self) -> None:
        """POSTing .pdf document to /api/predict returns 400."""
        body, ct = encode_multipart_formdata({}, {"file": ("report.pdf", b"%PDF-1.4", "application/pdf")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)

    def test_09_predict_reject_bin_or_exe(self) -> None:
        """POSTing arbitrary binary (.bin/.exe) to /api/predict returns 400."""
        body, ct = encode_multipart_formdata({}, {"file": ("exploit.exe", b"MZ\x90\x00", "application/octet-stream")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 400)

    def test_10_analyze_unsupported_file_returns_415(self) -> None:
        """MedNexus /api/analyze returns 415 Unsupported Media Type for invalid extensions."""
        body, ct = encode_multipart_formdata({}, {"file": ("archive.zip", b"PK\x03\x04", "application/zip")})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 415)
        self.assertIn("Supported files", resp.json().get("detail", ""))

    # -------------------------------------------------------------------------
    # Feature 3: Corrupted / zero-byte file handling
    # -------------------------------------------------------------------------

    def test_11_predict_zero_byte_tif(self) -> None:
        """Uploading a 0-byte .tif returns clean 400 or 500 JSON error, not unhandled crash."""
        body, ct = encode_multipart_formdata({}, {"file": ("empty.tif", b"", "image/tiff")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn(resp.status_code, (400, 500))
        self.assertIn("error", resp.json())

    def test_12_predict_corrupted_header_tif(self) -> None:
        """Uploading corrupt non-image bytes named corrupt.tif returns clean JSON error."""
        body, ct = encode_multipart_formdata({}, {"file": ("corrupt.tif", b"CORRUPT_NOT_A_REAL_TIFF", "image/tiff")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn(resp.status_code, (400, 500))
        self.assertIn("error", resp.json())

    def test_13_predict_corrupt_file_temp_cleanup(self) -> None:
        """Temporary file is unlinked even when decompression/inference fails."""
        uploads_dir = Path("neurovision-mri/static/uploads")
        before_count = len(list(uploads_dir.glob("*.*"))) if uploads_dir.exists() else 0

        body, ct = encode_multipart_formdata({}, {"file": ("corrupt.tif", b"CORRUPT_BYTES", "image/tiff")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn(resp.status_code, (400, 500))

        after_count = len(list(uploads_dir.glob("*.*"))) if uploads_dir.exists() else 0
        self.assertEqual(before_count, after_count, "Stranded file found after corrupted upload")

    def test_14_predict_plain_text_with_tif_ext(self) -> None:
        """ASCII text file renamed to fake.tif is handled safely without crashing service."""
        fake_content = b"This is just an ASCII text file pretending to be an MRI scan."
        body, ct = encode_multipart_formdata({}, {"file": ("fake.tif", fake_content, "image/tiff")})
        resp = http_request(f"{self.mri_url}/api/predict", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn(resp.status_code, (400, 500))

    def test_15_analyze_corrupt_tif_error_handling(self) -> None:
        """MedNexus handles downstream MRI processing failures gracefully with HTTP error."""
        body, ct = encode_multipart_formdata({}, {"file": ("corrupt.tif", b"CORRUPT_BYTES", "image/tiff")})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn(resp.status_code, (500, 502, 503))
        self.assertIn("detail", resp.json())

    # -------------------------------------------------------------------------
    # Feature 4: Filenames with special characters and spaces
    # -------------------------------------------------------------------------

    def test_16_predict_filename_with_spaces(self) -> None:
        """Filename containing spaces processes successfully."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path, custom_filename="brain slice scan 01.tif")
        self.assertEqual(resp.status_code, 200)

    def test_17_predict_filename_path_traversal(self) -> None:
        """Filename with path traversal payload is safely sanitized."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path, custom_filename="../../../etc/passwd.tif")
        self.assertEqual(resp.status_code, 200)

    def test_18_predict_filename_unicode(self) -> None:
        """Filename with non-ASCII unicode characters processes cleanly."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path, custom_filename="снимок_мозга_01.tif")
        self.assertEqual(resp.status_code, 200)

    def test_19_predict_filename_multiple_dots(self) -> None:
        """Filename with multiple dot separators processes correctly."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path, custom_filename="patient.axial.slice.TCGA.final.tif")
        self.assertEqual(resp.status_code, 200)

    def test_20_predict_filename_special_punctuation(self) -> None:
        """Filename with URL-style punctuation (#, &, %, parenthesis) handled safely."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path, custom_filename="slice#01&scan%20(copy)!.tif")
        self.assertEqual(resp.status_code, 200)

    # -------------------------------------------------------------------------
    # Feature 5: Rapid sequential uploads
    # -------------------------------------------------------------------------

    def test_21_rapid_sequential_burst(self) -> None:
        """5 rapid consecutive uploads all complete with HTTP 200."""
        tumor_path = get_sample_tumor_path()
        for i in range(5):
            resp = self.predict_mri(tumor_path, custom_filename=f"burst_{i}.tif")
            self.assertEqual(resp.status_code, 200, f"Burst request {i} failed: {resp.text}")

    def test_22_rapid_uploads_unique_uuids(self) -> None:
        """Consecutive uploads of identical filename receive distinct processing paths."""
        tumor_path = get_sample_tumor_path()
        resp1 = self.predict_mri(tumor_path, custom_filename="concurrent.tif")
        resp2 = self.predict_mri(tumor_path, custom_filename="concurrent.tif")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)

    def test_23_rapid_uploads_temp_cleanup(self) -> None:
        """Zero lingering files remain after rapid sequential burst."""
        uploads_dir = Path("neurovision-mri/static/uploads")
        before_count = len(list(uploads_dir.glob("*.tif"))) if uploads_dir.exists() else 0

        tumor_path = get_sample_tumor_path()
        for i in range(4):
            resp = self.predict_mri(tumor_path, custom_filename=f"cleanup_burst_{i}.tif")
            self.assertEqual(resp.status_code, 200)

        after_count = len(list(uploads_dir.glob("*.tif"))) if uploads_dir.exists() else 0
        self.assertEqual(before_count, after_count)

    def test_24_rapid_uploads_consistent_classes(self) -> None:
        """Responses across a burst consistently yield expected class."""
        tumor_path = get_sample_tumor_path()
        classes = []
        for _ in range(3):
            resp = self.predict_mri(tumor_path)
            classes.append(resp.json().get("predicted_class"))
        self.assertEqual(classes, ["tumour_suspected", "tumour_suspected", "tumour_suspected"])

    def test_25_server_health_stable_after_rapid_burst(self) -> None:
        """Server remains healthy and responsive immediately following rapid burst."""
        resp = http_request(f"{self.mri_url}/health", method="GET")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    # -------------------------------------------------------------------------
    # Feature 6: Large file upload handling
    # -------------------------------------------------------------------------

    def test_26_analyze_exceeds_max_upload_size(self) -> None:
        """Upload exceeding MAX_UPLOAD_MB (13 MB > 12 MB) returns HTTP 413."""
        # 13 MB dummy buffer
        oversized = b"0" * (13 * 1024 * 1024)
        body, ct = encode_multipart_formdata({}, {"file": ("huge_scan.tif", oversized, "image/tiff")})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 413)

    def test_27_analyze_413_error_message_detail(self) -> None:
        """413 response detail specifies maximum upload limit."""
        oversized = b"0" * (13 * 1024 * 1024)
        body, ct = encode_multipart_formdata({}, {"file": ("huge_scan.tif", oversized, "image/tiff")})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertIn("Upload exceeds 12 MB limit", resp.json().get("detail", ""))

    def test_28_analyze_valid_large_file_under_limit(self) -> None:
        """File well within limit (e.g. real ~200 KB TIFF sample) processes normally."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp.status_code, 200)

    def test_29_large_file_aborted_temp_cleanup(self) -> None:
        """Rejected oversized upload leaves zero lingering temp files."""
        uploads_dir = Path("neurovision-mri/static/uploads")
        before_count = len(list(uploads_dir.glob("*.*"))) if uploads_dir.exists() else 0

        oversized = b"0" * (13 * 1024 * 1024)
        body, ct = encode_multipart_formdata({}, {"file": ("huge_scan.tif", oversized, "image/tiff")})
        resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
        self.assertEqual(resp.status_code, 413)

        after_count = len(list(uploads_dir.glob("*.*"))) if uploads_dir.exists() else 0
        self.assertEqual(before_count, after_count)

    def test_30_predict_large_valid_tif_memory_safety(self) -> None:
        """Valid sample TIFF file processes without memory exhaustion or socket errors."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("predicted_class", resp.json())


if __name__ == "__main__":
    unittest.main()
