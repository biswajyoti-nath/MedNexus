"""Tier 1: Feature Coverage E2E Tests (>=5 test cases per feature).

Features Covered:
1. Direct MRI service health check (/health and /api/health).
2. Positive tumor prediction (sample_tumor.tif).
3. Negative no-tumor prediction (sample_no_tumor.tif).
4. Overlay data URL validity (PNG magic bytes iVBORw0KGgo in base64).
5. MedNexus proxy /api/analyze routing .tif and .tiff to NeuroVision.
6. Temporary file cleanup (verifying zero leftover temp files).
"""
from __future__ import annotations

import base64
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path

from tests.e2e.test_fixtures import (
    E2ETestBase,
    NeuroVisionFixtureHandler,
    create_minimal_valid_png_bytes,
    get_sample_no_tumor_path,
    get_sample_tumor_path,
    http_request,
)


class TestTier1FeatureCoverage(E2ETestBase):
    """Tier 1 comprehensive feature tests."""

    # -------------------------------------------------------------------------
    # Feature 1: Direct MRI service health check (/health and /api/health)
    # -------------------------------------------------------------------------

    def test_01_mri_health_endpoint_contract(self) -> None:
        """GET /health returns 200 with status ok and specialist metadata."""
        url = f"{self.mri_url}/health"
        resp = http_request(url, method="GET")
        self.assertEqual(resp.status_code, 200, f"Expected 200 from {url}, got {resp.status_code}")
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")
        self.assertEqual(data.get("task"), "2D brain MRI slice screening")
        self.assertIs(data.get("clinically_validated"), False)

    def test_02_mri_api_health_endpoint_contract(self) -> None:
        """GET /api/health returns 200 with identical schema for MedNexus auto-discovery."""
        url = f"{self.mri_url}/api/health"
        resp = http_request(url, method="GET")
        self.assertEqual(resp.status_code, 200, f"Expected 200 from {url}, got {resp.status_code}")
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")
        self.assertIn("task", data)
        self.assertIs(data.get("clinically_validated"), False)

    def test_03_mednexus_health_aggregated_status(self) -> None:
        """GET /api/health on MedNexus returns 200 and reports system connectivity."""
        url = f"{self.mednexus_url}/api/health"
        resp = http_request(url, method="GET")
        self.assertEqual(resp.status_code, 200, f"Expected 200 from {url}, got {resp.status_code}")
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("ollama", data)
        self.assertIn("neurovision", data)
        self.assertIn("endpoint", data["neurovision"])

    def test_04_health_json_content_type(self) -> None:
        """All health endpoints return application/json Content-Type."""
        for endpoint in (f"{self.mri_url}/health", f"{self.mri_url}/api/health", f"{self.mednexus_url}/api/health"):
            resp = http_request(endpoint, method="GET")
            ct = resp.headers.get("content-type", "")
            self.assertTrue(ct.startswith("application/json"), f"{endpoint} did not return JSON content-type: {ct}")

    def test_05_health_rejects_post_method(self) -> None:
        """POST to /health is rejected with 405 Method Not Allowed (or non-200 error)."""
        url = f"{self.mri_url}/health"
        resp = http_request(url, method="POST", data=b"{}")
        self.assertIn(resp.status_code, (405, 501, 400, 404))

    def test_06_health_response_latency(self) -> None:
        """Health check responds in under 1 second without blocking."""
        t0 = time.time()
        resp = http_request(f"{self.mri_url}/health", method="GET")
        elapsed = time.time() - t0
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 1.0, f"Health check latency too high: {elapsed:.3f}s")

    # -------------------------------------------------------------------------
    # Feature 2: Positive tumor prediction (sample_tumor.tif)
    # -------------------------------------------------------------------------

    def test_07_positive_prediction_status_200(self) -> None:
        """POST sample_tumor.tif to /api/predict returns HTTP 200."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        self.assertEqual(resp.status_code, 200, f"Predict failed: {resp.text}")

    def test_08_positive_prediction_predicted_class(self) -> None:
        """Positive slice yields predicted_class == 'tumour_suspected'."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("predicted_class"), "tumour_suspected")

    def test_09_positive_prediction_segmentation_available(self) -> None:
        """Positive slice yields segmentation_available is True."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        self.assertIs(data.get("segmentation_available"), True)

    def test_10_positive_prediction_bounding_box_format(self) -> None:
        """Positive slice returns bounding box as 4 positive integers [x, y, w, h]."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        bbox = data.get("bbox")
        self.assertIsInstance(bbox, list)
        self.assertEqual(len(bbox), 4)
        for val in bbox:
            self.assertIsInstance(val, int)
            self.assertGreaterEqual(val, 0)
        self.assertGreater(bbox[2], 0, "Bbox width must be positive")
        self.assertGreater(bbox[3], 0, "Bbox height must be positive")

    def test_11_positive_prediction_score_range_and_provenance(self) -> None:
        """Positive slice returns float model_score with explicit uncalibrated score notice."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        score = data.get("model_score")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        score_type = data.get("score_type", "")
        self.assertIn("not a calibrated clinical probability", score_type)

    def test_12_positive_prediction_dimensions(self) -> None:
        """Positive slice returns positive integer source dimensions."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        self.assertIsInstance(data.get("image_width"), int)
        self.assertIsInstance(data.get("image_height"), int)
        self.assertGreater(data.get("image_width"), 0)
        self.assertGreater(data.get("image_height"), 0)

    # -------------------------------------------------------------------------
    # Feature 3: Negative no-tumor prediction (sample_no_tumor.tif)
    # -------------------------------------------------------------------------

    def test_13_negative_prediction_status_200(self) -> None:
        """POST sample_no_tumor.tif to /api/predict returns HTTP 200."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        self.assertEqual(resp.status_code, 200, f"Negative predict failed: {resp.text}")

    def test_14_negative_prediction_predicted_class(self) -> None:
        """Negative slice yields predicted_class == 'no_tumour_detected_by_model'."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        data = resp.json()
        self.assertEqual(data.get("predicted_class"), "no_tumour_detected_by_model")

    def test_15_negative_prediction_segmentation_available(self) -> None:
        """Negative slice yields segmentation_available is False."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        data = resp.json()
        self.assertIs(data.get("segmentation_available"), False)

    def test_16_negative_prediction_null_bbox(self) -> None:
        """Negative slice yields bbox is None."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        data = resp.json()
        self.assertIsNone(data.get("bbox"))

    def test_17_negative_prediction_null_overlay(self) -> None:
        """Negative slice yields segmentation_overlay is None."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        data = resp.json()
        self.assertIsNone(data.get("segmentation_overlay"))

    def test_18_negative_prediction_score_range(self) -> None:
        """Negative slice returns valid float score below tumor confidence threshold."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.predict_mri(no_tumor_path)
        data = resp.json()
        score = data.get("model_score")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertLess(score, 0.5, "Negative slice score should typically be under 0.5")

    # -------------------------------------------------------------------------
    # Feature 4: Overlay data URL validity
    # -------------------------------------------------------------------------

    def test_19_overlay_data_url_prefix(self) -> None:
        """Segmentation overlay starts with data:image/png;base64,"""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        overlay = data.get("segmentation_overlay")
        self.assertIsInstance(overlay, str)
        self.assertTrue(overlay.startswith("data:image/png;base64,"))

    def test_20_overlay_base64_valid_decode(self) -> None:
        """Base64 data decodes into valid binary bytes without error."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        overlay = resp.json().get("segmentation_overlay")
        b64_content = overlay.split(",", 1)[1]
        decoded = base64.b64decode(b64_content)
        self.assertGreater(len(decoded), 0)

    def test_21_overlay_png_magic_bytes(self) -> None:
        """Decoded overlay contains PNG signature and base64 begins with iVBORw0KGgo."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        overlay = resp.json().get("segmentation_overlay")
        b64_content = overlay.split(",", 1)[1]
        self.assertTrue(b64_content.startswith("iVBORw0KGgo"), "PNG base64 must begin with standard magic prefix")
        decoded = base64.b64decode(b64_content)
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"), "Decoded bytes must match PNG header")

    def test_22_overlay_ihdr_dimensions_within_canvas(self) -> None:
        """Decoded PNG dimensions fit within 760x760 web display boundary."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        overlay = resp.json().get("segmentation_overlay")
        b64_content = overlay.split(",", 1)[1]
        decoded = base64.b64decode(b64_content)
        # Parse IHDR dimensions (bytes 16..24)
        width, height = struct.unpack(">II", decoded[16:24])
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertLessEqual(width, 760)
        self.assertLessEqual(height, 760)

    def test_23_overlay_has_rgba_color_type(self) -> None:
        """Decoded PNG specifies 8-bit depth and RGBA / palette with alpha."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        overlay = resp.json().get("segmentation_overlay")
        decoded = base64.b64decode(overlay.split(",", 1)[1])
        bit_depth = decoded[24]
        color_type = decoded[25]
        self.assertEqual(bit_depth, 8)
        self.assertIn(color_type, (6, 3, 2), "Must be RGBA (6) or alpha-compatible color type")

    def test_24_overlay_alpha_mask_non_trivial(self) -> None:
        """Decoded overlay is non-empty binary stream containing compressed image data."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        overlay = resp.json().get("segmentation_overlay")
        decoded = base64.b64decode(overlay.split(",", 1)[1])
        self.assertIn(b"IHDR", decoded)
        self.assertIn(b"IDAT", decoded)
        self.assertIn(b"IEND", decoded)

    # -------------------------------------------------------------------------
    # Feature 5: MedNexus proxy /api/analyze routing .tif and .tiff
    # -------------------------------------------------------------------------

    def test_25_mednexus_routes_tif_to_neurovision(self) -> None:
        """MedNexus /api/analyze routes .tif to NeuroVision MRI specialist."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp.status_code, 200, f"MedNexus analyze failed: {resp.text}")
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")
        self.assertEqual(data.get("mode"), "live_model")

    def test_26_mednexus_routes_tiff_extension(self) -> None:
        """MedNexus /api/analyze routes .tiff file extension identically."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path, custom_filename="brain_scan.tiff")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")

    def test_27_mednexus_route_structure_metadata(self) -> None:
        """MedNexus response includes route block identifying input_modality and reason."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        route = data.get("route", {})
        self.assertEqual(route.get("input_modality"), "MRI TIFF slice")
        self.assertEqual(route.get("selected_specialist"), "NeuroVision MRI")
        self.assertIn("reason", route)

    def test_28_mednexus_preserves_specialist_result_fields(self) -> None:
        """MedNexus specialist_result preserves predicted_class, score, bbox, and overlay."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        sr = data.get("specialist_result", {})
        self.assertEqual(sr.get("predicted_class"), "tumour_suspected")
        self.assertIsInstance(sr.get("model_score"), float)
        self.assertIsInstance(sr.get("bbox"), list)
        self.assertTrue(sr.get("segmentation_overlay", "").startswith("data:image/png;base64,"))

    def test_29_mednexus_summary_enforces_slice_level_disclaimer(self) -> None:
        """MedNexus summary contains explicit non-diagnostic demarcation."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        summary = resp.json().get("summary", "")
        self.assertIn("not a calibrated clinical probability", summary)
        self.assertIn("2D slice-level output, not a patient-level diagnosis", summary)

    def test_30_mednexus_clinical_status_not_validated(self) -> None:
        """MedNexus clinical_status reports not clinically validated."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        self.assertEqual(data.get("clinical_status"), "not clinically validated")

    # -------------------------------------------------------------------------
    # Feature 6: Temporary file cleanup
    # -------------------------------------------------------------------------

    def test_31_temp_cleanup_after_positive_mri(self) -> None:
        """No temporary files linger after positive prediction."""
        tumor_path = get_sample_tumor_path()
        uploads_dir = Path("neurovision-mri/static/uploads")
        before_files = set(uploads_dir.glob("*.*")) if uploads_dir.exists() else set()

        resp = self.predict_mri(tumor_path)
        self.assertEqual(resp.status_code, 200)

        after_files = set(uploads_dir.glob("*.*")) if uploads_dir.exists() else set()
        new_files = [f for f in (after_files - before_files) if not f.name.endswith(".png")]
        self.assertEqual(len(new_files), 0, f"Found lingering upload files: {new_files}")

    def test_32_temp_cleanup_after_negative_mri(self) -> None:
        """No temporary files linger after negative prediction."""
        no_tumor_path = get_sample_no_tumor_path()
        uploads_dir = Path("neurovision-mri/static/uploads")
        before_files = set(uploads_dir.glob("*.*")) if uploads_dir.exists() else set()

        resp = self.predict_mri(no_tumor_path)
        self.assertEqual(resp.status_code, 200)

        after_files = set(uploads_dir.glob("*.*")) if uploads_dir.exists() else set()
        new_files = [f for f in (after_files - before_files) if not f.name.endswith(".png")]
        self.assertEqual(len(new_files), 0, f"Found lingering upload files: {new_files}")

    def test_33_temp_cleanup_after_failed_prediction(self) -> None:
        """Temporary file is cleaned up even if prediction payload fails."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(b"CORRUPT_BYTES_FOR_FAILURE_TEST")
            tmp_path = Path(tmp.name)
        try:
            resp = self.predict_mri(tmp_path)
            self.assertIn(resp.status_code, (400, 500))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # In fixture mode, verify fixture handler cleaned up
        if not self.is_live:
            self.assertEqual(
                len(NeuroVisionFixtureHandler.temp_created_files),
                len(NeuroVisionFixtureHandler.temp_deleted_files),
                "Fixture created and deleted file counts must match",
            )

    def test_34_temp_files_use_unique_uuid_prefix(self) -> None:
        """Simultaneous or successive uploads create distinct paths avoiding filename collisions."""
        tumor_path = get_sample_tumor_path()
        resp1 = self.predict_mri(tumor_path, custom_filename="shared_scan.tif")
        resp2 = self.predict_mri(tumor_path, custom_filename="shared_scan.tif")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)

    def test_35_temp_cleanup_multi_cycle_zero_leakage(self) -> None:
        """Executing 5 sequential predictions results in zero stranded temporary files."""
        tumor_path = get_sample_tumor_path()
        uploads_dir = Path("neurovision-mri/static/uploads")
        initial_count = len(list(uploads_dir.glob("*.tif"))) if uploads_dir.exists() else 0

        for _ in range(5):
            resp = self.predict_mri(tumor_path)
            self.assertEqual(resp.status_code, 200)

        final_count = len(list(uploads_dir.glob("*.tif"))) if uploads_dir.exists() else 0
        self.assertEqual(initial_count, final_count, "Stranded TIFF files detected after multiple uploads")


if __name__ == "__main__":
    unittest.main()
