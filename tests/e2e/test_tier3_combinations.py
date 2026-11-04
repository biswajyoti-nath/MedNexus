"""Tier 3: Cross-Feature Combinations E2E Tests (Pairwise Matrix).

Features Covered:
1. Modality switching: Text question -> MRI TIFF slice -> PNG photograph -> Demo mode.
2. Health polling during active inference.
3. Multiple consecutive MRI slice screenings.
"""
from __future__ import annotations

import concurrent.futures
import time
import unittest
from pathlib import Path

from tests.e2e.test_fixtures import (
    E2ETestBase,
    create_minimal_valid_png_bytes,
    get_sample_no_tumor_path,
    get_sample_tumor_path,
    http_request,
)


class TestTier3CrossFeatureCombinations(E2ETestBase):
    """Tier 3 pairwise combination and integration tests."""

    # -------------------------------------------------------------------------
    # Combination 1: Modality switching
    # (Text -> MRI TIFF -> PNG photograph -> Demo mode)
    # -------------------------------------------------------------------------

    def test_01_modality_switch_text_to_mri(self) -> None:
        """Sequential switch from clinical text symptoms to MRI TIFF slice."""
        # 1. Text inquiry
        resp_text = self.analyze_mednexus(symptoms="Intermittent mild headache in right temporal region")
        self.assertEqual(resp_text.status_code, 200)
        data_text = resp_text.json()
        self.assertIn(data_text.get("specialist"), ("qwen3.5:0.8b", "text_assistant", "qwen"))

        # 2. MRI slice upload
        tumor_path = get_sample_tumor_path()
        resp_mri = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp_mri.status_code, 200)
        data_mri = resp_mri.json()
        self.assertEqual(data_mri.get("specialist"), "NeuroVision MRI")
        self.assertEqual(data_mri.get("route", {}).get("input_modality"), "MRI TIFF slice")

    def test_02_modality_switch_mri_to_vision(self) -> None:
        """Sequential switch from MRI TIFF slice to standard PNG photograph."""
        # 1. MRI slice
        tumor_path = get_sample_tumor_path()
        resp_mri = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp_mri.status_code, 200)
        self.assertEqual(resp_mri.json().get("specialist"), "NeuroVision MRI")

        # 2. PNG image
        png_bytes = create_minimal_valid_png_bytes(100, 100)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = Path(tmp.name)
        try:
            resp_vision = self.analyze_mednexus(filepath=tmp_path, content_type="image/png")
            self.assertEqual(resp_vision.status_code, 200)
            data_vision = resp_vision.json()
            self.assertIn(data_vision.get("specialist"), ("medgemma1.5", "vision_model", "medgemma"))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_03_modality_switch_vision_to_demo(self) -> None:
        """Switch from live image review to simulated demo fixture."""
        png_bytes = create_minimal_valid_png_bytes(64, 64)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = Path(tmp.name)
        try:
            resp_vision = self.analyze_mednexus(filepath=tmp_path, content_type="image/png")
            self.assertEqual(resp_vision.status_code, 200)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        resp_demo = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        self.assertEqual(resp_demo.status_code, 200)
        data_demo = resp_demo.json()
        self.assertEqual(data_demo.get("mode"), "simulated_demo")
        self.assertIn("SIMULATED DEMO DATA", data_demo.get("provenance", ""))

    def test_04_modality_switch_demo_to_mri(self) -> None:
        """Switch from demo fixture back to live MRI specialist inference."""
        resp_demo = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        self.assertEqual(resp_demo.status_code, 200)
        self.assertEqual(resp_demo.json().get("mode"), "simulated_demo")

        tumor_path = get_sample_tumor_path()
        resp_mri = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp_mri.status_code, 200)
        data_mri = resp_mri.json()
        self.assertEqual(data_mri.get("mode"), "live_model")
        self.assertEqual(data_mri.get("provenance"), "LIVE SPECIALIST INFERENCE")

    def test_05_modality_text_plus_mri_combined(self) -> None:
        """Providing both text symptoms and MRI TIFF routes to NeuroVision with symptoms preserved."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(
            symptoms="Morning visual disturbance and progressive dizziness",
            history="No prior surgical resection",
            filepath=tumor_path,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")
        self.assertEqual(data.get("specialist_result", {}).get("predicted_class"), "tumour_suspected")

    def test_06_session_isolation_across_modalities(self) -> None:
        """Cycling through text -> MRI -> demo -> MRI produces isolated responses without state leakage."""
        tumor_path = get_sample_tumor_path()
        no_tumor_path = get_sample_no_tumor_path()

        # Step 1: Text
        r1 = self.analyze_mednexus(symptoms="Isolated migraine")
        self.assertEqual(r1.status_code, 200)
        self.assertNotIn("specialist_result", r1.json())

        # Step 2: Positive MRI
        r2 = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json().get("specialist_result", {}).get("predicted_class"), "tumour_suspected")

        # Step 3: Demo fixture
        r3 = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json().get("mode"), "simulated_demo")

        # Step 4: Negative MRI
        r4 = self.analyze_mednexus(filepath=no_tumor_path)
        self.assertEqual(r4.status_code, 200)
        self.assertEqual(r4.json().get("specialist_result", {}).get("predicted_class"), "no_tumour_detected_by_model")
        self.assertIsNone(r4.json().get("specialist_result", {}).get("segmentation_overlay"))

    # -------------------------------------------------------------------------
    # Combination 2: Health polling during active inference
    # -------------------------------------------------------------------------

    def test_07_mri_health_polling_during_predict(self) -> None:
        """Health endpoints remain responsive when prediction requests are in-flight."""
        tumor_path = get_sample_tumor_path()

        def run_predict() -> int:
            resp = self.predict_mri(tumor_path)
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future = executor.submit(run_predict)
            time.sleep(0.01)

            # Poll health while prediction is executing
            health_codes = []
            for _ in range(3):
                h_resp = http_request(f"{self.mri_url}/health", method="GET")
                health_codes.append(h_resp.status_code)
                time.sleep(0.01)

            pred_status = future.result(timeout=180 if self.is_live else 10)

        self.assertEqual(pred_status, 200)
        self.assertEqual(health_codes, [200, 200, 200])

    def test_08_mednexus_health_during_mri_analyze(self) -> None:
        """MedNexus /api/health responds with 200 during active /api/analyze routing."""
        tumor_path = get_sample_tumor_path()

        def run_analyze() -> int:
            resp = self.analyze_mednexus(filepath=tumor_path)
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future = executor.submit(run_analyze)
            time.sleep(0.01)

            h_resp = http_request(f"{self.mednexus_url}/api/health", method="GET")
            analyze_status = future.result(timeout=180 if self.is_live else 10)

        self.assertEqual(analyze_status, 200)
        self.assertEqual(h_resp.status_code, 200)

    def test_09_health_readiness_consistency_before_during_after(self) -> None:
        """System health reports ok before, during, and after inference without degradation."""
        # Before
        h1 = http_request(f"{self.mri_url}/health", method="GET")
        self.assertEqual(h1.status_code, 200)
        self.assertEqual(h1.json().get("status"), "ok")

        # Inference
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        self.assertEqual(resp.status_code, 200)

        # After
        h2 = http_request(f"{self.mri_url}/health", method="GET")
        self.assertEqual(h2.status_code, 200)
        self.assertEqual(h2.json().get("status"), "ok")

    # -------------------------------------------------------------------------
    # Combination 3: Multiple consecutive MRI slice screenings
    # -------------------------------------------------------------------------

    def test_10_consecutive_tumor_then_no_tumor(self) -> None:
        """Screen positive tumor slice, then negative slice; assert zero mask/bbox leakage."""
        tumor_path = get_sample_tumor_path()
        no_tumor_path = get_sample_no_tumor_path()

        resp_pos = self.predict_mri(tumor_path)
        self.assertEqual(resp_pos.status_code, 200)
        data_pos = resp_pos.json()
        self.assertEqual(data_pos.get("predicted_class"), "tumour_suspected")
        self.assertIsNotNone(data_pos.get("bbox"))
        self.assertIsNotNone(data_pos.get("segmentation_overlay"))

        resp_neg = self.predict_mri(no_tumor_path)
        self.assertEqual(resp_neg.status_code, 200)
        data_neg = resp_neg.json()
        self.assertEqual(data_neg.get("predicted_class"), "no_tumour_detected_by_model")
        self.assertIsNone(data_neg.get("bbox"), "Negative scan must not inherit bounding box")
        self.assertIsNone(data_neg.get("segmentation_overlay"), "Negative scan must not inherit overlay")

    def test_11_consecutive_no_tumor_then_tumor(self) -> None:
        """Screen negative slice, then positive slice; assert mask and bbox are produced normally."""
        tumor_path = get_sample_tumor_path()
        no_tumor_path = get_sample_no_tumor_path()

        resp_neg = self.predict_mri(no_tumor_path)
        self.assertEqual(resp_neg.status_code, 200)
        self.assertEqual(resp_neg.json().get("predicted_class"), "no_tumour_detected_by_model")

        resp_pos = self.predict_mri(tumor_path)
        self.assertEqual(resp_pos.status_code, 200)
        data_pos = resp_pos.json()
        self.assertEqual(data_pos.get("predicted_class"), "tumour_suspected")
        self.assertIsNotNone(data_pos.get("bbox"))
        self.assertIsNotNone(data_pos.get("segmentation_overlay"))

    def test_12_consecutive_alternating_ten_slices(self) -> None:
        """Alternating 10 sequential predictions maintains 100% classification fidelity."""
        tumor_path = get_sample_tumor_path()
        no_tumor_path = get_sample_no_tumor_path()

        expected = []
        actual = []
        for i in range(10):
            if i % 2 == 0:
                expected.append("tumour_suspected")
                resp = self.predict_mri(tumor_path, custom_filename=f"cycle_{i}_tumor.tif")
            else:
                expected.append("no_tumour_detected_by_model")
                resp = self.predict_mri(no_tumor_path, custom_filename=f"cycle_{i}_no_tumor.tif")
            self.assertEqual(resp.status_code, 200)
            actual.append(resp.json().get("predicted_class"))

        self.assertEqual(actual, expected)

    def test_13_consecutive_bounding_box_stability(self) -> None:
        """Multiple successive predictions on same positive slice produce consistent bounding box dimensions."""
        tumor_path = get_sample_tumor_path()
        bboxes = []
        for _ in range(3):
            resp = self.predict_mri(tumor_path)
            self.assertEqual(resp.status_code, 200)
            bboxes.append(resp.json().get("bbox"))

        self.assertEqual(bboxes[0], bboxes[1])
        self.assertEqual(bboxes[1], bboxes[2])


if __name__ == "__main__":
    unittest.main()
