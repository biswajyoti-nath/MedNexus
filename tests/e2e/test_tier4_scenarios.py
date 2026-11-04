"""Tier 4: Real-World Application Scenarios (>=5 Workloads).

Workloads Covered:
1. Full 90-second judge walkthrough simulation from DEMO_SCRIPT.md.
2. Suspected glioma patient workflow (intake note + axial slice screening).
3. Negative screening workflow (patient follow-up note + negative axial slice).
4. Offline / simulated contingency demonstration.
5. Diagnostic disclaimer and non-diagnostic copy compliance assertion across all responses.
"""
from __future__ import annotations

import base64
import unittest
from pathlib import Path

from tests.e2e.test_fixtures import (
    E2ETestBase,
    encode_multipart_formdata,
    get_sample_no_tumor_path,
    get_sample_tumor_path,
    http_request,
)


class TestTier4RealWorldScenarios(E2ETestBase):
    """Tier 4 end-to-end clinical workflow and judge scenario tests."""

    # -------------------------------------------------------------------------
    # Workload 1: Full 90-second judge walkthrough simulation (DEMO_SCRIPT.md)
    # -------------------------------------------------------------------------

    def test_01_judge_walkthrough_step1_check_system_readiness(self) -> None:
        """Step 1 (Before judges arrive): UI loads and health check reports system readiness."""
        # 1. UI root loads
        resp_ui = http_request(f"{self.mednexus_url}/", method="GET")
        self.assertEqual(resp_ui.status_code, 200)
        self.assertIn("text/html", resp_ui.headers.get("content-type", ""))

        # 2. Health check reports ready status
        resp_health = http_request(f"{self.mednexus_url}/api/health", method="GET")
        self.assertEqual(resp_health.status_code, 200)
        data = resp_health.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("neurovision", data)

    def test_02_judge_walkthrough_step2_route_tiff_to_neurovision(self) -> None:
        """Step 2 (Show the route): Choose TIFF slice; verify routed to NeuroVision."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")
        self.assertEqual(data.get("route", {}).get("selected_specialist"), "NeuroVision MRI")
        self.assertEqual(data.get("route", {}).get("input_modality"), "MRI TIFF slice")

    def test_03_judge_walkthrough_step3_verify_evidence_panel(self) -> None:
        """Step 3 (Show the evidence): Show specialist label, slice result, score, overlay, bbox."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        sr = data.get("specialist_result", {})
        self.assertEqual(sr.get("predicted_class"), "tumour_suspected")
        self.assertIsInstance(sr.get("model_score"), float)
        self.assertIsInstance(sr.get("bbox"), list)
        self.assertEqual(len(sr.get("bbox")), 4)

        overlay = sr.get("segmentation_overlay", "")
        self.assertTrue(overlay.startswith("data:image/png;base64,iVBORw0KGgo"))

    def test_04_judge_walkthrough_step4_verify_provenance_and_scope(self) -> None:
        """Step 4 (Show the boundary): Research result on one slice; not a patient diagnosis."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        self.assertEqual(data.get("provenance"), "LIVE SPECIALIST INFERENCE")
        self.assertIn("not validated for broad neurological diagnosis", data.get("scope", ""))
        self.assertIs(data.get("specialist_result", {}).get("clinically_validated"), False)

    def test_05_judge_walkthrough_step5_wider_workflow_text_and_vision(self) -> None:
        """Step 5 (Show wider workflow): Text reports route to text model, maintaining specialist identities."""
        resp = self.analyze_mednexus(
            symptoms="Persistent mild headache, no neurological deficit.",
            history="No prior medical history.",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data.get("specialist"), ("qwen3.5:0.8b", "text_assistant", "qwen"))
        self.assertIn("clinician review required", data.get("summary", "").lower())

    def test_06_judge_walkthrough_step6_close_and_disclaimer_coherence(self) -> None:
        """Step 6 (Close): Output synthesizes specialist handoff with visible result and clear limits."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        summary = resp.json().get("summary", "")
        self.assertIn("Specialist output: tumour_suspected", summary)
        self.assertIn("not a calibrated clinical probability", summary)
        self.assertIn("2D slice-level output, not a patient-level diagnosis", summary)

    # -------------------------------------------------------------------------
    # Workload 2: Suspected glioma patient workflow (intake note + axial slice)
    # -------------------------------------------------------------------------

    def test_07_glioma_workflow_intake_submission(self) -> None:
        """Glioma intake: 48yo male with headache and hemiparesis paired with positive MRI."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(
            symptoms="Progressive morning headaches, nausea, mild left-sided hemiparesis over 3 weeks.",
            history="No previous seizures or craniotomy. Smoker (10 pack-years).",
            filepath=tumor_path,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")

    def test_08_glioma_workflow_classification_and_score(self) -> None:
        """Glioma intake yields tumour_suspected classification with valid score metadata."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        sr = resp.json().get("specialist_result", {})
        self.assertEqual(sr.get("predicted_class"), "tumour_suspected")
        self.assertGreaterEqual(sr.get("model_score"), 0.50)
        self.assertIn("not a calibrated clinical probability", sr.get("score_type", ""))

    def test_09_glioma_workflow_segmentation_overlay_and_bbox(self) -> None:
        """Glioma intake returns segmentation overlay and localized bounding box coordinates."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        sr = resp.json().get("specialist_result", {})
        self.assertIs(sr.get("segmentation_available"), True)
        bbox = sr.get("bbox")
        self.assertIsInstance(bbox, list)
        self.assertEqual(len(bbox), 4)
        x, y, w, h = bbox
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        self.assertTrue(sr.get("segmentation_overlay", "").startswith("data:image/png;base64,"))

    def test_10_glioma_workflow_non_diagnostic_intake_summary(self) -> None:
        """Glioma intake summary communicates slice-level localized evidence with uncertainty preserved."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(
            symptoms="Morning visual disturbance and progressive dizziness",
            filepath=tumor_path,
        )
        summary = resp.json().get("summary", "")
        self.assertNotIn("We diagnose glioblastoma", summary)
        self.assertIn("slice-level output", summary)

    # -------------------------------------------------------------------------
    # Workload 3: Negative screening workflow (follow-up note + negative slice)
    # -------------------------------------------------------------------------

    def test_11_negative_screening_follow_up_submission(self) -> None:
        """Negative screening: 12-month post-op surveillance scan with negative axial slice."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.analyze_mednexus(
            symptoms="Asymptomatic; routine 12-month post-treatment follow-up.",
            history="Prior surgical resection of low-grade glioma 12 months ago; stable exam.",
            filepath=no_tumor_path,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("specialist"), "NeuroVision MRI")

    def test_12_negative_screening_result_clean(self) -> None:
        """Negative screening yields no_tumour_detected_by_model with null mask and null bbox."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.analyze_mednexus(filepath=no_tumor_path)
        sr = resp.json().get("specialist_result", {})
        self.assertEqual(sr.get("predicted_class"), "no_tumour_detected_by_model")
        self.assertIs(sr.get("segmentation_available"), False)
        self.assertIsNone(sr.get("bbox"))
        self.assertIsNone(sr.get("segmentation_overlay"))

    def test_13_negative_screening_low_model_score(self) -> None:
        """Negative screening slice yields low model score (< 0.50)."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.analyze_mednexus(filepath=no_tumor_path)
        sr = resp.json().get("specialist_result", {})
        score = sr.get("model_score")
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.50)

    def test_14_negative_screening_disclaimer_boundary(self) -> None:
        """Negative screening explicitly bounds findings to single slice without whole-brain clearance."""
        no_tumor_path = get_sample_no_tumor_path()
        resp = self.analyze_mednexus(filepath=no_tumor_path)
        summary = resp.json().get("summary", "")
        self.assertIn("Specialist output: no_tumour_detected_by_model", summary)
        self.assertIn("2D slice-level output, not a patient-level diagnosis", summary)

    # -------------------------------------------------------------------------
    # Workload 4: Offline / simulated contingency demonstration
    # -------------------------------------------------------------------------

    def test_15_offline_contingency_demo_endpoint_contract(self) -> None:
        """Offline contingency: /api/demo returns simulated fixture with clear provenance."""
        resp = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("mode"), "simulated_demo")
        self.assertEqual(data.get("provenance"), "SIMULATED DEMO DATA · NO MODEL INFERENCE WAS RUN")
        self.assertEqual(data.get("specialist"), "Demo workflow simulator")

    def test_16_offline_contingency_fixture_copy_demarcation(self) -> None:
        """Demo fixture text explicitly identifies fictional patient and placeholder finding."""
        resp = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        data = resp.json()
        summary = data.get("summary", "")
        self.assertIn("fictional adult with a fictional headache history", summary)
        self.assertIn("placeholder MRI finding", summary)
        self.assertEqual(data.get("clinical_status"), "not a diagnosis; not patient data")

    def test_17_offline_contingency_missing_mri_service_error_handling(self) -> None:
        """When MRI specialist is offline, MedNexus returns 503 with setup instructions, not 500 crash."""
        dead_port_url = "http://127.0.0.1:59999/api/predict"
        # Test direct call against fixture simulating unreachable MRI specialist
        tumor_path = get_sample_tumor_path()
        with open(tumor_path, "rb") as f:
            content = f.read()

        body, ct = encode_multipart_formdata({}, {"file": ("scan.tif", content, "image/tiff")})
        # If in fixture mode, temporarily point target to dead port
        from tests.e2e.test_fixtures import MedNexusFixtureHandler
        orig_target = MedNexusFixtureHandler.mri_target_url
        try:
            MedNexusFixtureHandler.mri_target_url = dead_port_url
            resp = http_request(f"{self.mednexus_url}/api/analyze", method="POST", headers={"Content-Type": ct}, data=body)
            self.assertIn(resp.status_code, (503, 502, 500))
            data = resp.json()
            self.assertIn("detail", data)
            self.assertIn("NeuroVision MRI specialist", data.get("detail", ""))
        finally:
            MedNexusFixtureHandler.mri_target_url = orig_target

    def test_18_offline_contingency_spa_loads_without_backend_failure(self) -> None:
        """Static SPA index / loads cleanly with doctype and complete title."""
        resp = http_request(f"{self.mednexus_url}/", method="GET")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<!doctype html>", resp.text.lower())
        self.assertIn("mednexus", resp.text.lower())

    # -------------------------------------------------------------------------
    # Workload 5: Diagnostic disclaimer and non-diagnostic copy compliance
    # -------------------------------------------------------------------------

    def test_19_disclaimer_audit_mri_predict_response(self) -> None:
        """Audit /api/predict: score_type, scope, and clinically_validated False."""
        tumor_path = get_sample_tumor_path()
        resp = self.predict_mri(tumor_path)
        data = resp.json()
        self.assertIn("not a calibrated clinical probability", data.get("score_type", ""))
        self.assertIn("Research model trained on LGG MRI slices", data.get("scope", ""))
        self.assertIs(data.get("clinically_validated"), False)

    def test_20_disclaimer_audit_mri_health_response(self) -> None:
        """Audit /health: clinically_validated False."""
        resp = http_request(f"{self.mri_url}/health", method="GET")
        data = resp.json()
        self.assertIs(data.get("clinically_validated"), False)

    def test_21_disclaimer_audit_mednexus_analyze_response(self) -> None:
        """Audit /api/analyze: summary disclaimers and not clinically validated status."""
        tumor_path = get_sample_tumor_path()
        resp = self.analyze_mednexus(filepath=tumor_path)
        data = resp.json()
        summary = data.get("summary", "")
        self.assertIn("not a calibrated clinical probability", summary)
        self.assertIn("2D slice-level output, not a patient-level diagnosis", summary)
        self.assertEqual(data.get("clinical_status"), "not clinically validated")

    def test_22_disclaimer_audit_mednexus_demo_response(self) -> None:
        """Audit /api/demo: provenance and clinical_status indicate non-diagnostic fixture."""
        resp = http_request(f"{self.mednexus_url}/api/demo", method="GET")
        data = resp.json()
        self.assertEqual(data.get("clinical_status"), "not a diagnosis; not patient data")
        self.assertEqual(data.get("provenance"), "SIMULATED DEMO DATA · NO MODEL INFERENCE WAS RUN")

    def test_23_disclaimer_audit_absence_of_clinical_slop_words(self) -> None:
        """Audit all responses: strictly zero AI slop or definitive diagnostic claims."""
        tumor_path = get_sample_tumor_path()
        resp_mri = self.predict_mri(tumor_path)
        resp_analyze = self.analyze_mednexus(filepath=tumor_path)
        resp_demo = http_request(f"{self.mednexus_url}/api/demo", method="GET")

        slop_phrases = [
            "we diagnose",
            "definitive diagnosis",
            "autonomous doctor",
            "100% accuracy",
            "confirms glioblastoma multiforme",
            "clinically approved",
            "medical certification",
        ]

        for resp in (resp_mri, resp_analyze, resp_demo):
            body_text = resp.text.lower()
            for phrase in slop_phrases:
                self.assertNotIn(
                    phrase,
                    body_text,
                    f"Forbidden clinical slop phrase '{phrase}' detected in response: {resp.text}",
                )


if __name__ == "__main__":
    unittest.main()
