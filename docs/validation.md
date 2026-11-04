# Validation status and claims guide

## What has been checked in this project

The repository contains fixture-based application tests, project training/evaluation code, and a report of one project-held-out patient-level test split. These are useful engineering and research artifacts. They do not amount to a clinical evaluation or regulatory review.

| Evidence | What it can support | What it cannot establish by itself |
|---|---|---|
| Fixture-based application checks | Expected API routing, boundary handling, and UI/API response shapes for the fixtures | Clinical correctness or safety with real patients |
| Internal patient-level test report | Project-reported performance on that documented test split | External validity, calibration, prospective benefit, or regulatory status |
| Grad-CAM | A visual association between image regions and one classifier output | Causality, lesion localization, or a clinically faithful explanation |
| Segmentation agreement | Whether model masks overlap | Whether any model is correct |
| Dropout variation | Changes across repeated stochastic passes of one segmentor | A calibrated probability or complete uncertainty estimate |
| Health endpoints | Whether a local process responds | Whether its predictions are valid |

## Evaluation work still needed before a clinical-facing study

1. Freeze and version the intended use, model, preprocessing, threshold, and outputs.
2. Reproduce patient-level splits and report per-patient and per-slice results with uncertainty intervals.
3. Evaluate on independent institutions, scanners, protocols, populations, and relevant subgroups.
4. Measure calibration, failure modes, false positives/negatives, robustness to image variation, and missing-sequence behavior.
5. Compare outputs with qualified readers and clinically relevant reference standards.
6. Test the human workflow: comprehension, automation bias, escalation, override, and time-to-review.
7. Establish privacy, security, monitoring, incident response, change control, and jurisdiction-specific regulatory review.

## Presentation language

Prefer “the model flagged this single slice for additional review” and “project-reported score”. Avoid “diagnosed”, “detected a tumour” as a confirmed fact, “98% chance”, “clinically accurate”, or “FDA/HIPAA compliant”. A prototype UI safeguard is not a substitute for the work listed above.
