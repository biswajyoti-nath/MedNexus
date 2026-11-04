# NeuroVision MRI model card

## Summary

NeuroVision is an experimental image-classification and segmentation pipeline for **individual 2D MRI slices**. The intended project use is model development, demonstration, and research discussion. It is not intended for screening, diagnosis, triage, treatment selection, or autonomous clinical decisions.

> **Not clinically validated. Not for clinical use.** Scores and maps are preliminary model outputs. A qualified clinician must interpret the original imaging and complete study.

## Input and processing

- Project data documentation describes a lower-grade glioma dataset of 110 patients and 3,929 slices.
- The training code uses a deterministic patient-level split (seed 42; configured 80/10/10), intended to keep slices from one patient within one partition.
- Inputs are resized to 256 × 256 for the configured models. The exact preprocessing, weights, threshold, and code version affect outputs.
- The classifier produces a positive/negative research flag using the configured threshold (0.5 in the current source configuration).
- If the slice is flagged positive, available segmentation models produce masks that are averaged and thresholded. The reported box is in the transformed image frame.

## Model components

| Role | Current architecture in source | Notes |
|---|---|---|
| Classification | ResNet-34, EfficientNet-B0, DenseNet-121 | Sigmoid outputs are averaged as a soft-vote ensemble. The score is not calibrated. |
| Segmentation | U-Net with ResNet-34 encoder; U-Net with EfficientNet-B0 encoder; U-Net++ with DenseNet-121 encoder | Available per-model probability maps are averaged and thresholded. |
| Classifier explanation | Grad-CAM for the ResNet-34 classifier | Association/activation view, not causal evidence or a lesion boundary. |
| Segmentation comparison | Per-model overlap/agreement view | Shows model agreement, not correctness. |
| Segmentation variation | Monte Carlo dropout over repeated passes of the first segmentor (10 passes in current source) | Exploratory variation, not calibrated uncertainty. |

## Project-reported evaluation

The included project report describes an unseen patient-level test set of 349 slices. The following values are transcribed from that report and have not been independently re-computed for this publication:

| Task/metric | Project-reported value |
|---|---:|
| Classification accuracy | 95.70% |
| Classification precision | 96.15% |
| Classification recall | 92.59% |
| Classification F1 | 94.62% |
| Classification ROC AUC | 98.15% |
| Segmentation Dice | 88.33% |
| Segmentation IoU | 79.53% |

These are internal project results, not evidence of clinical performance. The repository does not establish independent replication, external-site/scanner generalization, calibration, prospective performance, subgroup safety, reader impact, or clinical utility. The report must be read together with its dataset, split, and threshold details.

## Known limitations and foreseeable risks

- Single-slice processing loses 3D sequence and study context; a model flag can be misread as a conclusion about the full scan.
- Dataset composition and labels constrain the populations and acquisition settings represented.
- Image resizing and preprocessing can affect the apparent size and coordinates of a mask or box.
- Grad-CAM, overlap maps, and dropout variation can look persuasive without being clinically validated explanations.
- Missing local checkpoints reduce the ensemble; checkpoint provenance and compatibility must be verified.
- A model score is not a calibrated probability of disease and must not be communicated as one.

## Local runtime requirement

The current NeuroVision app is configured to require CUDA and does not fall back to CPU inference. Checkpoints are excluded from Git due to size and require a separate authorized local setup. See [development setup](development.md).
