# MedNexus AI — Hackathon Demo Runbook

## One-line pitch

**MedNexus routes each piece of health evidence to the local specialist best suited to review it, then keeps the source, model output, and uncertainty visible together.**

Do not describe this prototype as diagnosing patients or autonomously spawning AI agents. It routes evidence by modality to configured model services and presents their separate outputs.

## Before the judges arrive

- Start Ollama and confirm Qwen and MedGemma are available.
- Start the NeuroVision Flask service with JSON routes at `/api/predict`, `/api/explain`, and `/health`.
- Start MedNexus at `http://127.0.0.1:8000`.
- Refresh the service indicators and confirm NeuroVision is marked ready.
- Keep a public/sample MRI slice ready, such as `neurovision-mri/static/uploads/sample_tumor.tif`.
- Avoid real patient information. Use the demo's clearly labelled fixture when the live specialists are unavailable.

## Judge walkthrough (about 90 seconds, plus automatic XAI time)

1. **Frame the problem (8 sec):** “Health evidence arrives in different forms. A text model should not improvise an MRI segmentation, and an MRI model should not interpret a lab report.”
2. **Show the route (12 sec):** Choose a TIFF slice. Point out that MedNexus routes it to NeuroVision, its dedicated 2D MRI specialist, rather than sending it to a general vision model.
3. **Show the evidence (20 sec):** Submit the sample and show the specialist label, slice-level result, explicitly uncalibrated model score, segmentation overlay, and any reported region coordinates. Explain that the overlay comes from the MRI segmentation model.
4. **Explore model views (20 sec):** After the MRI result, the original slice, classifier Grad-CAM, segmentation-ensemble agreement, and MC-dropout variance load automatically. Click a view to open its plain-language explanation. Describe these as exploratory model visualizations, not clinical explanations or measures of diagnostic certainty. This extra pass may take a little while.
5. **Show the boundary (10 sec):** Point to the visible prototype notice and scope. Say: “This is a research result on one 2D slice, not a patient-level diagnosis or a clinically validated probability.”
6. **Show the wider workflow (10 sec):** If time and services allow, submit a short report or text example to show that text is routed to Qwen and ordinary medical images to MedGemma. Each specialist remains identified in the result.
7. **Close (5 sec):** “The useful part is the evidence handoff: the right specialist, a visible result, and clear limits in one review workspace.”

## If a live model is unavailable

Use **See an example** and explicitly call it a simulated fixture. Do not imply that the fixture came from a live model. If NeuroVision is unavailable, explain the separate local service requirement and show a saved, clearly identified sample visualization only as an illustration—not as a new inference.

## Likely judge questions

**What is novel here?** A modality-aware local evidence workflow: MRI slices go to a segmentation specialist, ordinary images to a vision-language model, and reports/context to a text model. MedNexus keeps their outputs and provenance distinct in one review flow.

**Is it clinically validated?** No. It is a hackathon research prototype. The MRI model operates on 2D slices and its score is not calibrated as a clinical probability.

**Why local models?** The prototype demonstrates a privacy-conscious architecture and can run without sending evidence to a hosted inference API; local deployment still depends on the operator's machine and service configuration.

**Does it read a full MRI study?** Not currently. NeuroVision handles an individual TIFF slice from its research dataset, not a full DICOM series or patient-level interpretation.
