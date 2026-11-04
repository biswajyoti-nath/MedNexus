# NeuroVision JSON bridge

The MRI Flask app serves its own HTML at `/`. MedNexus connects to a JSON route at `POST /api/predict` with a multipart field named `file`.

1. Open the existing MRI project's `app.py`.
2. Add `jsonify` to the Flask imports and allow both `.tif` and `.tiff` in `ALLOWED_EXTENSIONS`.
3. Paste the snippet from `mri_json_endpoint_snippet.py` after the existing `index()` route and before the `if __name__ == '__main__':` block.
4. Restart the MRI app. It should continue to serve its original UI at `/` and expose JSON at `/api/predict`.
5. Run MedNexus with `MRI_SERVICE_URL=http://127.0.0.1:8080/api/predict`.

The current NeuroVision service also exposes `POST /api/explain` for Grad-CAM, segmentation-agreement, and MC-dropout variance visualizations. After a TIFF review returns, MedNexus automatically queues this additional pass and displays the views when ready. The response uses temporary image data URLs; uploaded slices and generated maps are removed after the response. This route is separate from `/api/predict`, whose JSON contract remains unchanged. Explainability generation is a potentially slower GPU inference pass and reuses the MRI process's loaded models.

The JSON route returns a slice-level prediction, an explicitly uncalibrated model score, segmentation status, bounding box, and (when a mask exists) a compact PNG overlay encoded as a data URL. MedNexus keeps those fields attached to the NeuroVision specialist result. The endpoint reports the LGG research scope and sets `clinically_validated` to `false`; it does not turn the output into a patient-level diagnosis.

The three XAI maps are experimental views of model behavior, not clinically validated explanations, lesion boundaries, or measures of clinical certainty. A missing map is shown as unavailable rather than inferred or substituted.
