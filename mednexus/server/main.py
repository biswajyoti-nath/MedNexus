from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import fitz
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen3.5:0.8b")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "medgemma1.5")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
MRI_SERVICE_URL = os.getenv("MRI_SERVICE_URL", "http://127.0.0.1:8080/api/predict")
MRI_EXPLAIN_URL = os.getenv("MRI_EXPLAIN_URL", MRI_SERVICE_URL.rsplit("/", 1)[0] + "/explain")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "12")) * 1024 * 1024

app = FastAPI(title="MedNexus AI", version="0.1.0")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")

SYSTEM_PROMPT = """You are the language interface for MedNexus AI, a research and hackathon clinical-evidence prototype.
You are NOT a doctor and must not diagnose, prescribe, or claim clinical validation. Do not infer a disease from symptoms alone.
Summarize only the information supplied. Separate: (1) information directly present, (2) possible questions or interpretations that need clinician review, and (3) missing information.
Never invent measurements, probabilities, imaging findings, citations, or model outputs. Do not turn a model score into a clinical probability.
For potentially urgent symptoms, advise immediate assessment by local emergency services or a qualified clinician without attempting remote triage.
Use plain language and be explicit about uncertainty. Finish with: 'Research/demo output; clinician review required.'"""


def ollama_tags() -> tuple[bool, list[str]]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return True, models
    except (requests.RequestException, ValueError):
        return False, []


def ollama_chat(model: str, messages: list[dict[str, Any]], images: list[str] | None = None) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
    }
    if images:
        payload["messages"][-1]["images"] = images
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=240)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"The configured local model ({model}) did not respond. Check Ollama and confirm that model is available, then retry.") from exc


def extract_pdf(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc).strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read this PDF. Check that it is a valid file with selectable text; scanned-PDF OCR is not enabled.") from exc


def image_to_b64(data: bytes, filename: str) -> str:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im.thumbnail((1400, 1400))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported or unreadable image: {filename}") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    ollama_up, models = ollama_tags()
    mri_up = False
    mri_error = None
    mri_device = None
    mri_gpu_name = None
    try:
        # The NeuroVision service may not expose a health route; this is only a connectivity hint.
        r = requests.get(MRI_SERVICE_URL.rsplit("/", 1)[0] + "/health", timeout=1.5)
        mri_up = r.ok
        if mri_up:
            mri_health = r.json()
            mri_device = mri_health.get("device")
            mri_gpu_name = mri_health.get("gpu_name")
    except requests.RequestException:
        mri_error = "NeuroVision is not responding. Start the MRI service on port 8080."
    return {
        "status": "ok",
        "ollama": {"connected": ollama_up, "models": models},
        "expected_models": {"text": TEXT_MODEL, "vision": VISION_MODEL},
        "neurovision": {
            "health_route_detected": mri_up,
            "endpoint": MRI_SERVICE_URL,
            "status": "responding" if mri_up else "offline",
            "device": mri_device,
            "gpu_name": mri_gpu_name,
            "note": mri_error or "Service health responds; this does not validate model quality or clinical suitability.",
        },
        "mode": "live" if ollama_up else "demo_available",
    }


@app.post("/api/explain")
async def explain_mri(file: UploadFile | None = File(None)) -> dict[str, Any]:
    """Proxy optional MRI model visualizations without changing the prediction contract."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Attach a .tif or .tiff MRI slice first.")
    filename = Path(file.filename).name
    if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(status_code=415, detail="Explainability views require a .tif or .tiff MRI slice.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    try:
        response = requests.post(
            MRI_EXPLAIN_URL,
            files={"file": (filename, content, "image/tiff")},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        raise HTTPException(status_code=503, detail="NeuroVision could not generate the optional views. Check the MRI service and its /api/explain route, then retry.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="NeuroVision is not responding on port 8080. Start the MRI service, then retry the optional views.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="NeuroVision returned an unreadable explainability response. Retry after checking the MRI service.") from exc


@app.post("/api/analyze")
async def analyze(
    symptoms: str = Form(""),
    history: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    content = b""
    filename = ""
    extracted_text = ""
    if file and file.filename:
        filename = Path(file.filename).name
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        suffix = Path(filename).suffix.lower()
        if suffix in {".tif", ".tiff"}:
            # Strict routing: TIFF MRI goes to the dedicated model, never an improvised fallback.
            try:
                res = requests.post(
                    MRI_SERVICE_URL,
                    files={"file": (filename, content, "image/tiff")},
                    data={"symptoms": symptoms, "history": history},
                    timeout=180,
                )
                res.raise_for_status()
                mri_result = res.json()
            except requests.HTTPError as exc:
                raise HTTPException(status_code=503, detail="NeuroVision rejected the MRI review. Check that its JSON bridge is enabled at /api/predict, then retry.") from exc
            except requests.RequestException as exc:
                raise HTTPException(status_code=503, detail="NeuroVision MRI is not responding on port 8080. Start the MRI service, then retry this slice.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=502, detail="NeuroVision returned an unreadable MRI response. Check the MRI service, then retry.") from exc
            predicted = mri_result.get('predicted_class', 'not supplied')
            score = mri_result.get('model_score')
            segmentation_available = bool(mri_result.get('segmentation_available'))
            bbox = mri_result.get('bbox')
            summary = (
                f"Specialist output: {predicted}\n"
                f"Model score: {score if score is not None else 'not supplied'} (not a calibrated clinical probability)\n"
                f"Segmentation: {'available' if segmentation_available else 'not reported'}\n"
                f"Bounding box: {bbox if bbox is not None else 'not supplied'}\n\n"
                "Interpret this only within the model's research scope. This is a 2D slice-level output, not a patient-level diagnosis."
            )
            return {
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
                    # Show the exact source slice without requesting a second inference pass.
                    "original_slice_preview": "data:image/jpeg;base64," + image_to_b64(content, filename),
                    "image_width": mri_result.get("image_width"),
                    "image_height": mri_result.get("image_height"),
                    "clinically_validated": bool(mri_result.get("clinically_validated", False)),
                },
                "scope": mri_result.get("scope", "2D research MRI slice model; check the model documentation"),
                "raw_result": {key: value for key, value in mri_result.items() if key != "segmentation_overlay"},
                "summary": summary,
                "clinical_status": "not clinically validated",
            }
        if suffix == ".pdf":
            extracted_text = extract_pdf(content)
            if not extracted_text:
                raise HTTPException(status_code=400, detail="No selectable text found in PDF. OCR is not enabled in this MVP.")
        elif suffix in {".txt", ".md", ".csv"}:
            extracted_text = content.decode("utf-8", errors="replace")[:30000]
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            encoded = image_to_b64(content, filename)
            prompt = f"Review this uploaded medical image as a research model. Describe only visible, supportable observations. Do not diagnose.\nSymptoms supplied: {symptoms or 'None'}\nRelevant history: {history or 'None'}"
            result = ollama_chat(VISION_MODEL, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], [encoded])
            return {
                "mode": "live_model", "specialist": VISION_MODEL,
                "provenance": "LIVE MODEL OUTPUT · NOT CLINICALLY VALIDATED",
                "summary": result, "file_name": filename,
                "scope": "General medical image interpretation; modality/task-specific validation not established.",
                "clinical_status": "not clinically validated",
            }
        else:
            raise HTTPException(status_code=415, detail="Supported files: TIFF for connected NeuroVision, PNG/JPG/WEBP for MedGemma, PDF/TXT/MD/CSV for report text.")

    user_text = f"Symptoms provided:\n{symptoms or 'Not provided'}\n\nRelevant history provided:\n{history or 'Not provided'}\n\nExtracted document text (if any):\n{extracted_text or 'No document uploaded.'}\n\nOrganize this information into: reported information, questions/missing context for clinician review, and cautious next-step discussion. Do not diagnose or prescribe."
    result = ollama_chat(TEXT_MODEL, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}])
    return {
        "mode": "live_model", "specialist": TEXT_MODEL,
        "provenance": "LIVE MODEL OUTPUT · NOT CLINICALLY VALIDATED",
        "summary": result, "file_name": filename or None,
        "document_text_extracted": bool(extracted_text),
        "clinical_status": "not clinically validated",
    }


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    return {
        "mode": "simulated_demo",
        "provenance": "SIMULATED DEMO DATA · NO MODEL INFERENCE WAS RUN",
        "specialist": "Demo workflow simulator",
        "summary": "This synthetic case demonstrates how MedNexus organizes supplied information. No real patient data or model inference is involved.\n\nReported information: The sample case includes a fictional adult with a fictional headache history and a placeholder MRI finding.\n\nQuestions for clinician review: Onset, progression, neurological examination, full imaging sequence and radiologist report would be needed.\n\nMissing context: No real laboratory results, DICOM series or verified clinical history are attached.\n\nResearch/demo output; clinician review required.",
        "scope": "Synthetic fixture for UI demonstration only.",
        "clinical_status": "not a diagnosis; not patient data",
    }
