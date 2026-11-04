# MedNexus AI (hackathon MVP)

A multimodal clinical-evidence demo shell with explicit provenance. It can call local Ollama models for text/image interpretation and an optional JSON adapter for the existing NeuroVision MRI Flask project.

> **Research/demo software only.** This is not a medical device and does not diagnose, prescribe, triage emergencies, or replace a qualified clinician. Demo fixtures are synthetic and visibly labelled as simulated.

## What works in this starter

- FastAPI backend and a responsive single-page UI (no frontend build step).
- Ollama integration for `qwen3.5:0.8b` text orchestration and `medgemma1.5` image interpretation.
- PDF/text report extraction (PDF support via PyMuPDF).
- TIFF-specific routing to NeuroVision MRI with structured findings, provenance, and an optional segmentation overlay.
- Explicit live/demo provenance in every result.
- Safe demo case for presenting the workflow before every model is installed.

## 1. Requirements

- Ubuntu 24.04, Python 3.10+
- Ollama (optional for demo mode; required for live local inference)
- Your existing NeuroVision repo/checkpoints for the MRI specialist

## 2. Setup

```bash
cd mednexus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install Ollama and pull the two models:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:0.8b
ollama pull medgemma1.5
```

Start MedNexus:

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000

## 3. GPU memory strategy

Do not keep MedGemma and the six-model NeuroVision ensemble resident together on a 6GB GPU. This starter asks Ollama to unload after each request (`keep_alive: 0`). Run one inference at a time. If memory is tight, set `OLLAMA_NUM_CTX=2048` or temporarily stop the Ollama model before MRI inference. The MRI service requires CUDA and never falls back to CPU.

## 4. Connect NeuroVision

The MRI repository's `app.py` serves HTML at `/`. MedNexus needs JSON. Follow `integration/README.md` and add the route snippet in `integration/mri_json_endpoint_snippet.py` to that Flask app. The snippet adds `/health` and `/api/predict`, returns the model's slice-level output and a compact segmentation overlay when one is produced, and cleans up its temporary upload. Use a Python environment compatible with the host OS; the included Windows virtual environment is not portable to Linux. Then set:

```env
MRI_SERVICE_URL=http://127.0.0.1:8080/api/predict
```

Start the NeuroVision Flask app in its own terminal, then start MedNexus. MedNexus sends `.tif` and `.tiff` files only to that endpoint. It does **not** silently substitute MedGemma if the MRI specialist is unavailable. Training data is not required for inference; preserve it separately if disk space or repository size matters.

## 5. Demo mode

Click **Load demo case**. Every result is marked `SIMULATED DEMO`, uses synthetic values, and is not evidence from an AI model. This is useful for demonstrating product flow while model integration is incomplete. Do not present simulated outputs as real inference.

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for a short judge-facing walkthrough and honest answers to likely questions.

## 6. Hardware note

A 6GB RTX 3050 is a tight fit for 4B multimodal inference. The quantized Ollama build, runtime, image processing, and context all need memory. Test with `nvidia-smi`; reduce context and concurrency before assuming a model will fit. This starter deliberately keeps requests sequential and unloads Ollama models after each call.
