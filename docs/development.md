# Development and local runbook

## Prerequisites

- Python versions compatible with the two `requirements.txt` files.
- NVIDIA GPU, current driver, and CUDA-enabled PyTorch for NeuroVision. CPU inference is intentionally disabled in the current service.
- Ollama running locally with the model tags configured for the MedNexus text and image routes.
- NeuroVision checkpoints stored at the filenames expected under `neurovision-mri/checkpoints/`. Weights are not part of this repository.

## MedNexus service

```bash
cd mednexus
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`. Edit `.env` for model tags and service URLs. Keep `.env` local and never commit it.

## NeuroVision MRI service

```bash
cd neurovision-mri
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Start with a working CUDA-enabled PyTorch installation before installing or replacing Torch through the requirements file. Verify `nvidia-smi`, PyTorch CUDA visibility, the configured device, checkpoint presence, and `GET http://127.0.0.1:8080/health`. The app intentionally raises if CUDA is unavailable; it will not run the model on CPU.

## Ollama

Start Ollama using its normal platform service, pull the model tags selected in `mednexus/.env`, then check `GET http://127.0.0.1:8000/api/health`. The exact tags and memory requirements depend on the local hardware and current configuration.

## Ports and health checks

| Service | Default port | Check |
|---|---:|---|
| MedNexus | 8000 | `/api/health` |
| NeuroVision | 8080 | `/health` |
| Ollama | 11434 | Ollama local API/model list |

Run services in separate terminals. Keep the three URLs consistent between `.env` and service startup. A health response confirms connectivity only; it does not validate model performance.

## Tests

Fixture-based checks live under `tests/e2e/`. They exercise routing, output structure, safety boundaries, and representative interactions without requiring the complete MRI dataset. Use the documented project commands and inspect their environment assumptions before running. A passing fixture suite does not establish clinical validity.

## Data and weights

The source repository intentionally omits `kaggle_3m/`, checkpoint files, uploaded scans, and virtual environments. Obtain dataset and weight files only from their authorized source, follow their license and access terms, and verify checksums/provenance before use. Never commit real patient data or access tokens.
