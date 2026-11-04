# MedNexus

**A local-first evidence review workspace connected to NeuroVision, an MRI research prototype.**

MedNexus brings an image, report, and a person's question into one review thread. It routes each input to the specialist service configured for that modality and presents the source beside a plain-language explanation. NeuroVision adds an experimental MRI slice classifier, segmentation ensemble, and automatically generated XAI views.

> **Research prototype · Not for clinical use · Not clinically validated**  
> Outputs are preliminary and unvalidated. They do not diagnose or guide treatment. Review original source material with a qualified clinician.

[Architecture](docs/architecture.md) · [Data flow and privacy](docs/data-flow-and-privacy.md) · [Model card](docs/model-card.md) · [Local setup](docs/development.md) · [Validation limits](docs/validation.md) · [Demo script](mednexus/DEMO_SCRIPT.md)

## Presentation demo

[Watch the final CodeFlare hackathon submission](output/video/MedNexus_CodeFlare_Hackathon_Submission.mp4). The video introduces the review workflow, MRI explainability views, profile and report tools, service checks, and research safeguards. The demo is illustrative and does not validate clinical performance.

## Project at a glance

| Layer | What it does | Default local address |
|---|---|---|
| **MedNexus** | FastAPI workflow API and browser interface for questions, reports, and ordinary images | `http://127.0.0.1:8000` |
| **NeuroVision MRI** | Flask service for 2D TIFF MRI slice classification, segmentation, and XAI | `http://127.0.0.1:8080` |
| **Ollama** | Local language and image model runtime selected by the MedNexus router | `http://127.0.0.1:11434` |

### What is included

- `mednexus/` — API, browser UI, configuration example, integration notes, and demo walkthrough.
- `neurovision-mri/` — MRI inference/training code, model definitions, XAI methods, service UI, and project documentation.
- `tests/` — fixture-based workflow and boundary checks.
- `docs/` — architecture, data handling, model scope, setup, and validation notes.
- `output/` — project sourcebook and presentation deck.

### Quick start

This repository does not bundle trained weights, datasets, or Python environments. Read [Local setup](docs/development.md) before starting services. NeuroVision is configured to require CUDA; it intentionally has no CPU inference fallback.

```bash
# MedNexus API and browser UI
cd mednexus
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Start Ollama with the models configured in `mednexus/.env`, and start NeuroVision separately using the instructions in [neurovision-mri/README.md](neurovision-mri/README.md). On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

### Research and data boundaries

- NeuroVision consumes **one 2D slice**, not a full MRI study or a radiology series.
- Its score is a model output, not a calibrated probability of disease.
- Grad-CAM indicates image regions associated with a classifier output; it is not causal evidence.
- Segmentation masks, agreement maps, variation maps, and boxes are model-generated visualizations, not confirmed lesion boundaries or clinical measurements.
- Project metrics are reported values from the included project report; they have not been independently re-computed or externally validated in this repository.
- The prototype has no user accounts or database. See [Data flow and privacy](docs/data-flow-and-privacy.md) for the local/session boundaries and cautions.

### What is deliberately not committed

Training datasets, model checkpoints, uploaded scans, virtual environments, `.env` files, caches, and temporary renders are excluded. See `.gitignore` and [Local setup](docs/development.md) for required local assets. Example XAI images are saved project illustrations; they are not a fresh prediction for a user.

### License

No license is currently declared. Public visibility does not grant permission to reuse the source, model artifacts, or example images. Dataset and third-party model terms must be checked separately before redistribution.
