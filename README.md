# magicpin AI Challenge

In-progress Python/FastAPI submission for the magicpin Vera AI Challenge.

This repository currently contains the Phase 1 scaffold:

- flexible Pydantic v2 schemas for the four-context framework
- FastAPI app with health and metadata endpoints
- idempotent `/v1/context` ingestion with version checks and extra-field retention
- local judge simulator and seed datasets for development

The response-generation endpoints are not complete yet. This snapshot is meant
to preserve progress and make the project easy to continue from GitHub.

## Project Structure

```text
.
├── bot.py                    # FastAPI challenge bot scaffold
├── judge_simulator.py         # Local judge/testing utility
├── requirements.txt           # Python dependencies
├── dataset/                   # Seed contexts used by the simulator
├── examples/                  # Reference examples and case studies
└── challenge-*.md             # Challenge and testing briefs
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the bot locally:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/v1/healthz
```

## Current API Surface

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`

## Current Ingestion Behavior

The context schemas intentionally allow extra fields so category-specific or
judge-provided data can survive validation and still be available for later LLM
composition.

For repeated context pushes:

- newer versions replace stored context
- same-version pushes return a no-op success response
- older versions return `409 stale_version`

Planned next endpoints:

- `POST /v1/tick`
- `POST /v1/reply`

## Notes

`judge_simulator.py` is configured for local development by default. Update the
configuration block at the top of that file before running a full simulation
against another model/provider.
