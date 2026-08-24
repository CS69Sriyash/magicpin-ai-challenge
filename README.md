# magicpin AI Challenge

In-progress Python/FastAPI submission for the magicpin Vera AI Challenge.

This repository currently contains an in-progress Phase 1 + Phase 2 scaffold:

- flexible Pydantic v2 schemas for the four-context framework
- FastAPI app with health and metadata endpoints
- idempotent `/v1/context` ingestion with version checks and extra-field retention
- LLM provider abstraction for local Ollama development and Groq production config
- draft `/v1/tick` proactive messaging flow
- draft `/v1/reply` reactive conversation flow
- local judge simulator and seed datasets for development

The response-generation flow is still being iterated on. This snapshot is meant
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

Run the local judge simulator against that bot:

```bash
BOT_URL=http://127.0.0.1:8080 python judge_simulator.py
```

For the real judge, `localhost` is not enough: deploy the bot and set
`BOT_URL` to the public HTTPS endpoint that the judge can reach.

For local LLM composition, run Ollama with the default model:

```bash
ollama pull qwen2:7b
ollama serve
```

Optional environment variables:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:7b
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
```

## Current API Surface

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

## Current Ingestion Behavior

The context schemas intentionally allow extra fields so category-specific or
judge-provided data can survive validation and still be available for later LLM
composition.

For repeated context pushes:

- newer versions replace stored context
- same-version pushes return a no-op success response
- older versions return `409 stale_version`

## Current Composition Behavior

`/v1/tick` resolves available triggers into merchant/category/customer context,
calls the active LLM provider, and returns proactive actions.

`/v1/reply` keeps lightweight in-memory conversation history, handles basic
deterministic stop/repeated-auto-reply cases, and otherwise delegates response
selection to the composer.

## Notes

`judge_simulator.py` is configured for local development by default. Update the
configuration block at the top of that file before running a full simulation
against another model/provider.
