"""
bot.py — magicpin Vera AI Challenge submission (Phase 1 + Phase 2 + Phase 3)

Phase 1 (unchanged): flexible Pydantic v2 context schemas, in-memory
CONTEXT_STORE, /v1/healthz, /v1/metadata, /v1/context, idempotency logic.

Phase 2 (unchanged): LLMProvider interface (OllamaProvider dev / GroqProvider
prod), /v1/tick, /v1/reply, conversation state tracking.

Phase 3 (this update) — EngagementComposer rewritten for eval tuning:
  1. Dynamic category routing — system prompt's tone directive is built
     per CategoryContext.slug rather than one fixed prompt for all verticals.
  2. Anchor-fact extraction — rather than just *telling* the LLM "don't
     hallucinate, quote exact numbers," the composer resolves the specific
     fact the trigger references (a digest item's citation, or the
     trigger's own payload numbers) and injects it as a labeled,
     already-correct block the model is instructed to quote verbatim. This
     is a stronger guardrail than a prompt instruction alone — it removes
     the retrieval step (and its hallucination risk) from the LLM's job
     entirely for the fact that most needs to be exact.
  3. CTA enforcement — every proactive message must end in exactly one
     binary/specific CTA, with a narrow, explicit exception (brief Appendix B)
     for customer-facing slot-booking replies where 2 concrete time slots
     are offered — that's one decision with two valid answers, not two CTAs.
  4. Few-shot — Appendix A (merchant-facing) and Appendix B (customer-facing)
     from challenge-brief.md are embedded verbatim as calibration examples.
  5. send_as correctness — deterministically "vera" for merchant-facing,
     "merchant_on_behalf" for customer-facing (brief §5), rather than left
     to the LLM's discretion.

Run locally:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Requires a running Ollama daemon with qwen2:7b pulled for local dev:
    ollama pull qwen2:7b && ollama serve
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Load a local .env file if one exists. `uv run` (and plain `python`/`uvicorn`
# invocations) do NOT auto-load .env files the way some tooling does — this
# is the actual, most common cause of "GROQ_API_KEY works in one terminal but
# not when launched via uv run/a process manager" reports. This call is a
# no-op if no .env is present, so it's safe in every environment including
# the judge's container where secrets are presumably injected as real env vars.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "[WARN] python-dotenv is not installed in this environment — .env "
        "files will NOT be loaded, so GROQ_API_KEY etc. must be set as real "
        "environment variables. Fix: run `uv add python-dotenv` in this "
        "project directory (this warning printed before logging is "
        "configured, hence plain print not logger)."
    )

APP_VERSION = "0.4.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vera_bot")


# =============================================================================
# PHASE 1 — Context schemas + ingestion (unchanged)
# =============================================================================


class FlexibleModel(BaseModel):
    """
    Base model that allows extra fields.
    This ensures we don't lose category-specific data (like 'delivery_orders_30d')
    when dumping the model back to a dict for the LLM.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class CategoryContext(FlexibleModel):
    slug: str
    offer_catalog: list[dict[str, Any]] = Field(default_factory=list)
    voice: dict[str, Any] = Field(default_factory=dict)
    peer_stats: dict[str, Any] = Field(default_factory=dict)
    digest: list[dict[str, Any]] = Field(default_factory=list)


class MerchantContext(FlexibleModel):
    merchant_id: str
    category_slug: str
    identity: dict[str, Any]
    subscription: dict[str, Any]
    performance: dict[str, Any]
    offers: list[dict[str, Any]] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    customer_aggregate: dict[str, Any] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)


class TriggerContext(FlexibleModel):
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int
    suppression_key: str
    expires_at: str


class CustomerContext(FlexibleModel):
    customer_id: str
    merchant_id: str
    identity: dict[str, Any]
    relationship: dict[str, Any]
    state: str
    preferences: dict[str, Any]
    consent: dict[str, Any]


SCOPE_MODELS: dict[str, type[FlexibleModel]] = {
    "category": CategoryContext,
    "merchant": MerchantContext,
    "trigger": TriggerContext,
    "customer": CustomerContext,
}


class ContextPushRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str | None = None


class ContextAcceptedResponse(BaseModel):
    accepted: Literal[True] = True
    ack_id: str
    stored_at: str


class ContextRejectedResponse(BaseModel):
    accepted: Literal[False] = False
    reason: str
    current_version: int | None = None
    details: str | None = None


app = FastAPI(title="Vera Challenge Bot", version=APP_VERSION)

_START_TIME = time.monotonic()

# (scope, context_id) -> {"version": int, "payload": dict}
CONTEXT_STORE: dict[tuple[str, str], dict[str, Any]] = {}

# conversation_id -> list of turn dicts: {"turn", "from_role", "message", "ts"}
CONVERSATIONS: dict[str, list[dict[str, Any]]] = {}

# conversation_id -> {"merchant_id", "customer_id", "category_slug", "trigger_id",
#                      "sent_bodies": set[str]}
CONVERSATION_META: dict[str, dict[str, Any]] = {}

# suppression_key -> True, once a trigger with this key has produced a send.
SENT_SUPPRESSION_KEYS: set[str] = set()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _get_context(scope: str, context_id: str | None) -> dict[str, Any] | None:
    """Look up a stored, validated payload by (scope, context_id). None if missing."""
    if not context_id:
        return None
    entry = CONTEXT_STORE.get((scope, context_id))
    return entry["payload"] if entry else None


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _context_id in CONTEXT_STORE:
        counts[scope] = counts.get(scope, 0) + 1

    return {
        "status": "ok",
        "uptime_seconds": int(time.monotonic() - _START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    # Introspect the actual live provider/model rather than a hardcoded
    # string — this used to always say "qwen2:7b (via Ollama)" regardless
    # of LLM_PROVIDER, which is a real misrepresentation once Groq is
    # actually the one composing every message.
    provider = COMPOSER.llm
    model_name = getattr(provider, "model", "unknown")
    return {
        "team_name": "Team Sriyash",
        "team_members": ["Sriyash"],
        "model": f"{model_name} (via {provider.name})",
        "approach": (
            "Flexible-schema ingestion + category-routed EngagementComposer "
            "with anchor-fact extraction (fights hallucination on specific "
            "numbers/citations) and enforced single-CTA structure, behind a "
            "swappable LLMProvider interface; local Ollama for dev, Groq for "
            "production."
        ),
        "version": app.version,
        "submitted_at": _utc_now_iso(),
    }


@app.post("/v1/context")
async def push_context(request: ContextPushRequest) -> JSONResponse:
    key = (request.scope, request.context_id)
    existing = CONTEXT_STORE.get(key)

    if existing is not None and request.version < existing["version"]:
        body = ContextRejectedResponse(
            reason="stale_version",
            current_version=existing["version"],
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=body.model_dump(exclude_none=True),
        )

    model_cls = SCOPE_MODELS.get(request.scope)
    if not model_cls:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_scope"},
        )

    try:
        validated = model_cls.model_validate(request.payload)
    except ValidationError as exc:
        logger.warning(
            "context push rejected (invalid_payload) | scope=%s context_id=%s",
            request.scope,
            request.context_id,
        )
        body = ContextRejectedResponse(reason="invalid_payload", details=str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=body.model_dump(exclude_none=True),
        )

    if existing is not None and request.version == existing["version"]:
        ack_id = f"ack_{request.context_id}_v{request.version}_noop"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"accepted": True, "ack_id": ack_id, "stored_at": _utc_now_iso()},
        )

    CONTEXT_STORE[key] = {
        "version": request.version,
        "payload": validated.model_dump(by_alias=True),
    }

    ack_id = f"ack_{request.context_id}_v{request.version}_{uuid.uuid4().hex[:8]}"
    body = ContextAcceptedResponse(ack_id=ack_id, stored_at=_utc_now_iso())
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())


# =============================================================================
# PHASE 2 — LLM provider interface (unchanged)
# =============================================================================


class LLMProvider(ABC):
    """
    Every backend (local or commercial) implements this one method.
    EngagementComposer depends only on this contract — swapping Ollama for
    Groq/OpenAI later is an env var change (LLM_PROVIDER), not a code change.
    """

    name: str = "base"

    @abstractmethod
    async def compose(self, prompt: str, system: str) -> str:
        """Return a single raw text completion. Raise LLMError on failure."""
        raise NotImplementedError


class LLMError(RuntimeError):
    """Raised for any provider failure: timeout, non-200, malformed response."""


class OllamaProvider(LLMProvider):
    """Local inference via the Ollama daemon — Phase 2/3 dev/iteration only."""

    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2:7b")
        self.timeout_seconds = timeout_seconds

    async def compose(self, prompt: str, system: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(f"ollama timeout after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama HTTP error: {exc}") from exc

        text = data.get("response")
        if not text:
            raise LLMError(f"ollama malformed response: {data}")
        return text


class _TokenRateLimiter:
    """
    Rolling 60s token-bucket limiter. Instead of guessing a flat inter-request
    sleep, this tracks actual (estimated) tokens spent in the trailing minute
    and only blocks a caller when the budget would genuinely be exceeded —
    so a burst of small requests isn't penalized the same as a burst of large
    ones. Shared across every call made through one provider instance (the
    process uses a single COMPOSER/provider singleton), so it correctly
    serializes both /v1/tick and /v1/reply traffic against the same quota.
    """

    def __init__(self, tokens_per_minute: int, safety_margin: float = 0.85) -> None:
        # Target a bit under the stated quota — Groq's window boundary
        # isn't guaranteed to align exactly with our own clock, so leaving
        # ~15% headroom avoids boundary-edge 429s.
        self.budget = max(1, int(tokens_per_minute * safety_margin))
        # Each entry is a *mutable* [timestamp, tokens] pair (not a tuple) so
        # acquire() can hand back a live reference that compose() later
        # corrects with the real usage Groq reports — otherwise every
        # reservation stays pinned at its conservative worst-case estimate
        # for the full 60s window even when the actual call used far fewer
        # tokens, causing us to self-throttle harder than Groq's own quota
        # would actually require.
        self._usage: deque[list[float | int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> list[float | int]:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._usage and now - self._usage[0][0] > 60:
                    self._usage.popleft()
                used = sum(tokens for _, tokens in self._usage)
                if used + estimated_tokens <= self.budget:
                    entry: list[float | int] = [now, estimated_tokens]
                    self._usage.append(entry)
                    return entry
                sleep_for = max(0.1, 60 - (now - self._usage[0][0]) + 0.05)
                logger.info(
                    "rate limiter: %d/%d used, pacing %.1fs before next call",
                    used,
                    self.budget,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)

    @staticmethod
    def adjust(entry: list[float | int], actual_tokens: int) -> None:
        """Replace a reservation's worst-case estimate with the real usage
        the provider reported, so subsequent budget checks reflect reality."""
        entry[1] = actual_tokens


class GroqProvider(LLMProvider):
    """
    Production swap target — sub-second inference. Paces itself against
    Groq's per-model TPM quota internally, so callers (tick/reply) don't
    need to know or care about rate limiting — they just await compose().
    """

    name = "groq"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        tokens_per_minute: int | None = None,
        max_tokens: int = 300,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        self.model = self._normalize_model(
            model or os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        )
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        tpm = tokens_per_minute or int(os.getenv("GROQ_TPM_LIMIT", "8000"))
        self._rate_limiter = _TokenRateLimiter(tpm)
        if not self.api_key:
            logger.error(
                "GroqProvider initialized with an EMPTY GROQ_API_KEY. Every call will "
                "401 until this is set. If you're launching via `uv run`, note it does "
                "NOT auto-load a .env file on its own — this file calls load_dotenv() "
                "at import time, so confirm a .env with GROQ_API_KEY=... exists in the "
                "working directory `uv run` is invoked from, or export the var in the "
                "same shell before running, or use `uv run --env-file .env ...`."
            )

    @staticmethod
    def _normalize_model(model: str) -> str:
        if model == "qwen3.6-27b":
            return "qwen/qwen3.6-27b"
        return model

    @staticmethod
    def _estimate_tokens(system: str, prompt: str, max_tokens: int) -> int:
        # Conservative ~4 chars/token heuristic for input, plus the full
        # completion budget as a worst-case reservation (we reserve before
        # we know the actual completion length, so we must assume the max).
        input_chars = len(system) + len(prompt)
        return (input_chars // 4) + max_tokens

    async def compose(self, prompt: str, system: str) -> str:
        estimated = self._estimate_tokens(system, prompt, self.max_tokens)
        reservation = await self._rate_limiter.acquire(estimated)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": self.max_tokens,
            # CRITICAL for qwen3 models on Groq: without this, the model
            # spends part (sometimes most) of max_tokens on hidden reasoning
            # before ever writing the actual answer, since thinking tokens
            # are billed against the same completion budget. With a tight
            # max_tokens=300, that reasoning overhead was truncating the
            # real JSON answer down to a garbage fragment. "none" disables
            # reasoning entirely for qwen3 models (Groq API reference).
            "reasoning_effort": "none",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(f"groq timeout after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            body_snippet = ""
            if exc.response is not None:
                body_snippet = f" body={exc.response.text[:300]!r}"
            raise LLMError(f"groq HTTP error: {exc}{body_snippet}") from exc

        # Correct our rate-limiter reservation from "worst-case estimate" to
        # what Groq actually reports — otherwise every call stays pinned at
        # its conservative upper bound for the full 60s window even when the
        # real completion was much shorter (likely now that reasoning is off).
        usage = data.get("usage") or {}
        actual_total = usage.get("total_tokens")
        if isinstance(actual_total, int) and actual_total > 0:
            self._rate_limiter.adjust(reservation, actual_total)

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"groq malformed response: {data}") from exc


class GeminiProvider(LLMProvider):
    """
    Second production option — Google's free tier has dramatically more TPM
    headroom than Groq's free tier (order of 250K-1M TPM vs. Groq's 8,000),
    which was our actual bottleneck. Two things worth knowing before relying
    on this:

    1. Gemini 2.5 Flash defaults to dynamic "thinking" (thinking_budget=-1)
       on Google's direct API — the same class of issue as qwen3's
       reasoning_effort on Groq, where hidden reasoning tokens can eat into
       the visible answer. We explicitly set thinking_budget=0 to disable it.
    2. There's a documented Gemini 2.5 Flash bug where finish_reason reports
       "STOP" (i.e. "completed normally") even when the output was silently
       truncated by the thinking/token-budget interaction — no error signal
       at all. We defend against this with a minimum-length sanity check on
       the returned text rather than trusting finish_reason.
    3. Free-tier RPM (5-30/min depending on model) is comparable to or worse
       than Groq's — TPM headroom doesn't mean unlimited throughput. Within
       one /v1/tick burst this can become the new binding constraint; our
       deadline-aware tick loop already handles that by returning partial
       results and retrying the rest on a later tick, so this isn't fatal,
       just worth knowing.

    Exact free-tier RPM/TPM figures vary by model and change over time —
    defaults below are deliberately conservative; verify current numbers on
    your own Google AI Studio dashboard and override via env vars if needed.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_output_tokens: int = 300,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        )
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        rpm = requests_per_minute or int(os.getenv("GEMINI_RPM_LIMIT", "10"))
        tpm = tokens_per_minute or int(os.getenv("GEMINI_TPM_LIMIT", "250000"))
        # Two independent rolling-window limiters: RPM (cost=1 per call) and
        # TPM (cost=estimated tokens per call) reuse the same tested limiter
        # class — a "requests per minute" cap is just a token-bucket where
        # every request costs exactly 1.
        self._rpm_limiter = _TokenRateLimiter(rpm, safety_margin=0.9)
        self._tpm_limiter = _TokenRateLimiter(tpm, safety_margin=0.85)
        if not self.api_key:
            logger.error(
                "GeminiProvider initialized with an EMPTY GEMINI_API_KEY. Every "
                "call will fail until this is set (via .env or a real env var)."
            )

    @staticmethod
    def _estimate_tokens(system: str, prompt: str, max_output_tokens: int) -> int:
        input_chars = len(system) + len(prompt)
        return (input_chars // 4) + max_output_tokens

    async def compose(self, prompt: str, system: str) -> str:
        estimated = self._estimate_tokens(system, prompt, self.max_output_tokens)
        await self._rpm_limiter.acquire(1)
        tpm_reservation = await self._tpm_limiter.acquire(estimated)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": self.max_output_tokens,
                # Disable dynamic thinking — see class docstring. Without
                # this, hidden reasoning tokens can consume the output
                # budget the same way qwen3's reasoning_effort did on Groq.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(f"gemini timeout after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            body_snippet = ""
            if exc.response is not None:
                body_snippet = f" body={exc.response.text[:300]!r}"
            raise LLMError(f"gemini HTTP error: {exc}{body_snippet}") from exc

        usage = data.get("usageMetadata") or {}
        actual_total = usage.get("totalTokenCount")
        if isinstance(actual_total, int) and actual_total > 0:
            self._tpm_limiter.adjust(tpm_reservation, actual_total)

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"gemini malformed response: {data}") from exc

        # Defend against the documented silent-truncation bug: finish_reason
        # can report normal completion even when thinking/token-budget
        # interaction cut the real answer short. A suspiciously tiny
        # response for what should be a JSON object with a real message
        # body is treated as a failure rather than trusted at face value.
        if len(text.strip()) < 20:
            raise LLMError(
                f"gemini response suspiciously short ({len(text)} chars) — "
                f"likely silent truncation, not a real answer: {text!r}"
            )

        return text


def get_llm_provider() -> LLMProvider:
    """
    The one place that reads LLM_PROVIDER. Default stays "ollama" for local
    dev; set LLM_PROVIDER=groq or LLM_PROVIDER=gemini (in your shell or
    .env) for the scored run — this stays config-driven rather than
    hardcoded so switching providers during debugging never requires a
    source edit.
    """
    provider_key = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider_key == "groq":
        return GroqProvider()
    if provider_key == "gemini":
        return GeminiProvider()
    if provider_key != "ollama":
        logger.warning("Unknown LLM_PROVIDER=%r, falling back to ollama", provider_key)
    return OllamaProvider()


# =============================================================================
# PHASE 3 — EngagementComposer (rewritten)
# =============================================================================

# --- 1. Dynamic category routing --------------------------------------------
#
# Hardcoded tone directives per the exact 5 verticals in the dataset, plus a
# generic fallback for any category we haven't seen. This is layered ON TOP
# of (not instead of) the category's own voice.tone/vocab_allowed/vocab_taboo
# fields from CategoryContext — the hardcoded directive sets the register,
# the live context data supplies the concrete vocabulary to use/avoid.

CATEGORY_VOICE_DIRECTIVES: dict[str, str] = {
    "dentists": (
        "Clinical, peer-to-peer, respectful. Write as one professional "
        "addressing another — the way a colleague would flag a relevant "
        "case study or guideline change, not the way a salesperson pitches. "
        'Use "Dr. {name}" where the merchant\'s identity includes a name. '
        "No hype, no overclaiming, no medical guarantees."
    ),
    "salons": (
        "Warm, practical, approachable-expert. Friendly without being "
        "gushy — like a trusted stylist giving straight advice, not a "
        "promo blast."
    ),
    "restaurants": (
        "Operator-to-operator, fellow-business-owner tone. Talk footfall, "
        "covers, and margins the way one restaurateur would talk to "
        "another — practical, a little busy, no food-blogger flourishes."
    ),
    "gyms": (
        "Motivational coach tone — energetic and disciplined, but grounded "
        "in the merchant's actual numbers, not generic hype. Coach-to-member "
        "register, not drill-sergeant."
    ),
    "pharmacies": (
        "Trustworthy, precise, neighbourhood-pharmacist tone. Calm and "
        "exact — this is a regulated, health-adjacent business; precision "
        "reads as competence here more than enthusiasm does."
    ),
}
_DEFAULT_VOICE_DIRECTIVE = (
    "Match the tone, register, and vocabulary given in this category's "
    "voice profile below as closely as possible."
)


def _category_voice_directive(category_slug: str) -> str:
    return CATEGORY_VOICE_DIRECTIVES.get(category_slug, _DEFAULT_VOICE_DIRECTIVE)


# --- 4. Few-shot examples (verbatim from challenge-brief.md Appendix A/B) ---

FEW_SHOT_BLOCK = """EXAMPLES (structure only — don't reuse these facts for a different merchant/customer):
Merchant-facing (send_as=vera): "Dr. Meera, JIDA's Oct issue landed — 2,100-patient trial: 3-month fluoride recall cuts caries recurrence 38% better than 6-month, relevant to your high-risk adults. Want the abstract + a patient WhatsApp draft? — JIDA Oct 2026 p.14"
Customer-facing (send_as=merchant_on_behalf, slot-booking exception applies): "Hi Priya, Dr. Meera's clinic here — it's been 5 months, your 6-month cleaning recall is due. 2 slots open: Wed 6pm or Thu 5pm. ₹299 cleaning + free fluoride. Reply 1 for Wed, 2 for Thu.\""""


# --- 2. Anchor-fact extraction (fights hallucination) -----------------------


def _resolve_digest_anchor(
    category: dict[str, Any], trigger: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Several trigger kinds (research_digest, regulation_change, cde_opportunity)
    reference a specific category.digest[] item by id rather than carrying
    the citation inline. Resolving it here — instead of asking the LLM to
    find it in a list — removes the single highest-risk hallucination point:
    a fabricated or mismatched source citation.
    """
    payload = trigger.get("payload", {}) or {}
    item_id = payload.get("top_item_id") or payload.get("digest_item_id")
    if not item_id:
        return None
    for item in category.get("digest", []) or []:
        if item.get("id") == item_id:
            return item
    logger.warning(
        "trigger references digest item %s but it wasn't found in category.digest",
        item_id,
    )
    return None


def _format_anchor_block(category: dict[str, Any], trigger: dict[str, Any]) -> str:
    digest_anchor = _resolve_digest_anchor(category, trigger)
    if digest_anchor:
        return (
            "ANCHOR FACT — this trigger references a specific digest item. "
            "Quote the 'source' field VERBATIM (exact citation text) and use "
            "the exact numbers below. Do not paraphrase the citation or round "
            "the numbers:\n"
            f"  title: {digest_anchor.get('title')}\n"
            f"  source (quote verbatim): {digest_anchor.get('source')}\n"
            f"  summary: {digest_anchor.get('summary')}\n"
            f"  trial_n: {digest_anchor.get('trial_n')}\n"
            f"  patient_segment: {digest_anchor.get('patient_segment')}\n"
            f"  actionable: {digest_anchor.get('actionable')}\n"
            f"  date: {digest_anchor.get('date')}\n"
            f"  credits: {digest_anchor.get('credits')}"
        )

    payload = trigger.get("payload", {}) or {}
    return (
        "ANCHOR FACT — every number, percentage, date, and named entity in "
        "this trigger payload MUST appear in your message in this exact "
        "form (e.g. write '-50%' not 'a significant drop'; write '12 days' "
        "not 'soon'). Do not summarize these away:\n"
        f"  {payload}"
    )


# --- System prompt builders --------------------------------------------------

_BASE_RUBRIC_RULES = """You are Vera's composer, magicpin's AI merchant-engagement assistant. Write ONE short message. Judged 0-10 on 5 axes (challenge-brief.md §8):
1. SPECIFICITY — use the exact number/date/citation from the ANCHOR FACT block below, word-for-word / digit-for-digit. Never invent a fact not given to you.
2. CATEGORY FIT — match the CATEGORY VOICE below; never use a taboo word.
3. MERCHANT FIT — use the merchant's real name/numbers/language; never fabricate merchant data.
4. TRIGGER RELEVANCE — reference the ANCHOR FACT directly; no generic nudges.
5. ENGAGEMENT — use one lever: loss aversion, social proof, effort-externalization ("I've already drafted X"), curiosity, reciprocity, or a direct question.
Never: a generic offer when a priced catalog item exists, a buried CTA, repeating a body already sent in this conversation, or exposing internal field/jargon names to the merchant."""

_CTA_RULES = """CTA RULES: end with exactly ONE call-to-action — a single binary choice ("Reply YES...", "Reply STOP...") or one specific question. Never two questions or a list of options."""


def _build_proactive_system_prompt(
    category_slug: str, allow_multi_slot_cta: bool
) -> str:
    voice_directive = _category_voice_directive(category_slug)
    slot_note = (
        "\n\nSLOT-BOOKING EXCEPTION APPLIES to this message: the trigger "
        "includes concrete available appointment slots for a customer. You "
        "may offer up to 2 of them as a single booking decision (see CTA "
        "RULES exception)."
        if allow_multi_slot_cta
        else "\n\nThe slot-booking exception does NOT apply here — use exactly "
        "one binary CTA or one specific question, no slot lists."
    )

    return f"""{_BASE_RUBRIC_RULES}

CATEGORY VOICE for "{category_slug}": {voice_directive}
(Also honor the specific voice.tone / vocab_allowed / vocab_taboo fields
given in the CATEGORY context block below — the directive above sets the
register, the context data supplies the exact words to use or avoid.)

{_CTA_RULES}{slot_note}

{FEW_SHOT_BLOCK}

OUTPUT FORMAT — return ONLY a single JSON object, no markdown fences, no
preamble, no text outside the JSON:
{{"body": "<the message text>", "cta": "<short cta label, e.g. binary_yes_no|reply_stop|slot_pick|specific_question>", "rationale": "<1-2 sentences, internal logging only>", "send_as": "vera_or_merchant_on_behalf_will_be_set_by_the_system_ignore_this_field"}}

Never leave "body" empty — an empty body is treated as malformed and
penalized. Return nothing but the JSON object."""


def _build_reply_system_prompt(category_slug: str | None) -> str:
    voice_directive = _category_voice_directive(category_slug or "")
    return f"""{_BASE_RUBRIC_RULES}

CATEGORY VOICE for "{category_slug or "unknown"}": {voice_directive}

{_CTA_RULES}

CONVERSATION BEHAVIOR:
- If the merchant gives explicit consent/intent ("yes", "let's do it", "go
  ahead", "sounds good"), do NOT ask another qualifying question — switch
  immediately to confirming the concrete next step or action taken.
- If the merchant is hostile or asks you to stop, do not argue or re-pitch;
  apologize briefly and offer to stop, or end if they explicitly asked to
  stop. If they go off-topic, politely redirect without being dismissive.

OUTPUT FORMAT — return ONLY a single JSON object, no markdown fences, no
preamble. Exactly one of:
{{"action": "send", "body": "<message text>", "cta": "<cta label>", "rationale": "<why>"}}
{{"action": "wait", "wait_seconds": <integer>, "rationale": "<why>"}}
{{"action": "end", "rationale": "<why>"}}

Never leave "body" empty when action is "send". Return nothing but the JSON object."""


class ComposerError(RuntimeError):
    """Raised when the composer cannot produce a usable action. Callers should
    catch this and skip (tick) or fall back to a safe default (reply)."""


def _extract_json(raw_text: str) -> dict[str, Any]:
    """
    Local 7B models often wrap JSON in markdown fences or add a stray
    sentence before/after. Strip fences, then grab the first {...} block.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ComposerError(f"no JSON object found in LLM output: {raw_text[:200]!r}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ComposerError(
            f"malformed JSON from LLM: {exc}; raw={raw_text[:200]!r}"
        ) from exc


class EngagementComposer:
    """Formats the 4 contexts into a prompt, calls the LLM, validates the JSON out."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    # ---- context formatting -------------------------------------------- #

    @staticmethod
    def _format_context_block(
        category: dict[str, Any],
        merchant: dict[str, Any],
        trigger: dict[str, Any],
        customer: dict[str, Any] | None,
    ) -> str:
        identity = merchant.get("identity", {})
        performance = merchant.get("performance", {})
        voice = category.get("voice", {})
        peer_stats = category.get("peer_stats", {})
        active_offers = [
            o for o in merchant.get("offers", []) if o.get("status") == "active"
        ]

        lines = [
            _format_anchor_block(category, trigger),
            "",
            f"CATEGORY: {category.get('slug')}",
            f"  voice tone: {voice.get('tone', 'n/a')}",
            f"  vocab allowed: {voice.get('vocab_allowed', [])[:8]}",
            f"  vocab taboo (never use): {voice.get('vocab_taboo', voice.get('taboos', []))}",
            f"  peer stats: {peer_stats}",
            "",
            f"MERCHANT: {identity.get('name')} (owner: {identity.get('owner_first_name', 'unknown')})",
            f"  location: {identity.get('locality')}, {identity.get('city')}",
            f"  languages: {identity.get('languages', [])}",
            f"  performance (last {performance.get('window_days', 30)}d): "
            f"views={performance.get('views')}, calls={performance.get('calls')}, "
            f"ctr={performance.get('ctr')}, delta_7d={performance.get('delta_7d', {})}",
            f"  active offers: {[o.get('title') for o in active_offers]}",
            f"  signals: {merchant.get('signals', [])}",
            f"  recent conversation: {merchant.get('conversation_history', [])[-3:]}",
            "",
            f"TRIGGER: kind={trigger.get('kind')} urgency={trigger.get('urgency')}",
            f"  payload: {trigger.get('payload', {})}",
        ]

        if customer:
            c_identity = customer.get("identity", {})
            c_relationship = customer.get("relationship", {})
            lines += [
                "",
                f"CUSTOMER: {c_identity.get('name')} (this trigger is customer-facing)",
                f"  language pref: {c_identity.get('language_pref')}",
                f"  relationship: state={customer.get('state')}, "
                f"visits_total={c_relationship.get('visits_total')}, "
                f"last_visit={c_relationship.get('last_visit')}",
            ]

        return "\n".join(lines)

    # ---- proactive (tick) ------------------------------------------------ #

    async def compose_proactive(
        self,
        category: dict[str, Any],
        merchant: dict[str, Any],
        trigger: dict[str, Any],
        customer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        allow_multi_slot_cta = bool(
            customer and (trigger.get("payload", {}) or {}).get("available_slots")
        )
        system_prompt = _build_proactive_system_prompt(
            category.get("slug", "unknown"), allow_multi_slot_cta
        )
        context_block = self._format_context_block(
            category, merchant, trigger, customer
        )
        prompt = (
            "Compose ONE proactive message using the context below. "
            "Follow the OUTPUT FORMAT exactly.\n\n"
            f"{context_block}"
        )
        try:
            raw = await self.llm.compose(prompt, system_prompt)
        except LLMError as exc:
            raise ComposerError(f"LLM call failed: {exc}") from exc

        parsed = _extract_json(raw)
        body = (parsed.get("body") or "").strip()
        if not body:
            raise ComposerError("LLM returned an empty body for a proactive message")

        # send_as is a structural fact (brief §5), not a creative choice —
        # determined deterministically from whether this is customer-facing,
        # never trusted to the model's own JSON output.
        send_as = "merchant_on_behalf" if customer else "vera"

        return {
            "body": body,
            "cta": parsed.get("cta") or "open_ended",
            "rationale": parsed.get("rationale") or "",
            "send_as": send_as,
        }

    # ---- reactive (reply) ------------------------------------------------- #

    @staticmethod
    def _detect_deterministic_signal(
        message: str, history: list[dict[str, Any]]
    ) -> str | None:
        """
        Fast, deterministic overrides for cases where relying on a local 7B
        model's judgment is too risky given the operational penalty for a
        malformed/empty response. Returns a signal name or None to fall
        through to the LLM.
        """
        normalized = message.strip().lower()

        explicit_stop_phrases = [
            "stop messaging",
            "unsubscribe",
            "leave me alone",
            "stop contacting",
            "remove me",
            "do not message",
            "don't message",
        ]
        if any(p in normalized for p in explicit_stop_phrases):
            return "explicit_stop"

        prior_incoming = [
            t["message"].strip().lower()
            for t in history
            if t.get("from_role") not in ("vera", "bot")
        ]
        repeat_count = sum(1 for m in prior_incoming if m == normalized)
        if repeat_count >= 1:
            return "repeated_auto_reply"

        return None

    async def compose_reply(
        self,
        *,
        category: dict[str, Any] | None,
        merchant: dict[str, Any] | None,
        customer: dict[str, Any] | None,
        history: list[dict[str, Any]],
        message: str,
        turn_number: int,
    ) -> dict[str, Any]:
        signal = self._detect_deterministic_signal(message, history)

        if signal == "explicit_stop":
            logger.info("reply: deterministic end (explicit_stop)")
            return {
                "action": "end",
                "rationale": "Merchant explicitly asked to stop messaging.",
            }

        if signal == "repeated_auto_reply":
            logger.info("reply: deterministic end (repeated_auto_reply)")
            return {
                "action": "end",
                "rationale": "Same canned reply seen twice — likely an unattended auto-responder; exiting gracefully.",
            }

        identity = (merchant or {}).get("identity", {})
        category_slug = (merchant or {}).get("category_slug") or (category or {}).get(
            "slug"
        )
        system_prompt = _build_reply_system_prompt(category_slug)
        history_lines = "\n".join(
            f"  [{t['from_role']}] {t['message']}" for t in history[-6:]
        )

        hints = []
        intent_phrases = [
            "let's do it",
            "lets do it",
            "go ahead",
            "sounds good",
            "ok proceed",
            "yes please",
            "do it",
        ]
        if any(p in message.lower() for p in intent_phrases):
            hints.append(
                "The merchant just gave explicit consent. Do NOT ask another "
                "qualifying question — action == 'send' confirming the concrete next step."
            )

        prompt = (
            "Decide the next action for this ongoing conversation. Follow the "
            "OUTPUT FORMAT exactly (action: send | wait | end).\n\n"
            f"MERCHANT: {identity.get('name', 'unknown')} "
            f"(languages: {identity.get('languages', [])})\n\n"
            f"CONVERSATION SO FAR (turn {turn_number}):\n{history_lines}\n\n"
            f'LATEST MESSAGE FROM THEM: "{message}"\n\n'
            + ("\n".join(hints) + "\n\n" if hints else "")
        )

        try:
            raw = await self.llm.compose(prompt, system_prompt)
            parsed = _extract_json(raw)
        except (LLMError, ComposerError) as exc:
            logger.warning("compose_reply falling back to 'wait': %s", exc)
            return {
                "action": "wait",
                "wait_seconds": 1800,
                "rationale": "Composer unavailable this turn; backing off rather than risking a malformed send.",
            }

        action = parsed.get("action")
        if action == "send":
            reply_body = (parsed.get("body") or "").strip()
            if not reply_body:
                logger.warning(
                    "compose_reply got action=send with empty body; falling back to wait"
                )
                return {
                    "action": "wait",
                    "wait_seconds": 900,
                    "rationale": "Model returned an empty body; avoiding a malformed send.",
                }
            return {
                "action": "send",
                "body": reply_body,
                "cta": parsed.get("cta") or "open_ended",
                "rationale": parsed.get("rationale") or "",
            }
        if action == "wait":
            return {
                "action": "wait",
                "wait_seconds": int(parsed.get("wait_seconds") or 1800),
                "rationale": parsed.get("rationale") or "",
            }
        if action == "end":
            return {"action": "end", "rationale": parsed.get("rationale") or ""}

        logger.warning(
            "compose_reply got unrecognized action=%r; falling back to wait", action
        )
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Unrecognized action from model; backing off safely.",
        }


COMPOSER = EngagementComposer(get_llm_provider())


# =============================================================================
# PHASE 2 — POST /v1/tick (unchanged)
# =============================================================================


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


MAX_ACTIONS_PER_TICK = 20


async def _compose_action_for_trigger(trigger_id: str) -> dict[str, Any] | None:
    trigger = _get_context("trigger", trigger_id)
    if not trigger:
        logger.info("tick: trigger %s not found in CONTEXT_STORE, skipping", trigger_id)
        return None

    suppression_key = trigger.get("suppression_key", "")
    if suppression_key and suppression_key in SENT_SUPPRESSION_KEYS:
        logger.info(
            "tick: trigger %s already sent (suppression_key=%s), skipping",
            trigger_id,
            suppression_key,
        )
        return None

    merchant_id = trigger.get("merchant_id")
    merchant = _get_context("merchant", merchant_id)
    if not merchant:
        logger.info(
            "tick: merchant %s not found for trigger %s, skipping",
            merchant_id,
            trigger_id,
        )
        return None

    category = _get_context("category", merchant.get("category_slug"))
    if not category:
        logger.info(
            "tick: category %s not found for merchant %s, skipping",
            merchant.get("category_slug"),
            merchant_id,
        )
        return None

    customer = None
    customer_id = trigger.get("customer_id")
    if trigger.get("scope") == "customer" and customer_id:
        customer = _get_context("customer", customer_id)

    try:
        composed = await COMPOSER.compose_proactive(
            category, merchant, trigger, customer
        )
    except ComposerError as exc:
        logger.warning("tick: composer failed for trigger %s: %s", trigger_id, exc)
        return None

    conversation_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}"
    CONVERSATION_META[conversation_id] = {
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "category_slug": merchant.get("category_slug"),
        "trigger_id": trigger_id,
        "sent_bodies": {composed["body"]},
    }
    CONVERSATIONS[conversation_id] = [
        {
            "turn": 1,
            "from_role": "vera",
            "message": composed["body"],
            "ts": _utc_now_iso(),
        }
    ]
    if suppression_key:
        SENT_SUPPRESSION_KEYS.add(suppression_key)

    identity = merchant.get("identity", {})
    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed["send_as"],
        "trigger_id": trigger_id,
        "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
        "template_params": [identity.get("name", ""), trigger.get("kind", "")],
        "body": composed["body"],
        "cta": composed["cta"],
        "suppression_key": suppression_key,
        "rationale": composed["rationale"],
    }


TICK_TIME_BUDGET_SECONDS = 25.0  # leave margin under the judge's 30s hard cap


@app.post("/v1/tick")
async def tick(body: TickRequest) -> dict[str, Any]:
    """
    Proactive messaging (testing-brief §2.2).

    Processes triggers one at a time against a shared deadline rather than
    `asyncio.gather` under one outer `wait_for`: with a rate-limited LLM
    provider (see GroqProvider._rate_limiter), concurrent calls would just
    queue up behind the same token budget anyway, so gather buys nothing —
    and critically, gather-inside-wait_for throws away every result
    (including ones that already succeeded) the moment the timeout fires.
    This loop instead tracks a real deadline and returns whatever it
    managed to compose the instant that deadline gets tight, so a slow
    trigger costs you that one trigger, never the whole batch. Any trigger
    not attempted this cycle simply isn't marked as sent (no suppression_key
    recorded for it), so it's fair game again on the next tick.
    """
    trigger_ids = body.available_triggers[:MAX_ACTIONS_PER_TICK]
    if not trigger_ids:
        return {"actions": []}

    deadline = time.monotonic() + TICK_TIME_BUDGET_SECONDS
    actions: list[dict[str, Any]] = []
    attempted = 0

    for trigger_id in trigger_ids:
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            logger.warning(
                "tick: time budget exhausted after %d/%d triggers attempted, "
                "returning %d partial action(s) — remaining triggers will be "
                "retried on a future tick",
                attempted,
                len(trigger_ids),
                len(actions),
            )
            break

        attempted += 1
        try:
            result = await asyncio.wait_for(
                _compose_action_for_trigger(trigger_id), timeout=remaining
            )
        except TimeoutError:
            logger.warning(
                "tick: trigger %s did not finish within the remaining tick budget "
                "(%.1fs left) — stopping here, will retry next tick",
                trigger_id,
                remaining,
            )
            break
        except Exception as exc:  # noqa: BLE001 - one bad trigger must never sink the tick
            logger.error(
                "tick: unexpected error composing trigger %s: %s", trigger_id, exc
            )
            continue

        if result is not None:
            actions.append(result)

    logger.info(
        "tick: %d/%d triggers attempted, %d produced a send",
        attempted,
        len(trigger_ids),
        len(actions),
    )
    return {"actions": actions[:MAX_ACTIONS_PER_TICK]}


# =============================================================================
# PHASE 2 — POST /v1/reply (unchanged)
# =============================================================================


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyRequest) -> dict[str, Any]:
    meta = CONVERSATION_META.get(body.conversation_id)
    merchant_id = body.merchant_id or (meta or {}).get("merchant_id")
    customer_id = body.customer_id or (meta or {}).get("customer_id")

    merchant = _get_context("merchant", merchant_id)
    category = (
        _get_context("category", merchant.get("category_slug")) if merchant else None
    )
    customer = _get_context("customer", customer_id) if customer_id else None

    history = CONVERSATIONS.setdefault(body.conversation_id, [])
    history.append(
        {
            "turn": body.turn_number,
            "from_role": body.from_role,
            "message": body.message,
            "ts": body.received_at,
        }
    )

    if meta is None:
        CONVERSATION_META[body.conversation_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "category_slug": merchant.get("category_slug") if merchant else None,
            "trigger_id": None,
            "sent_bodies": set(),
        }
        meta = CONVERSATION_META[body.conversation_id]

    result = await COMPOSER.compose_reply(
        category=category,
        merchant=merchant,
        customer=customer,
        history=history[:-1],
        message=body.message,
        turn_number=body.turn_number,
    )

    if result.get("action") == "send":
        sent_bodies: set[str] = meta.setdefault("sent_bodies", set())
        if result["body"] in sent_bodies:
            logger.warning(
                "reply: composer repeated a body verbatim in conversation %s — "
                "sending anyway, but this will incur an anti-repetition penalty",
                body.conversation_id,
            )
        sent_bodies.add(result["body"])
        history.append(
            {
                "turn": body.turn_number + 1,
                "from_role": "vera",
                "message": result["body"],
                "ts": _utc_now_iso(),
            }
        )

    logger.info(
        "reply: conversation=%s turn=%d action=%s",
        body.conversation_id,
        body.turn_number,
        result.get("action"),
    )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
