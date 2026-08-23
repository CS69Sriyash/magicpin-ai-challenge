"""
bot.py — magicpin Vera AI Challenge submission (Phase 1 + Phase 2)

Phase 1 (unchanged from prior version): flexible Pydantic v2 context
schemas, in-memory CONTEXT_STORE, /v1/healthz, /v1/metadata, /v1/context.

Phase 2 (this update):
  - LLMProvider abstract interface + OllamaProvider (dev) + GroqProvider
    (prod-ready, unused until GROQ_API_KEY is set) — swap via LLM_PROVIDER
    env var, no code changes needed.
  - EngagementComposer — single system prompt built from the 5-dimension
    rubric (challenge-brief.md §8) + compulsion levers (§10) + anti-patterns
    (§11); composes proactive sends and reactive replies.
  - POST /v1/tick   — proactive messaging (challenge-testing-brief.md §2.2)
  - POST /v1/reply  — reactive messaging (challenge-testing-brief.md §2.3)

Run locally:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Requires a running Ollama daemon with qwen2:7b pulled for local dev:
    ollama pull qwen2:7b && ollama serve
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

APP_VERSION = "0.2.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vera_bot")


# =============================================================================
# PHASE 1 — Context schemas + ingestion (unchanged in behavior)
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
# Lets /v1/reply recover which merchant/category a conversation belongs to
# even though the wire schema only guarantees merchant_id/customer_id on the
# very first /v1/reply call for a conversation the bot itself started.
CONVERSATION_META: dict[str, dict[str, Any]] = {}

# suppression_key -> True, once a trigger with this key has produced a send.
# Prevents re-firing the same nudge every tick while it's still "available".
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
    return {
        "team_name": "Team Sriyash",
        "team_members": ["Sriyash"],
        "model": os.getenv("OLLAMA_MODEL", "qwen2:7b") + " (via Ollama)",
        "approach": (
            "Flexible-schema ingestion + single-prompt EngagementComposer "
            "(rubric-driven system prompt) behind a swappable LLMProvider "
            "interface; local Ollama for dev, Groq for production."
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
# PHASE 2 — LLM provider interface
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
    """
    Local inference via the Ollama daemon — used for Phase 2 dev/iteration
    against judge_simulator.py. Free, zero rate limit, but not what we'd
    submit for the scored run (too slow under 10 req/s concurrency on a
    typical container CPU).
    """

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


class GroqProvider(LLMProvider):
    """
    Production swap target — sub-second Llama 3 inference. Not used until
    LLM_PROVIDER=groq and GROQ_API_KEY are set; kept here so the swap at
    submission time is a config change, not new code.
    """

    name = "groq"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            logger.warning("GroqProvider initialized without GROQ_API_KEY set")

    async def compose(self, prompt: str, system: str) -> str:
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
            "max_tokens": 700,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as exc:
            raise LLMError(f"groq timeout after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"groq HTTP error: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"groq malformed response: {data}") from exc


def get_llm_provider() -> LLMProvider:
    """The one place that reads LLM_PROVIDER. Default: ollama (Phase 2 dev)."""
    provider_key = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider_key == "groq":
        return GroqProvider()
    if provider_key != "ollama":
        logger.warning("Unknown LLM_PROVIDER=%r, falling back to ollama", provider_key)
    return OllamaProvider()


# =============================================================================
# PHASE 2 — EngagementComposer
# =============================================================================

SYSTEM_PROMPT = """You are the composer for Vera, magicpin's AI merchant-engagement assistant.
You write ONE message at a time — either a proactive nudge to a merchant/customer,
or a reply to something they just said. You are scored by an LLM judge on 5
dimensions, each 0-10 (challenge-brief.md §8):

1. SPECIFICITY — anchor on a concrete, verifiable fact from the context provided
   (a number, a date, a headline, a peer stat, a source citation). "Increase your
   sales" or "Flat 30% off" is generic and scores low. "Haircut @ ₹99" or
   "38% lower caries recurrence — JIDA Oct 2026 p.14" is specific and scores high.
   NEVER invent a number, date, or fact that isn't in the context you were given —
   fabrication is penalized more heavily than vagueness.

2. CATEGORY FIT — match the voice/vocabulary/register for this business's category.
   Dentists: clinical, peer-to-peer, respectful (use vocab like the ones given,
   avoid words on the taboo list). Salons: warm and practical. Restaurants:
   fellow-operator tone. Gyms: energetic, coach-to-member. Pharmacies: trustworthy,
   precise. Never use a taboo word from the category's voice profile.

3. MERCHANT FIT — personalize to THIS merchant: use their real name/owner name,
   their real numbers (views, calls, offers, signals), honor their language
   preference (code-mix Hindi-English naturally if their languages include "hi"),
   and reference their actual conversation history if relevant. Never fabricate
   merchant data not present in the context.

4. TRIGGER RELEVANCE — the message must make "why now" obvious. Reference the
   specific trigger's payload data directly, not a generic "check your profile"
   nudge that could apply to anyone at any time.

5. ENGAGEMENT COMPULSION — use at least one lever from this list (challenge-brief.md §10):
   loss aversion ("you're missing X"), social proof ("N other dentists in your
   locality did Y"), effort externalization ("I've already drafted X — just say go"),
   curiosity ("want to see who?"), reciprocity, asking the merchant a direct
   question, or a single binary CTA (never multiple choices in one message).

ANTI-PATTERNS TO NEVER PRODUCE (challenge-brief.md §11):
- Generic offers when a specific service+price exists in the offer catalog.
- More than one CTA in a single message.
- A buried CTA — the ask should land in the last sentence.
- Repeating a message body you've already sent in this same conversation.

CONVERSATION BEHAVIOR (for replies, not initial sends):
- If the merchant's message looks like a canned auto-reply (generic "thanks, our
  team will get back to you" wording) and this is the FIRST time you're seeing
  it, make ONE more attempt to reach a real person with a low-friction ask.
  If the SAME auto-reply pattern repeats, stop — end the conversation gracefully.
- If the merchant gives explicit consent/intent ("yes", "let's do it", "go
  ahead", "sounds good"), do NOT ask another qualifying question — switch
  immediately to confirming the concrete next step or action taken.
- If the merchant is hostile or asks you to stop, do not argue or re-pitch;
  either apologize briefly and offer to stop, or end the conversation if they
  explicitly asked you to stop messaging them. If they go off-topic (e.g. ask
  an unrelated question), politely redirect to what you can actually help with
  without being dismissive.

OUTPUT FORMAT — you must return ONLY a single JSON object, no markdown fences,
no preamble, no explanation outside the JSON. For a PROACTIVE message:
{"body": "<the message text>", "cta": "<short cta label, e.g. open_ended|binary_yes_no|reply_stop>", "rationale": "<1-2 sentences on why this message, for internal logging only>", "send_as": "vera"}

For a REPLY, return exactly one of these three shapes:
{"action": "send", "body": "<message text>", "cta": "<cta label>", "rationale": "<why>"}
{"action": "wait", "wait_seconds": <integer>, "rationale": "<why>"}
{"action": "end", "rationale": "<why>"}

Never leave "body" empty when action is "send" — an empty body is treated as
malformed and penalized. Return nothing but the JSON object."""


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
        context_block = self._format_context_block(
            category, merchant, trigger, customer
        )
        prompt = (
            "Compose ONE proactive message using the context below. "
            "Follow the OUTPUT FORMAT for a PROACTIVE message exactly.\n\n"
            f"{context_block}"
        )
        try:
            raw = await self.llm.compose(prompt, SYSTEM_PROMPT)
        except LLMError as exc:
            raise ComposerError(f"LLM call failed: {exc}") from exc

        parsed = _extract_json(raw)
        body = (parsed.get("body") or "").strip()
        if not body:
            raise ComposerError("LLM returned an empty body for a proactive message")

        return {
            "body": body,
            "cta": parsed.get("cta") or "open_ended",
            "rationale": parsed.get("rationale") or "",
            "send_as": parsed.get("send_as") or "vera",
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

        # Same exact incoming text seen before from this conversation ->
        # likely a canned auto-reply loop. Allow the first occurrence through
        # (bot gets one attempt per challenge-brief.md Pattern B), force end
        # from the second repeat onward.
        prior_incoming = [
            t["message"].strip().lower()
            for t in history
            if t.get("from_role") not in ("vera", "bot")
        ]
        repeat_count = sum(1 for m in prior_incoming if m == normalized)
        if repeat_count >= 1:  # this exact text has already appeared once before
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
        voice = (category or {}).get("voice", {})
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
            "OUTPUT FORMAT for a REPLY exactly (action: send | wait | end).\n\n"
            f"MERCHANT: {identity.get('name', 'unknown')} "
            f"(languages: {identity.get('languages', [])})\n"
            f"CATEGORY VOICE: tone={voice.get('tone', 'n/a')}, "
            f"taboo words: {voice.get('vocab_taboo', voice.get('taboos', []))}\n\n"
            f"CONVERSATION SO FAR (turn {turn_number}):\n{history_lines}\n\n"
            f'LATEST MESSAGE FROM THEM: "{message}"\n\n'
            + ("\n".join(hints) + "\n\n" if hints else "")
        )

        try:
            raw = await self.llm.compose(prompt, SYSTEM_PROMPT)
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
# PHASE 2 — POST /v1/tick
# =============================================================================


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


MAX_ACTIONS_PER_TICK = 20


async def _compose_action_for_trigger(trigger_id: str) -> dict[str, Any] | None:
    """
    Resolve trigger -> merchant -> category (+ optional customer), call the
    composer, and shape the result into the /v1/tick action schema. Returns
    None (never raises) if anything is missing or the composer fails, so one
    bad trigger can never take down the whole tick.
    """
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


@app.post("/v1/tick")
async def tick(body: TickRequest) -> dict[str, Any]:
    """
    Proactive messaging (testing-brief §2.2). Must return within 30s even
    under a 20-trigger tick; each composition runs concurrently and the
    whole batch is wrapped in an overall timeout so a stuck LLM call can
    never blow the judge's per-call budget — worst case we return
    {"actions": []} and pick it back up next tick.
    """
    trigger_ids = body.available_triggers[:MAX_ACTIONS_PER_TICK]
    if not trigger_ids:
        return {"actions": []}

    try:
        import asyncio

        results = await asyncio.wait_for(
            asyncio.gather(
                *(_compose_action_for_trigger(tid) for tid in trigger_ids),
                return_exceptions=True,
            ),
            timeout=25.0,
        )
    except TimeoutError:
        logger.warning(
            "tick: overall batch exceeded 25s budget, returning no actions this cycle"
        )
        return {"actions": []}

    actions: list[dict[str, Any]] = []
    for tid, result in zip(trigger_ids, results):
        if isinstance(result, Exception):
            logger.error("tick: unexpected error composing trigger %s: %s", tid, result)
            continue
        if result is not None:
            actions.append(result)

    logger.info("tick: %d/%d triggers produced a send", len(actions), len(trigger_ids))
    return {"actions": actions[:MAX_ACTIONS_PER_TICK]}


# =============================================================================
# PHASE 2 — POST /v1/reply
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
    """
    Reactive messaging (testing-brief §2.3). Handles both conversations the
    bot started (via /v1/tick, so CONVERSATION_META already has merchant/
    category context) and conversations the judge starts directly against
    /v1/reply (auto_reply_hell / intent_transition / hostile replay
    scenarios) — in the latter case merchant_id arrives on the request
    itself and we bootstrap the meta on first turn.
    """
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
        history=history[:-1],  # history *before* this incoming message
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
